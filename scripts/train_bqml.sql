-- ============================================================================
-- A4I 2026 - Challenge 1: Community Climate Resilience (Urban Heat)
-- City: New York (Brooklyn) [Adaptable to any campaign city]
-- Track: Hybrid — Budget Cycle (Cooling Centers) + Decade (Tree Canopy ROI)
--
-- Required Differentiator: BigQuery ML (LINEAR_REG with ML.EXPLAIN_PREDICT)
-- Predicts: temp_offset_f (degrees F above city campaign average), NOT raw temp.
-- Zero Data Leakage: strictly excludes temp_f, heat_index_f, race, and income
-- ============================================================================

-- ----------------------------------------------------------------------------
-- STEP 1: Enrich Census Tracts with Land Area, Elderly (65+) Population,
--         and Street Tree Canopy Density (NYC Tree Census 2015)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE TABLE `a4i_climate.tract_features` AS
WITH bbox AS (
  SELECT
    MIN(latitude) - 0.05 AS lat_min,
    MAX(latitude) + 0.05 AS lat_max,
    MIN(longitude) - 0.05 AS lon_min,
    MAX(longitude) + 0.05 AS lon_max,
    AVG(latitude) AS center_lat,
    AVG(longitude) AS center_lon
  FROM `a4i_climate.heat_observations`
),
geo_poly AS (
  SELECT
    geo_id,
    SAFE_CAST(internal_point_lat AS FLOAT64) AS geom_lat,
    SAFE_CAST(internal_point_lon AS FLOAT64) AS geom_lon,
    GREATEST(SAFE_DIVIDE(SAFE_CAST(area_land_meters AS FLOAT64), 1000000.0), 0.05) AS land_area_km2,
    SAFE_DIVIDE(
      SAFE_CAST(area_water_meters AS FLOAT64),
      GREATEST(SAFE_CAST(area_land_meters AS FLOAT64) + SAFE_CAST(area_water_meters AS FLOAT64), 1.0)
    ) * 100.0 AS water_area_pct,
    tract_geom
  FROM `bigquery-public-data.geo_census_tracts.us_census_tracts_national`
),
-- Section 9 bonus task: Sum the 12 ACS 65+ age columns (male_65_to_66 .. female_85_and_over)
acs_elderly AS (
  SELECT
    LPAD(geo_id, 11, '0') AS geo_id,
    (
      COALESCE(SAFE_CAST(male_65_to_66 AS FLOAT64), 0) +
      COALESCE(SAFE_CAST(male_67_to_69 AS FLOAT64), 0) +
      COALESCE(SAFE_CAST(male_70_to_74 AS FLOAT64), 0) +
      COALESCE(SAFE_CAST(male_75_to_79 AS FLOAT64), 0) +
      COALESCE(SAFE_CAST(male_80_to_84 AS FLOAT64), 0) +
      COALESCE(SAFE_CAST(male_85_and_over AS FLOAT64), 0) +
      COALESCE(SAFE_CAST(female_65_to_66 AS FLOAT64), 0) +
      COALESCE(SAFE_CAST(female_67_to_69 AS FLOAT64), 0) +
      COALESCE(SAFE_CAST(female_70_to_74 AS FLOAT64), 0) +
      COALESCE(SAFE_CAST(female_75_to_79 AS FLOAT64), 0) +
      COALESCE(SAFE_CAST(female_80_to_84 AS FLOAT64), 0) +
      COALESCE(SAFE_CAST(female_85_and_over AS FLOAT64), 0)
    ) AS pop_over_65
  FROM `bigquery-public-data.census_bureau_acs.censustract_2020_5yr`
),
-- NYC Street Tree Census (CC0 / Public Domain in bigquery-public-data)
nyc_trees_in_bbox AS (
  SELECT
    latitude,
    longitude,
    SAFE_CAST(tree_dbh AS FLOAT64) AS tree_dbh,
    ST_GEOGPOINT(longitude, latitude) AS tree_pt
  FROM `bigquery-public-data.new_york_trees.tree_census_2015`
  CROSS JOIN bbox
  WHERE status = 'Alive'
    AND latitude BETWEEN bbox.lat_min AND bbox.lat_max
    AND longitude BETWEEN bbox.lon_min AND bbox.lon_max
),
tract_tree_counts AS (
  SELECT
    d.geo_id,
    COUNT(t.tree_dbh) AS live_trees,
    AVG(t.tree_dbh) AS avg_tree_dbh
  FROM `a4i_climate.tract_demographics` d
  JOIN geo_poly g
    ON LPAD(d.geo_id, 11, '0') = g.geo_id
  LEFT JOIN nyc_trees_in_bbox t
    ON ST_INTERSECTS(g.tract_geom, t.tree_pt)
  GROUP BY d.geo_id
),
base_tracts AS (
  SELECT
    LPAD(d.geo_id, 11, '0') AS geo_id,
    d.tract_lat,
    d.tract_lon,
    g.land_area_km2,
    COALESCE(g.water_area_pct, 0.0) AS water_area_pct,
    d.total_pop,
    d.median_age,
    d.median_income,
    d.housing_units,
    d.median_year_structure_built,
    d.no_cars,
    d.households,
    COALESCE(d.poverty_rate, 0.0) AS poverty_rate,
    COALESCE(d.no_vehicle_rate, 0.0) AS no_vehicle_rate,
    COALESCE(e.pop_over_65, d.total_pop * 0.14) AS pop_over_65,
    SAFE_DIVIDE(COALESCE(e.pop_over_65, d.total_pop * 0.14), NULLIF(d.total_pop, 0)) AS over_65_rate,
    -- Built-environment density & structure age features
    SAFE_DIVIDE(d.total_pop, g.land_area_km2) AS pop_density_per_km2,
    SAFE_DIVIDE(d.housing_units, g.land_area_km2) AS housing_density_per_km2,
    GREATEST(0.0, 2026.0 - COALESCE(NULLIF(d.median_year_structure_built, 0), 1960.0)) AS building_age_years,
    -- Tree canopy metrics
    COALESCE(tc.live_trees, 0) AS live_trees,
    SAFE_DIVIDE(COALESCE(tc.live_trees, 0), g.land_area_km2) AS trees_per_km2,
    COALESCE(tc.avg_tree_dbh, 8.0) AS avg_tree_dbh,
    -- Distance from regional coast / water moderator (in km)
    ST_DISTANCE(
      ST_GEOGPOINT(d.tract_lon, d.tract_lat),
      ST_GEOGPOINT(bbox.lon_min, bbox.lat_min)
    ) / 1000.0 AS inland_dist_km,
    g.tract_geom
  FROM `a4i_climate.tract_demographics` d
  CROSS JOIN bbox
  LEFT JOIN geo_poly g
    ON LPAD(d.geo_id, 11, '0') = g.geo_id
  LEFT JOIN acs_elderly e
    ON LPAD(d.geo_id, 11, '0') = e.geo_id
  LEFT JOIN tract_tree_counts tc
    ON LPAD(d.geo_id, 11, '0') = LPAD(tc.geo_id, 11, '0')
  WHERE d.total_pop > 0
)
SELECT
  *,
  -- Canopy deficit score (0 = lushest canopy in city, 100 = severest tree scarcity)
  CASE
    WHEN MAX(trees_per_km2) OVER () > 0 THEN
      ROUND((1.0 - PERCENT_RANK() OVER (ORDER BY trees_per_km2 ASC)) * 100.0, 1)
    ELSE
      ROUND(PERCENT_RANK() OVER (ORDER BY housing_density_per_km2 ASC) * 100.0, 1)
  END AS canopy_deficit_pct
FROM base_tracts;

-- ----------------------------------------------------------------------------
-- STEP 2: Build Micro-Grid (~250m) Local Tree Counts & Street Training Table
--         Target: temp_offset_f = temp_f - AVG(temp_f) OVER ()
--         Zero data leakage: temp_f and heat_index_f are NOT model inputs!
-- ----------------------------------------------------------------------------
CREATE OR REPLACE TABLE `a4i_climate.heat_training_data` AS
WITH city_stats AS (
  SELECT
    AVG(temp_f) AS city_avg_temp_f,
    MIN(latitude) - 0.05 AS lat_min,
    MAX(latitude) + 0.05 AS lat_max,
    MIN(longitude) - 0.05 AS lon_min,
    MAX(longitude) + 0.05 AS lon_max
  FROM `a4i_climate.heat_observations`
),
-- Micro-block (~250m grid) live tree count for fine-grained intra-tract variation
grid_trees AS (
  SELECT
    ROUND(latitude, 3) AS grid_lat,
    ROUND(longitude, 3) AS grid_lon,
    COUNT(*) AS local_block_trees
  FROM `bigquery-public-data.new_york_trees.tree_census_2015`
  CROSS JOIN city_stats
  WHERE status = 'Alive'
    AND latitude BETWEEN city_stats.lat_min AND city_stats.lat_max
    AND longitude BETWEEN city_stats.lon_min AND city_stats.lon_max
  GROUP BY grid_lat, grid_lon
),
-- Spatially match each street observation to its enclosing census tract
matched_obs AS (
  SELECT
    h.latitude,
    h.longitude,
    h.city,
    h.temp_f,
    h.heat_index_f,
    h.humidity_pct,
    ROUND(h.temp_f - c.city_avg_temp_f, 3) AS temp_offset_f,
    c.city_avg_temp_f,
    t.geo_id,
    t.canopy_deficit_pct,
    t.trees_per_km2,
    t.housing_density_per_km2,
    t.pop_density_per_km2,
    t.building_age_years,
    t.inland_dist_km,
    t.water_area_pct,
    t.total_pop,
    t.pop_over_65,
    t.over_65_rate,
    t.poverty_rate,
    t.no_vehicle_rate,
    t.median_income,
    COALESCE(gt.local_block_trees, 0) AS local_block_trees,
    -- Local block tree scarcity (higher = fewer trees on this exact block)
    GREATEST(0.0, 50.0 - LEAST(COALESCE(gt.local_block_trees, 0), 50)) AS local_tree_scarcity
  FROM `a4i_climate.heat_observations` h
  CROSS JOIN city_stats c
  JOIN `a4i_climate.tract_features` t
    ON ST_INTERSECTS(t.tract_geom, ST_GEOGPOINT(h.longitude, h.latitude))
  LEFT JOIN grid_trees gt
    ON ROUND(h.latitude, 3) = gt.grid_lat
   AND ROUND(h.longitude, 3) = gt.grid_lon
)
SELECT * FROM matched_obs
WHERE temp_offset_f IS NOT NULL;

-- ----------------------------------------------------------------------------
-- STEP 3: Train the BigQuery ML Offset Model (LINEAR_REG)
--         Uses ONLY physical/built-environment features (no race, no income)
--         Supports ML.EXPLAIN_PREDICT and ML.GLOBAL_EXPLAIN on free tier
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MODEL `a4i_climate.heat_offset_model`
OPTIONS(
  model_type = 'LINEAR_REG',
  input_label_cols = ['temp_offset_f'],
  enable_global_explain = TRUE,
  calculate_p_values = TRUE,
  category_encoding_method = 'DUMMY_ENCODING',
  l2_reg = 0.1
) AS
SELECT
  temp_offset_f,
  canopy_deficit_pct,
  local_tree_scarcity,
  housing_density_per_km2,
  pop_density_per_km2,
  building_age_years,
  inland_dist_km,
  water_area_pct
FROM `a4i_climate.heat_training_data`;

-- ----------------------------------------------------------------------------
-- STEP 4: Precompute Tract-Level ML.EXPLAIN_PREDICT & Siting Priority Scores
--         Combines physical heat offset prediction with social vulnerability
-- ----------------------------------------------------------------------------
CREATE OR REPLACE TABLE `a4i_climate.tract_predictions_explained` AS
WITH tract_inputs AS (
  SELECT
    geo_id,
    tract_lat AS latitude,
    tract_lon AS longitude,
    canopy_deficit_pct,
    GREATEST(0.0, 50.0 - LEAST(SAFE_DIVIDE(trees_per_km2, 40.0), 50.0)) AS local_tree_scarcity,
    housing_density_per_km2,
    pop_density_per_km2,
    building_age_years,
    inland_dist_km,
    water_area_pct,
    trees_per_km2,
    live_trees,
    total_pop,
    pop_over_65,
    over_65_rate,
    median_age,
    median_income,
    poverty_rate,
    no_vehicle_rate,
    no_cars,
    households
  FROM `a4i_climate.tract_features`
),
explained AS (
  SELECT
    *
  FROM ML.EXPLAIN_PREDICT(
    MODEL `a4i_climate.heat_offset_model`,
    TABLE tract_inputs,
    STRUCT(5 AS top_k_features)
  )
)
SELECT
  geo_id,
  ROUND(latitude, 4) AS latitude,
  ROUND(longitude, 4) AS longitude,
  ROUND(predicted_temp_offset_f, 2) AS predicted_offset_f,
  ROUND(baseline_prediction_value, 2) AS baseline_offset_f,
  top_feature_attributions,
  canopy_deficit_pct,
  ROUND(trees_per_km2, 1) AS trees_per_km2,
  live_trees,
  ROUND(housing_density_per_km2, 1) AS housing_density_per_km2,
  ROUND(pop_density_per_km2, 1) AS pop_density_per_km2,
  ROUND(building_age_years, 1) AS building_age_years,
  ROUND(inland_dist_km, 2) AS inland_dist_km,
  ROUND(water_area_pct, 2) AS water_area_pct,
  CAST(total_pop AS INT64) AS total_pop,
  CAST(ROUND(pop_over_65) AS INT64) AS pop_over_65,
  ROUND(over_65_rate * 100.0, 1) AS over_65_pct,
  ROUND(median_age, 1) AS median_age,
  CAST(ROUND(median_income) AS INT64) AS median_income,
  ROUND(poverty_rate * 100.0, 1) AS poverty_pct,
  ROUND(no_vehicle_rate * 100.0, 1) AS no_vehicle_pct,
  CAST(ROUND(no_cars) AS INT64) AS no_vehicle_households,
  -- Composite Heat Risk & Vulnerability Priority Index (0-100 scale)
  ROUND(
    (PERCENT_RANK() OVER (ORDER BY predicted_temp_offset_f ASC) * 45.0) +
    (PERCENT_RANK() OVER (ORDER BY no_vehicle_rate ASC) * 20.0) +
    (PERCENT_RANK() OVER (ORDER BY poverty_rate ASC) * 20.0) +
    (PERCENT_RANK() OVER (ORDER BY over_65_rate ASC) * 15.0),
    1
  ) AS priority_score,
  -- Recommended physical intervention based on dominant feature attribution
  CASE
    WHEN canopy_deficit_pct >= 65 AND housing_density_per_km2 < 12000 THEN 'Street Tree Canopy Expansion + Cooling Center'
    WHEN canopy_deficit_pct >= 65 AND housing_density_per_km2 >= 12000 THEN 'Cool Pavement / Depaving + Permanent Cooling Center'
    WHEN housing_density_per_km2 >= 14000 THEN 'High-Capacity Cooling Center + Cool Roofs Retrofit'
    ELSE 'Targeted Pocket Canopy + Hydration / Library Extended Hours'
  END AS recommended_intervention
FROM explained;

-- ----------------------------------------------------------------------------
-- STEP 5: Isolated Post-Hoc Equity Audit Lookup Table (Section 9 Requirement)
--         "Strip the variable. Audit the outcome. Say you did both."
--         Race is NEVER in the model -- it exists ONLY here for post-hoc audit.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE TABLE `a4i_climate.tract_equity_audit_lookup` AS
SELECT
  LPAD(acs.geo_id, 11, '0') AS geo_id,
  SAFE_DIVIDE(SAFE_CAST(acs.black_pop AS FLOAT64), NULLIF(SAFE_CAST(acs.total_pop AS FLOAT64), 0)) * 100.0 AS black_pct,
  SAFE_DIVIDE(SAFE_CAST(acs.hispanic_pop AS FLOAT64), NULLIF(SAFE_CAST(acs.total_pop AS FLOAT64), 0)) * 100.0 AS hispanic_pct,
  SAFE_DIVIDE(SAFE_CAST(acs.asian_pop AS FLOAT64), NULLIF(SAFE_CAST(acs.total_pop AS FLOAT64), 0)) * 100.0 AS asian_pct,
  SAFE_DIVIDE(SAFE_CAST(acs.white_pop AS FLOAT64), NULLIF(SAFE_CAST(acs.total_pop AS FLOAT64), 0)) * 100.0 AS white_pct,
  SAFE_DIVIDE(
    COALESCE(SAFE_CAST(acs.black_pop AS FLOAT64), 0) +
    COALESCE(SAFE_CAST(acs.hispanic_pop AS FLOAT64), 0) +
    COALESCE(SAFE_CAST(acs.asian_pop AS FLOAT64), 0),
    NULLIF(SAFE_CAST(acs.total_pop AS FLOAT64), 0)
  ) * 100.0 AS bipoc_pct
FROM `bigquery-public-data.census_bureau_acs.censustract_2020_5yr` acs
JOIN `a4i_climate.tract_features` tf
  ON LPAD(acs.geo_id, 11, '0') = tf.geo_id;
