#!/usr/bin/env python3
"""
train_bqml.py — Orchestrates and verifies BigQuery ML pipeline for Challenge 1.

Usage:
    python3 scripts/train_bqml.py [--step 1|2|3|4|all] [--verify-only]
"""

import argparse
import os
import sys
import time
from google.cloud import bigquery

FORBIDDEN_LEAKAGE_FEATURES = {
    'temp_f',
    'heat_index_f',
    'humidity_pct',
    'median_income',
    'white_pop',
    'black_pop',
    'asian_pop',
    'hispanic_pop',
    'race',
}

def get_client(project_id=None):
    return bigquery.Client(project=project_id)

def run_step_1(client, dataset="a4i_climate"):
    print("Executing Step 1: Building a4i_climate.tract_features...")
    t0 = time.time()
    sql = f"""
    CREATE OR REPLACE TABLE `{dataset}.tract_features` AS
    WITH bbox AS (
      SELECT
        MIN(latitude) - 0.05 AS lat_min,
        MAX(latitude) + 0.05 AS lat_max,
        MIN(longitude) - 0.05 AS lon_min,
        MAX(longitude) + 0.05 AS lon_max,
        AVG(latitude) AS center_lat,
        AVG(longitude) AS center_lon
      FROM `{dataset}.heat_observations`
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
      FROM `{dataset}.tract_demographics` d
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
        SAFE_DIVIDE(d.total_pop, g.land_area_km2) AS pop_density_per_km2,
        SAFE_DIVIDE(d.housing_units, g.land_area_km2) AS housing_density_per_km2,
        GREATEST(0.0, 2026.0 - COALESCE(NULLIF(d.median_year_structure_built, 0), 1960.0)) AS building_age_years,
        COALESCE(tc.live_trees, 0) AS live_trees,
        SAFE_DIVIDE(COALESCE(tc.live_trees, 0), g.land_area_km2) AS trees_per_km2,
        COALESCE(tc.avg_tree_dbh, 8.0) AS avg_tree_dbh,
        ST_DISTANCE(
          ST_GEOGPOINT(d.tract_lon, d.tract_lat),
          ST_GEOGPOINT(bbox.lon_min, bbox.lat_min)
        ) / 1000.0 AS inland_dist_km,
        g.tract_geom
      FROM `{dataset}.tract_demographics` d
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
      CASE
        WHEN MAX(trees_per_km2) OVER () > 0 THEN
          ROUND((1.0 - PERCENT_RANK() OVER (ORDER BY trees_per_km2 ASC)) * 100.0, 1)
        ELSE
          ROUND(PERCENT_RANK() OVER (ORDER BY housing_density_per_km2 ASC) * 100.0, 1)
      END AS canopy_deficit_pct
    FROM base_tracts;
    """
    job = client.query(sql)
    job.result()
    print(f"Step 1 completed in {time.time() - t0:.1f}s.")

def run_step_2(client, dataset="a4i_climate"):
    print("Executing Step 2: Building a4i_climate.heat_training_data...")
    t0 = time.time()
    sql = f"""
    CREATE OR REPLACE TABLE `{dataset}.heat_training_data` AS
    WITH city_stats AS (
      SELECT
        AVG(temp_f) AS city_avg_temp_f,
        MIN(latitude) - 0.05 AS lat_min,
        MAX(latitude) + 0.05 AS lat_max,
        MIN(longitude) - 0.05 AS lon_min,
        MAX(longitude) + 0.05 AS lon_max
      FROM `{dataset}.heat_observations`
    ),
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
        GREATEST(0.0, 50.0 - LEAST(COALESCE(gt.local_block_trees, 0), 50)) AS local_tree_scarcity
      FROM `{dataset}.heat_observations` h
      CROSS JOIN city_stats c
      JOIN `{dataset}.tract_features` t
        ON ST_INTERSECTS(t.tract_geom, ST_GEOGPOINT(h.longitude, h.latitude))
      LEFT JOIN grid_trees gt
        ON ROUND(h.latitude, 3) = gt.grid_lat
       AND ROUND(h.longitude, 3) = gt.grid_lon
    )
    SELECT * FROM matched_obs
    WHERE temp_offset_f IS NOT NULL;
    """
    job = client.query(sql)
    job.result()
    print(f"Step 2 completed in {time.time() - t0:.1f}s.")

def run_step_3(client, dataset="a4i_climate"):
    print("Executing Step 3: Training BigQuery ML heat_offset_model...")
    t0 = time.time()
    sql = f"""
    CREATE OR REPLACE MODEL `{dataset}.heat_offset_model`
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
    FROM `{dataset}.heat_training_data`;
    """
    job = client.query(sql)
    job.result()
    print(f"Step 3 completed in {time.time() - t0:.1f}s.")

def run_step_4(client, dataset="a4i_climate"):
    print("Executing Step 4: Generating tract_predictions_explained...")
    t0 = time.time()
    sql = f"""
    CREATE OR REPLACE TABLE `{dataset}.tract_predictions_explained` AS
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
      FROM `{dataset}.tract_features`
    ),
    explained AS (
      SELECT
        *
      FROM ML.EXPLAIN_PREDICT(
        MODEL `{dataset}.heat_offset_model`,
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
      ROUND(
        (PERCENT_RANK() OVER (ORDER BY predicted_temp_offset_f ASC) * 45.0) +
        (PERCENT_RANK() OVER (ORDER BY no_vehicle_rate ASC) * 20.0) +
        (PERCENT_RANK() OVER (ORDER BY poverty_rate ASC) * 20.0) +
        (PERCENT_RANK() OVER (ORDER BY over_65_rate ASC) * 15.0),
        1
      ) AS priority_score,
      CASE
        WHEN canopy_deficit_pct >= 65 AND housing_density_per_km2 < 12000 THEN 'Street Tree Canopy Expansion + Cooling Center'
        WHEN canopy_deficit_pct >= 65 AND housing_density_per_km2 >= 12000 THEN 'Cool Pavement / Depaving + Permanent Cooling Center'
        WHEN housing_density_per_km2 >= 14000 THEN 'High-Capacity Cooling Center + Cool Roofs Retrofit'
        ELSE 'Targeted Pocket Canopy + Hydration / Library Extended Hours'
      END AS recommended_intervention
    FROM explained;
    """
    job = client.query(sql)
    job.result()
    print(f"Step 4 completed in {time.time() - t0:.1f}s.")

def run_step_5(client, dataset="a4i_climate"):
    print("Executing Step 5: Generating tract_equity_audit_lookup...")
    t0 = time.time()
    sql = f"""
    CREATE OR REPLACE TABLE `{dataset}.tract_equity_audit_lookup` AS
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
    JOIN `{dataset}.tract_features` tf
      ON LPAD(acs.geo_id, 11, '0') = tf.geo_id;
    """
    job = client.query(sql)
    job.result()
    print(f"Step 5 completed in {time.time() - t0:.1f}s.")

def verify_pipeline(client, dataset="a4i_climate"):
    print("\n" + "=" * 60)
    print("PIPELINE & ZERO DATA LEAKAGE VERIFICATION")
    print("=" * 60)

    # 1. Base tables
    for tbl in ['heat_observations', 'station_daily_temps', 'tract_demographics']:
        t = client.get_table(f"{client.project}.{dataset}.{tbl}")
        print(f"  [PASS] {tbl:25s} exists with {t.num_rows:,} rows")

    # 2. tract_features verification
    tf = client.get_table(f"{client.project}.{dataset}.tract_features")
    print(f"  [PASS] tract_features            exists with {tf.num_rows:,} rows")

    # 3. heat_training_data verification
    ht = client.get_table(f"{client.project}.{dataset}.heat_training_data")
    print(f"  [PASS] heat_training_data        exists with {ht.num_rows:,} rows")

    # 4. Check target label statistics
    stats = list(client.query(f"""
        SELECT
          ROUND(AVG(temp_offset_f), 4) AS mean_offset,
          ROUND(MIN(temp_offset_f), 2) AS min_offset,
          ROUND(MAX(temp_offset_f), 2) AS max_offset,
          ROUND(STDDEV(temp_offset_f), 3) AS stddev_offset
        FROM `{dataset}.heat_training_data`
    """).result())[0]
    print(f"  [STAT] Offset label: mean={stats.mean_offset} F (approx 0.0), range=[{stats.min_offset}, {stats.max_offset}] F, stddev={stats.stddev_offset}")
    assert abs(stats.mean_offset) < 0.05, f"Expected offset mean near 0, got {stats.mean_offset}"

    # 5. Zero Data Leakage check on trained model features
    weights = list(client.query(f"SELECT * FROM ML.WEIGHTS(MODEL `{dataset}.heat_offset_model`)").result())
    model_inputs = {w.processed_input for w in weights if w.processed_input != '__INTERCEPT__'}
    print(f"\n  Model Active Features: {sorted(list(model_inputs))}")

    leakage_detected = model_inputs.intersection(FORBIDDEN_LEAKAGE_FEATURES)
    if leakage_detected:
        print(f"  [FAIL] DATA LEAKAGE DETECTED! Forbidden features found: {leakage_detected}")
        sys.exit(1)
    else:
        print("  [PASS] ZERO DATA LEAKAGE VERIFIED: No temp_f, heat_index_f, race, or income features in model.")

    # 6. Evaluation metrics
    eval_row = list(client.query(f"SELECT * FROM ML.EVALUATE(MODEL `{dataset}.heat_offset_model`)").result())[0]
    print(f"\n  [EVAL] Model MAE: {eval_row.mean_absolute_error:.3f} F | R2 Score: {eval_row.r2_score:.4f} | MSE: {eval_row.mean_squared_error:.3f}")

    # 7. Global Feature Explanations (ML.GLOBAL_EXPLAIN)
    print("\n  [GLOBAL EXPLAIN] Top Feature Attributions:")
    global_explains = list(client.query(f"SELECT feature, ROUND(attribution, 4) as attribution FROM ML.GLOBAL_EXPLAIN(MODEL `{dataset}.heat_offset_model`) ORDER BY attribution DESC").result())
    for ge in global_explains:
        print(f"    - {ge.feature:<25s}: {ge.attribution:+.4f}")

    # 8. Predictions table
    tpe = client.get_table(f"{client.project}.{dataset}.tract_predictions_explained")
    print(f"\n  [PASS] tract_predictions_explained exists with {tpe.num_rows:,} tracts prioritized")

    # 9. Isolated Equity Audit table
    teal = client.get_table(f"{client.project}.{dataset}.tract_equity_audit_lookup")
    print(f"  [PASS] tract_equity_audit_lookup  exists with {teal.num_rows:,} rows")

    # 10. Post-Hoc Racial & Socioeconomic Equity Audit Results
    print("\n  [EQUITY AUDIT] Cohort Comparison:")
    audit_sql = f"""
    SELECT
      'All Brooklyn Tracts (Baseline)' AS cohort,
      COUNT(*) as tract_count,
      ROUND(AVG(l.bipoc_pct), 1) AS avg_bipoc_pct,
      ROUND(AVG(l.black_pct), 1) AS avg_black_pct,
      ROUND(AVG(l.hispanic_pct), 1) AS avg_hispanic_pct,
      ROUND(AVG(p.poverty_pct), 1) AS avg_poverty_pct,
      ROUND(AVG(p.no_vehicle_pct), 1) AS avg_no_vehicle_pct,
      ROUND(AVG(p.predicted_offset_f), 2) AS avg_predicted_offset_f
    FROM `{dataset}.tract_predictions_explained` p
    JOIN `{dataset}.tract_equity_audit_lookup` l ON p.geo_id = l.geo_id
    UNION ALL
    SELECT
      'Top Vulnerability Priority (Score >= 85)' AS cohort,
      COUNT(*) as tract_count,
      ROUND(AVG(l.bipoc_pct), 1) AS avg_bipoc_pct,
      ROUND(AVG(l.black_pct), 1) AS avg_black_pct,
      ROUND(AVG(l.hispanic_pct), 1) AS avg_hispanic_pct,
      ROUND(AVG(p.poverty_pct), 1) AS avg_poverty_pct,
      ROUND(AVG(p.no_vehicle_pct), 1) AS avg_no_vehicle_pct,
      ROUND(AVG(p.predicted_offset_f), 2) AS avg_predicted_offset_f
    FROM `{dataset}.tract_predictions_explained` p
    JOIN `{dataset}.tract_equity_audit_lookup` l ON p.geo_id = l.geo_id
    WHERE p.priority_score >= 85.0
    UNION ALL
    SELECT
      'Raw Heat Hotspots (Top 15% Offset)' AS cohort,
      COUNT(*) as tract_count,
      ROUND(AVG(l.bipoc_pct), 1) AS avg_bipoc_pct,
      ROUND(AVG(l.black_pct), 1) AS avg_black_pct,
      ROUND(AVG(l.hispanic_pct), 1) AS avg_hispanic_pct,
      ROUND(AVG(p.poverty_pct), 1) AS avg_poverty_pct,
      ROUND(AVG(p.no_vehicle_pct), 1) AS avg_no_vehicle_pct,
      ROUND(AVG(p.predicted_offset_f), 2) AS avg_predicted_offset_f
    FROM (
      SELECT *, PERCENT_RANK() OVER (ORDER BY predicted_offset_f ASC) as heat_rank
      FROM `{dataset}.tract_predictions_explained`
    ) p
    JOIN `{dataset}.tract_equity_audit_lookup` l ON p.geo_id = l.geo_id
    WHERE p.heat_rank >= 0.85;
    """
    audit_rows = list(client.query(audit_sql).result())
    print(f"    {'Cohort':<42s} | {'Count':<5s} | {'BIPOC %':<7s} | {'Black %':<7s} | {'Hisp %':<7s} | {'Pov %':<6s} | {'NoVeh %':<7s} | {'Offset F':<8s}")
    print("    " + "-" * 105)
    for r in audit_rows:
        print(f"    {r.cohort:<42s} | {r.tract_count:<5d} | {r.avg_bipoc_pct:<7.1f} | {r.avg_black_pct:<7.1f} | {r.avg_hispanic_pct:<7.1f} | {r.avg_poverty_pct:<6.1f} | {r.avg_no_vehicle_pct:<7.1f} | {r.avg_predicted_offset_f:<+8.2f}")

    print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="BigQuery ML Pipeline for A4I 2026 Challenge 1")
    parser.add_argument("--step", choices=["1", "2", "3", "4", "5", "all"], default="all", help="Pipeline step to run")
    parser.add_argument("--dataset", default="a4i_climate", help="BigQuery dataset name")
    parser.add_argument("--verify-only", action="store_true", help="Run verification without rebuilding tables")
    args = parser.parse_args()

    client = get_client()

    if args.verify_only:
        verify_pipeline(client, args.dataset)
        return

    if args.step in ["1", "all"]:
        run_step_1(client, args.dataset)
    if args.step in ["2", "all"]:
        run_step_2(client, args.dataset)
    if args.step in ["3", "all"]:
        run_step_3(client, args.dataset)
    if args.step in ["4", "all"]:
        run_step_4(client, args.dataset)
    if args.step in ["5", "all"]:
        run_step_5(client, args.dataset)

    verify_pipeline(client, args.dataset)

if __name__ == "__main__":
    main()

