-- ============================================================================
-- A4I 2026 - Challenge 1: Urban Heat Resilience (Brooklyn, NY)
-- 7-Year NOAA GHCN Station Frequency & Government Cooling Action Thresholds
--
-- Target User: Agent Person 1 (for `analyze_heat_frequency` & `dispatch_gov_cooling_action`)
-- Datasets: `a4i_climate.station_daily_temps` (2019-2025, 2,557 days)
--           `a4i_climate.tract_predictions_explained`
-- ============================================================================

-- ----------------------------------------------------------------------------
-- QUERY 1: Parameterized Street-Level 7-Year Frequency by Location Offset
-- Use this inside Agent Person 1's `analyze_heat_frequency` Python tool.
-- Replace @predicted_offset_f with the location's predicted/observed offset (e.g. 4.5)
-- Replace @station_id with benchmark station ('USW00014732' = LaGuardia AP, 'USW00094789' = JFK)
-- ----------------------------------------------------------------------------
WITH station_daily AS (
  SELECT
    station_id,
    station_name,
    observation_date,
    EXTRACT(YEAR FROM observation_date) AS obs_year,
    max_temp_f AS airport_max_temp_f,
    ROUND(max_temp_f + @predicted_offset_f, 1) AS street_max_temp_f
  FROM `a4i_climate.station_daily_temps`
  WHERE station_id = COALESCE(@station_id, 'USW00014732')
    AND EXTRACT(YEAR FROM observation_date) BETWEEN 2019 AND 2025
),
annual_rollups AS (
  SELECT
    obs_year,
    COUNT(*) AS total_summer_days,
    -- Airport Baseline Counts
    COUNTIF(airport_max_temp_f >= 90.0) AS airport_days_ge_90,
    COUNTIF(airport_max_temp_f >= 95.0) AS airport_days_ge_95,   -- NWS Heat Advisory baseline
    COUNTIF(airport_max_temp_f >= 100.0) AS airport_days_ge_100, -- Emergency Threshold baseline
    -- Local Street Microclimate Counts (Airport + Heat Island Offset)
    COUNTIF(street_max_temp_f >= 90.0) AS street_days_ge_90,
    COUNTIF(street_max_temp_f >= 95.0) AS street_days_ge_95,     -- Street Heat Advisory trigger
    COUNTIF(street_max_temp_f >= 98.0) AS street_days_ge_98,     -- NYCEM Misting/Hydration Van trigger
    COUNTIF(street_max_temp_f >= 100.0) AS street_days_ge_100    -- NYC Extreme Cooling Center dispatch
  FROM station_daily
  GROUP BY obs_year
)
SELECT
  obs_year,
  airport_days_ge_90,
  street_days_ge_90,
  airport_days_ge_95,
  street_days_ge_95,
  street_days_ge_98,
  airport_days_ge_100,
  street_days_ge_100,
  ROUND(SAFE_DIVIDE(street_days_ge_95 - airport_days_ge_95, NULLIF(airport_days_ge_95, 0)) * 100.0, 1) AS pct_increase_heat_advisory_days
FROM annual_rollups
ORDER BY obs_year;

-- ----------------------------------------------------------------------------
-- QUERY 2: 7-Year Aggregate Baseline vs. Hotspot Comparison (Full 7-Year Averages)
-- Demonstrates the multi-year frequency multiplier for story deck & agent reasoning.
-- ----------------------------------------------------------------------------
WITH scenarios AS (
  SELECT
    station_name,
    offset_label,
    offset_val,
    COUNTIF(max_temp_f >= 95.0) AS airport_days_ge_95,
    COUNTIF(max_temp_f + offset_val >= 95.0) AS street_days_ge_95,
    ROUND(COUNTIF(max_temp_f + offset_val >= 95.0) / 7.0, 1) AS avg_annual_days_ge_95,
    COUNTIF(max_temp_f >= 100.0) AS airport_days_ge_100,
    COUNTIF(max_temp_f + offset_val >= 100.0) AS street_days_ge_100,
    ROUND(COUNTIF(max_temp_f + offset_val >= 100.0) / 7.0, 1) AS avg_annual_days_ge_100
  FROM `a4i_climate.station_daily_temps`,
  UNNEST([
    STRUCT('Baseline (Airport)' AS offset_label, 0.0 AS offset_val),
    STRUCT('Mild Microclimate (+1.5°F)' AS offset_label, 1.5 AS offset_val),
    STRUCT('Moderate Urban Hotspot (+3.0°F)' AS offset_label, 3.0 AS offset_val),
    STRUCT('Severe Urban Hotspot (+4.5°F)' AS offset_label, 4.5 AS offset_val),
    STRUCT('Peak Campaign Hotspot (+5.35°F)' AS offset_label, 5.35 AS offset_val)
  ])
  WHERE station_id = 'USW00014732' -- LAGUARDIA AP
    AND EXTRACT(YEAR FROM observation_date) BETWEEN 2019 AND 2025
  GROUP BY 1, 2, 3
)
SELECT
  offset_label,
  offset_val,
  street_days_ge_95 AS total_7yr_days_ge_95,
  avg_annual_days_ge_95,
  street_days_ge_100 AS total_7yr_days_ge_100,
  avg_annual_days_ge_100,
  ROUND(SAFE_DIVIDE(street_days_ge_95, NULLIF(FIRST_VALUE(street_days_ge_95) OVER (ORDER BY offset_val), 0)), 2) AS multiplier_vs_airport_95F,
  ROUND(SAFE_DIVIDE(street_days_ge_100, NULLIF(FIRST_VALUE(street_days_ge_100) OVER (ORDER BY offset_val), 0)), 2) AS multiplier_vs_airport_100F
FROM scenarios
ORDER BY offset_val;

-- ----------------------------------------------------------------------------
-- QUERY 3: Direct Tract Join: Top Siting Priority Tracts with 7-Year Extreme Heat Days
-- Connects BQML predicted heat offset with NOAA GHCN-Daily history.
-- ----------------------------------------------------------------------------
WITH top_tracts AS (
  SELECT
    geo_id,
    latitude,
    longitude,
    predicted_offset_f,
    priority_score,
    poverty_pct,
    no_vehicle_pct,
    recommended_intervention
  FROM `a4i_climate.tract_predictions_explained`
  WHERE priority_score >= 80.0
),
tract_station_join AS (
  SELECT
    t.geo_id,
    t.predicted_offset_f,
    t.priority_score,
    t.poverty_pct,
    t.no_vehicle_pct,
    t.recommended_intervention,
    COUNTIF(s.max_temp_f >= 95.0) AS airport_7yr_ge_95,
    COUNTIF(s.max_temp_f + t.predicted_offset_f >= 95.0) AS street_7yr_ge_95,
    ROUND(COUNTIF(s.max_temp_f + t.predicted_offset_f >= 95.0) / 7.0, 1) AS street_annual_avg_ge_95,
    COUNTIF(s.max_temp_f + t.predicted_offset_f >= 98.0) AS street_7yr_ge_98,
    ROUND(COUNTIF(s.max_temp_f + t.predicted_offset_f >= 98.0) / 7.0, 1) AS street_annual_avg_ge_98,
    COUNTIF(s.max_temp_f + t.predicted_offset_f >= 100.0) AS street_7yr_ge_100,
    ROUND(COUNTIF(s.max_temp_f + t.predicted_offset_f >= 100.0) / 7.0, 1) AS street_annual_avg_ge_100
  FROM top_tracts t
  CROSS JOIN `a4i_climate.station_daily_temps` s
  WHERE s.station_id = 'USW00014732' -- LAGUARDIA AP
    AND EXTRACT(YEAR FROM s.observation_date) BETWEEN 2019 AND 2025
  GROUP BY 1, 2, 3, 4, 5, 6
)
SELECT * FROM tract_station_join
ORDER BY priority_score DESC;
