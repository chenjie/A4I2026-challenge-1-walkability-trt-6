# CoolBrooklyn AI — Agent Branch Additions & Architecture

**Challenge 1: Community Climate Resilience (Urban Heat) — Agents for Impact 2026**

This document details all contributions, architecture, tools, supervisor agents, and test suites added in the **`agent`** branch (`chenjie/A4I2026-challenge-1-walkability-trt-6`).

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [High-Level Architecture & What Was Added](#high-level-architecture--what-was-added)
3. [Chronological & Commit-Level Evolution](#chronological--commit-level-evolution)
4. [Detailed Breakdown of Major Additions](#detailed-breakdown-of-major-additions)
   - [1. Persona-Aware Supervisor Agent (`persona_agent.py`)](#1-persona-aware-supervisor-agent)
   - [2. Citizen Cooling Refuge Router & Google Maps Walking Navigation](#2-citizen-cooling-refuge-router--google-maps-navigation)
   - [3. Gemini Live Speech & Audio Briefing Integration](#3-gemini-live-speech--audio-briefing-integration)
   - [4. BigQuery ML Grounded Climate Tools (`agent/tools.py`)](#4-bigquery-ml-grounded-climate-tools)
   - [5. Standalone Google ADK CLI Runner (`scripts/run_adk.py`)](#5-standalone-google-adk-cli-runner)
   - [6. Interactive Spatial Workbench Web Application (`app.py`)](#6-interactive-spatial-workbench-web-application)
   - [7. Automated Test Suites (`tests/`)](#7-automated-test-suites)
5. [Complete File-by-File Inventory](#complete-file-by-file-inventory)
6. [The 3 Core Hackathon Criteria Verified](#the-3-core-hackathon-criteria-verified)
7. [How to Run and Verify](#how-to-run-and-verify)

---

## Executive Summary

The `agent` branch transforms the repository from raw climate and census data tables into an end-to-end, multi-persona AI decision engine for municipal climate resilience and citizen heat protection in Brooklyn, NY.

### Core Philosophy: "Strip the Variable. Audit the Outcome."
1. **Physical Thermodynamics, Not Demographics in the Model**: The BigQuery ML model (`a4i_climate.heat_offset_model`) deliberately excludes race and income from model training features to learn purely biophysical drivers (tree canopy deficit, impervious surface density, building vintage).
2. **Post-Hoc Algorithmic Equity Auditing**: Socioeconomic vulnerability (`no_vehicle_rate`, `pop_over_65`, `poverty_rate`) is applied downstream during multi-criteria prioritization. This catches systemic blind spots:
   - **The Industrial Heat Trap**: East Williamsburg IBZ runs at +5.6°F (raw heat rank `#1`), but has only 115 residents. Equity weighting demotes it to `#6`.
   - **The Coastal Senior Blind Spot**: Coney Island runs at a milder +2.9°F (ocean breeze, raw heat rank `#7`), but houses 1,365 seniors (26.3% of the tract, 75.8% carless). Post-hoc equity auditing elevates it to priority rank `#4` with a *Targeted Senior Equity Override*.

---

## High-Level Architecture & What Was Added

```mermaid
graph TD
    User[User / Municipal Operator / Citizen] --> UI[Spatial Workbench UI & Gemini Live Audio (app.py)]
    User --> CLI[Google ADK CLI Runner (scripts/run_adk.py)]
    
    UI --> Supervisor[Persona-Aware Supervisor Agent (agent/persona_agent.py)]
    CLI --> Supervisor
    
    subgraph "Operational Personas"
        P1[🏛️ City Planner (MOCEJ)]
        P2[🚨 Emergency Management (NYCEM/FDNY)]
        P3[🌳 Urban Forestry Officer (NYC Parks)]
        P4[📍 Vulnerable Resident / Live Location]
    end
    
    Supervisor --> P1
    Supervisor --> P2
    Supervisor --> P3
    Supervisor --> P4
    
    subgraph "11 Climate & Spatial Tools (agent/tools.py)"
        T1[predict_location_heat_offset (ML.EXPLAIN_PREDICT)]
        T2[recommend_interventions (Continuous #1..#N Ranking)]
        T3[run_equity_audit (Algorithmic Equity Audit)]
        T4[analyze_heat_frequency (7-Yr GHCN Station Analysis)]
        T5[compare_candidate_sites (Head-to-Head Tradeoffs)]
        T6[generate_placement_recommendation (Council Brief)]
        T7[dispatch_gov_cooling_action (Multi-Agency Routing)]
        T8[find_nearest_cooling_center_and_heat (Google Maps)]
        T9[geocode_street_or_intersection]
        T10[fetch_live_brooklyn_weather]
        T11[query_climate_dataset_summary]
    end
    
    P1 & P2 & P3 & P4 --> T1 & T2 & T3 & T4 & T5 & T6 & T7 & T8
    
    subgraph "Underlying Engines & APIs"
        BQ[BigQuery ML & NOAA GHCN Data]
        GMaps[Google Maps Platform Walking Directions API]
        Voice[Gemini Live / Web Speech Synthesis]
        Census[US Census ACS Demographics]
    end
    
    T1 & T2 & T4 --> BQ
    T8 --> GMaps
    T8 --> Voice
    T2 & T3 --> Census
```

---

## Chronological & Commit-Level Evolution

The `agent` branch contains 6 commits building upon `main` (`3aa3879`):

| Commit | Author | Description & Key Additions |
|---|---|---|
| `19507bb` | akshay-kap | **Foundation of Agent & Workbench**: Initial `agent/agent.py`, `agent/tools.py` (1,350+ lines), `app.py` (1,590+ lines), `Dockerfile`, `requirements.txt`, `PITCH_AND_DEMO_GUIDE.md`, `SLIDE_DECK_6_SLIDES.md`, `slides.html`, `scripts/deploy_cloud_run.sh`, `data/brooklyn_reference_data.json`. |
| `f7326f3` | Chenjie Ni | **Bug Fixes & Resilience**: Fixed `KeyError: 'heat_frequency_analysis'` in `analyze_heat_frequency()`. Made `generate_placement_recommendation` resilient to missing keys. Added `.gitignore` patterns for cache directories. |
| `c7b02aa` | Chenjie Ni | **Challenge Questions & Benchmark Suite**: Fixed challenge prompt handling, normalized NOAA GHCN frequency keys, and built first automated test suite (`tests/test_challenge_questions.py`) covering all canonical questions. |
| `bc479b4` | Chenjie Ni | **Citizen Router & Persona Architecture**: Created `agent/persona_agent.py` (655 lines) introducing the supervisor orchestrator. Built `find_nearest_cooling_center_and_heat` tool, cooling center registry, and Google Maps walking directions. |
| `c0bfafa` | Chenjie Ni | **4 Personas, Google Maps UI & Gemini Live**: Wired 4 operational personas into `app.py` with custom buttons, prompt chips, and CSS. Integrated browser GPS geolocation, Google Maps directions cards, and Gemini Live / Web Speech audio playback (`speakText`). Added 6 persona tests. |
| `8e31dbc` | Chenjie Ni | **Google ADK CLI Runner & Test Harness**: Built `scripts/run_adk.py` with native `google.adk.runners.Runner` integration (`--test`, `--query`, `--interactive`, `--server`). Created `tests/test_adk_runner.py`. Grounded all canonical test prompts in Brooklyn landmarks. 21/21 tests passing. |

---

## Detailed Breakdown of Major Additions

### 1. Persona-Aware Supervisor Agent (`agent/persona_agent.py`)
A top-level orchestrator class and dispatcher (`PersonaAwareSupervisorAgent`) that tailor reasoning, tool execution, and voice output across 4 municipal and community personas:

1. **🏛️ City Planner (NYC Mayor's Office of Climate & Environmental Justice)**:
   - Specializes in long-term capital allocation ($1.85M budget), 10-year street canopy expansion, multi-criteria spatial ranking (`#1..#8`), and algorithmic equity auditing.
2. **🚨 Emergency Management Officer (NYCEM & FDNY)**:
   - Specializes in acute 24–48h heat emergency response, inter-agency cooling dispatch, mobile hydration trucks, misting stations, transit corridor triage, and senior door-knocking.
3. **🌳 Urban Forestry Officer (NYC Parks & Recreation)**:
   - Evaluates tree canopy deficit (`canopy_deficit_pct`), street tree density (`trees_per_km2`), species survival, vaulted sidewalk constraints, and nature-based vs. cool pavement albedo trade-offs.
4. **📍 Vulnerable Resident / Live Location (Google Maps + Gemini Live)**:
   - Computes hyper-local street heat feel (+offset vs. ambient), finds the closest air-conditioned cooling center, generates walking directions via Google Maps, and produces speech-ready briefings.

### 2. Citizen Cooling Refuge Router & Google Maps Navigation
- **Brooklyn Cooling Center Registry**: Curated database of 10 designated cooling centers (Stone Ave Library, Brownsville Library, Coney Island Senior Center, Sunset Park Rec Center, etc.) with coordinates, HVAC capacity, hours, wheelchair accessibility, and amenities.
- **`find_nearest_cooling_center_and_heat` Tool**:
  - Resolves any Brooklyn address, intersection, or GPS latitude/longitude.
  - Calculates real-time microclimate street feel using `predict_location_heat_offset`.
  - Computes street walking distance using Haversine with urban Manhattan grid factor (1.25x).
  - Generates universal Google Maps walking navigation URLs:
    `https://www.google.com/maps/dir/?api=1&origin={lat,lon}&destination={dest_lat,dest_lon}&travelmode=walking`
  - Assesses danger tier (`EXTREME DANGER`, `HIGH DANGER`, `EXTREME CAUTION`).

### 3. Gemini Live Speech & Audio Briefing Integration
- **Spoken Script Generation**: Automatically drafts conversational, speech-optimized safety briefings:
  > *"Right now near Brownsville, the street feels like 103 degrees because of the local urban heat island effect—that's +5.4 degrees hotter than city forecasts. Your closest air-conditioned refuge is Stone Avenue Library, located 0.45 miles away. That is about a 6-minute walk. I have opened a Google Maps walking route for you."*
- **Workbench Audio Player**: Integrated Web Speech API audio synthesis in `app.py` with a **"▶️ Play Audio"** button and live voice readout.

### 4. BigQuery ML Grounded Climate Tools (`agent/tools.py`)
11 registered tools supporting the 4-stage orchestration pipeline:
- `predict_location_heat_offset`: Runs `ML.EXPLAIN_PREDICT` on `a4i_climate.heat_offset_model` to predict temperature offset (never raw weather) and outputs exact per-feature attributions (`canopy_deficit_pct`, `housing_density_per_km2`, etc.).
- `recommend_interventions`: Full continuous ranking (`#1..#N`) combining BQML offsets with census vulnerability.
- `run_equity_audit`: Post-hoc algorithmic audit exposing the Industrial Heat Trap and Coastal Senior Blind Spot.
- `analyze_heat_frequency`: 7-year NOAA GHCN-Daily weather station analysis (2019–2025, 2,557 days) quantifying days ≥95°F and the +36.3% warming trend.
- `compare_candidate_sites`: Head-to-head trade-off analysis between candidate sites.
- `generate_placement_recommendation`: Council-ready formal memo synthesis with budget breakdowns.
- `dispatch_gov_cooling_action`: Multi-agency work order routing across 3 clocks (Today, This Budget Cycle, This Decade).
- `geocode_street_or_intersection`: Resolves arbitrary Brooklyn addresses and cross-streets.
- `fetch_live_brooklyn_weather`: Fetches live ambient conditions to add to spatial offsets.
- `query_climate_dataset_summary`: Explains dataset validation and null `observed_at` rationale.
- `find_nearest_cooling_center_and_heat`: Direct citizen refuge and navigation tool.

### 5. Standalone Google ADK CLI Runner (`scripts/run_adk.py`)
A command-line interface wrapping `google.adk.runners.Runner` and `google.adk.sessions.InMemorySessionService`:
```bash
# Automated Brooklyn test suite execution
python scripts/run_adk.py --test

# Query the agent directly
python scripts/run_adk.py --query "Why is Pitkin Ave & Mother Gaston Blvd so much worse than two blocks north?"

# Interactive CLI REPL with persona selection
python scripts/run_adk.py --interactive

# Launch the interactive web workbench
python scripts/run_adk.py --server --port 8080
```

### 6. Interactive Spatial Workbench Web Application (`app.py`)
A single-file web application providing:
- **Interactive Leaflet Map**: Renders Brooklyn census tracts, heat offset choropleths, 0.5 km cooling catchment rings, and cooling center pins.
- **Persona Switcher Bar**: 1-click toggling between City Planner, Emergency Management, Urban Forestry, and Resident personas.
- **Dynamic Prompt Chips**: Pre-configured benchmark questions tailored to the active persona.
- **Live Attribution Visualizer**: Displays `ML.EXPLAIN_PREDICT` feature attribution bars (+X.X°F breakdown).
- **Google Maps Walking Route Cards**: Direct links opening turn-by-turn directions.
- **Gemini Live Audio Player**: Native voice playback for spoken safety briefings.
- **Live GPS Integration**: Browser `navigator.geolocation` button for instant "Heat Near Me" queries.

### 7. Automated Test Suites (`tests/`)
Comprehensive test coverage passing with 100% success rate (**21/21 tests passing**):
- **`tests/test_challenge_questions.py` (14 tests)**:
  - All 6 canonical Hackathon challenge prompts (Spatial comparison, Frequency, Senior intervention, Equity audit, Multi-agency dispatch, Unmeasured streets).
  - All 4 personas (`CITY_PLANNER`, `EMERGENCY_MANAGEMENT_OFFICER`, `URBAN_FORESTRY_OFFICER`, `RESIDENT_LIVE_LOCATION`).
  - Google Maps cooling center routing and speech generation.
- **`tests/test_adk_runner.py` (7 tests)**:
  - ADK Agent initialization and prompt validation.
  - Registration of all 11 climate and spatial tools.
  - Session service creation and runner execution.
  - CLI argument parsing and canned test runner execution.

---

## Complete File-by-File Inventory

| File Path | Status | Lines | Purpose / Contribution |
|---|---|---|---|
| `AGENTREADME.md` | **NEW** | ~350 | This documentation file detailing what the `agent` branch added. |
| `agent/persona_agent.py` | **NEW** | 657 | Persona-Aware Supervisor Agent (4 operational personas, Google Maps routing, Gemini Live speech). |
| `agent/agent.py` | **NEW** | 120 | Google ADK Agent configuration, model binding, and 11 registered tools. |
| `agent/tools.py` | **NEW** | 1,738 | 11 climate, spatial, BigQuery ML, and Google Maps cooling center tools. |
| `agent/mcp_toolbox_tools.yaml` | **NEW** | 136 | Google MCP Toolbox for Databases SQL tool definitions. |
| `agent/__init__.py` | **NEW** | 4 | Package initialization exposing `root_agent`, `build_adk_agent`, and `supervisor_agent`. |
| `agent/README.md` | **MODIFIED** | 48 | Technical documentation for the `agent/` folder and tool design. |
| `app.py` | **NEW** | 1,985 | Full-featured Spatial Workbench web server (Leaflet map, chat, personas, voice audio, Google Maps links). |
| `scripts/run_adk.py` | **NEW** | 301 | Standalone Google ADK CLI runner, interactive REPL, and automated test runner. |
| `tests/test_challenge_questions.py` | **NEW** | 304 | Automated test suite for all challenge questions and personas. |
| `tests/test_adk_runner.py` | **NEW** | 117 | Unit tests for ADK Agent, Runner, tool registration, and session service. |
| `data/brooklyn_reference_data.json` | **NEW** | 629 | Reference dataset for 761 Brooklyn census tracts and NOAA station history. |
| `Dockerfile` | **NEW** | 24 | Container build file for Cloud Run deployment. |
| `requirements.txt` | **NEW** | 11 | Python dependencies (`google-adk`, `google-genai`, `fastapi`, `uvicorn`, etc.). |
| `scripts/deploy_cloud_run.sh` | **NEW** | 33 | Deployment automation script for Google Cloud Run. |
| `PITCH_AND_DEMO_GUIDE.md` | **NEW** | 89 | 5-minute hackathon pitch and live demonstration script. |
| `SLIDE_DECK_6_SLIDES.md` | **NEW** | 139 | 6-slide executive presentation narrative. |
| `slides.html` | **NEW** | 958 | Interactive slide deck presentation viewer. |
| `.gitignore` | **MODIFIED** | 11 | Added ignore rules for `__pycache__` and `.pytest_cache`. |

---

## The 3 Core Hackathon Criteria Verified

| Criterion | Implementation in `agent` Branch |
|---|---|
| **1. Predicts Offset, NOT Temperature** | All tools enforce `predicted_offset_f` (+X.X°F vs. baseline). Predictions never confuse ambient forecast weather with local microclimate thermal mass. |
| **2. Calls Live BQML Prediction for Unmeasured Streets** | `predict_location_heat_offset` executes `ML.EXPLAIN_PREDICT` against `a4i_climate.heat_offset_model` on demand for any intersection or coordinate, returning feature attributions (`canopy_deficit_pct`, `housing_density_per_km2`, etc.). |
| **3. Respects Division of Labor Between Datasets** | Distinguishes single-day 1-hour NOAA Heat Watch spatial snapshot (`heat_observations`, explaining why `observed_at` is null) from 7-year NOAA GHCN-Daily temporal station history (`station_daily_temps`). |

---

## How to Run and Verify

### Run the Automated Test Suite (21 Tests)
```bash
python3 -m unittest discover tests
```
*Expected output: `Ran 21 tests ... OK`*

### Run the ADK CLI Runner
```bash
# Automated question validation
python3 scripts/run_adk.py --test

# Interactive query
python3 scripts/run_adk.py --query "Where should Brooklyn place emergency hydration vans today?"
```

### Launch the Spatial Workbench Web App
```bash
python3 app.py
```
*Opens on `http://localhost:8080` with interactive Leaflet map, persona selector, Google Maps walking directions, and Gemini Live audio player.*
