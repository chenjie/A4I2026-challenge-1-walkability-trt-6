# Brooklyn Urban Heat Resilience Planning Agent (`agent/`)

Built with **Google ADK (Agent Development Kit)**, **Gemini**, **BigQuery ML (`ML.EXPLAIN_PREDICT`)**, and **Google MCP Toolbox for Databases**.

## Architecture & Tool Design

| Tool | Type | What It Does |
|---|---|---|
| `predict_location_heat_offset` | Custom Python Tool (`tools.py`) | Resolves a Brooklyn street intersection, neighborhood, or `(lat, lon)` coordinate and executes `ML.EXPLAIN_PREDICT` against `a4i_climate.heat_offset_model`, returning the predicted offset (`+X.X°F` vs. city mean) and per-feature attributions. |
| `recommend_interventions` | Custom Python Tool (`tools.py`) | Ranks candidate locations for **Track 2 (Permanent Cooling Centers)** and **Track 3 (10-Year Street Canopy & Cool Pavement)** by combining BQML offset predictions with census vulnerability (`no_vehicle_rate`, 12-column `over_65_rate`, `poverty_rate`). |
| `analyze_heat_frequency` | Custom Python Tool (`tools.py`) | Queries 7 years of NOAA GHCN-Daily weather station history (`a4i_climate.station_daily_temps`, 2019–2025) to compute how many days/summer a `+offset_f` microclimate crosses 95°F and 100°F, and quantifies the 7-year warming trend (+36.3%). |
| `compare_candidate_sites` | Custom Python Tool (`tools.py`) | Performs a head-to-head trade-off analysis between two candidate sites (e.g., Brownsville vs. East Flatbush or Coney Island). |
| `run_equity_audit` | Custom Python Tool (`tools.py`) | Implements **"Strip the variable. Audit the outcome."** Compares recommended tracts against the Brooklyn baseline and exposes the **Industrial Heat Trap** (East Williamsburg IBZ +5.6°F) and **Coastal Senior Blind Spot** (Coney Island 26.3% 65+). |
| `generate_placement_recommendation` | Custom Python Tool (`tools.py`) | Synthesizes coordinates, `ML.EXPLAIN_PREDICT` attributions, demographics, 7-year station frequency, budget estimates, and equity audit findings into a council-ready **Placement Recommendation** brief. |
| `climate_planner_toolset` | Managed MCP Server (`mcp_toolbox_tools.yaml`) | Parameterized BigQuery SQL tools served via **Google MCP Toolbox for Databases** (`googleapis/mcp-toolbox`). |

## Running the Agent

### Option A: Interactive Spatial Workbench + Planner Chat (`app.py`)
```bash
python app.py
```
Opens on `http://localhost:8080` with an interactive Leaflet map of Brooklyn, `ML.EXPLAIN_PREDICT` attribution bars, live Planner Conversation (handling all 7 benchmark questions), Equity Audit scorecard, and 1-click City Council Placement Recommendation export.

### Option B: Google ADK Developer UI (`adk web`)
```bash
adk web .
```
