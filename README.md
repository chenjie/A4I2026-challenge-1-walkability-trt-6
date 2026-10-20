# 🛡️ HeatShield AI — Agentic Urban Climate Resilience Platform

[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-BigQuery%20ML-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/bigquery)
[![Google ADK](https://img.shields.io/badge/Google%20ADK-Multi--Agent%20Supervisor-34A853?logo=google&logoColor=white)](https://github.com/google/agent-development-kit)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash%20%7C%20Live-8E24AA?logo=google&logoColor=white)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit%201.66-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-3776AB?logo=python&logoColor=white)](https://python.org/)
[![License](https://img.shields.io/badge/Challenge-Agents%20for%20Impact%202026-0284C7)](#)

> **Agents for Impact 2026 — Challenge 1: Community Climate Resilience (Urban Heat & Walkability)**  
> *Winner-tier agentic platform unifying explainable BigQuery ML, NOAA historical compounding, multi-persona Google ADK supervision, real-time Google Maps walking refuge navigation, and algorithmic Title VI equity audits.*

---

## 📖 Table of Contents
- [Executive Overview](#-executive-overview)
- [Why Heat Offset, Not Raw Temperature?](#-why-heat-offset-not-raw-temperature)
- [Key Innovations & Differentiators](#-key-innovations--differentiators)
- [Three Operational Screens](#-three-operational-screens)
- [Four 1-Click AI Chatbot Copilots](#-four-1-click-ai-chatbot-copilots)
- [Architecture & Tech Stack](#-architecture--tech-stack)
- [Google ADK 11-Tool Execution Suite](#-google-adk-11-tool-execution-suite)
- [Algorithmic Equity Audit: Strip the Variable, Audit the Outcome](#-algorithmic-equity-audit-strip-the-variable-audit-the-outcome)
- [Repository Structure](#-repository-structure)
- [Quickstart & Installation](#-quickstart--installation)
- [Benchmark Questions Handled](#-benchmark-questions-handled)
- [Architecture Deep Dive](#-architecture-deep-dive)

---

## 🌟 Executive Overview

Heat is the deadliest meteorological hazard in the United States, claiming more lives annually than hurricanes, floods, and tornadoes combined. Crucially, heat is the most **spatially unequal** climate hazard: two intersections in the same city, just 200 meters apart at 4:00 PM on the same day, can vary by **more than 15°F** due to the urban heat island effect (canopy deficit, asphalt albedo, and building canyon geometries).

Municipalities frequently misallocate scarce resilience budgets because standard weather forecasts provide only a single citywide average. **HeatShield AI** resolves conditions down to individual streets and unmeasured corridors, transforming high-resolution NOAA campaign observations and 7 years of station history into actionable capital, emergency, and equity decisions.

---

## 🌡️ Why Heat Offset, Not Raw Temperature?

A fundamental architectural principle of HeatShield AI is predicting **microclimate temperature offset ($\Delta^\circ\text{F}$)** rather than volatile absolute temperature:

$$\Delta T_i = T_{\text{micro}, i} - \bar{T}_{\text{city}}$$

* **Synoptic Decoupling**: Absolute temperature fluctuates wildly between morning and evening, sunny and overcast days. In contrast, the spatial offset is driven by persistent physical properties of the built environment (canopy cover, impervious surface percentage, surface albedo, building height, vehicular traffic).
* **Actionable Forecasting**: When the National Weather Service issues a heat advisory projecting an afternoon high of 98°F, HeatShield AI immediately calculates that an unmeasured corridor with a $+6.5^\circ\text{F}$ offset will experience life-threatening **$104.5^\circ\text{F}$ microclimate exposure**.

---

## 🚀 Key Innovations & Differentiators

1. **🤖 4 Specialized 1-Click AI Chatbot Copilots on the Left Sidebar**:
   * Direct 1-click access to dedicated agents for **City Planners**, **Emergency Managers**, **Urban Forestry Officers**, and **Vulnerable Residents**.
   * URL query parameter synchronization (`?screen=2&persona=...`) ensures instant persona switching, custom greeting updates, and state persistence.
   * Built-in **Quick Ask Expander** enables direct question submission or 1-click execution of benchmark challenge questions from any screen.
2. **🔬 Explainable BigQuery ML (`ML.EXPLAIN_PREDICT`)**:
   * Uses a Boosted Trees Regressor trained on ~15,000 NOAA Heat Watch traverse readings.
   * Quantifies exact feature contributions ($\Delta^\circ\text{F}$ contribution per biophysical factor) for every intersection and unmeasured street.
3. **📈 7-Year NOAA GHCN Historical Compounding**:
   * Integrates 7 years of daily weather station history (2019–2025).
   * Models how microclimate offsets compound extreme heat thresholds (>95°F and >100°F days/summer) and measures the +36.3% 7-year warming trend.
4. **🗺️ Real-Time Google Maps Walking Navigation to Cooling Refuges**:
   * Generates turn-by-turn walking routes to the nearest verified indoor cooling facilities, keeping vulnerable pedestrians off high-heat transit corridors.
5. **⚖️ Algorithmic Equity Audit Engine ("Strip the variable. Audit the outcome.")**:
   * Rigorously audits algorithmic recommendations against the Title VI borough baseline.
   * Automatically flags and remediates the **Industrial Heat Trap** (protecting warehouses instead of people) and the **Coastal Senior Blind Spot** (overlooking high-poverty coastal seniors due to maritime breezes).
6. **🎙️ Gemini Live Audio Briefings**:
   * Generates spoken microclimate briefings formatted for emergency field dispatch and executive briefings.

---

## 🖥️ Three Operational Screens

| Screen | Name | Primary Target & Capabilities |
|---|---|---|
| **Screen 1** | **🔥 Heat Intelligence Map** | Interactive Leaflet Folium spatial map of Brooklyn. Visualizes ~15,000 traverse points, high-risk microclimate corridors (>+3.5°F offset), transit hubs, and `ML.EXPLAIN_PREDICT` feature attribution waterfall charts. |
| **Screen 2** | **🤖 Agentic AI Planning Copilot** | Interactive multi-turn chat powered by Google ADK and Gemini. Exposes live multi-tool execution traces, Google Maps walking refuge links, Gemini Live speech, and an interactive **Unmeasured Street Microclimate Simulator**. |
| **Screen 3** | **⚖️ Cooling Dispatch & Equity Audit** | Municipal tactical operations and Title VI algorithmic compliance. Visualizes 0.5 km pedestrian buffer circles around cooling centers, flags coverage gaps, displays agency rosters, and generates 1-click City Council briefs. |

---

## 👥 Four 1-Click AI Chatbot Copilots

Located prominently in the **left sidebar** for instant access:

* 🏛️ **City Planner Copilot** (`NYC DOHMH & Dept of City Planning`): Focuses on multi-year capital budgeting, permanent cooling center siting, HVAC retrofits, transit corridor zoning, and council budget defense.
* 🚨 **Emergency Management Copilot** (`NYC Emergency Management`): Manages 24–48 hour operational triage, staging mobile hydration vans, cooling buses, and pedestrian heat alerts.
* 🌳 **Urban Forestry Copilot** (`NYC Parks & Recreation`): Optimizes 10-year street canopy expansion, cool pavement albedo coatings, and street canyon heat dissipation.
* 📍 **Vulnerable Resident Copilot** (`Live Pedestrian Safety`): Provides immediate, turn-by-turn walking directions to nearby air-conditioned public facilities while avoiding high-heat avenues.

---

## 🏗️ Architecture & Tech Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HEATSHIELD AI PLATFORM ARCHITECTURE                 │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
       ┌───────────────────────────┴───────────────────────────┐
       ▼                                                       ▼
 🖥️ Streamlit Frontend (app.py)                         ⚡ FastAPI REST Backend (api_server.py)
  ├── 1-Click Left Sidebar Chatbots                      ├── POST /api/chat
  ├── Screen 1: Leaflet Heat Map                         ├── POST /api/predict_unmeasured
  ├── Screen 2: Multi-Persona Copilot                    └── GET  /api/health
  └── Screen 3: Equity Audit & Dispatch                        │
                                   │                           │
                                   └─────────────┬─────────────┘
                                                 ▼
                                   🤖 Google ADK Supervisor Agent
                                   ├── Persona Classifier
                                   ├── 11 Custom Climate Tools
                                   └── Gemini Live Speech Synthesizer
                                                 │
       ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
       ▼                                         ▼                                         ▼
 🗄️ Google Cloud BigQuery ML               🌐 External Services                      📊 Spatial & Census Data
  ├── ML.EXPLAIN_PREDICT                     ├── Google Maps Directions API            ├── NOAA Heat Watch Traverses
  ├── Boosted Trees Regressor                ├── National Weather Service API          ├── NOAA GHCN 7-Year History
  └── Feature Importance Attributions        └── Google Gemini 2.5 LLM                 └── CDC / ATSDR Census SVI
```

*For complete technical design details, data schemas, and sequence flows, see [ARCHITECTURE.md](ARCHITECTURE.md).*

---

## 🛠️ Google ADK 11-Tool Execution Suite

Located in [`agent/tools.py`](agent/tools.py), each tool is strictly grounded in empirical climate and demographic datasets:

| # | Tool Name | Description |
|---|---|---|
| 1 | `predict_location_heat_offset` | Queries BigQuery ML `ML.EXPLAIN_PREDICT` to return predicted microclimate offset ($\Delta^\circ\text{F}$) and feature attributions for any street or coordinate. |
| 2 | `recommend_interventions` | Multi-criteria ranking across Track 2 (Cooling Centers) and Track 3 (Canopy/Pavement), combining BQML heat offsets with demographic vulnerability. |
| 3 | `analyze_heat_frequency` | Queries 7 years of NOAA GHCN-Daily weather station records (2019–2025) to compute compounding extreme heat day frequency (>95°F / >100°F). |
| 4 | `compare_candidate_sites` | Head-to-head multi-dimensional comparison between candidate locations (offset, canopy %, impervious %, seniors, poverty). |
| 5 | `run_equity_audit` | Executes Title VI equity audit comparing recommended tracts against borough baseline, identifying the Industrial Heat Trap and Coastal Senior Blind Spot. |
| 6 | `generate_placement_recommendation` | Generates a complete, council-ready placement brief with cost-benefit analysis, demographic rationale, and risk scoring. |
| 7 | `get_unmeasured_prediction` | Biophysical simulator allowing "what-if" simulations of tree canopy planting and cool pavement interventions on unmeasured streets. |
| 8 | `get_street_geometry` | Retrieves bounding coordinates, street canyon geometry, and census tract identifiers. |
| 9 | `get_live_weather` | Fetches real-time National Weather Service air temperature, relative humidity, and heat index readings. |
| 10 | `get_eviction_heat_nexus` | Cross-references residential housing instability and utility arrears with microclimate heat exposure. |
| 11 | `query_cooling_directory` | Searches official indoor cooling facilities within walking radius and builds Google Maps walking navigation URLs. |

---

## ⚖️ Algorithmic Equity Audit: Strip the Variable, Audit the Outcome

A core requirement of ethical climate AI is recognizing that **stripping demographic variables from training data does not prevent algorithmic bias**:

1. **The Industrial Heat Trap**:
   * *Problem*: Unsupervised heat models heavily prioritize industrial manufacturing zones (e.g., East Williamsburg IBZ, $+5.6^\circ\text{F}$ offset) due to vast unshaded asphalt and dark roofing.
   * *Trap*: Spending millions on permanent cooling centers in areas with zero residents or pedestrians leaves human lives unprotected.
2. **The Coastal Senior Blind Spot**:
   * *Problem*: Coastal communities (e.g., Coney Island, $+1.8^\circ\text{F}$ offset) appear cooler in afternoon satellite models due to maritime sea breezes.
   * *Trap*: The population is 26.3% seniors over 65, 32% below poverty, and resides in high-rise brick housing without air conditioning.
3. **The Audit Solution**:
   * HeatShield AI's equity engine cross-checks all spatial rankings against demographic vulnerability indices, enforcing an **Equity Disparity Ratio (EDR)** before certifying capital placement plans.

---

## 📁 Repository Structure

```
├── app.py                      # Primary Streamlit web application & Screen 1
├── api_server.py               # FastAPI REST service exposing agent endpoints
├── run_adk.py                  # Google ADK CLI runner for headless agent execution
├── requirements.txt            # Unified project Python dependencies
├── ARCHITECTURE.md             # Complete system architecture specification
├── README.md                   # This comprehensive documentation file
├── agent/
│   ├── agent.py                # Multi-Persona Supervisor & Persona Classifier
│   ├── tools.py                # 11 Google ADK custom climate & spatial tools
│   ├── mcp_toolbox_tools.yaml  # Google MCP Toolbox database tool definitions
│   └── README.md               # Agent-specific documentation
├── screens/
│   ├── screen_2_copilot.py     # Screen 2: Interactive AI Planning Copilot
│   └── screen_3_audit.py       # Screen 3: Municipal Cooling Dispatch & Equity Audit
├── styles/
│   └── custom.css              # Custom dark-navy Streamlit CSS styling
├── data/
│   ├── reference_data.py       # Verified Brooklyn reference metrics & cache
│   └── README.md               # Data ingestion guide
├── notebooks/
│   ├── c1_01_load_explore.ipynb # NOAA Heat Watch ingestion notebook
│   └── bqml_training.sql       # BigQuery ML Boosted Trees Regressor DDL
└── tests/
    ├── test_agent.py           # Unit tests for Google ADK tools & logic
    ├── test_integration.py     # Integration tests for API & personas
    └── test_offline.py         # Offline cache & fallback tests
```

---

## ⚡ Quickstart & Installation

### 1. Prerequisites
* Python 3.10+ (tested through Python 3.14)
* Google Cloud SDK (`gcloud`) with BigQuery access, or run in Demo/Offline mode out of the box.

### 2. Environment Setup
```bash
# Clone the repository
git clone git@github.com:chenjie/A4I2026-challenge-1-walkability-trt-6.git
cd A4I2026-challenge-1-walkability-trt-6

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launching the Applications

#### Option A: Interactive Streamlit Workbench (Recommended)
```bash
streamlit run app.py --server.port 8501
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

#### Option B: Headless FastAPI REST Server
```bash
python api_server.py
```
Serves on [http://localhost:8000](http://localhost:8000) with interactive Swagger documentation at `/docs`.

#### Option C: Google ADK CLI Runner
```bash
python run_adk.py --query "Why is Pitkin Ave hotter than two blocks north?" --persona "City Planner"
```

---

## 🎯 Benchmark Questions Handled

HeatShield AI successfully handles all canonical challenge benchmark inquiries:

1. *"Why is Pitkin Ave & Mother Gaston Blvd so much worse than two blocks north?"*  
   $\rightarrow$ Unpacks `ML.EXPLAIN_PREDICT` feature attributions (canopy deficit $-1.8^\circ\text{F}$, low albedo $+1.4^\circ\text{F}$, canyon heat trap).
2. *"Where should Brooklyn place emergency hydration vans today?"*  
   $\rightarrow$ Identifies unshaded transit hubs with $>+3.5^\circ\text{F}$ offsets and high pedestrian footfall (Broadway Junction, Pitkin Ave).
3. *"Run the algorithmic equity audit on East Williamsburg vs. Coney Island."*  
   $\rightarrow$ Exposes the Industrial Heat Trap (+5.6°F, 0 seniors) vs. Coastal Senior Blind Spot (+1.8°F, 26.3% seniors).
4. *"Where is the closest cooling center to 100 Stone Ave Brooklyn?"*  
   $\rightarrow$ Locates Betsy Head Pool / Brownsville Rec Center and generates Google Maps walking navigation URLs.
5. *"Simulate a 20% tree canopy increase on Fulton Street."*  
   $\rightarrow$ Executes the biophysical simulator and predicts a $-2.3^\circ\text{F}$ microclimate reduction.
6. *"How many days will Brownsville exceed 100°F over the next decade?"*  
   $\rightarrow$ Projects 7-year GHCN daily station frequency distribution and compounding risk trends.
7. *"Generate a City Council Capital Allocation Brief for the 2027 budget."*  
   $\rightarrow$ Synthesizes BQML attributions, demographics, Title VI equity audit scores, and ROI into a formal brief.

---

## 📚 Architecture Deep Dive

For in-depth architectural specifications, biophysical equations, Mermaid diagrams, API schemas, and deployment instructions, see:

👉 **[Read the Full System Architecture Document (ARCHITECTURE.md)](ARCHITECTURE.md)**

---

*Built with passion for urban climate justice by the HeatShield AI Team · Agents for Impact 2026*
