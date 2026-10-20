# CoolBrooklyn AI — 6-Slide Demo Deck & Speaker Script

**Interactive HTML Deck:** Open [`slides.html`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/slides.html) directly in your browser (or visit `http://localhost:8080/slides` when running `python app.py`).
- Press **`←` / `→`** or **`Space`** to switch slides.
- Press **`N`** to toggle the floating **Presenter Script / Speaker Notes** drawer.
- Press **`Ctrl + P`** (`Save as PDF`, Landscape, Background Graphics ON) to export a crisp 6-slide PDF for the AngelHack portal.

---

## Slide 1: Title & The Brooklyn Urban Heat Crisis
**Headline:** CoolBrooklyn AI — Why Raw Heat Maps Fail City Planners  
**Subtitle:** Attribution-Driven Urban Heat Siting & Multi-Agency Cooling Dispatch (`New York — Brooklyn`)

### Key Visual Metrics (4 Hero Cards)
- **`15.7°F` Intra-Borough Heat Spread:** Across `15,142` street-level traverse points (`99.8°F` max vs. `84.1°F` min at the exact same afternoon hour).
- **`+5.4°F` vs. `-2.4°F` (2.5 Miles Apart):** **Brownsville** (`Pitkin Ave & Mother Gaston Blvd`, `+5.4°F` offset, `6` trees on block) vs. **Park Slope** (`7th Ave & 3rd St`, `-2.4°F` offset, `46` trees on block).
- **`761` Census Tracts (`2.64M` Residents):** Joined with `684,000+` NYC street trees, 12-column ACS `65+` senior population brackets, and 7-year NOAA GHCN daily station records (`2019–2025`).
- **`3` Municipal Decision Clocks:** Orchestrates **NYCEM** (48-hr mobile hydration), **DOHMH/DFTA** (this-summer cooling centers & senior checks), and **NYC Parks / DOT** (capital tree corridors & cool pavement).

### Left vs. Right Comparison
- **❌ Why Standard Heat Maps Fail City Planners:**
  1. **They don't tell you *WHAT* to build:** A raw hotspot map looks identical whether excess heat comes from **missing street tree canopy** (needs a 10-year NYC Parks tree corridor) or **elevated subway vaults & tenement masonry** where trees cannot be planted (needs NYC DOT high-albedo cool pavement & immediate cooling centers).
  2. **The Industrial Warehouse Trap:** Ranking strictly by physical degrees sends your `#1` cooling center to **East Williamsburg IBZ (`+5.6°F`)**—cooling concrete truck warehouses with only `115` seniors—while missing **1,365 carless seniors in Coney Island (`+2.9°F`)**.
- **✅ How CoolBrooklyn AI Solves It:**
  1. **Predicts Offset (`temp_offset_f`), Never Raw Temp:** BigQuery ML `heat_offset_model` ($R^2 = 0.642$, $\text{MAE} = 1.18^\circ\text{F}$) with zero data leakage.
  2. **Diagnoses Physical Drivers (`ML.EXPLAIN_PREDICT`):** Decomposes every street's offset into degree contributions.
  3. **End-to-End Agentic Dispatch:** Live unmeasured-street inference + 7-year NOAA frequency + post-hoc equity audit + 4-agency work-order dispatch.

> **🎤 Speaker Script (0:00 – 0:30):**
> *"Extreme heat is the #1 weather-related killer in the U.S. In Brooklyn, 15,142 street-level readings reveal a 15.7°F spread at the exact same hour—Brownsville runs +5.4°F hotter than the city mean while Park Slope runs -2.4°F cooler. Standard heat maps fail city planners in two ways: they don't explain WHY a street is hot, and ranking by raw heat alone sends your #1 cooling center to an empty industrial warehouse zone in East Williamsburg while missing 1,365 vulnerable seniors in Coney Island. CoolBrooklyn AI solves both."*

---

## Slide 2: System Architecture & The 3 Core Technical Foundations
**Headline:** 4-Source BigQuery ML Pipeline + Google ADK & MCP Orchestration  
**Subtitle:** Built to Nail Offset Prediction, Unmeasured Street Inference, and Spatial-Temporal Fusion

### Core Prediction Formula Banner
$$\text{Street Feel } (103.4^\circ\text{F}) = \text{Regional Forecast High } (98.0^\circ\text{F}) + \text{BQML Predicted Offset } (+5.4^\circ\text{F}\text{ vs. }91.4^\circ\text{F Mean})$$
*(Zero Data Leakage: raw `temp_f`, `heat_index_f`, race, and income are strictly excluded from `a4i_climate.heat_offset_model` training inputs.)*

### 3 Pillars (Directly Answering the 3 Judge Questions)
1. **Live Prediction on Unmeasured Streets (`Rule #1 & #2`):**
   - Clicking **any unmeasured coordinate `(lat, lon)`** on our Leaflet map triggers [`predict_location_heat_offset`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/agent/tools.py#L283-L387).
   - Performs Inverse-Distance-Weighted (`IDW`) spatial feature interpolation (`canopy_deficit_pct`, `housing_density_per_km2`, `local_tree_scarcity`, `building_age_years`, `inland_dist_km`) and runs live `ML.EXPLAIN_PREDICT` attribution—returning `"is_unmeasured_street_prediction": True`.
2. **Spatial (`WHERE`) vs. Temporal (`HOW OFTEN`) (`Rule #3`):**
   - **Why `observed_at = NULL` is Expected by Design:** `heat_observations` (`15,142` points) is a **single-day 1-hour campaign snapshot** resolving **WHERE** is hotter.
   - **7-Year NOAA GHCN Fusion (`2019–2025`):** [`get_station_heat_frequency`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/agent/tools.py#L510-L625) adds each site's spatial offset to `2,557` daily station highs (`station_daily_temps`).
   - **Finding:** Regional stations average `7.0 days/yr ≥ 95°F`, but Brownsville (`+5.4°F`) endures **23.1 days/yr ≥ 95°F (`3.3x`)**, with baseline hot days rising **+36.3%** from 2019–2021 to 2023–2025.
3. **Dual-Protocol Google ADK + MCP Toolbox Stack:**
   - **Data Layer:** Joins Heat Watch + 7-yr GHCN + **12-column ACS 65+ sum** (`male_65_to_66` through `female_85_and_over`) + **684k+ NYC Street Trees**.
   - **Agent Layer:** Google ADK `LlmAgent` ([`agent/agent.py`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/agent/agent.py)) + Declarative MCP Toolbox ([`agent/mcp_toolbox_tools.yaml`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/agent/mcp_toolbox_tools.yaml)) + 8 Python BQML tools ([`agent/tools.py`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/agent/tools.py)).

> **🎤 Speaker Script (0:30 – 1:05):**
> *"Our architecture nails three foundational rules: First, our BigQuery ML model predicts a temperature OFFSET (+5.4°F above the 91.4°F campaign mean), never raw temperature, with zero data leakage. Second, clicking any unmeasured street on our map calls predict_location_heat_offset to interpolate urban morphology and run live ML.EXPLAIN_PREDICT. Third, we separate WHERE from HOW OFTEN: observed_at is NULL in heat_observations because it's a single-day 1-hour snapshot, so our agent fuses that spatial offset with 7 years of daily NOAA GHCN station records—proving Brownsville endures 23.1 days per summer above 95°F, 3.3 times the official airport count."*

---

## Slide 3: Why `ML.EXPLAIN_PREDICT` Changes the Physical Fix
**Headline:** Why `ML.EXPLAIN_PREDICT` Changes the Engineering Fix on Every Street  
**Subtitle:** Per-Feature Degree Attributions Automatically Route Street Tree Corridors vs. Cool Pavement & Cooling Centers

### 3-Column Attribution Waterfall Comparison
| Street & Tract | Predicted Offset | Top `ML.EXPLAIN_PREDICT` Degree Attributions | Prescribed Engineering Fix |
| :--- | :--- | :--- | :--- |
| **BK-01 • Brownsville**<br>`Pitkin Ave & Mother Gaston Blvd`<br>Tract `36047091000` | **`+5.4°F`** | 1. `canopy_deficit_pct`: **`+2.35°F`** (`91.2%` deficit, `295 trees/km²`)<br>2. `housing_density_per_km2`: **`+1.62°F`**<br>3. `local_tree_scarcity`: **`+0.88°F`** (Only `6` trees on block)<br>4. `building_age_years`: **`+0.43°F`** (`74` yrs) | **`canopy_deficit_pct` dominates** $\rightarrow$ **NYC Parks 180-Tree Corridor (`$216k`)** + **Priority #1 Permanent Cooling Center (`$420k`)** for `986` seniors (`78.6%` carless). |
| **BK-03 • Bushwick**<br>`Myrtle Ave & Broadway`<br>Tract `36047042900` | **`+4.9°F`** | 1. `housing_density_per_km2`: **`+1.84°F`** (Dense tenements + elevated rail)<br>2. `canopy_deficit_pct`: **`+1.92°F`** (`85.4%` deficit)<br>3. `building_age_years`: **`+0.64°F`** (`82` yrs)<br>4. `local_tree_scarcity`: **`+0.38°F`** (Subway vaults restrict pits) | **Tenement canyons + elevated J/M/Z vaults block standard tree pits** $\rightarrow$ **NYC DOT High-Albedo Cool Pavement + Misting Transit Plaza + Cooling Center (`$552k`)**. |
| **BK-08 • Park Slope (Benchmark)**<br>`7th Ave & 3rd St`<br>Tract `36047012901` | **`-2.4°F`** | 1. `canopy_deficit_pct`: **`-2.15°F`** (`1,680 trees/km²` mature canopy)<br>2. `local_tree_scarcity`: **`-1.10°F`** (`46` trees on block arch)<br>3. `water_area_pct`: **`-0.15°F`** (Park lake)<br>4. `housing_density` + `age`: **`+1.00°F`** | **Empirical Proof:** Despite 95-year-old masonry brownstones, `1,680 trees/km²` delivers **`-3.25°F` of combined canopy cooling**. |

> **🎤 Speaker Script (1:05 – 1:40):**
> *"Why use ML.EXPLAIN_PREDICT? Because two blocks can both be +5°F hotter yet require opposite physical interventions. At Pitkin & Mother Gaston in Brownsville (+5.4°F), canopy deficit is the #1 attribution at +2.35°F—there are only 6 live trees on the block—so our agent routes a 180-tree corridor to NYC Parks paired with a permanent cooling center. At Myrtle & Broadway in Bushwick (+4.9°F), tenement thermal mass (+1.84°F) and the elevated J/M/Z train vaults restrict deep tree pits, so the agent prescribes NYC DOT high-albedo cool pavement and transit plaza misting."*

---

## Slide 4: Responsible AI — *"Strip the Variable. Audit the Outcome."*
**Headline:** "Strip the Variable. Audit the Outcome."  
**Subtitle:** How Post-Hoc Equity Auditing Avoids the Industrial Warehouse Trap & Rescues 1,365 Coney Island Seniors

### The Two Anomalies Caught by [`run_equity_audit`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/agent/tools.py#L705-L785)
- **🚫 Trap Avoided — East Williamsburg IBZ (`BK-07`, Tract `36047044900`):**
  - Has the **#1 highest raw physical heat offset in Brooklyn (`+5.6°F`)**, `96.5%` canopy deficit.
  - **Audit Finding:** Industrial Business Zone of concrete warehouses and truck yards with only **`115` seniors (`8.1%`)** and an **`$89,400` median income**.
  - **Decision:** Rejects spending `$400k` on a public cooling center in an industrial yard; routes to commercial Cool Roofs zoning mandate instead.
- **✅ Blind Spot Caught — Coney Island (`BK-06`, Tract `36047034000`):**
  - Only a **moderate physical offset (`+2.9°F`, Rank #6)** because Atlantic Ocean breezes provide `-1.00°F` cooling moderation.
  - **Audit Finding:** Home to **`1,365` seniors (`26.3%` age 65+)** and **`75.8%` carless households** (`$32,400` median income) in 1960s high-rise towers vulnerable to elevator/power outages.
  - **Decision:** Triggers a **Dedicated Senior Equity Override** at Surf Ave & Stillwell Ave.

### Post-Hoc Equity Audit Matrix (`tract_equity_audit_lookup`)
| Siting Strategy | Avg Offset | Seniors (65+) Protected / Tract | Carless Households | Median Income | Poverty Rate | Post-Hoc BIPOC Share |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **✅ CoolBrooklyn AI Portfolio (Vulnerability-Weighted)** | **`+5.0°F`** | **`1,069` seniors / tract** | **`75.5%`** | **`$36,225`** | **`34.1%`** | **`90.6%` BIPOC** |
| **❌ Naive Raw Heat-Only Top Sites (Includes IBZ Trap)** | `+5.3°F` | `678` seniors / tract (`-37%`) | `68.9%` | `$54,300` | `26.1%` | `78.1%` BIPOC |
| **📊 Brooklyn Borough Baseline (`761` Tracts)** | `0.0°F` | `518` seniors / tract | `56.2%` | `$74,692` | `17.8%` | `63.4%` BIPOC |

> **🎤 Speaker Script (1:40 – 2:15):**
> *"We operationalized 'Strip the Variable. Audit the Outcome.' Race, income, and poverty are completely stripped from BQML training so the model learns pure urban physics. Then, our post-hoc equity auditor joins predictions against ACS demographics and catches two major planning traps: First, it rejects East Williamsburg IBZ—the hottest raw offset at +5.6°F, but an industrial warehouse zone with only 115 seniors. Second, it rescues Coney Island—where ocean breezes mask the raw offset at +2.9°F, hiding 1,365 low-income, carless seniors living in high-rise towers."*

---

## Slide 5: Multi-Agency Gov Cooling Dispatch Across 3 Decision Clocks
**Headline:** Multi-Agency Municipal Cooling Dispatch Across 3 Clocks  
**Subtitle:** How [`dispatch_gov_cooling_action`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/agent/tools.py#L1030-L1170) Connects BigQuery ML Attributions Directly to NYC Operations

### 3 Decision Clocks Orchestrated by the Agent
1. **🚨 Clock 1 — Next 48 Hours (`NYC Emergency Management & DEP`):**
   - **Trigger:** Projected street-level feel (`Forecast High + Predicted Offset`) $\ge 98.0^\circ\text{F}$ (e.g., `103.4°F` at Pitkin Ave on a `98°F` forecast day).
   - **Action:** Dispatches **2 Mobile Hydration Vans**, solar misting stations at subway/bus transfers, and DEP fire-hydrant spray caps.
2. **🏥 Clock 2 — This Summer (`NYC DOHMH, DFTA & Brooklyn Public Library`):**
   - **Trigger:** High `no_vehicle_pct` (`>70%`) + high 12-column `pop_over_65` count.
   - **Action:** Extends library/senior center cooling hours to **10:00 PM** and dispatches **DFTA bilingual door-to-door senior wellness checks** (`4,231` seniors across top sites).
3. **🌳 Clock 3 — Capital Budget Cycle & Decade (`NYC Parks` vs. `NYC DOT`):**
   - **Trigger:** Top `ML.EXPLAIN_PREDICT` feature attribution.
   - **Action:** Routes **Street Tree Canopy Corridors** to `NYC Parks` when `canopy_deficit_pct` is `#1` (`-1.8°F to -2.3°F` projected maturity cooling), or **High-Albedo Cool Pavement & Depaving** to `NYC DOT` when `housing_density_per_km2` / subway vaults dominate.

> **🎤 Speaker Script (2:15 – 2:40):**
> *"Instead of giving generic chatbot advice, our agent executes dispatch_gov_cooling_action to route structured work orders across three municipal clocks: Clock 1 dispatches NYC Emergency Management hydration vans within 48 hours whenever street feel crosses 98°F; Clock 2 activates DOHMH extended cooling hours and door-to-door senior checks; and Clock 3 routes capital tree orders to NYC Parks or cool pavement orders to NYC DOT based on each street's top ML.EXPLAIN_PREDICT driver."*

---

## Slide 6: `$1.85M` City Council Capital Plan & Honest Scientific Guardrails
**Headline:** `$1.85M` City Council Placement Plan & Scientific Guardrails  
**Subtitle:** Ready-to-Fund FY2027 Municipal Capital Allocation + Honest Model Limitations

### FY2027 Funded Portfolio (`$1,847,000` Total Capital Budget)
| Priority Rank & Site | Offset & 7-Yr Freq (`≥95°F`) | Primary `ML.EXPLAIN_PREDICT` Driver | Vulnerability Profile | Prescribed Agency Fix | Capital Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **#1 Brownsville (`BK-01`)**<br>`Pitkin Ave & Mother Gaston Blvd` | **`+5.4°F`**<br>`23.1 days/yr ≥95°F` | `canopy_deficit_pct` (**`+2.35°F`**)<br>Only `6` trees on block | **`986` seniors** (`18.2%`)<br>`78.6%` carless • `$31.2k` inc | Permanent Cooling Center + 180-Tree Corridor + Cool Pavement | **`$636,000`** |
| **#2 East New York (`BK-02`)**<br>`Fulton St & Pennsylvania Ave` | **`+5.1°F`**<br>`21.8 days/yr ≥95°F` | `canopy_deficit_pct` (**`+2.18°F`**)<br>`88.5%` arterial deficit | **`1,038` seniors** (`16.8%`)<br>`72.4%` carless • `$36.8k` inc | Permanent Cooling Center + 220-Tree Broadway Junction Corridor | **`$659,000`** |
| **#3 Bushwick (`BK-03`)**<br>`Myrtle Ave & Broadway` | **`+4.9°F`**<br>`20.4 days/yr ≥95°F` | `housing_density` (**`+1.84°F`**)<br>Elevated J/M/Z rail + tenements | **`842` seniors** (`14.3%`)<br>`76.1%` carless • `$44.5k` inc | High-Albedo Cool Pavement under Elevated Rail + Cooling Center | **`$552,000`** |
| **⚡ Equity Override: Coney Island (`BK-06`)**<br>`Surf Ave & Stillwell Ave` | **`+2.9°F`**<br>Ocean breeze (`-1.0°F`) | `canopy_deficit_pct` (**`+1.72°F`**)<br>High-rise concrete plazas | **`1,365` seniors (`26.3%`)**<br>`75.8%` carless • `$32.4k` inc | Dedicated Senior Cooling Center + Backup Power Resilience Grant | **DOHMH / DFTA Ops** |

### 3 Honest Scientific Limitations & Scope Guardrails
1. **Single-Day Spatial Traverse vs. Multi-Year Temporal Station Record:** `heat_observations` captures a 1-hour afternoon traverse on a single summer day (`observed_at = NULL`); multi-year frequency (`days/yr ≥ 95°F`) relies on joining our 7-year NOAA GHCN station baseline.
2. **Observational Attribution vs. Causal Simulation:** `ML.EXPLAIN_PREDICT` on `LINEAR_REG` ($R^2 = 0.642$) diagnoses cross-tract physical drivers, not a dynamic fluid-dynamics (CFD) simulation of post-planting evapotranspiration.
3. **Hard Refusal Guardrails in Live Chat:** Our agent actively refuses out-of-footprint cities (e.g., *Phoenix*), refuses point-in-time future weather predictions (e.g., *July 24, 2030*), and evaluates `0.5 km` senior pedestrian walk catchments rather than naive tract centroids.

> **🎤 Speaker Script (2:40 – 3:00):**
> *"Finally, Slide 6 delivers our budget-ready $1.85M City Council Capital Plan protecting 4,231 seniors across Brownsville, East New York, Bushwick, and Coney Island—alongside three explicit scientific guardrails: our agent distinguishes single-day spatial traverses from 7-year temporal frequency, separates observational ML.EXPLAIN_PREDICT drivers from causal CFD simulation, and enforces hard refusal guardrails against out-of-footprint cities or future weather forecasting."*
