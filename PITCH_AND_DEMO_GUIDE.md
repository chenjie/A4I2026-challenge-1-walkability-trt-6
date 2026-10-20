# Pitch Deck, 3-Minute Demo Script & Judging Guide
**Agents for Impact 2026 — Challenge 1: Community Climate Resilience (Urban Heat)**  
**Project Name:** **CoolBrooklyn AI** — *Attribution-Driven Urban Heat & Equity Siting Workbench*

---

## 1. The 5-Slide Pitch Deck Structure

### Slide 1: The Problem — "Why Heat Maps Alone Fail City Planners"
- **The Stakes:** Extreme heat is the #1 weather-related killer in the U.S. Within **Brooklyn, NY**, two neighborhoods 2.5 miles apart experience an **8.6°F afternoon temperature gap** (`97.4°F` in Brownsville vs. `88.8°F` near Prospect Park).
- **The Planner's Dilemma:** A municipal sustainability director has budget for **3 cooling interventions** this cycle. Looking at a raw heat map fails in two ways:
  1. **It doesn't tell you *what* to build:** Should a hot block get street trees (10-year shade horizon) or a cooling center / cool pavement (this summer)?
  2. **It creates demographic blind spots:** If you rank strictly by physical degrees (`temp_offset_f`), you build your #1 cooling center in the **East Williamsburg Industrial Business Zone (`+5.6°F`)**—cooling concrete warehouses instead of seniors.

### Slide 2: Our Technical Architecture — "Explainable BQML + Google ADK + MCP"
- **4-Source BigQuery Spatial Pipeline:**
  1. `heat_observations`: **85,420** mobile traverse points (spatial pattern on campaign day, `2021-07-24`).
  2. `station_daily_temps`: **2,557 days** of NOAA GHCN-Daily (`USW00094728`, `2019–2025`) to translate spatial offsets into **days/year ≥ 90°F & ≥ 95°F**.
  3. `bigquery-public-data.census_bureau_acs` + `geo_census_tracts`: Land density, pre-1960 housing stock, transit dependence, and the **12-column ACS 65+ senior sum**.
  4. `bigquery-public-data.new_york_trees.tree_census_2015`: Physical street tree density and canopy survival rates (`684,000+` trees).
- **BigQuery ML (`heat_offset_model`):** Predicts **Temperature Offset (`temp_offset_f`)** relative to the citywide mean. Uses **`ML.EXPLAIN_PREDICT`** so every recommendation is backed by per-feature degree attributions (`tree_deficit_attribution_f` vs. `building_heat_trap_attribution_f`).
- **Google ADK + Managed MCP Toolbox (`agent/`):** Multi-tool orchestration with live street geocoding (`geocode_street_or_intersection`), counterfactual site comparison, equity auditing, and Council proposal generation.

### Slide 3: Responsible AI — "Strip the Variable. Audit the Outcome."
- **Step 1 (Strip the Variable):** We explicitly exclude `pct_nonwhite`, `median_income`, and `poverty_rate` from the BQML training features (`scripts/train_bqml.sql`). The model learns **pure urban physics** (canopy deficit, building density, pre-1960 brick heat storage, water proximity).
- **Step 2 (Audit the Outcome):** Our `run_equity_audit` tool joins model predictions back against ACS socioeconomic data post-hoc and uncovers **two critical planning traps**:
  1. **The Industrial Heat Trap (`tract 36047044900`):** East Williamsburg IBZ has the **#1 highest physical offset (`+5.6°F`)**, but only **115 seniors (4.8%)** and an **$89,400 median income**.
  2. **The Coastal Senior Blind Spot (`tract 36047032600`):** Coney Island (`Mermaid Ave & Stillwell Ave`) has only a **moderate physical offset (`+2.9°F`)** due to Atlantic Ocean breezes—so a temperature-only ranking misses it—yet **26.3% of residents (1,365 people) are seniors 65+** and **75.8% have no vehicle**.

### Slide 4: Hybrid Horizon Siting — "This Budget Cycle vs. This Decade"
- **`ML.EXPLAIN_PREDICT` prescribes the exact intervention:**
  - **When `tree_deficit_attribution_f` dominates (`+2.18°F` in Brownsville / `+1.94°F` in Bushwick):** Recommend a **10-Year Street Tree Canopy Corridor** (`45 trees/sq km` target -> `-1.8°F` long-term offset reduction) paired with an immediate **Priority 1 Cooling Center** where senior vulnerability is high.
  - **When `building_heat_trap_attribution_f` or subterranean infrastructure blocks trees (`+1.52°F` under the Broadway elevated train in Bushwick / high-rise towers in Coney Island):** Recommend **This-Summer Cool Pavement Coating + Misting Transit Shelters + Senior AC Retrofit Grants**.

### Slide 5: Actionable Output — The City Council Placement Recommendation
- **Top 3 Recommended Interventions (Budget Cycle 2026):**
  1. **Site #1 — Brownsville (`Pitkin Ave & Mother Gaston Blvd`, Tract `36047091000`):** `+5.4°F` offset (`38 days/yr ≥ 90°F`), `78.4%` carless households, `782` seniors. **Action:** Permanent Cooling Center + High-Albedo Cool Roof/Pavement Retrofits + 10-Yr Pitkin Canopy Corridor.
  2. **Site #2 — East New York (`Livonia Ave & Pennsylvania Ave`, Tract `36047115600`):** `+5.1°F` offset (`35 days/yr ≥ 90°F`), `74.1%` carless, `736` seniors. **Action:** Permanent Cooling Center beneath the elevated Livonia Ave line + Cool Pavement.
  3. **Site #3 — Coney Island (`Mermaid Ave & Stillwell Ave`, Tract `36047032600`):** Elevated from #6 physical rank to **#3 Equity Priority** to protect **1,365 transit-dependent seniors** (`26.3%` of tract population).

---

## 2. 3-Minute Live Demo Script

### Preparation Before Recording / Presenting
1. Start the interactive workbench:
   ```bash
   python app.py
   ```
   Open **http://localhost:8080** in your browser.
2. Have a second tab open (optional) with `adk web` or `scripts/train_bqml.sql` to show the underlying BigQuery ML pipeline.

### Minute 0:00 – 0:45: Spatial Overview & Physical Attribution (`ML.EXPLAIN_PREDICT`)
- **Action:** Point to the interactive Brooklyn Leaflet map on the left. Click the deep-red marker in **Brownsville (`Pitkin Ave & Mother Gaston Blvd`)**, then click the **"1. Hotspots + ML.EXPLAIN_PREDICT"** prompt chip on the right.
- **What to Say:**
  > "This is **CoolBrooklyn AI**, built on Google ADK, MCP Toolbox, and BigQuery ML. A city planner doesn't just need to know *where* it's hot—they need to know *why* it's hot so they pick the right intervention. When we query our `heat_offset_model` with `ML.EXPLAIN_PREDICT`, we see Pitkin & Mother Gaston is **+5.4°F warmer** than the Brooklyn average (`97.4°F` vs `92.0°F`). Crucially, `ML.EXPLAIN_PREDICT` attributes **+2.18°F** of that excess heat to **tree canopy deficit** (only 14.2 trees/sq km) and **+1.45°F** to **building thermal mass**."

### Minute 0:45 – 1:30: Separating Spatial Pattern from Temporal Frequency (7-Year NOAA History)
- **Action:** Click the **"4. 7-Year Frequency"** prompt chip (`"How often does Brownsville actually cross 90°F and 95°F over a 7-year period?"`).
- **What to Say:**
  > "Notice that our mobile traverse dataset (`heat_observations`) covers a single campaign day (`July 24, 2021`). You cannot compute heatwave frequency from one day! Our agent automatically separates **spatial pattern** from **temporal frequency** by joining Brownsville's `+5.4°F` spatial offset onto 7 years (`2,557 days`) of NOAA GHCN-Daily weather station records (`2019–2025`). While the regional station averages **16.7 days/year ≥ 90°F**, Brownsville residents endure **38 days/year ≥ 90°F** and **13.5 days/year ≥ 95°F**—more than double the city baseline."

### Minute 1:30 – 2:15: Responsible AI — "Strip the Variable. Audit the Outcome."
- **Action:** Click the **"3. Equity Audit"** prompt chip (`"Run the equity audit ('Strip the variable. Audit the outcome.')"`) and point to the **Equity Audit Scatter Matrix** in the bottom-left panel.
- **What to Say:**
  > "We implemented the Responsible AI principle *'Strip the variable. Audit the outcome.'* Race and income were excluded from our BQML model features so the model only learns physical heat mechanics. Then, our `run_equity_audit` tool audits who lives in the predicted hotspots. Look at the two anomalies it catches:
  > First, **East Williamsburg IBZ** has the highest physical offset at **+5.6°F**, but it's an industrial warehouse zone with only 115 seniors and an $89k median income—putting a cooling center there wastes taxpayer funds.
  > Second, **Coney Island (`Mermaid & Stillwell`)** is only **+2.9°F** because of ocean breezes, so a temperature-only ranking misses it completely. Yet **26.3% of its residents—1,365 people—are seniors over 65**, and **75.8% have no car**. Our vulnerability-weighted index elevates Coney Island into our top 3 priority sites."

### Minute 2:15 – 3:00: Counterfactual Comparison, Proposal Generation & Scope Guardrail
- **Action:**
  1. Click **"2. Compare 2 Sites"** (`Brownsville vs. East Williamsburg IBZ`) or **"5. Council Proposal"** (`3 cooling centers`).
  2. Click **"7. Test Guardrail"** (`"Predict the exact temperature in Brooklyn on July 24, 2030"`).
- **What to Say:**
  > "Planners can compare any two street intersections side-by-side or generate a budget-ready City Council memo in one click. Finally, watch what happens when we ask the agent to predict the exact temperature on July 24, 2030, or approve a HVAC vendor contract: the agent enforces strict scientific guardrails—refusing point-in-time weather forecasting, explaining the difference between spatial offset and weather models, and pivoting back to evidence-based 7-year historical frequency."

---

## 3. Judge Rubric Verification Checklist

| Judging Criterion | How Our Solution Satisfies It | Where to Show Judges |
| :--- | :--- | :--- |
| **1. Predicts Temperature Offset (Not Raw Temp)** | Target is `temp_offset_f = temp_f - AVG(temp_f) OVER ()`. Raw `temp_f` and `heat_index_f` are stripped from features to prevent data leakage. | [`scripts/train_bqml.sql`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/scripts/train_bqml.sql#L118-L148) |
| **2. Uses `ML.EXPLAIN_PREDICT` for Siting & Intervention Type** | `tract_predictions_explained` runs `ML.EXPLAIN_PREDICT(MODEL heat_offset_model, ...)` and compares tree deficit vs. building mass attributions to prescribe **Tree Corridors** vs. **Cool Pavement / Cooling Centers**. | [`scripts/train_bqml.sql`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/scripts/train_bqml.sql#L194-L260), [`agent/tools.py`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/agent/tools.py#L220-L315) |
| **3. Separates Spatial Pattern (`heat_observations`) from Temporal Frequency (`station_daily_temps`)** | `analyze_heat_frequency` never counts days in `heat_observations` (1 day). It adds the tract's spatial `temp_offset_f` to `2,557` daily `max_temp_f` observations in `station_daily_temps`. | [`agent/tools.py`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/agent/tools.py#L318-L398) |
| **4. 12-Column ACS 65+ Elderly Sum** | Explicitly sums all 6 male + 6 female ACS age brackets (`male_65_to_66` through `female_85_and_over`). | [`scripts/train_bqml.sql`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/scripts/train_bqml.sql#L48-L62) |
| **5. Responsible AI: "Strip the Variable. Audit the Outcome."** | Race/income are excluded from `heat_offset_model` training and joined post-hoc in `tract_equity_audit_lookup` and `run_equity_audit()`. | [`scripts/train_bqml.sql`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/scripts/train_bqml.sql#L265-L310), [`agent/tools.py`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/agent/tools.py#L485-L565) |
| **6. Google ADK + MCP Toolbox Integration** | Full `LlmAgent` in `agent/agent.py` with 8 tools + declarative MCP Toolbox config in `agent/mcp_toolbox_tools.yaml`. | [`agent/agent.py`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/agent/agent.py), [`agent/mcp_toolbox_tools.yaml`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/agent/mcp_toolbox_tools.yaml) |
| **7. Street-Level Specificity & Scope Guardrails** | Includes OpenStreetMap Nominatim + Census Geocoder tool (`geocode_street_or_intersection`) and explicit refusal guardrails for weather forecasting, structural engineering, and procurement. | [`agent/tools.py`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/agent/tools.py#L645-L725), [`app.py`](file:///c:/Users/Akshay/OneDrive/Desktop/Hackathon/A4I2026-challenge-1-walkability-trt-6/app.py#L62-L162) |
