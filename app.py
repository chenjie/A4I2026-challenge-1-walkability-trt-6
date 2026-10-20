"""
HeatShield AI — Screen 1: Brooklyn Heat Intelligence & Dispatch Map
-------------------------------------------------------------------
Google Agents for Impact 2026 Hackathon: Challenge 1 (Urban Heat & Walkability)

A modular, production-grade Streamlit application integrating:
  - Folium + streamlit-folium Leaflet spatial microclimate mapping
  - BigQuery ML ML.EXPLAIN_PREDICT feature attribution visualizer
  - Multi-Agency Cooling Resource Dispatch & 0.5 km Catchment Overlays
  - Multi-Persona Strategic Decision Engine (City Planner, Emergency Management, Forestry)
  - Seamless session-state handoff for Screen 2 (AI Planning Copilot)
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Service Layer
from services.data_service import (
    get_intersections,
    get_intersection_details,
    get_heat_explanations,
    get_agent_recommendations
)

# UI Components
from components.kpi_cards import render_kpi_cards
from components.heat_map import render_heat_map
from components.intersection_details import render_intersection_details
from components.heat_explanations import render_heat_explanations
from components.priority_table import render_priority_table
from components.historical_heat import render_historical_heat
from components.live_climate import render_live_climate

# Screen 2 & Screen 3 Main Screen Logic
from screens.screen_2_copilot import render_screen_2
from screens.screen_3_dispatch_equity import render_screen_3

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & STYLESHEET
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HeatShield AI — Brooklyn Heat Intelligence & Dispatch Map",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).resolve().parent / "styles" / "custom.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = "New York — Brooklyn"

if "selected_persona" not in st.session_state:
    st.session_state["selected_persona"] = "City Planner"

if "selected_intersection_id" not in st.session_state:
    st.session_state["selected_intersection_id"] = "BK-01"

if "selected_question" not in st.session_state:
    st.session_state["selected_question"] = None

if "data_mode" not in st.session_state:
    st.session_state["data_mode"] = "Demo"

if "show_recommendations" not in st.session_state:
    st.session_state["show_recommendations"] = True

if "show_coverage" not in st.session_state:
    st.session_state["show_coverage"] = False

if "show_heat_markers" not in st.session_state:
    st.session_state["show_heat_markers"] = True

if "show_ai_recommended" not in st.session_state:
    st.session_state["show_ai_recommended"] = True

if "show_cooling_resources" not in st.session_state:
    st.session_state["show_cooling_resources"] = True

if "show_coverage_rings" not in st.session_state:
    st.session_state["show_coverage_rings"] = False

if "forecast_high_f" not in st.session_state:
    st.session_state["forecast_high_f"] = 98

if "active_screen" not in st.session_state:
    st.session_state["active_screen"] = "Screen 1"

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & TARGET CITY SELECTION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px 0 16px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.9rem;">🛡️</span>
                <div>
                    <h2 style="margin: 0; color: #FFFFFF; font-size: 1.35rem; font-weight: 700;">HeatShield AI</h2>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


    # Synchronize screen state with query parameters
    if "screen" in st.query_params:
        qp = str(st.query_params["screen"]).lower()
        if "3" in qp or "equity" in qp or "dispatch" in qp:
            st.session_state["active_screen"] = "Screen 3"
        elif "2" in qp or "copilot" in qp or "agent" in qp:
            st.session_state["active_screen"] = "Screen 2"
        elif "1" in qp or "heat" in qp:
            st.session_state["active_screen"] = "Screen 1"

    curr_screen = st.session_state.get("active_screen", "Screen 1")

    st.markdown("<p style='font-size: 0.72rem; text-transform: uppercase; color: #64748B; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 6px;'>Navigation</p>", unsafe_allow_html=True)

    nav_1_class = "nav-item nav-active" if curr_screen == "Screen 1" else "nav-item nav-inactive"
    nav_1_badge = '<span style="font-size: 0.68rem; background: #0284C7; color: white; padding: 2px 7px; border-radius: 4px; font-weight: 600;">ACTIVE</span>' if curr_screen == "Screen 1" else '<span class="badge-soon">Screen 1</span>'

    nav_2_class = "nav-item nav-active" if curr_screen == "Screen 2" else "nav-item nav-inactive"
    nav_2_badge = '<span style="font-size: 0.68rem; background: #0284C7; color: white; padding: 2px 7px; border-radius: 4px; font-weight: 600;">ACTIVE</span>' if curr_screen == "Screen 2" else '<span class="badge-soon">Screen 2</span>'

    nav_3_class = "nav-item nav-active" if curr_screen == "Screen 3" else "nav-item nav-inactive"
    nav_3_badge = '<span style="font-size: 0.68rem; background: #0284C7; color: white; padding: 2px 7px; border-radius: 4px; font-weight: 600;">ACTIVE</span>' if curr_screen == "Screen 3" else '<span class="badge-soon">Screen 3</span>'

    st.markdown(f"""
        <a href="?screen=1" target="_self" class="{nav_1_class}">
            <span>🔥 Heat Intelligence</span>
            {nav_1_badge}
        </a>
        <a href="?screen=2" target="_self" class="{nav_2_class}">
            <span>🤖 AI Planning Copilot</span>
            {nav_2_badge}
        </a>
        <a href="?screen=3" target="_self" class="{nav_3_class}">
            <span>⚖️ Equity Audit</span>
            {nav_3_badge}
        </a>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: #1E293B; margin: 20px 0 14px 0;'>", unsafe_allow_html=True)
    
    # City Selector (Analysis Target City)
    st.markdown("<p style='font-size: 0.72rem; text-transform: uppercase; color: #94A3B8; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 4px;'>Analysis Target City</p>", unsafe_allow_html=True)
    city_options = ["New York — Brooklyn", "Dallas, Texas", "Chicago, Illinois (Locked)"]
    selected_city_val = st.selectbox(
        "City",
        city_options,
        index=0 if st.session_state["selected_city"] not in city_options else city_options.index(st.session_state["selected_city"]),
        label_visibility="collapsed"
    )
    st.session_state["selected_city"] = selected_city_val

    # Decision Persona Selector
    st.markdown("<p style='font-size: 0.72rem; text-transform: uppercase; color: #94A3B8; font-weight: 700; letter-spacing: 0.08em; margin-top: 14px; margin-bottom: 4px;'>Decision Persona</p>", unsafe_allow_html=True)
    persona_options = [
        "City Planner",
        "Emergency Management Officer",
        "Sustainability / Urban Forestry Officer"
    ]
    curr_persona_idx = persona_options.index(st.session_state["selected_persona"]) if st.session_state["selected_persona"] in persona_options else 0
    selected_persona_val = st.selectbox(
        "Decision Persona",
        persona_options,
        index=curr_persona_idx,
        label_visibility="collapsed"
    )
    st.session_state["selected_persona"] = selected_persona_val

    # Data Mode Toggle Indicator
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    mode_col1, mode_col2 = st.columns([1, 1])
    with mode_col1:
        st.markdown("<span style='font-size: 0.75rem; color: #94A3B8; font-weight: 600;'>Data Mode:</span>", unsafe_allow_html=True)
    with mode_col2:
        is_live = st.toggle("Live BQ", value=False, help="Toggle between Demo (synthetic) mode and Live BigQuery ML backend.")
        st.session_state["data_mode"] = "Live" if is_live else "Demo"

    st.markdown("""
        <div style="background-color: #1E293B; border-radius: 8px; padding: 12px; margin-top: 10px; border: 1px solid #334155;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
                <span class="badge-demo">Demo Mode</span>
                <span style="font-size: 0.70rem; color: #38BDF8; font-weight: 600;">NOAA Heat Watch</span>
            </div>
            <p style="font-size: 0.74rem; color: #CBD5E1; margin: 0; line-height: 1.35;">
                Simulated campaign data (~15,000 traverse points). Live BigQuery endpoint: <code>a4i_climate.heat_observations</code>.
            </p>
        </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. DYNAMIC TARGET CITY COMPUTATION & DATA INGESTION
# -----------------------------------------------------------------------------
# Derive dynamic target city heading from selected city in session state
selected_city_raw = st.session_state.get("selected_city", "New York — Brooklyn")
city_clean = selected_city_raw.replace("(Locked)", "").replace("(Active)", "").strip()

if "—" in city_clean:
    city_heading = city_clean.split("—")[-1].strip()
elif "," in city_clean:
    city_heading = city_clean.split(",")[0].strip()
else:
    city_heading = city_clean

screen_heading = f"{city_heading} Heat Intelligence & Dispatch Map"

# Load city-specific dataset
city_key = "dallas" if "dallas" in city_heading.lower() else "brooklyn"
df_intersections = get_intersections(city=city_key)
agent_recommendations = get_agent_recommendations(city=city_key)

# Validate selected ID exists in currently loaded city dataset
all_ids = df_intersections["intersection_id"].tolist()
if st.session_state["selected_intersection_id"] not in all_ids:
    st.session_state["selected_intersection_id"] = all_ids[0]


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 5. SCREEN ROUTER
# -----------------------------------------------------------------------------
if st.session_state["active_screen"] == "Screen 3":
    render_screen_3()
elif st.session_state["active_screen"] == "Screen 2":
    render_screen_2()
else:
    # -----------------------------------------------------------------------------
    # 5. HEADER SECTION (SCREEN 1)
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    header_col1, header_col2 = st.columns([2.8, 1.2])

    with header_col1:
        st.markdown(f"""
            <div style="margin-bottom: 6px;">
                <h1 style="color: #0F172A; font-size: 2.1rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;">
                    {screen_heading}
                </h1>
            </div>
        """, unsafe_allow_html=True)

    with header_col2:
        st.markdown("""
            <div style="text-align: right; padding-top: 4px;">
                <span class="badge-demo" style="font-size: 0.80rem; padding: 4px 12px;">
                    ⚠️ Demo Mode — Synthetic Data
                </span>
            </div>
        """, unsafe_allow_html=True)


    # -----------------------------------------------------------------------------
    # 6. EXECUTIVE FOCUS BANNER (TOP OF MAIN SCREEN)
    # -----------------------------------------------------------------------------
    persona_context = {
        "City Planner": {
            "icon": "🏛️",
            "persona_title": "City Planner",
            "agency": "NYC DOHMH & Department of City Planning",
            "prioritization": "High-heat census tracts (> +3.5°F offset) lacking indoor public cooling facilities within 0.5 km of transit corridors.",
            "permanent_solution": "Capital funding & zoning for permanent DOHMH indoor cooling centers, public facility HVAC retrofits, and Council budget defense.",
            "badge_color": "#1769E0",
            "zoom_desc": "🔭 Zoomed Out (11.5x) · Boroughwide Siting View",
            "zoom_pill_style": "background-color: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE;",
            "kpi_metrics": {
                "title1": "Highest Heat Offset",
                "val1": f"+{df_intersections['predicted_offset_f'].max():.1f}°F",
                "sub1": f"{df_intersections.loc[df_intersections['predicted_offset_f'].idxmax()]['neighborhood']} (Peak Microclimate)",
                "cls1": "hot",
                "title2": "Lowest Heat Offset",
                "val2": f"{df_intersections['predicted_offset_f'].min():.1f}°F",
                "sub2": f"{df_intersections.loc[df_intersections['predicted_offset_f'].idxmin()]['neighborhood']} (Baseline Reference)",
                "cls2": "emerald",
                "title3": "Target Cooling Sites",
                "val3": "3 Sites",
                "sub3": "DOHMH Permanent Centers Needed",
                "cls3": "primary",
                "title4": "Capital Priority Sites",
                "val4": f"{(df_intersections['priority_score'] >= 80).sum()} / {len(df_intersections)}",
                "sub4": "Priority Score ≥ 80 (Urgent)",
                "cls4": "warning"
            }
        },
        "Emergency Management Officer": {
            "icon": "⚡",
            "persona_title": "Emergency Management Officer",
            "agency": "NYC Emergency Management (NYCEM)",
            "prioritization": "Acute heatwave epicenters (offsets ≥ +4.5°F) with high elderly vulnerability (36,100 seniors) and low vehicle access.",
            "permanent_solution": "Permanent municipal misting station hookups, rapid mobile hydration truck staging corridors, and automated wellness dispatch.",
            "badge_color": "#DC2626",
            "zoom_desc": "🎯 Zoomed In (13.8x) · Tactical Dispatch Hotspots",
            "zoom_pill_style": "background-color: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5;",
            "kpi_metrics": {
                "title1": "Acute Heatwave Peak",
                "val1": f"+{df_intersections['predicted_offset_f'].max():.1f}°F",
                "sub1": "Brownsville Heat Epicenter",
                "cls1": "hot",
                "title2": "Tactical Fleet Assigned",
                "val2": "2 Units",
                "sub2": "NYCEM Hydration Trucks Staged",
                "cls2": "hot",
                "title3": "Vulnerable Residents",
                "val3": "36,100",
                "sub3": "High-Risk Seniors in Hotspot Tracts",
                "cls3": "warning",
                "title4": "Peak Zero-Vehicle Rate",
                "val4": f"{df_intersections['no_vehicle_rate'].max():.1f}%",
                "sub4": "Coney Island (Transit Dependent)",
                "cls4": "primary"
            }
        },
        "Sustainability / Urban Forestry Officer": {
            "icon": "🌳",
            "persona_title": "Sustainability / Urban Forestry Officer",
            "agency": "NYC Parks & Mayor's Office of Climate & Environmental Justice",
            "prioritization": "Severe urban heat corridors with tree canopy deficit (< 15% vs 30% goal) and high asphalt impervious surface ratio (> 70%).",
            "permanent_solution": "High-albedo cool pavement resurfacing, continuous street tree canopy expansion, and bioswale micro-cooling corridors.",
            "badge_color": "#059669",
            "zoom_desc": "🌳 Corridor Zoom (12.6x) · Canopy Deficit & Albedo View",
            "zoom_pill_style": "background-color: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0;",
            "kpi_metrics": {
                "title1": "Corridor Heat Deficit",
                "val1": f"+{df_intersections[df_intersections['predicted_offset_f'] > 2.0]['predicted_offset_f'].mean():.1f}°F",
                "sub1": "Avg Offset in Unshaded Strips",
                "cls1": "hot",
                "title2": "Tree Canopy Coverage",
                "val2": "14.2%",
                "sub2": "Severe Deficit vs 30% City Goal",
                "cls2": "emerald",
                "title3": "Impervious Surface Ratio",
                "val3": "74.8%",
                "sub3": "Asphalt & Thermal Mass Burden",
                "cls3": "warning",
                "title4": "Cool Pavement Candidates",
                "val4": "4 Corridors",
                "sub4": "Red Hook, Bushwick, ENY, Brownsville",
                "cls4": "primary"
            }
        }
    }
    p_info = persona_context[st.session_state["selected_persona"]]

    st.markdown(f"""<div class="focus-hero-card" style="border-left: 5px solid {p_info['badge_color']};">
    <div class="focus-meta-grid" style="margin-top: 0;">
    <div class="focus-meta-item" style="justify-content: center; align-items: flex-start;">
    <span style="background-color: {p_info['badge_color']}15; color: {p_info['badge_color']}; font-weight: 800; font-size: 0.92rem; padding: 8px 16px; border-radius: 6px; text-transform: uppercase; letter-spacing: 0.04em; display: inline-flex; align-items: center; gap: 7px; border: 1px solid {p_info['badge_color']}30; white-space: nowrap;">
    {p_info['icon']} PERSONA: {st.session_state['selected_persona'].upper()}
    </span>
    </div>
    <div class="focus-meta-item">
    <span class="focus-meta-label">Lead Agency:</span>
    <span class="focus-meta-val">{p_info['agency']}</span>
    </div>
    <div class="focus-meta-item">
    <span class="focus-meta-label">Prioritization:</span>
    <span class="focus-meta-val">{p_info['prioritization']}</span>
    </div>
    <div class="focus-meta-item">
    <span class="focus-meta-label">Permanent Solution:</span>
    <span class="focus-meta-val" style="color: {p_info['badge_color']}; font-weight: 700;">{p_info['permanent_solution']}</span>
    </div>
    </div>
    </div>""", unsafe_allow_html=True)



    # -----------------------------------------------------------------------------
    # 8. DASHBOARD KPI CARDS (DYNAMICALLY TAILORED TO ACTIVE PERSONA)
    # -----------------------------------------------------------------------------
    render_kpi_cards(df_intersections, data_mode=st.session_state["data_mode"], custom_kpis=p_info["kpi_metrics"])

    # -----------------------------------------------------------------------------
    # 8.5 LIVE CLIMATE INTELLIGENCE (OPEN-METEO & NWS TELEMETRY)
    # -----------------------------------------------------------------------------
    render_live_climate(city=city_key)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)


    # -----------------------------------------------------------------------------
    # 9. MAP INTERACTION CONTROLS (DESIGNER-POLISHED ALIGNMENT & TYPOGRAPHY)
    # -----------------------------------------------------------------------------
    control_col1, control_col2 = st.columns([1.0, 1.0], gap="medium")

    with control_col1:
        st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                <span style="font-size: 0.84rem; font-weight: 700; color: #0F172A;">
                    🌡️ City Forecast High (Illustrative Scenario)
                </span>
                <span style="font-size: 0.72rem; color: #64748B;">
                    Base Reference Temperature
                </span>
            </div>
        """, unsafe_allow_html=True)
        forecast_high_val = st.slider(
            "City Forecast High (Illustrative Scenario)",
            min_value=90,
            max_value=106,
            value=st.session_state["forecast_high_f"],
            step=1,
            format="%d°F",
            label_visibility="collapsed",
            help="Illustrative scenario slider. Combines with predicted offset: scenario_value = forecast_high_f + predicted_offset_f. Not a validated street-level temperature forecast."
        )
        st.session_state["forecast_high_f"] = forecast_high_val

    with control_col2:
        st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                <span style="font-size: 0.84rem; font-weight: 700; color: #0F172A;">
                    📍 Select Intersection to Inspect
                </span>
                <span style="font-size: 0.72rem; color: #1769E0; font-weight: 600;">
                    Map Synchronized
                </span>
            </div>
        """, unsafe_allow_html=True)
        dropdown_labels = [f"{r['intersection_id']}: {r['neighborhood']} ({r['intersection_name']})" for _, r in df_intersections.iterrows()]
        curr_index = 0
        for idx, r in df_intersections.iterrows():
            if r["intersection_id"] == st.session_state["selected_intersection_id"]:
                curr_index = idx
                break
            
        selected_choice = st.selectbox(
            "Select Intersection to Inspect",
            options=dropdown_labels,
            index=curr_index,
            label_visibility="collapsed",
            help="Synchronized with map markers. Updates Intersection Intelligence and Feature Attributions."
        )
        # Extract selected ID
        st.session_state["selected_intersection_id"] = selected_choice.split(":")[0].strip()


    # -----------------------------------------------------------------------------
    # 10. MAIN SECTION: FOLIUM HEAT MAP & DETAILS / EXPLANATIONS PANEL
    # -----------------------------------------------------------------------------
    map_col, details_col = st.columns([1.0, 1.0], gap="medium")

    with map_col:
        st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; flex-wrap: wrap; gap: 4px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <h3 style="margin: 0; font-size: 1.10rem; font-weight: 700; color: #0F172A;">
                        Interactive {city_heading} Heat & Dispatch Map (Leaflet)
                    </h3>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="font-size: 0.72rem; color: #64748B;">NOAA Campaign Canvas</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Interactive Map Layer Controls (4 Toggles)
        lay_col1, lay_col2 = st.columns(2)
        with lay_col1:
            st.session_state["show_heat_markers"] = st.toggle(
                "Show Heat Offset Markers",
                value=st.session_state["show_heat_markers"],
                key="tog_heat_offsets",
                help="Toggle circular heat-offset markers (+5.6°F to -3.1°F)"
            )
            st.session_state["show_cooling_resources"] = st.toggle(
                "Show Proposed Cooling Resources",
                value=st.session_state["show_cooling_resources"],
                key="tog_cooling_resources",
                help="Toggle intervention icons (🚚 Hydration Trucks, ❄️ Cooling Centers, 🌳 Tree Corridors)"
            )
        with lay_col2:
            st.session_state["show_ai_recommended"] = st.toggle(
                "Show AI-Recommended Locations",
                value=st.session_state["show_ai_recommended"],
                key="tog_ai_recommended",
                help="Toggle purple target outline for locations recommended by AI Copilot"
            )
            st.session_state["show_coverage_rings"] = st.toggle(
                "Show Approximate Coverage Rings",
                value=st.session_state["show_coverage_rings"],
                key="tog_coverage_rings",
                help="Toggle 500-meter straight-line service catchment circles"
            )

        render_heat_map(
            df=df_intersections,
            selected_id=st.session_state["selected_intersection_id"],
            recommendations=agent_recommendations,
            show_recommendations=st.session_state["show_ai_recommended"],
            show_coverage=st.session_state["show_coverage_rings"],
            forecast_high_f=st.session_state["forecast_high_f"],
            persona=st.session_state["selected_persona"],
            city_key=city_key,
            show_heat_markers=st.session_state["show_heat_markers"],
            show_ai_recommended=st.session_state["show_ai_recommended"],
            show_cooling_resources=st.session_state["show_cooling_resources"],
            show_coverage_rings=st.session_state["show_coverage_rings"]
        )

    with details_col:
        selected_record = get_intersection_details(st.session_state["selected_intersection_id"], city=city_key)
        if selected_record:
            render_intersection_details(
                intersection=selected_record,
                recommendations=agent_recommendations,
                show_recommendations=st.session_state["show_ai_recommended"],
                forecast_high_f=st.session_state["forecast_high_f"]
            )
        
            # Section 9: Why is this location hotter?
            explanations = get_heat_explanations(st.session_state["selected_intersection_id"])
            render_heat_explanations(explanations, data_mode=st.session_state["data_mode"])


    # -----------------------------------------------------------------------------
    # 11. PRIORITY INTERSECTION TABLE
    # -----------------------------------------------------------------------------
    st.markdown("<hr style='border-color: #E2E8F0; margin: 24px 0 16px 0;'>", unsafe_allow_html=True)
    render_priority_table(df_intersections, selected_id=st.session_state["selected_intersection_id"])


    # -----------------------------------------------------------------------------
    # 12. SEVEN-YEAR HISTORICAL HEAT INTELLIGENCE (GHCN-DAILY)
    # -----------------------------------------------------------------------------
    st.markdown("<hr style='border-color: #E2E8F0; margin: 28px 0 20px 0;'>", unsafe_allow_html=True)
    render_historical_heat(city=city_key, data_mode=st.session_state["data_mode"])




    # -----------------------------------------------------------------------------
    # 13. CLOUD RUN ARCHITECTURE FOOTER
    # -----------------------------------------------------------------------------
    st.markdown("""
        <div style="margin-top: 36px; padding: 18px; background-color: #FFFFFF; border-radius: 10px; border: 1px solid #E2E8F0; text-align: center; color: #64748B; font-size: 0.80rem;">
            <p style="margin: 0 0 4px 0; font-weight: 700; color: #0F172A;">
                HeatShield AI — Screen 1: Brooklyn Heat Intelligence & Dispatch Map
            </p>
            <p style="margin: 0; line-height: 1.4;">
                Google Agents for Impact 2026 Hackathon · Screen 1 Frontend MVP · Ready for Google Cloud Run containerization and BigQuery ML / ADK backend connection.
            </p>
        </div>
    """, unsafe_allow_html=True)
