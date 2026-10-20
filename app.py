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

# Screen 3 Main Screen Logic
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
    st.session_state["show_coverage"] = True

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
        elif "1" in qp or "heat" in qp:
            st.session_state["active_screen"] = "Screen 1"

    is_screen_1 = (st.session_state.get("active_screen", "Screen 1") == "Screen 1")

    st.markdown("<p style='font-size: 0.72rem; text-transform: uppercase; color: #64748B; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 6px;'>Navigation</p>", unsafe_allow_html=True)

    if is_screen_1:
        st.markdown("""
            <a href="?screen=1" target="_self" class="nav-item nav-active">
                <span>🔥 Heat Intelligence</span>
                <span style="font-size: 0.68rem; background: #0284C7; color: white; padding: 2px 7px; border-radius: 4px; font-weight: 600;">ACTIVE</span>
            </a>
            <div class="nav-item nav-inactive">
                <span>🤖 AI Planning Copilot</span>
                <span class="badge-soon">Coming Soon</span>
            </div>
            <a href="?screen=3" target="_self" class="nav-item nav-inactive">
                <span>⚖️ Equity Audit</span>
                <span class="badge-soon">Screen 3</span>
            </a>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <a href="?screen=1" target="_self" class="nav-item nav-inactive">
                <span>🔥 Heat Intelligence</span>
                <span class="badge-soon">Screen 1</span>
            </a>
            <div class="nav-item nav-inactive">
                <span>🤖 AI Planning Copilot</span>
                <span class="badge-soon">Coming Soon</span>
            </div>
            <a href="?screen=3" target="_self" class="nav-item nav-active">
                <span>⚖️ Equity Audit</span>
                <span style="font-size: 0.68rem; background: #0284C7; color: white; padding: 2px 7px; border-radius: 4px; font-weight: 600;">ACTIVE</span>
            </a>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: #1E293B; margin: 20px 0 14px 0;'>", unsafe_allow_html=True)
    
    # City Selector (Analysis Target City)
    st.markdown("<p style='font-size: 0.72rem; text-transform: uppercase; color: #64748B; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 4px;'>Analysis Target City</p>", unsafe_allow_html=True)
    city_options = ["New York — Brooklyn", "Dallas, Texas", "Chicago, Illinois (Locked)"]
    selected_city_val = st.selectbox(
        "City",
        city_options,
        index=0 if st.session_state["selected_city"] not in city_options else city_options.index(st.session_state["selected_city"]),
        label_visibility="collapsed"
    )
    st.session_state["selected_city"] = selected_city_val

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
else:
    # -----------------------------------------------------------------------------
    # 5. HEADER SECTION (SCREEN 1)
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    header_col1, header_col2 = st.columns([2.8, 1.2])

    with header_col1:
        st.markdown(f"""
            <div style="margin-bottom: 2px;">
                <h1 style="color: #0F172A; font-size: 2.1rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;">
                    {screen_heading}
                </h1>
                <p style="color: #475569; font-size: 0.95rem; margin: 4px 0 0 0;">
                    Explore street-level heat differences, community vulnerability, and AI-recommended cooling interventions.
                </p>
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
    # 6. PERSONA SELECTOR
    # -----------------------------------------------------------------------------
    persona_options = [
        "City Planner",
        "Emergency Management Officer",
        "Sustainability / Urban Forestry Officer"
    ]

    try:
        active_persona = st.segmented_control(
            "Active Decision Persona",
            options=persona_options,
            default=st.session_state["selected_persona"],
            label_visibility="collapsed"
        )
        if active_persona:
            st.session_state["selected_persona"] = active_persona
    except AttributeError:
        active_persona = st.radio(
            "Active Decision Persona",
            options=persona_options,
            index=persona_options.index(st.session_state["selected_persona"]),
            horizontal=True,
            label_visibility="collapsed"
        )
        st.session_state["selected_persona"] = active_persona

    persona_context = {
        "City Planner": {
            "focus": "Permanent DOHMH Cooling Center Siting & Council Budget Defense",
            "badge_color": "#1769E0",
            "desc": "Prioritizing permanent public facility investments in transit-accessible, high-offset neighborhoods.",
            "zoom_desc": "🔭 Zoomed Out (11.5x) · Boroughwide Siting View",
            "zoom_pill_style": "background-color: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE;"
        },
        "Emergency Management Officer": {
            "focus": "NYCEM Tactical Hydration Truck Deployment & Senior Wellness Checks",
            "badge_color": "#DC2626",
            "desc": "Deploying emergency mobile assets to streets crossing extreme heat thresholds during hot afternoons.",
            "zoom_desc": "🎯 Zoomed In (13.8x) · Tactical Dispatch Hotspots",
            "zoom_pill_style": "background-color: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5;"
        },
        "Sustainability / Urban Forestry Officer": {
            "focus": "NYC Parks Tree Corridors & Cool Pavement Capital Planning",
            "badge_color": "#059669",
            "desc": "Targeting interventions where impervious surface fraction and canopy deficit drive microclimate heat.",
            "zoom_desc": "🌳 Corridor Zoom (12.6x) · Canopy Deficit & Albedo View",
            "zoom_pill_style": "background-color: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0;"
        }
    }
    p_info = persona_context[st.session_state["selected_persona"]]

    st.markdown(f"""
        <div style="background-color: #FFFFFF; border-radius: 8px; padding: 10px 16px; margin: 8px 0 14px 0; border: 1px solid #E2E8F0; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                <strong style="color: #0F172A; font-size: 0.88rem;">Active Persona:</strong>
                <span style="background-color: {p_info['badge_color']}15; color: {p_info['badge_color']}; font-weight: 700; font-size: 0.82rem; padding: 2px 8px; border-radius: 4px;">
                    {st.session_state['selected_persona']}
                </span>
                <span style="color: #64748B; font-size: 0.82rem;">{p_info['desc']}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="font-size: 0.76rem; font-weight: 700; padding: 3px 8px; border-radius: 4px; {p_info['zoom_pill_style']}">
                    {p_info['zoom_desc']}
                </div>
                <div style="font-size: 0.78rem; color: #334155; font-weight: 600; background: #F1F5F9; padding: 3px 8px; border-radius: 4px;">
                    🎯 <strong>Focus</strong>: {p_info['focus']}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)




    # -----------------------------------------------------------------------------
    # 8. DASHBOARD KPI CARDS
    # -----------------------------------------------------------------------------
    render_kpi_cards(df_intersections, data_mode=st.session_state["data_mode"])

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)


    # -----------------------------------------------------------------------------
    # 9. MAP INTERACTION CONTROLS
    # -----------------------------------------------------------------------------
    control_col1, control_col2 = st.columns([1.2, 1.2], gap="large")

    with control_col1:
        forecast_high_val = st.slider(
            "City Forecast High (Illustrative Scenario)",
            min_value=90,
            max_value=106,
            value=st.session_state["forecast_high_f"],
            step=1,
            format="%d°F",
            help="Illustrative scenario slider. Combines with predicted offset: scenario_value = forecast_high_f + predicted_offset_f. Not a validated street-level temperature forecast."
        )
        st.session_state["forecast_high_f"] = forecast_high_val

    with control_col2:
        # Synchronized Dropdown for reliable selection
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
            help="Synchronized with map markers. Updates Intersection Intelligence and Feature Attributions."
        )
        # Extract selected ID
        st.session_state["selected_intersection_id"] = selected_choice.split(":")[0].strip()


    # -----------------------------------------------------------------------------
    # 10. MAIN SECTION: FOLIUM HEAT MAP & DETAILS / EXPLANATIONS PANEL
    # -----------------------------------------------------------------------------
    map_col, details_col = st.columns([1.65, 1.0], gap="medium")

    with map_col:
        st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; flex-wrap: wrap; gap: 4px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <h3 style="margin: 0; font-size: 1.10rem; font-weight: 700; color: #0F172A;">
                        Interactive {city_heading} Heat & Dispatch Map (Leaflet)
                    </h3>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 4px; {p_info['zoom_pill_style']}">
                        {p_info['zoom_desc']}
                    </span>
                    <span style="font-size: 0.72rem; color: #64748B;">NOAA Campaign Canvas</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        render_heat_map(
            df=df_intersections,
            selected_id=st.session_state["selected_intersection_id"],
            recommendations=agent_recommendations,
            show_recommendations=st.session_state["show_recommendations"],
            show_coverage=st.session_state["show_coverage"],
            forecast_high_f=st.session_state["forecast_high_f"],
            persona=st.session_state["selected_persona"],
            city_key=city_key
        )

    with details_col:
        selected_record = get_intersection_details(st.session_state["selected_intersection_id"], city=city_key)
        if selected_record:
            render_intersection_details(
                intersection=selected_record,
                recommendations=agent_recommendations,
                show_recommendations=st.session_state["show_recommendations"],
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
    # 12. PERSONA-SPECIFIC QUESTIONS (HANDOFF FOR SCREEN 2: AI PLANNING COPILOT)
    # -----------------------------------------------------------------------------
    st.markdown("<hr style='border-color: #E2E8F0; margin: 24px 0 16px 0;'>", unsafe_allow_html=True)

    st.markdown("""
        <div style="margin-bottom: 8px;">
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #0F172A;">
                Questions to Explore with HeatShield AI
            </h3>
            <p style="margin: 2px 0 12px 0; color: #64748B; font-size: 0.82rem;">
                Click a persona-tailored question to stage it in session state for the <strong>AI Planning Copilot</strong> (Screen 2).
            </p>
        </div>
    """, unsafe_allow_html=True)

    persona_questions = {
        "City Planner": [
            "Where should we establish three cooling centers this summer?",
            "Why is this intersection hotter than nearby locations two blocks north?",
            "Which communities should receive capital resilience investment first?"
        ],
        "Emergency Management Officer": [
            "Which intersections should receive hydration trucks tomorrow?",
            "Where should senior wellness checks be prioritized during peak heat?",
            "Which communities have high heat exposure and limited vehicle access?"
        ],
        "Sustainability / Urban Forestry Officer": [
            "Which locations should be evaluated for urban tree canopy expansion?",
            "Where does impervious surface contribute most to microclimate heat?",
            "Which unshaded commercial corridors are primary candidates for cool pavement?"
        ]
    }

    questions = persona_questions.get(st.session_state["selected_persona"], persona_questions["City Planner"])
    q_col1, q_col2, q_col3 = st.columns(3)

    for idx, (col, q_text) in enumerate(zip([q_col1, q_col2, q_col3], questions)):
        with col:
            if st.button(f"💬 {q_text}", key=f"btn_q_{idx}", use_container_width=True):
                st.session_state["selected_question"] = q_text
                st.toast(f"Staged question for Copilot: '{q_text}'", icon="🤖")

    if st.session_state["selected_question"]:
        st.markdown(f"""
            <div style="background-color: #F0FDF4; border: 1px solid #86EFAC; border-radius: 8px; padding: 12px 16px; margin-top: 10px; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-size: 0.75rem; font-weight: 700; color: #166534; text-transform: uppercase;">
                        Ready for AI Planning Copilot (Screen 2)
                    </span>
                    <p style="margin: 2px 0 0 0; color: #14532D; font-weight: 600; font-size: 0.88rem;">
                        "{st.session_state['selected_question']}"
                    </p>
                    <span style="font-size: 0.72rem; color: #15803D;">
                        Context attached: Persona = <strong>{st.session_state['selected_persona']}</strong> | Target Location = <strong>{st.session_state['selected_intersection_id']} ({st.session_state['selected_city']})</strong>
                    </span>
                </div>
                <span style="background-color: #16A34A; color: white; font-size: 0.72rem; font-weight: 700; padding: 4px 10px; border-radius: 6px;">
                    STAGED IN SESSION STATE
                </span>
            </div>
        """, unsafe_allow_html=True)


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
