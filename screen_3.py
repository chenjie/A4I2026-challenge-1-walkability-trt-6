"""
HeatShield AI — Screen 3: Government Cooling Dispatch & Equity Audit
-------------------------------------------------------------------
Google Agents for Impact 2026 Hackathon: Challenge 1 (Urban Heat & Walkability)

Purpose:
  Convert AI recommendations into actionable interventions.

Features:
  - Recommended dispatch locations & multi-agency resource allocations (NYCEM, DOHMH, Parks / Dallas OEM, DCHHS)
  - Interactive Leaflet dispatch map with 0.5 km (500m) walking coverage rings and secondary buffers
  - Spatial equity gaps identification (unserved high-vulnerability hotspots)
  - Algorithmic Title VI / Justice40 equity audit scorecards and demographic benchmarks
  - Downloadable Council Action Proposal (.md), CAD Dispatch Manifest (.csv), and Spec (.json)
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Screen 3 Main Screen Logic
from screens.screen_3_dispatch_equity import render_screen_3

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLESHEET
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HeatShield AI — Screen 3: Cooling Dispatch & Equity Audit",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
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

if "selected_intersection_id" not in st.session_state:
    st.session_state["selected_intersection_id"] = "BK-01"

if "data_mode" not in st.session_state:
    st.session_state["data_mode"] = "Demo"

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('''
        <div style="padding: 10px 0 16px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.9rem;">🛡️</span>
                <div>
                    <h2 style="margin: 0; color: #FFFFFF; font-size: 1.35rem; font-weight: 700;">HeatShield AI</h2>
                    <span style="font-size: 0.70rem; color: #34D399; font-weight: 600;">Urban Resilience Platform</span>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown("<p style='font-size: 0.72rem; text-transform: uppercase; color: #64748B; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 6px;'>Active View</p>", unsafe_allow_html=True)

    st.markdown('''
        <div class="nav-item nav-inactive">
            <span>🔥 Heat Intelligence</span>
            <span style="font-size: 0.68rem; color: #64748B;">Screen 1</span>
        </div>
        <div class="nav-item nav-inactive">
            <span>🤖 AI Planning Copilot</span>
            <span style="font-size: 0.68rem; color: #64748B;">Screen 2</span>
        </div>
        <div class="nav-item nav-active" style="border-left-color: #059669; background-color: #064E3B25;">
            <span style="color: #34D399; font-weight: 700;">⚖️ Cooling Dispatch & Equity Audit</span>
            <span style="font-size: 0.68rem; background: #059669; color: white; padding: 2px 7px; border-radius: 4px; font-weight: 600;">SCREEN 3</span>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #1E293B; margin: 18px 0 14px 0;'>", unsafe_allow_html=True)

    # City Selector
    st.markdown("<p style='font-size: 0.72rem; text-transform: uppercase; color: #64748B; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 4px;'>Analysis Target City</p>", unsafe_allow_html=True)
    city_options = ["New York — Brooklyn", "Dallas, Texas", "Chicago, Illinois (Locked)"]
    selected_city_val = st.selectbox(
        "City",
        city_options,
        index=0 if st.session_state["selected_city"] not in city_options else city_options.index(st.session_state["selected_city"]),
        label_visibility="collapsed"
    )
    st.session_state["selected_city"] = selected_city_val

    # Data Mode Toggle
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    mode_col1, mode_col2 = st.columns([1, 1])
    with mode_col1:
        st.markdown("<span style='font-size: 0.75rem; color: #94A3B8; font-weight: 600;'>Data Mode:</span>", unsafe_allow_html=True)
    with mode_col2:
        is_live = st.toggle("Live BQ", value=False, help="Toggle between Demo (synthetic) mode and Live BigQuery ML backend.")
        st.session_state["data_mode"] = "Live" if is_live else "Demo"

    st.markdown('''
        <div style="background-color: #1E293B; border-radius: 8px; padding: 12px; margin-top: 12px; border: 1px solid #334155;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
                <span style="background: #064E3B; color: #6EE7B7; font-size: 0.68rem; font-weight: 700; padding: 2px 6px; border-radius: 4px;">Audited</span>
                <span style="font-size: 0.70rem; color: #38BDF8; font-weight: 600;">Title VI & Justice40</span>
            </div>
            <p style="font-size: 0.74rem; color: #CBD5E1; margin: 0; line-height: 1.35;">
                Demographic parity verification benchmarked against municipal census baselines.
            </p>
        </div>
    ''', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. RENDER SCREEN 3
# -----------------------------------------------------------------------------
render_screen_3()

# -----------------------------------------------------------------------------
# 5. FOOTER
# -----------------------------------------------------------------------------
st.markdown('''
    <div style="margin-top: 36px; padding: 18px; background-color: #FFFFFF; border-radius: 10px; border: 1px solid #E2E8F0; text-align: center; color: #64748B; font-size: 0.80rem;">
        <p style="margin: 0 0 4px 0; font-weight: 700; color: #0F172A;">
            HeatShield AI — Screen 3: Government Cooling Dispatch & Equity Audit
        </p>
        <p style="margin: 0; line-height: 1.4;">
            Google Agents for Impact 2026 Hackathon · Operational Dispatch & Title VI Civil Rights Equity Auditing.
            Converting microclimate AI into verified municipal interventions.
        </p>
    </div>
''', unsafe_allow_html=True)
