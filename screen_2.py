"""
HeatShield AI — Screen 2: Agentic AI Planning Copilot
-----------------------------------------------------
Google Agents for Impact 2026 Hackathon: Challenge 1 (Urban Heat & Walkability)

Purpose:
  Demonstrate real ADK agent execution with custom tool invocations,
  unmeasured-location predictions, execution timelines, and ranked recommendations.
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path

from screens.screen_2_copilot import render_screen_2

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLESHEET
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HeatShield AI — Screen 2: Agentic AI Planning Copilot",
    page_icon="🤖",
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

if "selected_persona" not in st.session_state:
    st.session_state["selected_persona"] = "City Planner"

if "forecast_high_f" not in st.session_state:
    st.session_state["forecast_high_f"] = 98.0

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="padding: 12px 4px 16px 4px;">
            <div style="font-size: 0.72rem; font-weight: 800; letter-spacing: 0.08em; color: #94A3B8; text-transform: uppercase;">
                Navigation
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <a href="app.py?screen=1" target="_self" class="nav-item nav-inactive">
            <span>🔥 Heat Intelligence</span>
            <span class="badge-soon">Screen 1</span>
        </a>
        <a href="screen_2.py" target="_self" class="nav-item nav-active">
            <span>🤖 AI Planning Copilot</span>
            <span style="font-size: 0.68rem; background: #0284C7; color: white; padding: 2px 7px; border-radius: 4px; font-weight: 600;">ACTIVE</span>
        </a>
        <a href="app.py?screen=3" target="_self" class="nav-item nav-inactive">
            <span>⚖️ Equity Audit</span>
            <span class="badge-soon">Screen 3</span>
        </a>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. RENDER SCREEN 2
# -----------------------------------------------------------------------------
render_screen_2()
