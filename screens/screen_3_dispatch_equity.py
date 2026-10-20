"""
Screen 3: Government Cooling Dispatch & Equity Audit
===================================================
Converts AI recommendations into actionable interventions.
Equips city emergency managers, public health directors, and city council members
with multi-agency resource allocations, 0.5 km coverage walking buffers,
equity disparity benchmarks, BQML feature attributions, and council briefing exports.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from services.data_service import (
    get_intersections,
    get_intersection_details,
    get_agent_recommendations,
    get_heat_explanations,
    get_equity_audit,
    get_equity_gaps
)
from components.dispatch_map import render_dispatch_map
from components.dispatch_roster import render_dispatch_roster
from components.equity_audit import render_equity_audit
from components.dispatch_explanations import render_dispatch_explanations
from components.action_proposal import render_action_proposal

def render_screen_3() -> None:
    """Main render function for Screen 3."""
    city_key = st.session_state.get("selected_city", "brooklyn")
    data_mode = st.session_state.get("data_mode", "Demo")
    city_label = "Brooklyn, New York" if "brooklyn" in city_key else "Dallas, Texas"
    agencies_str = "NYCEM · DOHMH · NYC Parks" if "brooklyn" in city_key else "Dallas OEM · DCHHS · Dallas Parks"

    # Screen Header
    st.markdown(f'''
        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border-radius: 12px; padding: 22px 24px; margin-bottom: 20px; color: #FFFFFF; border: 1px solid #334155; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                        <span style="background: #10B981; color: #064E3B; font-weight: 800; font-size: 0.70rem; padding: 3px 8px; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.05em;">
                            SCREEN 3 · OPERATIONAL DISPATCH
                        </span>
                        <span style="background: #334155; color: #94A3B8; font-weight: 700; font-size: 0.70rem; padding: 3px 8px; border-radius: 4px;">
                            {agencies_str}
                        </span>
                    </div>
                    <h1 style="margin: 0 0 6px 0; font-size: 1.7rem; font-weight: 800; color: #F8FAFC;">
                        Government Cooling Dispatch & Equity Audit
                    </h1>
                    <p style="margin: 0; color: #94A3B8; font-size: 0.88rem; max-width: 780px;">
                        Convert microclimate AI predictions into operational field deployments. Audit coverage radii,
                        defend equity reach under Title VI standards, and export council briefing proposals.
                    </p>
                </div>
                <div style="text-align: right; background: rgba(255,255,255,0.06); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Operational Status</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #34D399;">Active Deployment Ready</div>
                    <div style="font-size: 0.72rem; color: #CBD5E1;">Jurisdiction: <strong>{city_label}</strong></div>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # 1. Load Data
    intersections_df = get_intersections(city_key)
    recommendations = get_agent_recommendations(city_key)
    equity_audit = get_equity_audit(city_key, recommendations)
    equity_gaps = get_equity_gaps(city_key, recommendations, buffer_km=0.8)

    # Ensure selected intersection
    if "selected_intersection_id" not in st.session_state or not st.session_state["selected_intersection_id"]:
        default_id = recommendations[0]["intersection_id"] if recommendations else "BK-01"
        st.session_state["selected_intersection_id"] = default_id

    selected_id = st.session_state["selected_intersection_id"]
    selected_record = get_intersection_details(selected_id, city_key)
    explanations = get_heat_explanations(selected_id)

    # 2. Main Two-Column Layout: Map on Left, Assignments & Explanations on Right
    map_col, roster_col = st.columns([1.15, 0.85])

    with map_col:
        st.markdown('''
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                <h3 style="margin: 0; font-size: 1.1rem; font-weight: 800; color: #0F172A;">
                    Dispatch Map & Catchment Buffer Analysis
                </h3>
                <span style="font-size: 0.74rem; color: #64748B;">
                    0.5 km (500m) Pedestrian Radius
                </span>
            </div>
        ''', unsafe_allow_html=True)

        # Map layer control toggles
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
        with ctrl_col1:
            show_primary_rings = st.checkbox("0.5 km Walk Rings", value=True, key="chk_primary_rings")
        with ctrl_col2:
            show_secondary_rings = st.checkbox("1.0 km Transit Buffer", value=False, key="chk_sec_rings")
        with ctrl_col3:
            show_gaps = st.checkbox("⚠️ Show Equity Gaps", value=True, key="chk_show_gaps")

        render_dispatch_map(
            df=intersections_df,
            recommendations=recommendations,
            equity_gaps=equity_gaps,
            selected_id=selected_id,
            show_primary_rings=show_primary_rings,
            show_secondary_rings=show_secondary_rings,
            show_gaps=show_gaps,
            city_key=city_key
        )

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        # BigQuery ML Explanations under the map
        render_dispatch_explanations(
            explanations=explanations,
            selected_record=selected_record,
            recommendations=recommendations,
            data_mode=data_mode
        )

    with roster_col:
        # Multi-agency Dispatch Assignments Roster
        render_dispatch_roster(
            recommendations=recommendations,
            intersections_df=intersections_df,
            selected_id=selected_id,
            city_key=city_key
        )

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 3. Comprehensive Equity Audit Section (Full Width)
    render_equity_audit(
        audit=equity_audit,
        equity_gaps=equity_gaps,
        city_key=city_key
    )

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 4. Downloadable Action Proposal & Council Memorandum (Full Width)
    render_action_proposal(
        city=city_label,
        recommendations=recommendations,
        equity_audit=equity_audit,
        equity_gaps=equity_gaps
    )
