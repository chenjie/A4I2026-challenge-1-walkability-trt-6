"""
Agency Dispatch Roster Component for Screen 3
---------------------------------------------
Displays multi-agency assignments, operational statuses, and resource allocations
across Emergency Management, Public Health, and Parks departments.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any

def render_dispatch_roster(
    recommendations: List[Dict[str, Any]],
    intersections_df: pd.DataFrame,
    selected_id: str,
    city_key: str = "brooklyn"
) -> str:
    """
    Renders filterable Multi-Agency Dispatch Roster with operational metrics,
    demographics reach, and interactive site selection.
    """
    st.markdown('''
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 2px;">
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #0F172A;">
                    Recommended Cooling Dispatch Locations & Agency Assignments
                </h3>
                <span style="font-size: 0.74rem; color: #1769E0; font-weight: 700;">
                    ADK Multi-Agent Inter-Agency Matrix
                </span>
            </div>
            <p style="margin: 0 0 12px 0; color: #64748B; font-size: 0.82rem;">
                Operational staging roster assigning designated municipal agencies to AI-identified microclimate heat hot spots.
            </p>
        </div>
    ''', unsafe_allow_html=True)

    if not recommendations:
        st.info("No dispatch recommendations found for this jurisdiction.")
        return selected_id

    # 1. Filter Bar
    filter_col1, filter_col2, filter_col3 = st.columns([1.2, 1.2, 1.0])
    
    # Extract unique agencies & types
    agencies = sorted(list(set(r.get("agency", "Unknown") for r in recommendations)))
    agency_options = ["All Agencies"] + agencies

    interventions = sorted(list(set(r.get("intervention", "Unknown") for r in recommendations)))
    intervention_options = ["All Types"] + interventions

    priority_options = ["All Priorities", "Urgent", "High", "Moderate"]

    with filter_col1:
        sel_agency = st.selectbox("Filter Agency", options=agency_options, index=0, key="dispatch_agency_filter")
    with filter_col2:
        sel_intervention = st.selectbox("Filter Intervention", options=intervention_options, index=0, key="dispatch_type_filter")
    with filter_col3:
        sel_priority = st.selectbox("Filter Priority", options=priority_options, index=0, key="dispatch_prio_filter")

    # Apply filters
    filtered_recs = []
    for r in recommendations:
        if sel_agency != "All Agencies" and r.get("agency") != sel_agency:
            continue
        if sel_intervention != "All Types" and r.get("intervention") != sel_intervention:
            continue
        if sel_priority != "All Priorities" and r.get("priority") != sel_priority:
            continue
        filtered_recs.append(r)

    st.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)

    # 2. Render Cards / Table
    for r in filtered_recs:
        r_id = r["intersection_id"]
        matched = intersections_df[intersections_df["intersection_id"] == r_id]
        if not matched.empty:
            m_row = matched.iloc[0]
            offset_str = m_row["offset_str"]
            vuln_str = m_row["vuln_pop_str"]
            no_veh_str = m_row["no_veh_str"]
            prio_score = m_row["priority_score"]
        else:
            offset_str = "N/A"
            vuln_str = "N/A"
            no_veh_str = "N/A"
            prio_score = 75

        is_selected = (r_id == selected_id)
        border_color = "#1769E0" if is_selected else "#E2E8F0"
        bg_card = "#F8FAFC" if is_selected else "#FFFFFF"

        agency_name = r.get("agency", "AGENCY")
        agency_cls = "agency-nycem" if ("NYCEM" in agency_name or "OEM" in agency_name) else ("agency-dohmh" if ("DOHMH" in agency_name or "HEALTH" in agency_name or "DCHHS" in agency_name) else "agency-parks")

        status_text = r.get("status", "Staged for Dispatch")
        status_bg = "#DCFCE7" if "Operational" in status_text else ("#FEF3C7" if "Staged" in status_text else "#E0E7FF")
        status_fg = "#166534" if "Operational" in status_text else ("#92400E" if "Staged" in status_text else "#3730A3")

        card_col_left, card_col_right = st.columns([3.8, 1.2])

        with card_col_left:
            st.markdown(f'''
                <div style="background: {bg_card}; border: 1.5px solid {border_color}; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px; flex-wrap: wrap; gap: 6px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 0.92rem; font-weight: 800; color: #0F172A;">{r_id}: {r.get('neighborhood')}</span>
                            <span class="agency-badge {agency_cls}">{agency_name}</span>
                            <span style="background: {status_bg}; color: {status_fg}; font-size: 0.70rem; font-weight: 700; padding: 2px 7px; border-radius: 4px;">
                                {status_text}
                            </span>
                        </div>
                        <div>
                            <span style="font-size: 0.76rem; font-weight: 700; color: #DC2626; background: #FEE2E2; padding: 2px 8px; border-radius: 4px;">
                                Heat Offset: {offset_str}
                            </span>
                        </div>
                    </div>
                    <div style="font-size: 0.86rem; font-weight: 700; color: #1E293B; margin-bottom: 4px;">
                        🎯 {r.get('intervention_title')}
                    </div>
                    <div style="font-size: 0.78rem; color: #64748B; margin-bottom: 8px;">
                        📍 <em>{r.get('intersection_name')}</em> · Operational Window: <strong>{r.get('target_hours', 'Daily')}</strong> · Est. Budget: <strong>{r.get('est_budget', 'N/A')}</strong>
                    </div>
                    <p style="font-size: 0.80rem; color: #334155; margin: 0 0 8px 0; line-height: 1.4;">
                        {r.get('reason')}
                    </p>
                    <div style="display: flex; align-items: center; gap: 14px; font-size: 0.76rem; color: #475569; padding-top: 6px; border-top: 1px solid #F1F5F9; flex-wrap: wrap;">
                        <span>👥 Vulnerable Residents: <strong>{vuln_str}</strong></span>
                        <span>🚗 Zero-Vehicle Rate: <strong>{no_veh_str}</strong></span>
                        <span>⭐ Priority Score: <strong>{prio_score}/100</strong></span>
                        <span>🔵 Service Radius: <strong>{r.get('catchment_radius_m', 500)}m (0.5 km)</strong></span>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

        with card_col_right:
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            if st.button(
                f"{'📍 Selected' if is_selected else 'Inspect Site'}",
                key=f"btn_roster_sel_{r_id}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state["selected_intersection_id"] = r_id
                st.rerun()

    return st.session_state.get("selected_intersection_id", selected_id)
