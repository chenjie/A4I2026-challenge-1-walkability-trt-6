"""
Priority Table Component for HeatShield AI
------------------------------------------
Displays the ranked priority table combining microclimate heat offset,
community vulnerability, and vehicle access.
"""

import streamlit as st
import pandas as pd

def render_priority_table(df: pd.DataFrame, selected_id: str, persona: str = "City Planner") -> None:
    """Renders ranked dataframe table sorted dynamically by persona priorities."""
    if persona == "Emergency Management Officer":
        sorted_df = df.sort_values(by=["predicted_offset_f", "priority_score"], ascending=[False, False]).reset_index(drop=True)
        sort_label = "Sorted by Acute Heat Offset & Emergency Vulnerability · Tactical Dispatch"
    elif persona == "Sustainability / Urban Forestry Officer":
        sorted_df = df.sort_values(by="predicted_offset_f", ascending=False).reset_index(drop=True)
        sort_label = "Sorted by Microclimate Heat Deficit · Canopy & Cool Pavement Targets"
    else:
        sorted_df = df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)
        sort_label = "Sorted by Multi-Factor Priority Score · Permanent Facility Siting"

    st.markdown(f"""
        <div style="margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; flex-wrap: wrap; gap: 6px;">
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #0F172A;">
                    Priority Heat Locations
                </h3>
                <span style="font-size: 0.74rem; color: #1E40AF; background: #EFF6FF; border: 1px solid #BFDBFE; padding: 2px 8px; border-radius: 4px; font-weight: 600;">
                    {sort_label}
                </span>
            </div>
            <p style="margin: 0 0 12px 0; color: #64748B; font-size: 0.80rem;">
                Heuristic prioritization matrix combining positive microclimate heat offset, socio-demographic vulnerability, and transit dependency.
            </p>
        </div>
    """, unsafe_allow_html=True)

    sorted_df["Rank"] = [f"#{i+1}" for i in range(len(sorted_df))]

    display_df = sorted_df[[
        "Rank",
        "intersection_id",
        "intersection_name",
        "neighborhood",
        "offset_str",
        "vuln_pop_str",
        "no_veh_str",
        "priority_score",
        "risk_category"
    ]].rename(columns={
        "intersection_id": "Intersection ID",
        "intersection_name": "Intersection Name",
        "neighborhood": "Neighborhood",
        "offset_str": "Predicted Heat Offset (°F)",
        "vuln_pop_str": "Vulnerable Population",
        "no_veh_str": "No-Vehicle Rate",
        "priority_score": "Priority Score",
        "risk_category": "Risk Category"
    })

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width="small"),
            "Intersection ID": st.column_config.TextColumn("ID", width="small"),
            "Intersection Name": st.column_config.TextColumn("Intersection Name", width="medium"),
            "Neighborhood": st.column_config.TextColumn("Neighborhood", width="small"),
            "Predicted Heat Offset (°F)": st.column_config.TextColumn("Predicted Heat Offset (°F)", width="medium"),
            "Vulnerable Population": st.column_config.TextColumn("Vulnerable Pop", width="small"),
            "No-Vehicle Rate": st.column_config.TextColumn("No-Vehicle %", width="small"),
            "Priority Score": st.column_config.NumberColumn("Priority Score (0-100)", width="small"),
            "Risk Category": st.column_config.TextColumn("Risk Category", width="small"),
        }
    )

    st.markdown("""
        <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 4px; line-height: 1.35;">
            * <strong>Prioritization Heuristic Note:</strong> Priority Score (0–100) = [Normalized Heat Offset (40%)] + [Vulnerable Population Ratio (35%)] + [Zero-Vehicle Rate (25%)].
            This is an illustrative planning heuristic for resource staging, not a clinically calibrated epidemiological risk index.
        </div>
    """, unsafe_allow_html=True)
