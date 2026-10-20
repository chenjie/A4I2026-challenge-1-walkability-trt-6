"""
Unmeasured Location Heat Offset Predictor Component for Screen 2
----------------------------------------------------------------
Provides an interactive interface to query unmeasured streets and coordinates,
invoking the custom ADK tool `predict_location_heat_offset()` with BigQuery ML
feature attributions (`ML.EXPLAIN_PREDICT`).
"""

from __future__ import annotations

import streamlit as st
from typing import Any, Dict, Optional
from agent.tools import predict_location_heat_offset


def render_unmeasured_predictor(forecast_high_f: float = 98.0) -> Optional[Dict[str, Any]]:
    """
    Renders the Unmeasured Location Predictor card.
    Returns the prediction dictionary if a prediction was just executed, else None.
    """
    st.markdown('''
        <div style="background: #FFFFFF; border-radius: 10px; border: 1px solid #E2E8F0; padding: 16px 18px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.2rem;">📍</span>
                    <h3 style="margin: 0; font-size: 1.05rem; font-weight: 800; color: #0F172A;">
                        Predict Heat Offset for Unmeasured Locations
                    </h3>
                </div>
                <span class="tool-badge-pill">
                    🛠️ predict_location_heat_offset()
                </span>
            </div>
            <p style="margin: 0 0 12px 0; color: #64748B; font-size: 0.80rem; line-height: 1.4;">
                NOAA volunteers drove fixed routes on one day. Use BigQuery ML to predict microclimate offsets for 
                unmeasured streets and pedestrian corridors across the borough.
            </p>
        </div>
    ''', unsafe_allow_html=True)

    preset_options = [
        "Fulton St & Utica Ave (Bed-Stuy)",
        "Church Ave & Flatbush Ave (Flatbush)",
        "Eastern Pkwy & Nostrand Ave (Crown Heights)",
        "Atlantic Ave & 4th Ave (Downtown Brooklyn / Transit Hub)",
        "Custom Coordinates (Enter Lat, Lon)"
    ]

    p_col1, p_col2 = st.columns([1.3, 1.0])
    with p_col1:
        sel_preset = st.selectbox(
            "Select Unmeasured Candidate Location",
            preset_options,
            index=0,
            key="pred_preset_sel"
        )

    with p_col2:
        if sel_preset == "Custom Coordinates (Enter Lat, Lon)":
            custom_input = st.text_input("Enter Latitude, Longitude", value="40.6782, -73.9442", key="custom_coords_input")
            target_query = custom_input.strip()
        else:
            target_query = sel_preset.split("(")[0].strip()
            st.markdown(f"<div style='padding-top: 26px; font-size: 0.76rem; color: #64748B;'>Target: <code>{target_query}</code></div>", unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns([1, 2])
    with btn_col1:
        predict_clicked = st.button("⚡ Predict Heat Offset", type="primary", use_container_width=True, key="btn_run_pred")

    # Maintain last prediction in session state
    if "last_unmeasured_prediction" not in st.session_state:
        # Pre-seed with default prediction
        st.session_state["last_unmeasured_prediction"] = predict_location_heat_offset(
            location_query="Fulton St & Utica Ave",
            forecast_high_f=forecast_high_f
        )

    if predict_clicked:
        with st.spinner(f"Invoking BigQuery ML for '{target_query}'..."):
            pred_result = predict_location_heat_offset(
                location_query=target_query,
                forecast_high_f=forecast_high_f
            )
            st.session_state["last_unmeasured_prediction"] = pred_result
            return pred_result

    pred = st.session_state.get("last_unmeasured_prediction")
    if pred and pred.get("status") == "success":
        offset = pred.get("predicted_offset_f", 0.0)
        offset_sign = "+" if offset > 0 else ""
        expected_temp = pred.get("expected_intersection_temp_f", forecast_high_f + offset)
        neighborhood = pred.get("neighborhood", "Brooklyn")
        matched_loc = pred.get("matched_location", target_query)
        prio_score = pred.get("priority_score", 84.5)
        rec_interv = pred.get("recommended_intervention", "High-Capacity Cooling Center")
        attributions = pred.get("ml_explain_predict_attributions", [])

        st.markdown(f'''
            <div style="background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 14px 16px; margin-top: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <span style="font-size: 0.70rem; color: #0284C7; font-weight: 800; text-transform: uppercase;">
                            UNMEASURED LOCATION PREDICTION · {neighborhood}
                        </span>
                        <div style="font-size: 1.0rem; font-weight: 800; color: #0F172A;">
                            {matched_loc}
                        </div>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <span style="background: #FEE2E2; color: #991B1B; font-weight: 800; font-size: 0.95rem; padding: 4px 10px; border-radius: 6px; border: 1px solid #FCA5A5;">
                            {offset_sign}{offset:.1f}°F Offset
                        </span>
                        <span style="background: #EFF6FF; color: #1E40AF; font-weight: 700; font-size: 0.82rem; padding: 4px 10px; border-radius: 6px; border: 1px solid #BFDBFE;">
                            {expected_temp:.1f}°F Street Feel
                        </span>
                    </div>
                </div>

                <div style="margin: 8px 0; font-size: 0.78rem; color: #475569; line-height: 1.45;">
                    <em>{pred.get("predicted_offset_statement", "")}</em>
                </div>

                <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #E2E8F0; font-size: 0.76rem; color: #334155;">
                    <strong>Recommended Municipal Intervention:</strong> {rec_interv} · <strong>Priority Score:</strong> <code>{prio_score}</code>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        # Attributions list
        if attributions:
            with st.expander("🔬 View ML.EXPLAIN_PREDICT Physical Feature Attributions", expanded=False):
                for attr in attributions[:4]:
                    val = attr.get("attribution", 0.0)
                    sign = "+" if val > 0 else ""
                    st.markdown(f"- **`{sign}{val:.2f}°F`** from **{attr.get('feature', 'Feature')}**: {attr.get('explanation', '')}")

    return None
