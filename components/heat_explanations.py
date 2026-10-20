"""
Heat Explanations Component for HeatShield AI
---------------------------------------------
Visualizes BigQuery ML ML.EXPLAIN_PREDICT feature attributions
explaining why an intersection runs hotter or cooler than average.
"""

import streamlit as st
from typing import List, Dict, Any

def render_heat_explanations(explanations: List[Dict[str, Any]], data_mode: str = "Demo"):
    """Renders compact feature attribution bars modeled after ML.EXPLAIN_PREDICT."""
    st.markdown("""
        <div class="content-box">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                <h4 style="margin: 0; font-size: 0.95rem; font-weight: 700; color: #0F172A;">
                    What's Driving the Heat Difference?
                </h4>
                <span style="font-size: 0.68rem; color: #1769E0; font-weight: 700; text-transform: uppercase;">
                    ML.EXPLAIN_PREDICT
                </span>
            </div>
            <p style="font-size: 0.74rem; color: #64748B; margin-bottom: 12px; line-height: 1.35;">
                Feature attributions isolate the physical drivers (pavement, canopy, geometry) elevating or mitigating microclimate heat.
            </p>
    """, unsafe_allow_html=True)

    if not explanations:
        st.markdown("<p style='font-size: 0.78rem; color: #94A3B8;'>No explanation data available for this location.</p>", unsafe_allow_html=True)
    else:
        for item in explanations:
            factor = item.get("factor", "Unknown")
            attr_f = item.get("attribution_f", 0.0)
            share_pct = item.get("share_pct", 0)
            desc = item.get("description", "")
            
            c_str = f"+{attr_f:.1f}°F" if attr_f >= 0 else f"{attr_f:.1f}°F"
            bar_color = "#DC2626" if attr_f >= 2.0 else ("#EA580C" if attr_f > 0 else "#059669")
            width_pct = min(100, max(14, abs(share_pct)))

            st.markdown(f"""
                <div class="factor-row">
                    <div class="factor-header">
                        <span style="color: #334155; font-size: 0.80rem; font-weight: 600;">{factor}</span>
                        <strong style="color: {bar_color}; font-size: 0.80rem;">{c_str}</strong>
                    </div>
                    <div class="factor-bar-bg">
                        <div class="factor-bar-fill" style="width: {width_pct}%; background-color: {bar_color};"></div>
                    </div>
                    <p style="font-size: 0.70rem; color: #64748B; margin: 2px 0 0 0;">{desc}</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("""
        <p style="font-size: 0.68rem; color: #94A3B8; margin-top: 10px; font-style: italic; line-height: 1.3;">
            * Illustrative synthetic explanation data. In Live mode, values map to BigQuery ML attributions. Statistical associations do not imply direct causal certainty.
        </p>
        </div>
    """, unsafe_allow_html=True)
