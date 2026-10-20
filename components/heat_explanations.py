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
    factors_html = ""
    if not explanations:
        factors_html = "<p style='font-size: 0.82rem; color: #94A3B8;'>No explanation data available for this location.</p>"
    else:
        for item in explanations:
            factor = item.get("factor", "Unknown")
            attr_f = item.get("attribution_f", 0.0)
            share_pct = item.get("share_pct", 0)
            desc = item.get("description", "")
            
            c_str = f"+{attr_f:.1f}°F" if attr_f >= 0 else f"{attr_f:.1f}°F"
            bar_color = "#DC2626" if attr_f >= 2.0 else ("#EA580C" if attr_f > 0 else "#059669")
            width_pct = min(100, max(14, abs(share_pct)))

            factors_html += f"""
<div class="factor-row" style="margin-bottom: 14px;">
    <div class="factor-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
        <span style="color: #0F172A; font-size: 0.88rem; font-weight: 700;">{factor}</span>
        <strong style="color: {bar_color}; font-size: 0.88rem; font-weight: 800;">{c_str}</strong>
    </div>
    <div class="factor-bar-bg" style="background-color: #E2E8F0; height: 8px; border-radius: 4px; overflow: hidden;">
        <div class="factor-bar-fill" style="width: {width_pct}%; background-color: {bar_color}; height: 100%; border-radius: 4px;"></div>
    </div>
    <p style="font-size: 0.76rem; color: #64748B; margin: 4px 0 0 0; line-height: 1.35;">{desc}</p>
</div>"""

    full_html = f"""<div class="content-box">
<div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
    <h4 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: #0F172A;">
        What is driving the heat difference?
    </h4>
    <span style="font-size: 0.70rem; color: #1769E0; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;">
        ML.EXPLAIN_PREDICT
    </span>
</div>
<p style="font-size: 0.78rem; color: #64748B; margin: 4px 0 16px 0; line-height: 1.4;">
    Feature attributions isolate the physical drivers (pavement, canopy, geometry) elevating or mitigating microclimate heat.
</p>
{factors_html}
<p style="font-size: 0.70rem; color: #94A3B8; margin-top: 12px; margin-bottom: 0; font-style: italic; line-height: 1.35; border-top: 1px solid #F1F5F9; padding-top: 8px;">
    * Illustrative synthetic explanation data. In Live mode, values map to BigQuery ML attributions. Statistical associations do not imply direct causal certainty.
</p>
</div>"""

    st.markdown(full_html, unsafe_allow_html=True)

