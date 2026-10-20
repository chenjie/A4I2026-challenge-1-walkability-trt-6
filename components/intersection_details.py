"""
Intersection Details Component for HeatShield AI
------------------------------------------------
Displays granular intelligence for the selected intersection:
demographics, priority classification, and AI recommendations.
"""

import streamlit as st
import textwrap
from typing import Dict, Any, List

def render_intersection_details(
    intersection: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    show_recommendations: bool = True,
    forecast_high_f: int = 98
):
    """Renders the Intersection Intelligence card for the selected location."""
    i_id = intersection["intersection_id"]
    name = intersection["intersection_name"]
    neighborhood = intersection["neighborhood"]
    offset = intersection["predicted_offset_f"]
    offset_str = intersection["offset_str"]
    vuln_pop_str = intersection["vuln_pop_str"]
    total_pop_str = intersection["total_pop_str"]
    no_veh_str = intersection["no_veh_str"]
    median_age = intersection["median_age"]
    priority_score = intersection["priority_score"]
    risk_cat = intersection["risk_category"]

    # Comparative narrative
    if offset > 0:
        rel_text = f"This location is predicted to be <strong>+{offset:.1f}°F hotter</strong> than the city reference average."
    elif offset < 0:
        rel_text = f"This location is predicted to be <strong>{abs(offset):.1f}°F cooler</strong> than the city reference average."
    else:
        rel_text = "This location is predicted to be <strong>at the city reference average</strong>."

    # Risk badge class
    p_badge = "badge-high" if priority_score >= 80 else ("badge-elevated" if priority_score >= 60 else "badge-cool")

    # Match recommendation
    rec_matched = next((r for r in recommendations if r["intersection_id"] == i_id), None)

    color_offset = '#DC2626' if offset >= 4.0 else ('#059669' if offset <= -2.0 else '#2563EB')

    st.markdown(f"""<div class="content-box">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
<div>
<span style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; color: #64748B;">
{i_id} · Intersection Intelligence
</span>
<h3 style="margin: 2px 0 0 0; color: #0F172A; font-size: 1.25rem; font-weight: 800;">{neighborhood}</h3>
<p style="margin: 0; color: #64748B; font-size: 0.78rem;">{name}</p>
</div>
<span class="badge-pill {p_badge}">Priority {priority_score}/100</span>
</div>
<div style="background: #F8FAFC; border-radius: 8px; padding: 12px 14px; margin-bottom: 12px; border: 1px solid #E2E8F0;">
<div style="font-size: 0.70rem; color: #64748B; font-weight: 700; text-transform: uppercase;">
Predicted Heat Offset (°F)
</div>
<div style="font-size: 2.2rem; font-weight: 800; color: {color_offset}; line-height: 1.1;">
{offset_str}
</div>
<p style="font-size: 0.78rem; color: #475569; margin: 6px 0 0 0; line-height: 1.35;">
{rel_text}
</p>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px;">
<div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 8px 10px; border-radius: 6px;">
<div style="font-size: 0.68rem; color: #64748B; font-weight: 600;">Population</div>
<div style="font-size: 1.05rem; font-weight: 700; color: #0F172A;">{total_pop_str}</div>
</div>
<div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 8px 10px; border-radius: 6px;">
<div style="font-size: 0.68rem; color: #64748B; font-weight: 600;">Zero-Vehicle Rate</div>
<div style="font-size: 1.05rem; font-weight: 700; color: #0F172A;">{no_veh_str}</div>
</div>
<div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 8px 10px; border-radius: 6px;">
<div style="font-size: 0.68rem; color: #64748B; font-weight: 600;">Median Age</div>
<div style="font-size: 1.05rem; font-weight: 700; color: #0F172A;">{median_age} yrs</div>
</div>
<div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 8px 10px; border-radius: 6px;">
<div style="font-size: 0.68rem; color: #64748B; font-weight: 600;">Vulnerable Pop</div>
<div style="font-size: 1.05rem; font-weight: 700; color: #DC2626;">{vuln_pop_str}</div>
</div>
</div>
""", unsafe_allow_html=True)



    st.markdown("</div>", unsafe_allow_html=True)
