"""
KPI Cards Component for HeatShield AI
-------------------------------------
Computes and renders 4 primary metrics dynamically from the dataset:
  1. Highest Predicted Heat Offset
  2. Lowest Predicted Heat Offset
  3. Elevated Heat Locations (count > 0°F)
  4. High-Priority Locations
"""

import streamlit as st
import pandas as pd

def render_kpi_cards(df: pd.DataFrame, data_mode: str = "Demo", custom_kpis: dict = None):
    """Renders 4 attractive KPI metric cards derived dynamically from dataset & active persona."""
    highest_offset = df["predicted_offset_f"].max()
    lowest_offset = df["predicted_offset_f"].min()
    elevated_count = (df["predicted_offset_f"] > 0).sum()
    high_priority_count = (df["priority_score"] >= 80).sum()
    
    highest_row = df.loc[df["predicted_offset_f"].idxmax()]
    lowest_row = df.loc[df["predicted_offset_f"].idxmin()]

    if custom_kpis:
        m1 = custom_kpis["title1"], custom_kpis["val1"], custom_kpis["sub1"], custom_kpis["cls1"]
        m2 = custom_kpis["title2"], custom_kpis["val2"], custom_kpis["sub2"], custom_kpis["cls2"]
        m3 = custom_kpis["title3"], custom_kpis["val3"], custom_kpis["sub3"], custom_kpis["cls3"]
        m4 = custom_kpis["title4"], custom_kpis["val4"], custom_kpis["sub4"], custom_kpis["cls4"]
    else:
        m1 = "Highest Predicted Heat Offset", f"+{highest_offset:.1f}°F", f"{highest_row['neighborhood']} ({highest_row['intersection_id']})", "hot"
        m2 = "Lowest Predicted Heat Offset", f"{lowest_offset:.1f}°F", f"{lowest_row['neighborhood']} ({lowest_row['intersection_id']})", "emerald"
        m3 = "Elevated Heat Locations", f"{elevated_count} <span style='font-size: 0.95rem; font-weight: 500; color: #64748B;'>/ {len(df)}</span>", "Intersections with offset &gt; 0.0°F", "warning"
        m4 = "High-Priority Locations", str(high_priority_count), "Priority score &ge; 80 (Urgent Action)", "primary"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-title">{m1[0]}</div>
                    <div class="kpi-value {m1[3]}">{m1[1]}</div>
                </div>
                <p class="kpi-subtitle">{m1[2]}</p>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-title">{m2[0]}</div>
                    <div class="kpi-value {m2[3]}">{m2[1]}</div>
                </div>
                <p class="kpi-subtitle">{m2[2]}</p>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-title">{m3[0]}</div>
                    <div class="kpi-value {m3[3]}">{m3[1]}</div>
                </div>
                <p class="kpi-subtitle">{m3[2]}</p>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-title">{m4[0]}</div>
                    <div class="kpi-value {m4[3]}">{m4[1]}</div>
                </div>
                <p class="kpi-subtitle">{m4[2]}</p>
            </div>
        """, unsafe_allow_html=True)

    if data_mode == "Demo":
        st.markdown("""
            <div style="font-size: 0.72rem; color: #94A3B8; text-align: right; margin-top: 4px; font-style: italic;">
                * Illustrative synthetic data — dynamically tailored to active persona operational priorities.
            </div>
        """, unsafe_allow_html=True)
