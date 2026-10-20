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

def render_kpi_cards(df: pd.DataFrame, data_mode: str = "Demo"):
    """Renders 4 attractive KPI metric cards derived from data."""
    highest_offset = df["predicted_offset_f"].max()
    lowest_offset = df["predicted_offset_f"].min()
    elevated_count = (df["predicted_offset_f"] > 0).sum()
    high_priority_count = (df["priority_score"] >= 80).sum()
    
    highest_row = df.loc[df["predicted_offset_f"].idxmax()]
    lowest_row = df.loc[df["predicted_offset_f"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-title">Highest Predicted Heat Offset</div>
                    <div class="kpi-value hot">+{highest_offset:.1f}°F</div>
                </div>
                <p class="kpi-subtitle">{highest_row['neighborhood']} ({highest_row['intersection_id']})</p>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-title">Lowest Predicted Heat Offset</div>
                    <div class="kpi-value emerald">{lowest_offset:.1f}°F</div>
                </div>
                <p class="kpi-subtitle">{lowest_row['neighborhood']} ({lowest_row['intersection_id']})</p>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-title">Elevated Heat Locations</div>
                    <div class="kpi-value warning">{elevated_count} <span style="font-size: 0.95rem; font-weight: 500; color: #64748B;">/ {len(df)}</span></div>
                </div>
                <p class="kpi-subtitle">Intersections with offset &gt; 0.0°F</p>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-title">High-Priority Locations</div>
                    <div class="kpi-value primary">{high_priority_count}</div>
                </div>
                <p class="kpi-subtitle">Priority score &ge; 80 (Urgent Action)</p>
            </div>
        """, unsafe_allow_html=True)

    if data_mode == "Demo":
        st.markdown("""
            <div style="font-size: 0.72rem; color: #94A3B8; text-align: right; margin-top: 4px; font-style: italic;">
                * Illustrative synthetic data — not actual BigQuery predictions.
            </div>
        """, unsafe_allow_html=True)
