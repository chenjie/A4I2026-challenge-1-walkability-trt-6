"""
Historical Heat Intelligence Component for HeatShield AI
---------------------------------------------------------
Visualizes seven-year GHCN-Daily weather station historical extreme heat analysis (2019-2025).
Provides:
  - Plotly 7-year vertical bar chart with dynamic threshold (90°F vs 95°F)
  - Color-accented peak frequency year highlight (Vivid Orange)
  - Three data-derived historical KPI cards
  - Scientific explanation panel clarifying NOAA Heat Watch vs GHCN-Daily
  - Clarification on observed_at = NULL in Heat Watch campaigns
  - Technical disclosure on GHCN station geography vs street microclimates
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any
from services.data_service import get_historical_heat_data


def render_historical_heat(city: str = "brooklyn", data_mode: str = "Demo") -> None:
    """Renders the Seven-Year Historical Heat Intelligence section."""
    
    # 1. Section Header & Subtitle
    st.markdown("""
        <div style="margin-top: 14px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 8px;">
                <div>
                    <h3 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #0F172A; display: flex; align-items: center; gap: 8px;">
                        <span>📈</span> Historical Heat Intelligence
                    </h3>
                    <p style="margin: 4px 0 0 0; color: #64748B; font-size: 0.88rem; line-height: 1.45;">
                        Seven years of extreme-heat patterns reveal how frequently Brooklyn experiences elevated temperatures.
                    </p>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="font-size: 0.72rem; color: #64748B; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; background: #F1F5F9; padding: 4px 10px; border-radius: 6px;">
                        NOAA GHCN-Daily (2019–2025)
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 2. Retrieve Data
    try:
        df_hist = get_historical_heat_data(city=city, data_mode=data_mode)
    except Exception as e:
        st.error(f"⚠️ Unable to load historical GHCN weather station records: {e}")
        return

    if df_hist.empty:
        st.warning("No historical station data returned for the selected region.")
        return

    # Ensure required columns
    years = df_hist["year"].tolist()
    station_name = df_hist["station_name"].iloc[0] if "station_name" in df_hist.columns else "NOAA GHCN-Daily Station"
    station_id = df_hist["station_id"].iloc[0] if "station_id" in df_hist.columns else "USW00094789"

    # 3. Analytics Card Layout: Left 70% Chart, Right 30% KPIs
    col_chart, col_kpis = st.columns([2.3, 1.0], gap="large")

    with col_chart:
        c_head1, c_head2 = st.columns([1.6, 1.4], gap="small")
        with c_head1:
            st.markdown("""
                <h4 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: #0F172A;">
                    Annual Extreme Heat Days (2019–2025)
                </h4>
            """, unsafe_allow_html=True)
        with c_head2:
            threshold_choice = st.selectbox(
                "Threshold Selection",
                options=["≥ 95°F (Extreme Heat Warning)", "≥ 90°F (Heat Advisory Threshold)"],
                index=0,
                label_visibility="collapsed",
                key="hist_threshold_select",
                help="Toggle temperature threshold for historical frequency count."
            )

        is_95 = "95" in threshold_choice
        val_col = "days_over_95f" if is_95 else "days_over_90f"
        threshold_deg = "95°F" if is_95 else "90°F"
        threshold_label = "≥ 95°F" if is_95 else "≥ 90°F"

        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 2px; margin-bottom: 8px;">
                <span style="font-size: 0.74rem; color: #64748B;">
                    Threshold: <strong style="color: #1769E0;">{threshold_label}</strong> · {station_name}
                </span>
                <span style="font-size: 0.72rem; color: #94A3B8;">
                    Station ID: {station_id}
                </span>
            </div>
        """, unsafe_allow_html=True)

        # Dynamic KPI Calculations from the actual dataset
        y_values = df_hist[val_col].tolist()
        total_days = int(sum(y_values))
        max_days = max(y_values)
        max_idx = y_values.index(max_days)
        peak_year = years[max_idx]
        avg_days = total_days / len(years) if len(years) > 0 else 0.0

        # Build Bar Colors: Navy/Blue for standard years, Vivid Orange Accent for peak year
        bar_colors = ["#EA580C" if v == max_days else "#1A73E8" for v in y_values]

        # Hover text data
        hover_texts = [
            f"<b>Year: {yr}</b><br>"
            f"Extreme Heat Days: <b>{v} days</b> ({threshold_label})<br>"
            f"Threshold: {threshold_deg}<br>"
            f"Station: {station_name} ({station_id})<br>"
            f"{'🏆 Peak Historical Record' if v == max_days else 'Standard Observation Year'}"
            for yr, v in zip(years, y_values)
        ]

        # Plotly Figure
        fig = go.Figure(data=[
            go.Bar(
                x=[str(y) for y in years],
                y=y_values,
                marker=dict(
                    color=bar_colors,
                    line=dict(width=0),
                    cornerradius=5
                ),
                text=[f"<b>{v}d</b>" for v in y_values],
                textposition="outside",
                textfont=dict(size=12, color="#0F172A", family="Inter, sans-serif"),
                hoverinfo="text",
                hovertext=hover_texts
            )
        ])

        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=25, b=25),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title=dict(text="Observation Calendar Year", font=dict(size=11, color="#64748B")),
                tickfont=dict(size=12, color="#0F172A", family="Inter, sans-serif"),
                showgrid=False,
                zeroline=False,
                fixedrange=True
            ),
            yaxis=dict(
                title=dict(text=f"Days {threshold_label}", font=dict(size=11, color="#64748B")),
                tickfont=dict(size=11, color="#64748B"),
                gridcolor="#F1F5F9",
                gridwidth=1,
                zeroline=False,
                range=[0, max_days * 1.28],
                fixedrange=True
            ),
            hoverlabel=dict(
                bgcolor="#0F172A",
                font_size=12,
                font_color="#FFFFFF",
                font_family="Inter, sans-serif"
            ),
            bargap=0.35
        )

        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with col_kpis:
        st.markdown("""
            <div style="font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em; color: #64748B; margin-bottom: 8px;">
                Historical Insights (7-Yr Window)
            </div>
        """, unsafe_allow_html=True)

        # KPI 1: Total Extreme Heat Days
        st.markdown(f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;">
                <div style="font-size: 0.70rem; color: #64748B; font-weight: 700; text-transform: uppercase;">
                    Total Extreme Heat Days
                </div>
                <div style="font-size: 1.65rem; font-weight: 800; color: #0F172A; line-height: 1.15; margin-top: 2px;">
                    {total_days} Days
                </div>
                <div style="font-size: 0.72rem; color: #475569; margin-top: 3px;">
                    Sum of days {threshold_label} across 2019–2025
                </div>
            </div>
        """, unsafe_allow_html=True)

        # KPI 2: Highest Heat-Frequency Year
        st.markdown(f"""
            <div style="background: #FFF7ED; border: 1px solid #FED7AA; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;">
                <div style="font-size: 0.70rem; color: #C2410C; font-weight: 700; text-transform: uppercase;">
                    Highest Heat-Frequency Year
                </div>
                <div style="font-size: 1.65rem; font-weight: 800; color: #EA580C; line-height: 1.15; margin-top: 2px;">
                    {peak_year} <span style="font-size: 1.05rem; font-weight: 700;">({max_days} Days)</span>
                </div>
                <div style="font-size: 0.72rem; color: #9A3412; margin-top: 3px;">
                    Peak borough macro-heatwave frequency
                </div>
            </div>
        """, unsafe_allow_html=True)

        # KPI 3: Seven-Year Average
        st.markdown(f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 14px; margin-bottom: 0;">
                <div style="font-size: 0.70rem; color: #64748B; font-weight: 700; text-transform: uppercase;">
                    Seven-Year Average
                </div>
                <div style="font-size: 1.65rem; font-weight: 800; color: #1769E0; line-height: 1.15; margin-top: 2px;">
                    {avg_days:.1f} <span style="font-size: 0.95rem; font-weight: 600;">Days / Yr</span>
                </div>
                <div style="font-size: 0.72rem; color: #475569; margin-top: 3px;">
                    Mean annual recurrence expectation
                </div>
            </div>
        """, unsafe_allow_html=True)

    # 4. Scientific Explanation: Understanding Our Two Heat Datasets
    st.markdown("""
        <hr style="border: 0; border-top: 1px solid #E2E8F0; margin: 18px 0 16px 0;">
        <div style="margin-bottom: 12px;">
            <h4 style="margin: 0 0 6px 0; font-size: 0.98rem; font-weight: 800; color: #0F172A; display: flex; align-items: center; gap: 7px;">
                <span>🔬</span> Understanding Our Two Heat Datasets
            </h4>
            <p style="margin: 0 0 12px 0; font-size: 0.80rem; color: #64748B; line-height: 1.45;">
                HeatShield AI pairs high-resolution spatial sensor data with multi-year climatological station records to answer two distinct municipal planning questions:
            </p>
        </div>
    """, unsafe_allow_html=True)

    ds_col1, ds_col2 = st.columns(2, gap="medium")
    with ds_col1:
        st.markdown("""
            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 14px 16px; height: 100%;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    <span style="font-size: 1.1rem;">🚗</span>
                    <strong style="color: #166534; font-size: 0.88rem;">Street-Level Heat Watch</strong>
                </div>
                <p style="margin: 0 0 6px 0; font-size: 0.84rem; font-weight: 700; color: #15803D;">
                    "One campaign captures how heat varies between streets."
                </p>
                <p style="margin: 0; font-size: 0.78rem; color: #166534; line-height: 1.4;">
                    Traverse sensor points across Brooklyn identify acute microclimate heat islands (e.g. +5.6°F in Brownsville vs -3.1°F in Park Slope) driven by pavement density, tree canopy deficit, and urban geometry.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with ds_col2:
        st.markdown("""
            <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 14px 16px; height: 100%;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    <span style="font-size: 1.1rem;">🏛️</span>
                    <strong style="color: #1E40AF; font-size: 0.88rem;">Seven-Year Weather History</strong>
                </div>
                <p style="margin: 0 0 6px 0; font-size: 0.84rem; font-weight: 700; color: #1D4ED8;">
                    "Daily station records show how often extreme heat occurs."
                </p>
                <p style="margin: 0; font-size: 0.78rem; color: #1E40AF; line-height: 1.4;">
                    Continuous NOAA GHCN-Daily weather station records quantify multi-year recurrence rates (e.g., 11 extreme-heat days in 2024), establishing seasonal risk baselines for municipal emergency staging.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Important Note on observed_at = NULL
    st.markdown("""
        <div style="margin-top: 14px; background: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #F59E0B; border-radius: 6px; padding: 10px 14px;">
            <p style="margin: 0; font-size: 0.78rem; color: #92400E; line-height: 1.45;">
                <strong>⚠️ Data Governance Notice:</strong> Some Heat Watch campaigns do not record individual observation timestamps. Therefore, <code>observed_at = NULL</code> may be expected and does not necessarily indicate missing temperature measurements.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Expandable "Learn More" Section for Technical Climatological Context
    with st.expander("ℹ️ Learn More: Weather Station Placement & Data Interpretation"):
        st.markdown(f"""
            <div style="font-size: 0.80rem; color: #475569; line-height: 1.55; padding-top: 4px;">
                <p style="margin-bottom: 8px;">
                    <strong>Station Geography Disclosure:</strong> The historical station observations depicted in this chart originate from 
                    <strong>{station_name}</strong> (Station ID: <code>{station_id}</code>). Official NOAA airport weather stations are located in open-exposure terrain (such as coastal airfield perimeters) and may not be situated directly within dense interior Brooklyn street blocks.
                </p>
                <p style="margin-bottom: 8px;">
                    <strong>Spatial Granularity vs. Temporal Frequency:</strong> This chart represents calibrated station-level measurements and should 
                    <em>not</em> be interpreted as historical temperatures recorded at every individual Brooklyn intersection. 
                    Rather, the 7-year GHCN history provides city planners with empirical recurrence intervals, while the NOAA Heat Watch trained regression model predicts how local surface factors either amplify or mitigate that regional heat burden at each street corner.
                </p>
                <p style="margin-bottom: 0;">
                    <strong>Data Pipeline:</strong> In Live Mode, this frequency analysis is populated directly by querying BigQuery <code>bigquery-public-data.ghcn_d.ghcnd_20*</code> tables spanning 2019–2025.
                </p>
            </div>
        """, unsafe_allow_html=True)
