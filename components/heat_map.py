"""
Interactive Brooklyn Heat Map Component
---------------------------------------
Built with Folium and streamlit-folium.
Renders microclimate heat-offset markers, agency recommendations,
and 0.5 km cooling catchment rings.
"""

import streamlit as st
import pandas as pd
import folium
from folium import plugins
from streamlit_folium import st_folium
from typing import Dict, List, Any, Optional

def get_offset_color(offset: float) -> str:
    """Returns hex color code based on predicted_offset_f."""
    if offset >= 5.0:
        return "#DC2626"  # Crimson (+5.6°F)
    elif offset >= 3.5:
        return "#EA580C"  # Vibrant Orange
    elif offset >= 2.0:
        return "#F59E0B"  # Amber
    elif offset >= 0.0:
        return "#EAB308"  # Warm Yellow
    elif offset <= -2.5:
        return "#059669"  # Emerald Green (-3.1°F Park Slope)
    else:
        return "#2563EB"  # Cool Blue


def render_heat_map(
    df: pd.DataFrame,
    selected_id: str,
    recommendations: List[Dict[str, Any]],
    show_recommendations: bool = True,
    show_coverage: bool = True,
    forecast_high_f: int = 98,
    persona: str = "City Planner",
    city_key: str = "brooklyn"
) -> Optional[str]:
    """
    Renders Folium Leaflet Map with Persona-driven Dynamic Zoom & Focus:
      - City Planner: Zoomed OUT (11.5x) across entire city for facility siting.
      - Emergency Management: Zoomed IN (13.8x) directly to acute heat hotspots.
      - Sustainability: Mid-Zoom (12.6x) framing canopy deficit & pavement corridors.
    """
    # -------------------------------------------------------------------------
    # Persona-driven Dynamic Map Zoom and Centroid Focus
    # -------------------------------------------------------------------------
    if not df.empty and "latitude" in df.columns and "longitude" in df.columns:
        city_mean_lat = float(df["latitude"].mean())
        city_mean_lon = float(df["longitude"].mean())

        if persona == "Emergency Management Officer":
            # Tactical zoom-in on critical acute heat hotspots (offset >= 4.5°F or priority >= 85)
            hotspots = df[(df["predicted_offset_f"] >= 4.5) | (df["priority_score"] >= 88)]
            if not hotspots.empty:
                map_center = [float(hotspots["latitude"].mean()), float(hotspots["longitude"].mean())]
            else:
                map_center = [city_mean_lat, city_mean_lon]
            zoom_start = 13.8

        elif persona == "Sustainability / Urban Forestry Officer":
            # Corridor mid-zoom focusing on moderate-to-high built environment heat corridors
            corridors = df[df["predicted_offset_f"].between(1.5, 4.5)]
            if not corridors.empty:
                map_center = [float(corridors["latitude"].mean()), float(corridors["longitude"].mean())]
            else:
                map_center = [city_mean_lat, city_mean_lon]
            zoom_start = 12.6

        else:  # City Planner
            # Wide-angle macro overview of entire city / borough (Zoomed Out)
            map_center = [city_mean_lat, city_mean_lon]
            zoom_start = 11.5
    else:
        map_center = [40.6650, -73.9550]
        zoom_start = 11.5

    # Initialize Folium Map with standard OpenStreetMap tiles (no API key required)
    m = folium.Map(
        location=map_center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True
    )

    # Dictionary of recommendations by ID for fast lookup
    rec_by_id = {r["intersection_id"]: r for r in recommendations}

    # 1. Catchment Overlays Layer (0.5 km = 500m straight-line buffer)
    if show_coverage and show_recommendations:
        for rec in recommendations:
            matched = df[df["intersection_id"] == rec["intersection_id"]]
            if matched.empty:
                continue
            row = matched.iloc[0]
            
            # Distinct styling by agency / intervention
            agency = rec.get("agency", "")
            if "NYCEM" in agency:
                circle_color = "#DC2626"
                fill_color = "#FCA5A5"
            elif "DOHMH" in agency:
                circle_color = "#1D4ED8"
                fill_color = "#93C5FD"
            else:
                circle_color = "#059669"
                fill_color = "#86EFAC"

            folium.Circle(
                location=[row["latitude"], row["longitude"]],
                radius=500,  # 0.5 km straight-line radius
                color=circle_color,
                weight=1.5,
                fill=True,
                fill_color=fill_color,
                fill_opacity=0.22,
                popup=folium.Popup(
                    f"<b>Approximate 500 m straight-line coverage</b><br/>"
                    f"Hub: {row['neighborhood']} ({rec['intervention_title']})<br/>"
                    f"<em>* Note: Straight-line approximation, not verified walking distance.</em>",
                    max_width=280
                ),
                tooltip=f"0.5 km Senior Walk Zone ({rec['agency']})"
            ).add_to(m)

    # 2. Main Intersection Markers
    for _, row in df.iterrows():
        i_id = row["intersection_id"]
        lat = row["latitude"]
        lon = row["longitude"]
        offset = row["predicted_offset_f"]
        color = get_offset_color(offset)
        is_selected = (i_id == selected_id)
        is_recommended = (i_id in rec_by_id)

        expected_temp = forecast_high_f + offset
        
        # Tooltip HTML
        tooltip_html = (
            f"<strong>{i_id} · {row['neighborhood']}</strong><br/>"
            f"{row['intersection_name']}<br/>"
            f"Predicted Heat Offset: <strong>{row['offset_str']}</strong><br/>"
            f"Illustrative Scenario Temp: <strong>{expected_temp:.1f}°F</strong> (at {forecast_high_f}°F base)<br/>"
            f"Priority Score: <strong>{row['priority_score']}/100</strong>"
        )

        # Popup HTML
        rec_info = rec_by_id.get(i_id)
        rec_snippet = ""
        if rec_info and show_recommendations:
            rec_snippet = (
                f"<div style='margin-top:6px; padding:4px 6px; background:#EFF6FF; border-left:3px solid #1769E0; font-size:11px;'>"
                f"<strong>AI Recommendation:</strong> {rec_info['intervention_title']}<br/>"
                f"<em>{rec_info['reason']}</em>"
                f"</div>"
            )

        popup_html = (
            f"<div style='font-family:sans-serif; font-size:12px; width:220px;'>"
            f"<div style='font-weight:bold; font-size:13px; color:#0F172A;'>{i_id}: {row['neighborhood']}</div>"
            f"<div style='color:#64748B; font-size:11px; margin-bottom:4px;'>{row['intersection_name']}</div>"
            f"<div>Predicted Heat Offset: <b style='color:{color}; font-size:14px;'>{row['offset_str']}</b></div>"
            f"<div>Vulnerable Pop: <b>{row['vuln_pop_str']}</b></div>"
            f"<div>Zero-Vehicle Rate: <b>{row['no_veh_str']}</b></div>"
            f"<div>Priority Score: <b>{row['priority_score']} ({row['risk_category']})</b></div>"
            f"{rec_snippet}"
            f"</div>"
        )

        # Visual marker settings
        marker_radius = 8 if not is_selected else 12
        border_weight = 3 if is_selected else (2 if is_recommended else 1)
        border_color = "#0F172A" if is_selected else ("#0284C7" if (is_recommended and show_recommendations) else "#FFFFFF")

        folium.CircleMarker(
            location=[lat, lon],
            radius=marker_radius,
            color=border_color,
            weight=border_weight,
            fill=True,
            fill_color=color,
            fill_opacity=0.90,
            tooltip=folium.Tooltip(tooltip_html),
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)

        # Highlight ring for selected location
        if is_selected:
            folium.CircleMarker(
                location=[lat, lon],
                radius=18,
                color="#0F172A",
                weight=2,
                fill=False,
                dash_array="4, 4"
            ).add_to(m)

    # 3. Floating HTML Legend
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 24px; left: 24px; width: 220px; 
        background-color: white; 
        border:1px solid #CBD5E1; 
        border-radius: 8px;
        z-index:9999; 
        font-size:11px;
        padding: 10px 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        line-height: 1.4;
        ">
        <strong style="color:#0F172A; font-size:12px;">Predicted Heat Offset</strong><br/>
        <span style="color:#64748B; font-size:10px;">0°F = City Reference Baseline</span>
        <div style="margin-top: 6px;">
            <div style="display:flex; align-items:center; gap:6px; margin-bottom:3px;">
                <span style="background:#DC2626; width:11px; height:11px; border-radius:50%; display:inline-block;"></span>
                <span>+5.0°F+ Crimson (Brownsville)</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px; margin-bottom:3px;">
                <span style="background:#EA580C; width:11px; height:11px; border-radius:50%; display:inline-block;"></span>
                <span>+3.5°F to +4.9°F Orange</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px; margin-bottom:3px;">
                <span style="background:#EAB308; width:11px; height:11px; border-radius:50%; display:inline-block;"></span>
                <span>0.0°F to +3.4°F Near Ref</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px; margin-bottom:3px;">
                <span style="background:#059669; width:11px; height:11px; border-radius:50%; display:inline-block;"></span>
                <span>-3.1°F Emerald (Park Slope)</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <span style="background:#2563EB; width:11px; height:11px; border-radius:50%; display:inline-block;"></span>
                <span>Cooler than Reference</span>
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Render through streamlit_folium with dynamic key based on persona and city
    # to trigger immediate zoom-in / zoom-out transitions upon persona selection
    map_output = st_folium(
        m,
        width="100%",
        height=520,
        key=f"folium_map_{persona}_{city_key}",
        returned_objects=["last_object_clicked_popup"]
    )
    
    # Notice below map regarding straight-line distance
    if show_coverage and show_recommendations:
        st.markdown("""
            <div style="font-size: 0.74rem; color: #64748B; margin-top: 4px;">
                * <strong>Coverage Rings:</strong> Approximate 500 m straight-line coverage — not verified walking distance or network isochrone.
            </div>
        """, unsafe_allow_html=True)

    return None
