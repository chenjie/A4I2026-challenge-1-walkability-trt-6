"""
Interactive Government Cooling Dispatch & Coverage Rings Map
-------------------------------------------------------------
Built with Folium and streamlit-folium.
Visualizes:
  - Multi-Agency Cooling Resource Dispatch Locations (NYCEM, DOHMH, Parks / Dallas OEM, DCHHS)
  - Approximate 0.5 km (500m) walking coverage rings and toggleable 1.0 km secondary buffers
  - Spatial Equity Gap overlays highlighting unserved high-vulnerability heat islands
"""

import streamlit as st
import pandas as pd
import folium
from folium import plugins
from streamlit_folium import st_folium
from typing import Dict, List, Any, Optional

def get_agency_colors(agency_name: str) -> Dict[str, str]:
    """Returns stroke and fill colors based on agency name."""
    agency_upper = str(agency_name).upper()
    if "NYCEM" in agency_upper or "OEM" in agency_upper:
        return {
            "stroke": "#DC2626",  # Emergency Crimson
            "fill": "#FCA5A5",
            "icon": "truck",
            "name": "Emergency Management (Mobile Units)"
        }
    elif "DOHMH" in agency_upper or "HEALTH" in agency_upper or "DCHHS" in agency_upper:
        return {
            "stroke": "#1D4ED8",  # Health Royal Blue
            "fill": "#93C5FD",
            "icon": "plus-square",
            "name": "Public Health (Cooling Centers)"
        }
    else:  # Parks / Forestry / Infrastructure
        return {
            "stroke": "#059669",  # Forest Emerald
            "fill": "#86EFAC",
            "icon": "tree",
            "name": "Parks & Infrastructure (Canopy / Pavement)"
        }


def render_dispatch_map(
    df: pd.DataFrame,
    recommendations: List[Dict[str, Any]],
    equity_gaps: List[Dict[str, Any]],
    selected_id: str,
    show_primary_rings: bool = True,
    show_secondary_rings: bool = False,
    show_gaps: bool = True,
    city_key: str = "brooklyn"
) -> Optional[str]:
    """
    Renders Folium Leaflet Map with:
      - Agency-color coded dispatch markers
      - 0.5 km (500m) primary walking coverage rings
      - 1.0 km secondary coverage buffer (optional)
      - Highlighted Equity Gap locations
    """
    # Map Centroid
    if not df.empty and "latitude" in df.columns and "longitude" in df.columns:
        center_lat = float(df["latitude"].mean())
        center_lon = float(df["longitude"].mean())
    else:
        center_lat, center_lon = (40.6650, -73.9550) if "brooklyn" in city_key else (32.7800, -96.8000)

    zoom_start = 11.8 if "brooklyn" in city_key else 11.2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True
    )

    rec_by_id = {r["intersection_id"]: r for r in recommendations}
    gap_by_id = {g["intersection_id"]: g for g in equity_gaps}

    # 1. PRIMARY COVERAGE RINGS (0.5 km = 500m straight-line radius)
    if show_primary_rings:
        for rec in recommendations:
            matched = df[df["intersection_id"] == rec["intersection_id"]]
            if matched.empty:
                continue
            row = matched.iloc[0]
            theme = get_agency_colors(rec.get("agency", ""))

            folium.Circle(
                location=[row["latitude"], row["longitude"]],
                radius=500,  # 500 meters (~6 min walk)
                color=theme["stroke"],
                weight=2,
                fill=True,
                fill_color=theme["fill"],
                fill_opacity=0.25,
                popup=folium.Popup(
                    f"<div style='font-family: sans-serif; font-size: 12px; width: 220px;'>"
                    f"<strong style='color:{theme['stroke']}; font-size: 13px;'>0.5 km Primary Coverage Ring</strong><br/>"
                    f"<b>Location:</b> {row['neighborhood']} ({rec['intersection_id']})<br/>"
                    f"<b>Lead Agency:</b> {rec.get('agency', 'Agency')}<br/>"
                    f"<b>Intervention:</b> {rec.get('intervention_title', '')}<br/>"
                    f"<em>* 500m straight-line pedestrian walking buffer.</em>"
                    f"</div>",
                    max_width=260
                ),
                tooltip=f"0.5 km Walking Catchment: {rec.get('agency')} ({row['neighborhood']})"
            ).add_to(m)

    # 2. SECONDARY COVERAGE BUFFER (1.0 km)
    if show_secondary_rings:
        for rec in recommendations:
            matched = df[df["intersection_id"] == rec["intersection_id"]]
            if matched.empty:
                continue
            row = matched.iloc[0]
            theme = get_agency_colors(rec.get("agency", ""))

            folium.Circle(
                location=[row["latitude"], row["longitude"]],
                radius=1000,  # 1.0 km (~12-15 min walk or bike)
                color=theme["stroke"],
                weight=1,
                dash_array="5, 6",
                fill=True,
                fill_color=theme["fill"],
                fill_opacity=0.08,
                tooltip=f"1.0 km Secondary Buffer: {rec.get('agency')} ({row['neighborhood']})"
            ).add_to(m)

    # 3. EQUITY GAPS LAYER (Unserved High-Risk Zones)
    if show_gaps:
        for gap in equity_gaps:
            g_id = gap["intersection_id"]
            lat, lon = gap["latitude"], gap["longitude"]
            severity_color = "#B91C1C" if gap["severity"] == "Critical Gap" else "#D97706"

            gap_popup = (
                f"<div style='font-family: sans-serif; font-size: 12px; width: 230px;'>"
                f"<span style='background:#FEE2E2; color:#991B1B; font-weight:700; font-size:10px; padding:2px 6px; border-radius:4px;'>"
                f"⚠️ {gap['severity'].upper()}"
                f"</span>"
                f"<h4 style='margin:4px 0 2px 0; font-size:13px; color:#0F172A;'>{gap['neighborhood']}</h4>"
                f"<div style='color:#64748B; font-size:11px; margin-bottom:6px;'>{gap['intersection_name']} ({g_id})</div>"
                f"<div>Heat Offset: <strong style='color:#DC2626;'>+{gap['predicted_offset_f']:.1f}°F</strong></div>"
                f"<div>Vulnerable Population: <strong>{gap['vulnerable_population']:,}</strong></div>"
                f"<div>Zero-Vehicle Rate: <strong>{gap['no_vehicle_rate']:.1f}%</strong></div>"
                f"<div>Nearest Cooling Asset: <strong style='color:#B91C1C;'>{gap['min_dist_km']} km away</strong> (Unserved)</div>"
                f"<div style='margin-top:6px; padding:4px 6px; background:#FEF3C7; border-left:3px solid #F59E0B; font-size:11px;'>"
                f"<strong>Recommended Remedy:</strong> {gap['suggested_remedy']}"
                f"</div>"
                f"</div>"
            )

            # Warning diamond ring
            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                color=severity_color,
                weight=2,
                dash_array="3, 3",
                fill=True,
                fill_color="#FEF3C7",
                fill_opacity=0.85,
                tooltip=f"⚠️ Equity Gap: {gap['neighborhood']} ({gap['min_dist_km']} km unserved)",
                popup=folium.Popup(gap_popup, max_width=260)
            ).add_to(m)

    # 4. DISPATCH LOCATION MARKERS (Active Interventions)
    for rec in recommendations:
        matched = df[df["intersection_id"] == rec["intersection_id"]]
        if matched.empty:
            continue
        row = matched.iloc[0]
        i_id = rec["intersection_id"]
        lat, lon = row["latitude"], row["longitude"]
        theme = get_agency_colors(rec.get("agency", ""))
        is_selected = (i_id == selected_id)

        # Popup
        popup_html = (
            f"<div style='font-family: sans-serif; font-size: 12px; width: 230px;'>"
            f"<span style='background:{theme['stroke']}18; color:{theme['stroke']}; font-weight:700; font-size:10px; padding:2px 6px; border-radius:4px;'>"
            f"{rec.get('agency', 'AGENCY')} DISPATCH"
            f"</span>"
            f"<h4 style='margin:4px 0 2px 0; font-size:13px; color:#0F172A;'>{rec.get('intervention_title', '')}</h4>"
            f"<div style='color:#64748B; font-size:11px; margin-bottom:6px;'>{row['neighborhood']} ({row['intersection_name']})</div>"
            f"<div>Predicted Heat Offset: <strong style='color:#DC2626;'>{row['offset_str']}</strong></div>"
            f"<div>Priority Score: <strong>{row['priority_score']}/100</strong></div>"
            f"<div>Catchment Radius: <strong>{rec.get('catchment_radius_m', 500)}m (0.5 km)</strong></div>"
            f"<div>Vulnerable Residents Reached: <strong>{row['vuln_pop_str']}</strong></div>"
            f"<div>Zero-Vehicle Transit Rate: <strong>{row['no_veh_str']}</strong></div>"
            f"<div style='margin-top:6px; font-size:11px; color:#334155; line-height:1.3;'>"
            f"<em>{rec.get('reason', '')}</em>"
            f"</div>"
            f"</div>"
        )

        # Draw circle marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=12 if is_selected else 9,
            color="#0F172A" if is_selected else theme["stroke"],
            weight=3 if is_selected else 2,
            fill=True,
            fill_color=theme["stroke"],
            fill_opacity=0.95,
            tooltip=f"📍 {rec.get('agency')}: {rec.get('intervention_title')} ({row['neighborhood']})",
            popup=folium.Popup(popup_html, max_width=260)
        ).add_to(m)

        # Selection halo
        if is_selected:
            folium.CircleMarker(
                location=[lat, lon],
                radius=19,
                color="#0F172A",
                weight=2,
                fill=False,
                dash_array="4, 4"
            ).add_to(m)

    # 5. Non-Dispatched Baseline Reference Markers
    for _, row in df.iterrows():
        i_id = row["intersection_id"]
        if i_id in rec_by_id or i_id in gap_by_id:
            continue
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color="#94A3B8",
            weight=1,
            fill=True,
            fill_color="#CBD5E1",
            fill_opacity=0.70,
            tooltip=f"{i_id}: {row['neighborhood']} ({row['offset_str']})"
        ).add_to(m)

    # Render through streamlit_folium
    map_data = st_folium(
        m,
        width=None,
        height=480,
        returned_objects=["last_object_clicked_tooltip"]
    )

    # Render clean visual map legend
    st.markdown('''
        <div style="background: #FFFFFF; border-radius: 8px; padding: 10px 14px; margin-top: 8px; border: 1px solid #E2E8F0; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 0.74rem;">
            <div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap;">
                <span style="font-weight: 700; color: #0F172A;">Map Legend:</span>
                <span style="display: inline-flex; align-items: center; gap: 5px;">
                    <span style="width: 10px; height: 10px; border-radius: 50%; background: #DC2626; display: inline-block;"></span>
                    <strong>NYCEM / OEM</strong> (Mobile Hydration)
                </span>
                <span style="display: inline-flex; align-items: center; gap: 5px;">
                    <span style="width: 10px; height: 10px; border-radius: 50%; background: #1D4ED8; display: inline-block;"></span>
                    <strong>DOHMH / Health</strong> (Cooling Center)
                </span>
                <span style="display: inline-flex; align-items: center; gap: 5px;">
                    <span style="width: 10px; height: 10px; border-radius: 50%; background: #059669; display: inline-block;"></span>
                    <strong>Parks</strong> (Cool Pavement / Tree)
                </span>
                <span style="display: inline-flex; align-items: center; gap: 5px;">
                    <span style="width: 10px; height: 10px; border-radius: 50%; border: 2px dashed #B91C1C; background: #FEF3C7; display: inline-block;"></span>
                    <strong>⚠️ Equity Gap</strong> (Unserved Hotspot)
                </span>
            </div>
            <div style="color: #64748B;">
                🔵 Shaded Circle = <strong>0.5 km (500m) Walk Catchment</strong>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    return None
