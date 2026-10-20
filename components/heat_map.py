"""
Interactive Brooklyn Heat Map Component
---------------------------------------
Built with Folium and streamlit-folium.
Renders:
  - Persona-driven dynamic zoom & focus
  - Microclimate heat-offset markers (Group A)
  - AI-Recommended priority locations (Group B)
  - Proposed cooling intervention icons (Hydration Truck 🚚, Cooling Center ❄️, Tree Corridor 🌳)
  - Approximate 500 m service coverage rings
  - Rich recommendation popups with full planning disclosures
  - Comprehensive two-group floating legend (Map Layers & AI Recommendations)
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
        return "#DC2626"  # Crimson (+5.6°F Brownsville)
    elif offset >= 3.5:
        return "#EA580C"  # Vibrant Orange
    elif offset >= 2.0:
        return "#F59E0B"  # Amber
    elif offset >= 0.0:
        return "#EAB308"  # Warm Yellow / Neutral
    elif offset <= -2.5:
        return "#059669"  # Emerald Green (-3.1°F Park Slope)
    else:
        return "#2563EB"  # Cool Blue


def render_heat_map(
    df: pd.DataFrame,
    selected_id: str,
    recommendations: List[Dict[str, Any]],
    show_recommendations: bool = True,
    show_coverage: bool = False,
    forecast_high_f: int = 98,
    persona: str = "City Planner",
    city_key: str = "brooklyn",
    show_heat_markers: bool = True,
    show_ai_recommended: bool = True,
    show_cooling_resources: bool = True,
    show_coverage_rings: bool = False
) -> Optional[str]:
    """
    Renders Folium Leaflet Map with Persona-driven Dynamic Zoom & Focus:
      - Layer toggles dynamically govern visibility of Heat Markers, AI Recs, Cooling Resources, and Coverage Rings.
      - Floats comprehensive dual-group legend: Group A (Heat Offset) & Group B (Proposed Interventions).
    """
    # Fallback to legacy argument names if only those were passed
    effective_ai_rec = show_ai_recommended if show_ai_recommended is not None else show_recommendations
    effective_coverage = show_coverage_rings if show_coverage_rings is not None else show_coverage

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

    # Initialize Folium Map
    m = folium.Map(
        location=map_center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True
    )

    # Dictionary of recommendations by ID for fast lookup
    rec_by_id = {r["intersection_id"]: r for r in recommendations}

    # -------------------------------------------------------------------------
    # 1. Approximate Service Coverage Rings (500 m Straight-Line Buffer)
    # -------------------------------------------------------------------------
    if effective_coverage and recommendations:
        for rec in recommendations:
            matched = df[df["intersection_id"] == rec["intersection_id"]]
            if matched.empty:
                continue
            row = matched.iloc[0]
            
            agency = rec.get("agency", "")
            itype = rec.get("intervention", "").lower()

            if "hydration" in itype or "truck" in itype:
                circle_color = "#2563EB"
                fill_color = "#93C5FD"
                dash_style = "5, 5"
                service_type = "Mobile Hydration Support"
            elif "cooling" in itype or "center" in itype:
                circle_color = "#1D4ED8"
                fill_color = "#93C5FD"
                dash_style = None
                service_type = "Indoor Cooling Center"
            else:
                circle_color = "#059669"
                fill_color = "#86EFAC"
                dash_style = "4, 4"
                service_type = "Urban Shade Intervention"

            folium.Circle(
                location=[row["latitude"], row["longitude"]],
                radius=500,  # 0.5 km = 500 m straight-line radius
                color=circle_color,
                weight=1.8,
                dash_array=dash_style,
                fill=True,
                fill_color=fill_color,
                fill_opacity=0.20,
                popup=folium.Popup(
                    f"<div style='font-family:-apple-system, sans-serif; font-size:12px; line-height:1.4;'>"
                    f"<strong style='color:#0F172A; font-size:13px;'>Approximate 500 m Service Catchment</strong><br/>"
                    f"<strong>Hub:</strong> {row['neighborhood']} ({rec['intervention_title']})<br/>"
                    f"<strong>Service:</strong> {service_type}<br/>"
                    f"<span style='color:#64748B; font-size:11px; margin-top:4px; display:block;'>"
                    f"* Coverage rings show approximate 500 m straight-line distance, not verified walking routes or actual service accessibility."
                    f"</span>"
                    f"</div>",
                    max_width=290
                ),
                tooltip=f"500 m Catchment: {rec['intervention_title']} ({rec.get('agency', 'Agency')})"
            ).add_to(m)

    # -------------------------------------------------------------------------
    # 2. Main Intersection Heat Markers & Recommendation Overlays
    # -------------------------------------------------------------------------
    for _, row in df.iterrows():
        i_id = row["intersection_id"]
        lat = row["latitude"]
        lon = row["longitude"]
        offset = row["predicted_offset_f"]
        color = get_offset_color(offset)
        is_selected = (i_id == selected_id)
        is_recommended = (i_id in rec_by_id)
        rec_info = rec_by_id.get(i_id)

        expected_temp = forecast_high_f + offset
        
        # Tooltip HTML
        tooltip_html = (
            f"<strong>{i_id} · {row['neighborhood']}</strong><br/>"
            f"{row['intersection_name']}<br/>"
            f"Predicted Heat Offset: <strong>{row['offset_str']}</strong><br/>"
            f"Illustrative Scenario Temp: <strong>{expected_temp:.1f}°F</strong> (at {forecast_high_f}°F base)<br/>"
            f"Priority Score: <strong>{row['priority_score']}/100</strong>"
        )
        if is_recommended and effective_ai_rec:
            tooltip_html += f"<br/><strong style='color:#9333EA;'>🎯 Recommended by AI Copilot: {rec_info['intervention_title']}</strong>"

        # Rich Popup HTML
        rec_section = ""
        if rec_info and effective_ai_rec:
            itype = rec_info.get("intervention", "").lower()
            icon_sym = "🚚" if "hydration" in itype else ("❄️" if "cooling" in itype else "🌳")
            rec_section = f"""
            <div style="margin-top: 8px; padding: 8px 10px; background: #EFF6FF; border-left: 3px solid #1769E0; border-radius: 4px;">
                <div style="font-size: 10px; font-weight: 800; color: #1E40AF; text-transform: uppercase; letter-spacing: 0.04em;">
                    {icon_sym} AI-Recommended Intervention
                </div>
                <div style="font-weight: 700; color: #0F172A; font-size: 12px; margin-top: 2px;">
                    {rec_info['intervention_title']}
                </div>
                <div style="font-size: 11px; color: #334155; margin-top: 4px; line-height: 1.35;">
                    <strong>Reason:</strong> {rec_info['reason']}
                </div>
                <div style="font-size: 10px; color: #64748B; margin-top: 6px; padding-top: 4px; border-top: 1px dashed #BFDBFE;">
                    <strong>Source:</strong> Google ADK Planning Copilot (Illustrative)<br/>
                    <em>* Proposed municipal intervention. Not an approved government dispatch.</em>
                </div>
            </div>
            """

        popup_html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 12px; width: 250px; line-height: 1.4;">
            <div style="font-weight: 800; font-size: 13.5px; color: #0F172A; border-bottom: 1px solid #E2E8F0; padding-bottom: 4px; margin-bottom: 6px;">
                {i_id} — {row['neighborhood']}
            </div>
            <div style="color: #64748B; font-size: 11px; margin-bottom: 6px;">{row['intersection_name']}</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span style="color: #475569;">Predicted Heat Offset:</span>
                <strong style="color: {color}; font-size: 13px;">{row['offset_str']}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span style="color: #475569;">Vulnerable Pop:</span>
                <strong>{row['vuln_pop_str']}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span style="color: #475569;">Zero-Vehicle Rate:</span>
                <strong>{row['no_veh_str']}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span style="color: #475569;">Priority Score:</span>
                <strong>{row['priority_score']} / 100 ({row['risk_category']})</strong>
            </div>
            {rec_section}
        </div>
        """

        # A. Heat Offset Circle Markers (Layer Toggle: show_heat_markers)
        if show_heat_markers:
            marker_radius = 8 if not is_selected else 12
            border_weight = 3 if is_selected else (2 if (is_recommended and effective_ai_rec) else 1)
            border_color = "#0F172A" if is_selected else ("#9333EA" if (is_recommended and effective_ai_rec) else "#FFFFFF")

            folium.CircleMarker(
                location=[lat, lon],
                radius=marker_radius,
                color=border_color,
                weight=border_weight,
                fill=True,
                fill_color=color,
                fill_opacity=0.90,
                tooltip=folium.Tooltip(tooltip_html),
                popup=folium.Popup(popup_html, max_width=280)
            ).add_to(m)

        # B. AI-Selected Priority Marker (Layer Toggle: show_ai_recommended)
        if is_recommended and effective_ai_rec:
            folium.CircleMarker(
                location=[lat, lon],
                radius=16,
                color="#9333EA",
                weight=2.5,
                dash_array="3, 3",
                fill=False,
                tooltip=f"🎯 AI-Selected Priority Location: {i_id} ({row['neighborhood']})",
                popup=folium.Popup(popup_html, max_width=280)
            ).add_to(m)

        # C. Proposed Cooling Resource Icons (Layer Toggle: show_cooling_resources)
        if is_recommended and show_cooling_resources and rec_info:
            itype = rec_info.get("intervention", "").lower()
            if "hydration" in itype:
                icon_sym = "🚚"
                badge_bg = "#2563EB"
                badge_title = "Hydration Truck"
            elif "cooling" in itype:
                icon_sym = "❄️"
                badge_bg = "#1D4ED8"
                badge_title = "Cooling Center"
            else:
                icon_sym = "🌳"
                badge_bg = "#059669"
                badge_title = "Tree Corridor"

            # Offset slightly to upper-right of the intersection node
            icon_html = f"""
            <div style="
                background-color: {badge_bg};
                color: white;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 13px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.30);
                border: 2px solid white;
                cursor: pointer;
            " title="{badge_title}: {rec_info['intervention_title']}">
                {icon_sym}
            </div>
            """
            folium.Marker(
                location=[lat + 0.0016, lon + 0.0016],
                icon=folium.DivIcon(
                    html=icon_html,
                    icon_size=(24, 24),
                    icon_anchor=(12, 12)
                ),
                tooltip=f"{icon_sym} Proposed {badge_title}: {rec_info['intervention_title']}",
                popup=folium.Popup(popup_html, max_width=280)
            ).add_to(m)

        # D. Highlight Ring for Selected Location
        if is_selected:
            folium.CircleMarker(
                location=[lat, lon],
                radius=20,
                color="#0F172A",
                weight=2,
                fill=False,
                dash_array="4, 4"
            ).add_to(m)

    # -------------------------------------------------------------------------
    # 3. Floating Dual-Group Map Legend (Map Layers & AI Recommendations)
    # -------------------------------------------------------------------------
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 20px; left: 20px; width: 310px; 
        background-color: rgba(255, 255, 255, 0.96); 
        backdrop-filter: blur(4px);
        border: 1px solid #CBD5E1; 
        border-radius: 8px;
        z-index: 9999; 
        font-size: 11px;
        padding: 10px 12px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.16);
        line-height: 1.35;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <div style="border-bottom: 1px solid #E2E8F0; padding-bottom: 5px; margin-bottom: 7px;">
            <strong style="color: #0F172A; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.04em;">
                🗺️ Map Layers & AI Recommendations
            </strong>
        </div>

        <!-- Group A: Predicted Heat Offset -->
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                <span style="font-weight: 700; color: #475569; font-size: 10px; text-transform: uppercase;">
                    Group A · Predicted Heat Offset
                </span>
                <span style="color: #64748B; font-size: 9px;">0°F = ref average</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px 6px;">
                <div style="display: flex; align-items: center; gap: 5px;">
                    <span style="background: #DC2626; width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0;"></span>
                    <span style="color: #0F172A; font-size: 10px;">Crimson: Signif. hotter</span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <span style="background: #EA580C; width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0;"></span>
                    <span style="color: #0F172A; font-size: 10px;">Orange: Mod. hotter</span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <span style="background: #EAB308; width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0;"></span>
                    <span style="color: #0F172A; font-size: 10px;">Neutral: Near avg</span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <span style="background: #059669; width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0;"></span>
                    <span style="color: #0F172A; font-size: 10px;">Blue/Green: Cooler ref</span>
                </div>
            </div>
        </div>

        <!-- Group B: Proposed Cooling Interventions -->
        <div style="border-top: 1px solid #F1F5F9; padding-top: 6px;">
            <span style="font-weight: 700; color: #475569; font-size: 10px; text-transform: uppercase; display: block; margin-bottom: 4px;">
                Group B · Proposed Interventions
            </span>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px 6px;">
                <div style="display: flex; align-items: center; gap: 5px;">
                    <span style="font-size: 11px;">🚚</span>
                    <span style="color: #0F172A; font-size: 10px;"><strong>Hydration Truck</strong></span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <span style="font-size: 11px;">❄️</span>
                    <span style="color: #0F172A; font-size: 10px;"><strong>Cooling Center</strong></span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <span style="font-size: 11px;">🌳</span>
                    <span style="color: #0F172A; font-size: 10px;"><strong>Tree Corridor</strong></span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <span style="font-size: 11px;">🎯</span>
                    <span style="color: #9333EA; font-size: 10px;"><strong>AI Priority Site</strong></span>
                </div>
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Render through streamlit_folium with dynamic key based on persona and city
    map_output = st_folium(
        m,
        width="100%",
        height=520,
        key=f"folium_map_{persona}_{city_key}_{show_heat_markers}_{effective_ai_rec}_{show_cooling_resources}_{effective_coverage}",
        returned_objects=["last_object_clicked_popup"]
    )
    
    # -------------------------------------------------------------------------
    # 4. Mandatory Notice Below Map
    # -------------------------------------------------------------------------
    if effective_coverage:
        st.markdown("""
            <div style="font-size: 0.74rem; color: #64748B; margin-top: 4px; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 6px 10px; border-radius: 6px;">
                * <strong>Coverage Rings:</strong> Approximate 500 m straight-line distance, not verified walking routes or actual service accessibility.
            </div>
        """, unsafe_allow_html=True)

    return None
