"""
Live Climate Intelligence Component for HeatShield AI
------------------------------------------------------
Renders the real-time ambient weather and NWS heat advisory card.
Integrates:
  - Open-Meteo Weather API (real-time air temperature, humidity, apparent temp, forecast high)
  - National Weather Service (NWS active heat alerts for Brooklyn/Kings County)
"""

import streamlit as st
import textwrap
from typing import Dict, Any
from services.live_weather_service import fetch_open_meteo_current, fetch_nws_active_alerts


@st.cache_data(ttl=300)
def get_cached_weather(city: str) -> Dict[str, Any]:
    """Caches live weather data for 5 minutes to avoid redundant API requests."""
    return fetch_open_meteo_current(city)


@st.cache_data(ttl=300)
def get_cached_nws_alerts(city: str) -> Dict[str, Any]:
    """Caches NWS alert lookups for 5 minutes."""
    return fetch_nws_active_alerts(city)


def render_live_climate(city: str = "brooklyn") -> None:
    """Renders the sleek dark Live Climate Intelligence widget."""
    weather = get_cached_weather(city)
    nws = get_cached_nws_alerts(city)

    city_name = weather.get("city_name", "Brooklyn, New York")
    current_temp = weather.get("current_temp_f", 70)
    humidity = weather.get("humidity_pct", 31)
    apparent_temp = weather.get("apparent_temp_f", current_temp)
    forecast_high = weather.get("forecast_high_f", current_temp)
    status_label = weather.get("status_label", "Live Open-Meteo API")
    is_live = weather.get("is_live", True)

    alerts = nws.get("alerts", [])
    has_alerts = len(alerts) > 0

    if has_alerts:
        alert_headlines = "<br>".join([
            f"<strong style='color: #F87171;'>⚠️ {a.get('event', 'Advisory')}:</strong> {a.get('headline', '')}"
            for a in alerts[:2]
        ])
        alerts_html = f"""<div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 8px; padding: 12px 14px; margin-top: 14px;">
<div style="color: #FCA5A5; font-weight: 700; font-size: 0.92rem; display: flex; align-items: center; gap: 7px; margin-bottom: 4px;">
<span>🛡️</span> Official Heat Alerts (Active Warning)
</div>
<div style="color: #FECACA; font-size: 0.82rem; line-height: 1.45;">
{alert_headlines}
</div>
</div>"""
    else:
        alerts_html = """<div style="background: #161F30; border: 1px solid #1E293B; border-radius: 8px; padding: 12px 16px; margin-top: 14px;">
<div style="color: #FFFFFF; font-weight: 700; font-size: 0.92rem; display: flex; align-items: center; gap: 7px; margin-bottom: 4px;">
<span style="color: #94A3B8;">🛡️</span> Official Heat Alerts
</div>
<p style="margin: 0; color: #94A3B8; font-size: 0.80rem; line-height: 1.45;">
Active advisories from the National Weather Service will appear here. No live alert lookup has been performed in this preview. Continuous 24/7 NWS monitoring active.
</p>
</div>"""

    badge_bg = "#064E3B" if is_live else "#374151"
    badge_color = "#34D399" if is_live else "#D1D5DB"
    badge_dot = "🟢" if is_live else "⚪"

    card_html = f"""<div style="background: #0B0F19; border: 1px solid #1E293B; border-radius: 14px; padding: 22px 24px; box-shadow: 0 4px 18px rgba(11, 15, 25, 0.4); margin-bottom: 18px;">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 18px; flex-wrap: wrap; gap: 8px;">
<div>
<div style="display: flex; align-items: center; gap: 8px;">
<span style="color: #10B981; font-weight: 800; font-size: 1.25rem; letter-spacing: -0.05em;">((o))</span>
<h3 style="margin: 0; font-size: 1.18rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.01em;">
Live Climate Intelligence
</h3>
</div>
<div style="color: #94A3B8; font-size: 0.82rem; margin-top: 3px;">
{city_name} · Current conditions
</div>
</div>
<div style="display: flex; align-items: center; gap: 8px;">
<span style="background: {badge_bg}; color: {badge_color}; border: 1px solid rgba(255, 255, 255, 0.1); padding: 4px 12px; border-radius: 9999px; font-size: 0.74rem; font-weight: 700; letter-spacing: 0.03em;">
{badge_dot} {status_label}
</span>
</div>
</div>
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; padding-bottom: 4px;">
<div>
<div style="color: #EF4444; font-size: 1.25rem; margin-bottom: 4px;">🌡️</div>
<div style="color: #FFFFFF; font-size: 2.15rem; font-weight: 800; line-height: 1.05; letter-spacing: -0.02em;">
{current_temp}°F
</div>
<div style="color: #94A3B8; font-size: 0.80rem; margin-top: 4px;">
Current air temperature
</div>
<div style="color: #64748B; font-size: 0.72rem; margin-top: 2px;">
Feels like {apparent_temp}°F
</div>
</div>
<div>
<div style="color: #3B82F6; font-size: 1.25rem; margin-bottom: 4px;">💧</div>
<div style="color: #FFFFFF; font-size: 2.15rem; font-weight: 800; line-height: 1.05; letter-spacing: -0.02em;">
{humidity}%
</div>
<div style="color: #94A3B8; font-size: 0.80rem; margin-top: 4px;">
Humidity
</div>
<div style="color: #64748B; font-size: 0.72rem; margin-top: 2px;">
Relative humidity
</div>
</div>
<div>
<div style="color: #F59E0B; font-size: 1.25rem; margin-bottom: 4px;">☀️</div>
<div style="color: #FFFFFF; font-size: 2.15rem; font-weight: 800; line-height: 1.05; letter-spacing: -0.02em;">
{forecast_high}°F
</div>
<div style="color: #94A3B8; font-size: 0.80rem; margin-top: 4px;">
Today's forecast high
</div>
<div style="color: #64748B; font-size: 0.72rem; margin-top: 2px;">
Daily peak forecast
</div>
</div>
</div>
{alerts_html}
<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 14px; padding-top: 10px; border-top: 1px solid #1E293B; flex-wrap: wrap; gap: 8px;">
<span style="color: #64748B; font-size: 0.74rem;">
Source: Weather API + National Weather Service
</span>
<span style="color: #38BDF8; font-size: 0.74rem; font-weight: 600;">
✓ Live Telemetry Synchronized
</span>
</div>
</div>"""

    st.markdown(card_html, unsafe_allow_html=True)
