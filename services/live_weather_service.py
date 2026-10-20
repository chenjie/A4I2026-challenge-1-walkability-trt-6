"""
Live Weather and Alert Telemetry Service for HeatShield AI
-----------------------------------------------------------
Integrates:
1. Open-Meteo Weather API (free, open-access, zero-key public weather API)
   - Real-time current air temperature, humidity, apparent temperature, and daily forecast high/low.
2. National Weather Service (NWS) Active Alerts API
   - Real-time heat advisories, excessive heat watches/warnings for Kings County (Brooklyn, NY).
"""

import urllib.request
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Coordinates lookup for supported regions
CITY_COORDINATES: Dict[str, Dict[str, Any]] = {
    "brooklyn": {
        "name": "Brooklyn, New York",
        "county": "Kings County",
        "latitude": 40.6782,
        "longitude": -73.9442,
        "timezone": "America/New_York",
    },
    "dallas": {
        "name": "Dallas, Texas",
        "county": "Dallas County",
        "latitude": 32.7767,
        "longitude": -96.7970,
        "timezone": "America/Chicago",
    }
}


def fetch_open_meteo_current(city: str = "brooklyn") -> Dict[str, Any]:
    """
    Fetches real-time current conditions and daily forecast from Open-Meteo API.
    URL: https://api.open-meteo.com/v1/forecast
    """
    city_key = "dallas" if "dallas" in city.lower() else "brooklyn"
    coord = CITY_COORDINATES[city_key]

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={coord['latitude']}&longitude={coord['longitude']}"
        f"&current=temperature_2m,relative_humidity_2m,apparent_temperature"
        f"&daily=temperature_2m_max,temperature_2m_min"
        f"&temperature_unit=fahrenheit"
        f"&timezone={coord['timezone']}"
    )

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "HeatShieldAI-ClimatePlatform/1.0 (Google Agents for Impact 2026)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode("utf-8"))

        current = data.get("current", {})
        daily = data.get("daily", {})

        temp_f = round(float(current.get("temperature_2m", 0.0)))
        humidity = round(float(current.get("relative_humidity_2m", 0.0)))
        apparent_f = round(float(current.get("apparent_temperature", temp_f)))
        
        max_temps = daily.get("temperature_2m_max", [temp_f])
        min_temps = daily.get("temperature_2m_min", [temp_f - 15])
        forecast_high = round(float(max_temps[0])) if max_temps else temp_f
        forecast_low = round(float(min_temps[0])) if min_temps else temp_f - 15

        return {
            "city_name": coord["name"],
            "county": coord["county"],
            "current_temp_f": temp_f,
            "humidity_pct": humidity,
            "apparent_temp_f": apparent_f,
            "forecast_high_f": forecast_high,
            "forecast_low_f": forecast_low,
            "is_live": True,
            "status_label": "Live Open-Meteo API",
            "source": "Open-Meteo Weather API",
            "time": current.get("time", "")
        }
    except Exception as e:
        logger.warning(f"Open-Meteo live API lookup failed: {e}. Using calibrated baseline fallback.")
        # Fallback calibrated default values for resilient presentation
        default_high = 92 if city_key == "dallas" else 84
        default_current = 88 if city_key == "dallas" else 78
        default_humidity = 48 if city_key == "dallas" else 58
        return {
            "city_name": coord["name"],
            "county": coord["county"],
            "current_temp_f": default_current,
            "humidity_pct": default_humidity,
            "apparent_temp_f": default_current + 2,
            "forecast_high_f": default_high,
            "forecast_low_f": default_high - 16,
            "is_live": False,
            "status_label": "Offline Fallback",
            "source": "Open-Meteo Weather API (Simulated)",
            "time": "Simulated"
        }


def fetch_nws_active_alerts(city: str = "brooklyn") -> Dict[str, Any]:
    """
    Queries National Weather Service (NWS) api.weather.gov for active alerts at coordinate point.
    """
    city_key = "dallas" if "dallas" in city.lower() else "brooklyn"
    coord = CITY_COORDINATES[city_key]

    url = f"https://api.weather.gov/alerts/active?point={coord['latitude']},{coord['longitude']}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "HeatShieldAI-ClimatePlatform/1.0 (climate-resilience-hackathon; contact@heatshield.internal)",
            "Accept": "application/geo+json"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode("utf-8"))

        features = data.get("features", [])
        heat_alerts: List[Dict[str, Any]] = []

        for f in features:
            props = f.get("properties", {})
            event_name = props.get("event", "")
            # Filter or prioritize heat advisories, excessive heat warnings, or air quality
            heat_alerts.append({
                "headline": props.get("headline", event_name),
                "event": event_name,
                "severity": props.get("severity", "Unknown"),
                "certainty": props.get("certainty", "Unknown"),
                "urgency": props.get("urgency", "Unknown"),
                "description": props.get("description", ""),
                "instruction": props.get("instruction", "")
            })

        return {
            "alerts": heat_alerts,
            "count": len(heat_alerts),
            "is_live": True,
            "source": "National Weather Service (api.weather.gov)"
        }
    except Exception as e:
        logger.warning(f"NWS live alerts lookup failed: {e}.")
        return {
            "alerts": [],
            "count": 0,
            "is_live": False,
            "source": "National Weather Service (Preview)"
        }
