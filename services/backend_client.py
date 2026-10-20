"""
Backend Client Interface for HeatShield AI
------------------------------------------
Configurable client for communication with Google Cloud Run / ADK Agent Runtime.
In Live mode, calls remote microservices over HTTP.
In Demo / Local mode, connects directly in-process to agent.persona_agent and agent.tools.
Never silently replaces failed Live requests with synthetic data.
"""

import os
from typing import Dict, Any, Optional
import httpx

BACKEND_URL = os.environ.get("HEATSHIELD_BACKEND_URL", "").strip()
API_KEY = os.environ.get("HEATSHIELD_API_KEY", "").strip()


class BackendClient:
    def __init__(self, base_url: str = BACKEND_URL, api_key: str = API_KEY):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.is_live_configured = bool(self.base_url)

    def get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def fetch_live_predictions(self, city: str = "brooklyn") -> Dict[str, Any]:
        """Queries BigQuery ML prediction endpoint on Cloud Run."""
        if not self.is_live_configured:
            raise RuntimeError("Live backend URL not configured. Switch to Demo Mode.")
        
        url = f"{self.base_url}/api/v1/predict/heat-offsets"
        response = httpx.get(url, params={"city": city}, headers=self.get_headers(), timeout=10.0)
        response.raise_for_status()
        return response.json()

    def fetch_live_recommendations(self, city: str = "brooklyn", persona: str = "City Planner") -> Dict[str, Any]:
        """Queries Google ADK agent for cooling resource recommendations."""
        if not self.is_live_configured:
            raise RuntimeError("Live backend URL not configured. Switch to Demo Mode.")
        
        url = f"{self.base_url}/api/v1/agent/recommendations"
        response = httpx.post(url, json={"city": city, "persona": persona}, headers=self.get_headers(), timeout=15.0)
        response.raise_for_status()
        return response.json()

    def fetch_live_historical_temps(self, city: str = "brooklyn"):
        """Queries live GHCN station_daily_temps endpoint on Cloud Run / BigQuery."""
        import pandas as pd
        if not self.is_live_configured:
            raise RuntimeError("Live backend URL not configured. Switch to Demo Mode.")
        
        url = f"{self.base_url}/api/v1/weather/ghcn-history"
        response = httpx.get(url, params={"city": city}, headers=self.get_headers(), timeout=12.0)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data.get("history", []))

    def execute_chat(
        self,
        query: str,
        persona: str = "city_planner",
        forecast_high_f: float = 98.0,
        user_coords: Optional[tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """Queries Google ADK Agent for conversational multi-persona response."""
        if self.is_live_configured:
            url = f"{self.base_url}/api/v1/agent/chat"
            payload = {
                "message": query,
                "persona": persona,
                "forecast_high_f": forecast_high_f,
                "user_coords": list(user_coords) if user_coords else None,
            }
            response = httpx.post(url, json=payload, headers=self.get_headers(), timeout=20.0)
            response.raise_for_status()
            return response.json()

        # In-process direct execution fallback
        from agent.persona_agent import execute_persona_agent_turn
        return execute_persona_agent_turn(
            query=query,
            persona=persona,
            forecast_high_f=forecast_high_f,
            user_coords=user_coords,
        )

    def predict_location_offset(
        self,
        location_query: str,
        forecast_high_f: float = 98.0,
    ) -> Dict[str, Any]:
        """Predicts temperature offset using BigQuery ML ML.EXPLAIN_PREDICT."""
        if self.is_live_configured:
            url = f"{self.base_url}/api/v1/predict/location"
            payload = {
                "location_query": location_query,
                "forecast_high_f": forecast_high_f,
            }
            response = httpx.post(url, json=payload, headers=self.get_headers(), timeout=10.0)
            response.raise_for_status()
            return response.json()

        from agent.tools import predict_location_heat_offset
        return predict_location_heat_offset(
            location_query=location_query,
            forecast_high_f=forecast_high_f,
        )


backend_client = BackendClient()
