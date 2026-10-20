"""
Backend Client Interface for HeatShield AI
------------------------------------------
Configurable HTTP client for communication with Google Cloud Run / ADK Agent Runtime.
In Live mode, calls remote microservices. In Demo mode, defers to local data service.
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

backend_client = BackendClient()
