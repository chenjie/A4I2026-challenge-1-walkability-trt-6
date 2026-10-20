"""
Unit and integration tests for Screen 2 (Agentic AI Planning Copilot)
and Backend Client in-process connectivity.
"""

import os
import unittest

from agent.persona_agent import Persona, execute_persona_agent_turn
from agent.tools import predict_location_heat_offset
from screens.screen_2_copilot import render_screen_2
from services.backend_client import backend_client


class TestScreen2Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["DISABLE_LIVE_LLM"] = "1"

    def test_screen_2_importable(self):
        """Verify render_screen_2 is callable and valid."""
        self.assertTrue(callable(render_screen_2))

    def test_backend_client_in_process_chat(self):
        """Verify BackendClient fallback executes multi-persona agent turns in-process."""
        res = backend_client.execute_chat(
            query="Why is Pitkin Ave & Mother Gaston Blvd hotter than 2 blocks away?",
            persona=Persona.CITY_PLANNER,
            forecast_high_f=98.0
        )
        self.assertIn("reply", res)
        self.assertIn("tools_called", res)
        self.assertTrue(any("predict_location_heat_offset" in t or "compare_candidate_sites" in t for t in res["tools_called"]))

    def test_resident_persona_google_maps_routing(self):
        """Verify Resident Live Location persona returns Google Maps refuge URL."""
        res = backend_client.execute_chat(
            query="Find the nearest cooling center to 100 Stone Ave Brooklyn.",
            persona=Persona.RESIDENT_LIVE_LOCATION,
            forecast_high_f=98.0
        )
        self.assertIn("reply", res)
        self.assertIsNotNone(res.get("google_maps_url"))
        self.assertTrue("google.com/maps/dir" in res["google_maps_url"])
        self.assertIsNotNone(res.get("gemini_live_speech"))

    def test_unmeasured_street_prediction_tool(self):
        """Verify predict_location_heat_offset produces biophysical feature attributions."""
        pred = backend_client.predict_location_offset(
            location_query="Pitkin Ave & Mother Gaston Blvd",
            forecast_high_f=98.0
        )
        self.assertIn("predicted_offset_f", pred)
        self.assertGreater(pred["predicted_offset_f"], 0.0)
        attrs = pred.get("ml_explain_predict_attributions") or pred.get("top_feature_attributions")
        self.assertIsNotNone(attrs)
        self.assertGreater(len(attrs), 0)


if __name__ == "__main__":
    unittest.main()
