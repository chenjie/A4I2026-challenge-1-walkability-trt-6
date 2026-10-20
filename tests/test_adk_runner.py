"""
Unit tests for Google ADK Agent, Runner, and Brooklyn Tool Execution.
Verifies native ADK components, session creation, and tool integrations.
"""

from __future__ import annotations

import unittest
from agent.agent import MODEL_NAME, SYSTEM_INSTRUCTION, CUSTOM_TOOLS, build_adk_agent, root_agent
from agent.tools import (
    analyze_heat_frequency,
    compare_candidate_sites,
    dispatch_gov_cooling_action,
    find_nearest_cooling_center_and_heat,
    generate_placement_recommendation,
    predict_location_heat_offset,
    recommend_interventions,
    run_equity_audit,
)
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


class TestADKAgentAndRunner(unittest.TestCase):
    def setUp(self):
        self.agent = root_agent or build_adk_agent()

    def test_agent_initialization(self):
        """Ensure Google ADK Agent is properly created with name, model, and instructions."""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.name, "brooklyn_heat_resilience_planner")
        self.assertEqual(self.agent.model, MODEL_NAME)
        self.assertIn("Brooklyn Urban Heat Resilience", self.agent.instruction)
        self.assertIn("Pitkin Ave & Mother Gaston Blvd", self.agent.instruction)

    def test_adk_tools_registration(self):
        """Ensure all 11 climate and spatial tools are registered on the ADK Agent."""
        self.assertEqual(len(self.agent.tools), 11)
        tool_names = [getattr(t, "__name__", str(t)) for t in self.agent.tools]
        self.assertIn("predict_location_heat_offset", tool_names)
        self.assertIn("recommend_interventions", tool_names)
        self.assertIn("run_equity_audit", tool_names)
        self.assertIn("analyze_heat_frequency", tool_names)
        self.assertIn("compare_candidate_sites", tool_names)
        self.assertIn("dispatch_gov_cooling_action", tool_names)

    def test_adk_session_service_and_runner(self):
        """Verify ADK InMemorySessionService and Runner instantiate without errors."""
        session_service = InMemorySessionService()
        session = session_service.create_session_sync(
            app_name=self.agent.name,
            user_id="test_city_planner",
        )
        self.assertIsNotNone(session.id)

        runner = Runner(
            agent=self.agent,
            app_name=self.agent.name,
            session_service=session_service,
            auto_create_session=True,
        )
        self.assertIsNotNone(runner)

    def test_tool_predict_location_heat_offset(self):
        """Test BQML linear regression prediction tool for Pitkin Ave hotspot."""
        res = predict_location_heat_offset("Pitkin Ave & Mother Gaston Blvd, Brownsville", forecast_high_f=98.0)
        self.assertEqual(res["neighborhood"], "Brownsville")
        self.assertEqual(res["geo_id"], "36047091000")
        self.assertAlmostEqual(res["predicted_offset_f"], 5.4, places=1)
        self.assertAlmostEqual(res["expected_intersection_temp_f"], 103.4, places=1)
        features = [a["feature"] for a in res["ml_explain_predict_attributions"]]
        self.assertIn("canopy_deficit_pct", features)

    def test_tool_recommend_interventions_continuous_ranking(self):
        """Test continuous ranking across all 8 Brooklyn corridors with rank shift."""
        res = recommend_interventions(track_mode="hybrid", num_sites=8, forecast_high_f=98.0)
        self.assertEqual(len(res["recommendations"]), 8)
        # Check rank shift calculation
        for site in res["recommendations"]:
            self.assertIn("rank_shift", site)
            self.assertIn("priority_rank", site)
            self.assertIn("raw_heat_rank", site)

    def test_tool_run_equity_audit(self):
        """Test post-hoc equity audit comparing Top 3 priority sites vs raw heat ranking."""
        res = run_equity_audit(top_n=3)
        self.assertIn("comparison_vs_brooklyn_baseline", res)
        self.assertIn("critical_audit_finding_what_we_caught", res)
        self.assertIn("Industrial Heat Trap", res["critical_audit_finding_what_we_caught"])
        self.assertIn("Coastal Senior Blind Spot", res["critical_audit_finding_what_we_caught"])

    def test_tool_analyze_heat_frequency(self):
        """Test 7-year GHCN station frequency and local crossing multiplier."""
        res = analyze_heat_frequency(location_offset_f=5.4, danger_threshold_f=95.0)
        self.assertAlmostEqual(res["multiplier_vs_city_baseline"], 2.7, places=1)
        self.assertEqual(len(res["yearly_breakdown"]), 7)
        years = [row["year"] for row in res["yearly_breakdown"]]
        self.assertIn(2025, years)

    def test_tool_compare_candidate_sites(self):
        """Test head-to-head comparison between Brownsville and East Flatbush."""
        res = compare_candidate_sites("Brownsville", "East Flatbush", forecast_high_f=98.0)
        self.assertIn("Brownsville", res["site_a"]["name"])
        self.assertIn("East Flatbush", res["site_b"]["name"])
        self.assertIn("tradeoff_analysis", res)

    def test_tool_dispatch_gov_cooling_action(self):
        """Test multi-agency government dispatch for Brooklyn priority sites."""
        res = dispatch_gov_cooling_action(forecast_high_f=98.0)
        self.assertIn("dispatches", res)
        self.assertTrue(len(res["dispatches"]) >= 3)
        actions = res["dispatches"][0]["agency_actions"]
        self.assertTrue(any("NYCEM" in a.get("agency", "") or "DOHMH" in a.get("agency", "") or "DOT" in a.get("agency", "") for a in actions))


if __name__ == "__main__":
    unittest.main()
