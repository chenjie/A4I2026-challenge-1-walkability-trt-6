"""
Automated verification test suite for the CoolBrooklyn AI Urban Climate Resilience Agent.
Tests all 6 canonical Hackathon challenge questions against `orchestrate_agent_turn`
to guarantee that responses are rigorous, grounded in BigQuery ML and Census data,
and execute with zero errors or unhandled exceptions.
"""

import sys
import unittest

from app import orchestrate_agent_turn
from agent.tools import (
    analyze_heat_frequency,
    compare_candidate_sites,
    generate_placement_recommendation,
    predict_location_heat_offset,
    run_equity_audit,
)


class TestChallengeQuestions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        cls._prev_llm = os.environ.get("DISABLE_LIVE_LLM")
        os.environ["DISABLE_LIVE_LLM"] = "1"

    @classmethod
    def tearDownClass(cls):
        import os
        if cls._prev_llm is None:
            os.environ.pop("DISABLE_LIVE_LLM", None)
        else:
            os.environ["DISABLE_LIVE_LLM"] = cls._prev_llm

    def setUp(self):
        self.forecast_high_f = 98.0

    def test_question_1_spatial_heat_comparison(self):
        """
        Q1 (Brooklyn Ready): 'Why is Pitkin Ave & Mother Gaston Blvd so much worse than two blocks north?'
        Must analyze Brooklyn's flagship hotspot (Pitkin Ave & Mother Gaston Blvd in Brownsville),
        compare to Atlantic Ave / Park Slope, and detail physical root causes via ML.EXPLAIN_PREDICT.
        """
        q = "Why is Pitkin Ave & Mother Gaston Blvd so much worse than two blocks north?"
        res = orchestrate_agent_turn(q, history=[], forecast_high_f=self.forecast_high_f)

        self.assertIn("reply", res)
        reply = res["reply"]

        # Ensure no crash
        self.assertTrue(len(reply) > 200)

        # Flagship hotspot
        self.assertIn("Pitkin Ave & Mother Gaston Blvd", reply)

        # Scientific feature attributions
        self.assertIn("ML.EXPLAIN_PREDICT", reply)
        self.assertTrue("canopy_deficit_pct" in reply or "Tree Canopy Deficit" in reply)
        self.assertTrue("housing_density_per_km2" in reply or "Impervious Asphalt" in reply)
        self.assertIn("+5.4°F", reply)
        self.assertIn("Two Blocks North", reply)
        self.assertIn("Park Slope", reply)

        # Check honest model boundary
        self.assertIn("Honest Model Limitation", reply)

    def test_question_1_legacy_dallas_prompt_mapping(self):
        """
        Q1 (Legacy Hackathon Prompt Compatibility): 'Why is 4th and Marsalis so much worse than two blocks north?'
        Must explain the prompt's Dallas context, map to Brooklyn's flagship hotspot (Pitkin Ave),
        and detail the physical root causes via ML.EXPLAIN_PREDICT feature attributions.
        """
        q = "Why is 4th and Marsalis so much worse than two blocks north?"
        res = orchestrate_agent_turn(q, history=[], forecast_high_f=self.forecast_high_f)

        self.assertIn("reply", res)
        reply = res["reply"]

        self.assertTrue(len(reply) > 200)
        self.assertTrue(
            "Challenge Prompt Context" in reply or "4th & Marsalis" in reply or "Marsalis" in reply
        )
        self.assertIn("Pitkin Ave & Mother Gaston Blvd", reply)
        self.assertIn("ML.EXPLAIN_PREDICT", reply)
        self.assertIn("+5.4°F", reply)
        self.assertIn("Two Blocks North", reply)
        self.assertIn("Park Slope", reply)
        self.assertIn("Honest Model Limitation", reply)

    def test_question_2_heat_frequency_and_trend(self):
        """
        Q2 (Brooklyn Ready): 'How often does Pitkin Ave & Mother Gaston Blvd in Brownsville actually hit dangerous heat? Is it getting worse?'
        Must resolve 7-year station history, local crossing frequency, exposure multiplier,
        yearly breakdown table, and 7-year trend without KeyError.
        """
        q = "How often does Pitkin Ave & Mother Gaston Blvd in Brownsville actually hit dangerous heat? Is it getting worse?"
        res = orchestrate_agent_turn(q, history=[], forecast_high_f=self.forecast_high_f)

        self.assertIn("reply", res)
        reply = res["reply"]

        # Ensure no crash and table rendered
        self.assertIn("7-Year Heat Frequency & Trend Analysis", reply)
        self.assertIn("station_daily_temps", reply)
        self.assertIn("Regional Airport Station Baseline", reply)
        self.assertIn("days/summer", reply)
        self.assertIn("more often", reply)
        self.assertIn("Is It Getting Worse?", reply)

        # Ensure table rows rendered for 2019-2025
        for yr in (2019, 2021, 2023, 2025):
            self.assertIn(str(yr), reply)

        # Verify no double signs like '+-'
        self.assertNotIn("+-", reply)

    def test_question_3_demographics_and_vulnerability(self):
        """
        Q3 (Brooklyn Ready): 'Who lives around Pitkin Ave & Mother Gaston Blvd—how many people, how old, how many without a car?'
        Must provide demographic counts: total population, seniors 65+, carless households,
        poverty rate, and explain why carlessness compounds heat risk.
        """
        q = "Who lives around Pitkin Ave & Mother Gaston Blvd—how many people, how old, how many without a car?"
        res = orchestrate_agent_turn(q, history=[], forecast_high_f=self.forecast_high_f)

        self.assertIn("reply", res)
        reply = res["reply"]

        # Population and demographic facts
        self.assertIn("5,420 residents", reply)
        self.assertIn("986 seniors", reply)
        self.assertIn("18.2%", reply)
        self.assertIn("78.6%", reply)
        self.assertIn("carless households", reply)
        self.assertIn("38.4%", reply)
        self.assertIn("poverty rate", reply)
        self.assertIn("Pitkin Ave", reply)

    def test_question_4_head_to_head_tradeoff(self):
        """
        Q4 (Brooklyn Ready): 'I'm down to two Brooklyn sites—Brownsville and East Flatbush—and have to pick one. Make the case for each.'
        Must compare Site A (Brownsville) vs Site B (East Flatbush) with side-by-side table,
        trade-off analysis, Case for A, Case for B, and planner recommendation.
        """
        q = "I'm down to two Brooklyn sites—Brownsville and East Flatbush—and have to pick one. Make the case for each."
        res = orchestrate_agent_turn(q, history=[], forecast_high_f=self.forecast_high_f)

        self.assertIn("reply", res)
        reply = res["reply"]

        self.assertIn("Head-to-Head Siting Trade-Off", reply)
        self.assertIn("Pitkin Ave & Mother Gaston Blvd", reply)
        self.assertIn("Church Ave & Nostrand Ave", reply)
        self.assertIn("The Case for", reply)
        self.assertIn("Planner Recommendation", reply)

    def test_question_5_council_proposal_draft(self):
        """
        Q5 (Brooklyn Ready): 'Draft the proposal I'm taking to the New York City Council and Brooklyn Borough President in the morning.'
        Must generate City Council briefing memo with budget request, executive summary,
        continuous #1-#3 ranking table, and multi-agency dispatch.
        """
        q = "Draft the proposal I'm taking to the New York City Council and Brooklyn Borough President in the morning."
        res = orchestrate_agent_turn(q, history=[], forecast_high_f=self.forecast_high_f)

        self.assertIn("reply", res)
        reply = res["reply"]

        self.assertIn("CITY COUNCIL PLACEMENT RECOMMENDATION BRIEF", reply)
        self.assertIn("Borough of Brooklyn", reply)
        self.assertIn("Total Capital Request", reply)
        self.assertIn("Executive Summary & Siting Table", reply)
        self.assertIn("Pitkin Ave & Mother Gaston Blvd", reply)
        self.assertNotIn("+-", reply)

    def test_question_6_equity_audit(self):
        """
        Q6 (Brooklyn Ready): 'Before I send it—did we actually reach the Brooklyn neighborhoods that need this most, like Brownsville, East New York, and Coney Island?'
        Must run post-hoc equity audit ('Strip the variable. Audit the outcome.'),
        produce the equity comparison table, and highlight the Industrial Heat Trap
        and Coastal Senior Blind Spot.
        """
        q = "Before I send it—did we actually reach the Brooklyn neighborhoods that need this most, like Brownsville, East New York, and Coney Island?"
        res = orchestrate_agent_turn(q, history=[], forecast_high_f=self.forecast_high_f)

        self.assertIn("reply", res)
        reply = res["reply"]

        self.assertIn("Post-Hoc Equity Audit", reply)
        self.assertIn("Strip the Variable. Audit the Outcome.", reply)
        self.assertIn("2.0x", reply)  # 2.0x poverty rate
        self.assertIn("75.7%", reply)  # carless households
        self.assertIn("Industrial Heat Trap", reply)
        self.assertIn("Coastal Senior Blind Spot", reply)
        self.assertIn("Coney Island", reply)

    def test_question_7_continuous_ranking_and_siting(self):
        """
        Q7 (Brooklyn Ready): 'We can fund three cooling centers in Brooklyn this summer. Where should they go?'
        Must provide continuous #1..#8 ranking, rank shift (Delta), and Tier 1 top 3 capital siting.
        """
        q = "We can fund three cooling centers in Brooklyn this summer. Where should they go?"
        res = orchestrate_agent_turn(q, history=[], forecast_high_f=self.forecast_high_f)

        self.assertIn("reply", res)
        reply = res["reply"]

        self.assertIn("Full Continuous Ranking", reply)
        self.assertIn("Priority Rank", reply)
        self.assertIn("Raw Heat Rank", reply)
        self.assertIn("Pitkin Ave & Mother Gaston Blvd", reply)
        self.assertIn("Tier 1 Immediate Capital", reply)
        self.assertTrue(len(res.get("highlight_sites", [])) >= 3)

    def test_persona_resident_live_location_and_maps(self):
        """
        Persona 4: Resident / Live Location ('Where I am present')
        Must determine street heat feel vs city forecast, find nearest cooling center,
        calculate walking ETA, generate clickable Google Maps directions URL,
        and provide a clean Gemini Live speech briefing.
        """
        q = "Based on where I am present at 40.6725, -73.9142, what is the heatwave near to me and where is the nearest cooling center?"
        res = orchestrate_agent_turn(
            q,
            history=[],
            forecast_high_f=self.forecast_high_f,
            persona="resident",
            user_coords=(40.6725, -73.9142),
        )

        self.assertEqual(res.get("persona"), "resident")
        self.assertIn("reply", res)
        reply = res["reply"]

        # Check nearest cooling center resolution
        self.assertIn("Brownsville Library", reply)
        self.assertIn("Walking Distance", reply)
        self.assertIn("Estimated Walk Time", reply)
        self.assertIn("Google Maps", reply)

        # Check Google Maps URL
        maps_url = res.get("google_maps_url")
        self.assertIsNotNone(maps_url)
        self.assertTrue(maps_url.startswith("https://www.google.com/maps/dir/"))
        self.assertIn("destination=", maps_url)

        # Check Gemini Live Speech audio string
        speech = res.get("gemini_live_speech")
        self.assertIsNotNone(speech)
        self.assertTrue(len(speech) > 40)
        self.assertNotIn("|", speech)
        self.assertNotIn("##", speech)

    def test_persona_emergency_management_officer(self):
        """
        Persona 2: Emergency Management Officer (NYCEM / FDNY)
        Must provide tactical 24-48h emergency response, hydration vans,
        misting tents, transit corridor triage, and senior welfare checks.
        """
        q = "As an emergency management officer, dispatch mobile hydration vans and misting tents to high-vulnerability transit stops."
        res = orchestrate_agent_turn(
            q,
            history=[],
            forecast_high_f=self.forecast_high_f,
            persona="emergency_manager",
        )

        self.assertEqual(res.get("persona"), "emergency_manager")
        self.assertIn("reply", res)
        reply = res["reply"]

        self.assertIn("NYCEM", reply)
        self.assertIn("Mobile Hydration", reply)
        self.assertIn("Misting Tent", reply)
        self.assertIn("dispatch_gov_cooling_action", str(res.get("tools_called")))

    def test_persona_urban_forestry_officer(self):
        """
        Persona 3: Sustainability / Urban Forestry Officer (NYC Parks)
        Must provide canopy deficit analysis, street tree density,
        10-year planting corridor priorities, and vaulted sidewalk trade-offs.
        """
        q = "As an urban forestry officer, what is the canopy deficit and street tree density at Pitkin Ave & Mother Gaston Blvd?"
        res = orchestrate_agent_turn(
            q,
            history=[],
            forecast_high_f=self.forecast_high_f,
            persona="forestry_officer",
        )

        self.assertEqual(res.get("persona"), "forestry_officer")
        self.assertIn("reply", res)
        reply = res["reply"]

        self.assertIn("NYC Parks", reply)
        self.assertTrue("canopy_deficit_pct" in reply or "Canopy Deficit" in reply)
        self.assertTrue("trees_per_km2" in reply or "trees/km²" in reply)
        self.assertIn("Pitkin Ave", reply)

    def test_guardrail_out_of_bounds_phoenix(self):
        """
        Guardrail: Out-of-Footprint City Refusal
        Must refuse out-of-bounds city (Phoenix) without fabricating numbers.
        """
        q = "Where should we put cooling centers in Phoenix tomorrow?"
        res = orchestrate_agent_turn(q, history=[], forecast_high_f=self.forecast_high_f)

        self.assertIn("reply", res)
        reply = res["reply"]

        self.assertIn("Out-of-Footprint Refusal", reply)
        self.assertIn("Phoenix", reply)
        self.assertIn("Brooklyn", reply)


if __name__ == "__main__":
    unittest.main()
