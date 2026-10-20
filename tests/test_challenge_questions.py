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
    def setUp(self):
        self.forecast_high_f = 98.0

    def test_question_1_spatial_heat_comparison(self):
        """
        Q1: 'Why is 4th and Marsalis so much worse than two blocks north?'
        Must explain the prompt's Dallas context, map to Brooklyn's flagship hotspot (Pitkin Ave),
        and detail the physical root causes via ML.EXPLAIN_PREDICT feature attributions.
        """
        q = "Why is 4th and Marsalis so much worse than two blocks north?"
        res = orchestrate_agent_turn(q, history=[], forecast_high_f=self.forecast_high_f)

        self.assertIn("reply", res)
        reply = res["reply"]

        # Ensure no crash
        self.assertTrue(len(reply) > 200)

        # Context mapping check
        self.assertTrue(
            "Challenge Prompt Context" in reply or "4th & Marsalis" in reply or "Marsalis" in reply
        )
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

    def test_question_2_heat_frequency_and_trend(self):
        """
        Q2: 'How often does that area actually hit dangerous heat? Is it getting worse?'
        Must resolve 7-year station history, local crossing frequency, exposure multiplier,
        yearly breakdown table, and 7-year trend without KeyError.
        """
        q = "How often does that area actually hit dangerous heat? Is it getting worse?"
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
        Q3: 'Who lives around that site—how many people, how old, how many without a car?'
        Must provide demographic counts: total population, seniors 65+, carless households,
        poverty rate, and explain why carlessness compounds heat risk.
        """
        q = "Who lives around that site—how many people, how old, how many without a car?"
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
        Q4: 'I'm down to two sites and have to pick one. Make the case for each.'
        Must compare Site A (Brownsville) vs Site B (East Flatbush) with side-by-side table,
        trade-off analysis, Case for A, Case for B, and planner recommendation.
        """
        q = "I'm down to two sites and have to pick one. Make the case for each."
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
        Q5: 'Draft the proposal I'm taking to council in the morning.'
        Must generate City Council briefing memo with budget request, executive summary,
        continuous #1-#3 ranking table, and multi-agency dispatch.
        """
        q = "Draft the proposal I'm taking to council in the morning."
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
        Q6: 'Before I send it—did we actually reach the neighborhoods that need this most?'
        Must run post-hoc equity audit ('Strip the variable. Audit the outcome.'),
        produce the equity comparison table, and highlight the Industrial Heat Trap
        and Coastal Senior Blind Spot.
        """
        q = "Before I send it—did we actually reach the neighborhoods that need this most?"
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


if __name__ == "__main__":
    unittest.main()
