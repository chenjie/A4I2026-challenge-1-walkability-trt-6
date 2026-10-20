#!/usr/bin/env python3
"""
A4I 2026 - Challenge 1: Community Climate Resilience (Urban Heat)
Google ADK (Agent Development Kit) Runner, Tester & Server CLI (`scripts/run_adk.py`).

Provides programmatic Google ADK integration to:
  1. Initialize the Google ADK Agent (`root_agent` from `agent/agent.py`)
  2. Run single queries or interactive CLI turns with `google.adk.runners.Runner`
  3. Execute automated test runs across the Brooklyn-ready canonical questions
  4. Launch the local Spatial Workbench application (`app.py`)

Usage:
    # Run automated Brooklyn test suite
    python scripts/run_adk.py --test

    # Run a single query through the agent
    python scripts/run_adk.py --query "Why is Pitkin Ave & Mother Gaston Blvd so much worse than two blocks north?"

    # Start interactive CLI chat
    python scripts/run_adk.py --interactive

    # Launch the interactive web workbench
    python scripts/run_adk.py --server --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is on Python sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.agent import MODEL_NAME, SYSTEM_INSTRUCTION, CUSTOM_TOOLS, build_adk_agent, root_agent
from app import orchestrate_agent_turn

# ---------------------------------------------------------------------------
# Canonical Brooklyn-Ready Challenge Questions
# ---------------------------------------------------------------------------
BROOKLYN_TEST_QUESTIONS = [
    {
        "id": "Q1",
        "name": "Spatial Heat Comparison (Root Causes)",
        "query": "Why is Pitkin Ave & Mother Gaston Blvd so much worse than two blocks north?",
        "expected_keywords": ["Pitkin Ave & Mother Gaston Blvd", "ML.EXPLAIN_PREDICT", "+5.4°F", "Two Blocks North"],
    },
    {
        "id": "Q1-Legacy",
        "name": "Legacy Prompt Table Compatibility (Dallas Context)",
        "query": "Why is 4th and Marsalis so much worse than two blocks north?",
        "expected_keywords": ["Pitkin Ave & Mother Gaston Blvd", "Challenge Prompt Context", "ML.EXPLAIN_PREDICT"],
    },
    {
        "id": "Q2",
        "name": "7-Year Heat Frequency & Trend",
        "query": "How often does Pitkin Ave & Mother Gaston Blvd in Brownsville actually hit dangerous heat? Is it getting worse?",
        "expected_keywords": ["7-Year Heat Frequency", "station_daily_temps", "days/summer"],
    },
    {
        "id": "Q3",
        "name": "Demographics & Vulnerability (12-Col ACS)",
        "query": "Who lives around Pitkin Ave & Mother Gaston Blvd—how many people, how old, how many without a car?",
        "expected_keywords": ["5,420 residents", "986 seniors", "78.6%", "carless households"],
    },
    {
        "id": "Q4",
        "name": "Head-to-Head 2-Site Siting Trade-Off",
        "query": "I'm down to two Brooklyn sites—Brownsville and East Flatbush—and have to pick one. Make the case for each.",
        "expected_keywords": ["Head-to-Head Siting Trade-Off", "Pitkin Ave & Mother Gaston Blvd", "Church Ave & Nostrand Ave"],
    },
    {
        "id": "Q5",
        "name": "City Council & Borough President Briefing Memo",
        "query": "Draft the proposal I'm taking to the New York City Council and Brooklyn Borough President in the morning.",
        "expected_keywords": ["CITY COUNCIL PLACEMENT RECOMMENDATION BRIEF", "Borough of Brooklyn", "Total Capital Request"],
    },
    {
        "id": "Q6",
        "name": "Post-Hoc Equity Audit (Bias Check)",
        "query": "Before I send it—did we actually reach the Brooklyn neighborhoods that need this most, like Brownsville, East New York, and Coney Island?",
        "expected_keywords": ["Post-Hoc Equity Audit", "Strip the Variable", "Industrial Heat Trap", "Coastal Senior Blind Spot"],
    },
    {
        "id": "Q7",
        "name": "Continuous Siting Ranking (#1..#8)",
        "query": "We can fund three cooling centers in Brooklyn this summer. Where should they go?",
        "expected_keywords": ["Full Continuous Ranking", "Priority Rank", "Raw Heat Rank", "Pitkin Ave & Mother Gaston Blvd"],
    },
    {
        "id": "Persona-Resident",
        "name": "Resident Persona (GPS + Walking Directions)",
        "query": "Based on where I am present at 40.6725, -73.9142, what is the heatwave near to me and where is the nearest cooling center?",
        "expected_keywords": ["Brownsville Library", "Walking Distance", "Google Maps"],
        "persona": "resident",
        "coords": (40.6725, -73.9142),
    },
    {
        "id": "Persona-Emergency",
        "name": "Emergency Manager Persona (NYCEM / FDNY)",
        "query": "As an emergency management officer, dispatch mobile hydration vans and misting tents to high-vulnerability transit stops.",
        "expected_keywords": ["NYCEM", "Mobile Hydration", "Misting Tent"],
        "persona": "emergency_manager",
    },
    {
        "id": "Guardrail-Refusal",
        "name": "Scientific Guardrail (Out-of-Footprint City Refusal)",
        "query": "Where should we put cooling centers in Phoenix tomorrow?",
        "expected_keywords": ["Out-of-Footprint Refusal", "Phoenix", "Brooklyn"],
    },
]


def create_adk_runner(agent_instance: Any = None):
    """
    Construct a Google ADK Runner instance configured with in-memory sessions.
    Falls back gracefully if google.adk is not installed.
    """
    agent = agent_instance or root_agent
    if agent is None:
        agent = build_adk_agent()
    if agent is None:
        return None

    try:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        session_service = InMemorySessionService()
        runner = Runner(
            agent=agent,
            app_name=getattr(agent, "name", "brooklyn_heat_resilience_planner"),
            session_service=session_service,
            auto_create_session=True,
        )
        return runner
    except Exception as exc:
        print(f"[Notice] Google ADK Runner initialization note: {exc}", file=sys.stderr)
        return None


def execute_turn(
    query: str,
    persona: str | None = None,
    user_coords: tuple[float, float] | None = None,
    forecast_high_f: float = 98.0,
) -> dict[str, Any]:
    """
    Execute a turn against the agent orchestrator with full tool support.
    """
    return orchestrate_agent_turn(
        message=query,
        history=[],
        forecast_high_f=forecast_high_f,
        persona=persona,
        user_coords=user_coords,
    )


def run_tests() -> int:
    """
    Run automated verification across all Brooklyn-ready questions.
    Returns 0 on success, 1 on any failure.
    """
    print("=" * 80)
    print("🧪 RUNNING BROOKLYN-READY ADK AGENT TEST SUITE")
    print("=" * 80)
    print(f"Model: {MODEL_NAME} | Tools: {len(CUSTOM_TOOLS)} registered | Persona Router: Active\n")

    passed = 0
    failed = 0
    start_total = time.time()

    for idx, test in enumerate(BROOKLYN_TEST_QUESTIONS, 1):
        test_id = test["id"]
        name = test["name"]
        query = test["query"]
        expected = test["expected_keywords"]
        persona = test.get("persona")
        coords = test.get("coords")

        t0 = time.time()
        try:
            res = execute_turn(query=query, persona=persona, user_coords=coords)
            dt = time.time() - t0
            reply = res.get("reply", "")
            tools = res.get("tools_called", [])

            missing = [kw for kw in expected if kw not in reply]
            if not missing and len(reply) > 100:
                print(f"[{test_id}] ✅ PASS ({dt:.2f}s) — {name}")
                print(f"       Query: \"{query}\"")
                print(f"       Tools: {tools[:2]}")
                passed += 1
            else:
                print(f"[{test_id}] ❌ FAIL ({dt:.2f}s) — {name}")
                print(f"       Missing keywords: {missing}")
                print(f"       Reply snippet: {reply[:200]}...")
                failed += 1
        except Exception as e:
            dt = time.time() - t0
            print(f"[{test_id}] 💥 ERROR ({dt:.2f}s) — {name}: {e}")
            failed += 1

    total_time = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {passed}/{len(BROOKLYN_TEST_QUESTIONS)} passed, {failed} failed in {total_time:.2f}s")
    print("=" * 80)
    return 0 if failed == 0 else 1


def run_single_query(query: str, persona: str | None = None) -> None:
    """Execute a single query and print formatted agent output."""
    print(f"\nQuery: {query}")
    print(f"Persona: {persona or 'Default / Auto-classify'}\n")
    print("-" * 80)

    res = execute_turn(query, persona=persona)
    print(res.get("reply", ""))
    print("\n" + "-" * 80)
    print("Tools Called:")
    for t in res.get("tools_called", []):
        print(f"  • {t}")
    if res.get("highlight_sites"):
        print(f"Map Highlights: {res.get('highlight_sites')}")
    if res.get("google_maps_url"):
        print(f"Directions URL: {res.get('google_maps_url')}")


def run_interactive() -> None:
    """Start an interactive CLI session with the agent."""
    print("=" * 80)
    print("💬 CoolBrooklyn AI — Interactive ADK Planning Agent")
    print("Type your questions below. Enter 'exit' or 'quit' to end session.")
    print("=" * 80 + "\n")

    while True:
        try:
            user_input = input("\nPlanner > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nExiting session. Stay cool!")
                break
            run_single_query(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nSession aborted.")
            break


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Launch the interactive web workbench application (`app.py`)."""
    from http.server import ThreadingHTTPServer
    from app import ClimateWorkbenchHandler

    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, ClimateWorkbenchHandler)
    print(f"\n🚀 Launching CoolBrooklyn Workbench on http://{host}:{port}")
    print("   Press Ctrl+C to stop the server.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(
        description="CoolBrooklyn AI: Google ADK Runner, Tester & Workbench Server CLI."
    )
    parser.add_argument("--test", action="store_true", help="Run automated Brooklyn test suite")
    parser.add_argument("--query", type=str, help="Single query to ask the agent")
    parser.add_argument("--persona", type=str, default=None, help="Optional persona (city_planner, emergency_manager, forestry_officer, resident)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI chat session")
    parser.add_argument("--server", action="store_true", help="Launch the web workbench app server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")

    args = parser.parse_args()

    if args.test:
        sys.exit(run_tests())
    elif args.query:
        run_single_query(args.query, persona=args.persona)
    elif args.interactive:
        run_interactive()
    elif args.server:
        run_server(host=args.host, port=args.port)
    else:
        # Default behavior when run with no arguments: run tests
        sys.exit(run_tests())


if __name__ == "__main__":
    main()
