"""
Screen 2: Agentic AI Planning Copilot
====================================
Interactive Google ADK Multi-Agent Execution Environment.
Connects the Streamlit UI directly to the Persona-Aware Supervisor Agent,
BigQuery ML ML.EXPLAIN_PREDICT feature attribution, NOAA GHCN station frequency,
Google Maps walking refuge routing, and Gemini Live audio briefings.
"""

import os
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from agent.persona_agent import (
    Persona,
    PERSONA_REGISTRY,
    execute_persona_agent_turn,
)
from agent.tools import (
    predict_location_heat_offset,
    find_nearest_cooling_center_and_heat,
    analyze_heat_frequency,
    run_equity_audit,
)
from services.data_service import get_intersections


def render_screen_2() -> None:
    """Main render function for Screen 2."""
    city_key = st.session_state.get("selected_city", "brooklyn")
    forecast_high_f = float(st.session_state.get("forecast_high_f", 98))
    data_mode = st.session_state.get("data_mode", "Demo")
    selected_persona_name = st.session_state.get("selected_persona", "City Planner")

    # Map frontend persona names to internal Persona keys
    persona_key_map = {
        "City Planner": Persona.CITY_PLANNER,
        "Emergency Management Officer": Persona.EMERGENCY_MANAGEMENT_OFFICER,
        "Sustainability / Urban Forestry Officer": Persona.URBAN_FORESTRY_OFFICER,
        "Vulnerable Resident / Live Location": Persona.RESIDENT_LIVE_LOCATION,
    }
    active_persona_key = persona_key_map.get(selected_persona_name, Persona.CITY_PLANNER)
    persona_meta = PERSONA_REGISTRY.get(active_persona_key, PERSONA_REGISTRY[Persona.CITY_PLANNER])

    # -------------------------------------------------------------------------
    # Screen Header
    # -------------------------------------------------------------------------
    st.markdown(f'''
        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border-radius: 12px; padding: 22px 24px; margin-bottom: 20px; color: #FFFFFF; border: 1px solid #334155; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                        <span style="background: #0284C7; color: #FFFFFF; font-weight: 800; font-size: 0.70rem; padding: 3px 8px; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.05em;">
                            SCREEN 2 · AGENTIC AI COPILOT
                        </span>
                        <span style="background: #334155; color: #94A3B8; font-weight: 700; font-size: 0.70rem; padding: 3px 8px; border-radius: 4px;">
                            Google ADK · Gemini 2.5 · BigQuery ML
                        </span>
                    </div>
                    <h1 style="margin: 0 0 6px 0; font-size: 1.7rem; font-weight: 800; color: #F8FAFC;">
                        Agentic Climate Resilience Planning Copilot
                    </h1>
                    <p style="margin: 0; color: #94A3B8; font-size: 0.88rem; max-width: 780px;">
                        Conversational multi-persona AI orchestrator grounded in <code>ML.EXPLAIN_PREDICT</code> biophysical attributions,
                        7-year NOAA GHCN station trends, Google Maps walking refuge navigation, and Title VI equity audits.
                    </p>
                </div>
                <div style="text-align: right; background: rgba(255,255,255,0.06); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Active Persona</div>
                    <div style="font-size: 1.05rem; font-weight: 800; color: #38BDF8;">{persona_meta['badge']}</div>
                    <div style="font-size: 0.70rem; color: #CBD5E1;">Agency: <strong>{persona_meta['agency']}</strong></div>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Layout: 2 Columns (Left: Chat & Tool Trace | Right: Unmeasured Predictor & Persona Intel)
    # -------------------------------------------------------------------------
    chat_col, intel_col = st.columns([1.35, 0.95])

    # Initialize chat history in session state
    if "copilot_messages" not in st.session_state:
        st.session_state["copilot_messages"] = [
            {
                "role": "assistant",
                "content": (
                    f"👋 **Greetings! I am the {persona_meta['title']} ({persona_meta['agency']}).**\n\n"
                    "I am powered by Google ADK with 11 custom spatial and climate tools connected to BigQuery ML.\n\n"
                    "**How can I assist your heat resilience planning today?** You can select a benchmark question below or ask about any street, cooling intervention, or equity trade-off."
                ),
                "tools_called": ["agent_bootstrap", "persona_classifier"],
                "google_maps_url": None,
                "gemini_live_speech": f"Hello. I am the {persona_meta['title']} with the {persona_meta['agency']}. Ready for urban heat planning.",
            }
        ]

    # Handler function for executing query
    def submit_query(user_query: str):
        if not user_query.strip():
            return
        # Append user message
        st.session_state["copilot_messages"].append({
            "role": "user",
            "content": user_query.strip(),
            "tools_called": [],
            "google_maps_url": None,
            "gemini_live_speech": None
        })
        
        # Execute agent turn
        result = execute_persona_agent_turn(
            query=user_query.strip(),
            persona=active_persona_key,
            forecast_high_f=forecast_high_f
        )
        
        # Append assistant response
        st.session_state["copilot_messages"].append({
            "role": "assistant",
            "content": result.get("reply", "No response generated."),
            "tools_called": result.get("tools_called", []),
            "google_maps_url": result.get("google_maps_url"),
            "gemini_live_speech": result.get("gemini_live_speech")
        })

    with chat_col:
        # Prompt Chips Header
        st.markdown("""
            <div style="margin-bottom: 8px;">
                <span style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; color: #64748B; letter-spacing: 0.05em;">
                    ⚡ Canonical Benchmark Challenge Questions (1-Click Query)
                </span>
            </div>
        """, unsafe_allow_html=True)

        # 4 Canonical Questions as Buttons
        chip1, chip2 = st.columns(2)
        with chip1:
            if st.button("🌡️ Why is Pitkin & Mother Gaston hotter than 2 blocks away?", use_container_width=True):
                submit_query("Why is Pitkin Ave & Mother Gaston Blvd so much worse than two blocks north?")
                st.rerun()
            if st.button("🚨 Where should NYCEM stage hydration vans today?", use_container_width=True):
                submit_query("Where should Brooklyn place emergency hydration vans today for high-vulnerability transit corridors?")
                st.rerun()

        with chip2:
            if st.button("⚖️ Run algorithmic equity audit on Coney Island vs E. Williamsburg", use_container_width=True):
                submit_query("Run the algorithmic equity audit on East Williamsburg vs Coney Island.")
                st.rerun()
            if st.button("📍 Find nearest cooling center from 100 Stone Ave (Directions)", use_container_width=True):
                submit_query("Where is the closest cooling center to 100 Stone Ave Brooklyn with walking navigation?")
                st.rerun()

        st.markdown("<hr style='border-color: #E2E8F0; margin: 16px 0 16px 0;'>", unsafe_allow_html=True)

        # Chat Conversation Container
        chat_container = st.container(height=520)
        with chat_container:
            for idx, msg in enumerate(st.session_state["copilot_messages"]):
                is_user = (msg["role"] == "user")
                with st.chat_message(msg["role"], avatar="👤" if is_user else "🛡️"):
                    st.markdown(msg["content"])

                    # If tools were called, display visible evidence badge list
                    if msg.get("tools_called"):
                        tools_count = len(msg["tools_called"])
                        with st.expander(f"🛠️ Tool Execution Evidence ({tools_count} tool{'s' if tools_count > 1 else ''} called)", expanded=False):
                            for tc in msg["tools_called"]:
                                st.code(tc, language="python")

                    # If Google Maps Walking Navigation URL exists, show card
                    if msg.get("google_maps_url"):
                        st.markdown(f"""
                            <div style="margin-top: 10px; padding: 12px 14px; background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px;">
                                <div style="font-size: 0.82rem; font-weight: 700; color: #1E40AF; margin-bottom: 4px;">
                                    🚶 Google Maps Platform Walking Directions
                                </div>
                                <div style="font-size: 0.78rem; color: #1E3A8A; margin-bottom: 8px;">
                                    Real-time pedestrian route calculated with urban Manhattan grid factor.
                                </div>
                                <a href="{msg['google_maps_url']}" target="_blank" style="display: inline-block; background: #2563EB; color: white; padding: 6px 14px; border-radius: 6px; font-size: 0.80rem; font-weight: 600; text-decoration: none;">
                                    Open Turn-by-Turn Route in Google Maps ↗
                                </a>
                            </div>
                        """, unsafe_allow_html=True)

                    # If Gemini Live Speech script exists, display playback audio button
                    if msg.get("gemini_live_speech"):
                        clean_speech = msg['gemini_live_speech'].replace("'", "\\'").replace('"', '\\"')
                        st.markdown(f"""
                            <div style="margin-top: 8px; display: flex; align-items: center; gap: 10px;">
                                <button onclick="window.speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance('{clean_speech}'); u.rate = 1.05; window.speechSynthesis.speak(u);" style="background: #0F172A; color: #38BDF8; border: 1px solid #334155; padding: 5px 12px; border-radius: 6px; font-size: 0.76rem; font-weight: 700; cursor: pointer;">
                                    🔊 Play Spoken Briefing (Gemini Live)
                                </button>
                                <span style="font-size: 0.72rem; color: #64748B;">Speech-optimized citizen safety readout</span>
                            </div>
                        """, unsafe_allow_html=True)

        # Chat Input Box
        user_input = st.chat_input("Ask the AI Planning Copilot anything about Brooklyn urban heat resilience...")
        if user_input:
            submit_query(user_input)
            st.rerun()

    # -------------------------------------------------------------------------
    # Right Column: Unmeasured Location Predictor & Active Persona Intel
    # -------------------------------------------------------------------------
    with intel_col:
        # Card 1: Unmeasured Location Heat Predictor
        st.markdown("""
            <div style="background: #FFFFFF; border-radius: 10px; border: 1px solid #E2E8F0; padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                    <h3 style="margin: 0; font-size: 1.05rem; font-weight: 800; color: #0F172A;">
                        📍 Unmeasured Street Predictor
                    </h3>
                    <span style="font-size: 0.68rem; background: #FEF3C7; color: #92400E; font-weight: 700; padding: 2px 6px; border-radius: 4px;">
                        ML.EXPLAIN_PREDICT
                    </span>
                </div>
                <p style="margin: 0 0 12px 0; color: #64748B; font-size: 0.78rem;">
                    Run on-demand inference for streets or intersections volunteers never drove during the NOAA campaign.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Location input selection
        sample_streets = [
            "Pitkin Ave & Mother Gaston Blvd",
            "Rockaway Ave & Pitkin Ave",
            "Pennsylvania Ave & Atlantic Ave",
            "Surf Ave & Stillwell Ave",
            "Utica Ave & Eastern Pkwy",
            "Bushwick Ave & Myrtle Ave",
            "Flatbush Ave & Church Ave",
            "Custom Address / Coordinates..."
        ]
        chosen_street = st.selectbox("Select Target Location", sample_streets, index=0)
        
        target_location = chosen_street
        if chosen_street == "Custom Address / Coordinates...":
            target_location = st.text_input("Enter Address or Lat, Lon", value="40.6698, -73.9125")

        # Run Prediction Button
        if st.button("⚡ Run BQML Heat Attribution Inference", use_container_width=True, type="primary"):
            with st.spinner("Executing BQML ML.EXPLAIN_PREDICT..."):
                pred_result = predict_location_heat_offset(target_location, forecast_high_f=forecast_high_f)
                st.session_state["latest_pred"] = pred_result

        # Display latest prediction results if available
        latest_pred = st.session_state.get("latest_pred")
        if not latest_pred:
            # Default run for Pitkin
            latest_pred = predict_location_heat_offset("Pitkin Ave & Mother Gaston Blvd", forecast_high_f=forecast_high_f)

        if latest_pred:
            offset = latest_pred.get("predicted_offset_f", 0.0)
            street_feel = latest_pred.get("street_feel_f", forecast_high_f + offset)
            neighborhood = latest_pred.get("neighborhood", "Brooklyn")
            attributions = latest_pred.get("ml_explain_predict_attributions") or latest_pred.get("top_feature_attributions", [])

            color = "#DC2626" if offset >= 4.0 else ("#EA580C" if offset >= 2.5 else "#2563EB")

            st.markdown(f"""
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; margin-top: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <span style="font-size: 0.72rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Location</span>
                            <div style="font-size: 0.92rem; font-weight: 800; color: #0F172A;">{latest_pred.get('resolved_location', target_location)}</div>
                            <div style="font-size: 0.75rem; color: #64748B;">{neighborhood}</div>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 0.72rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Predicted Offset</span>
                            <div style="font-size: 1.45rem; font-weight: 900; color: {color};">+{offset:.1f}°F</div>
                            <div style="font-size: 0.75rem; color: #DC2626; font-weight: 600;">{street_feel:.1f}°F street feel</div>
                        </div>
                    </div>
                    <hr style="border-color: #E2E8F0; margin: 10px 0;">
                    <div style="font-size: 0.76rem; font-weight: 700; color: #0F172A; margin-bottom: 6px;">
                        🔬 Biophysical Driver Attribution Breakdown:
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if attributions:
                tot_attr = sum(float(a.get("attribution", a.get("attribution_f", 0.0))) for a in attributions) or 1.0
                for attr in attributions:
                    feat = attr.get("feature", "Biophysical driver")
                    val = float(attr.get("attribution", attr.get("attribution_f", 0.0)))
                    share = attr.get("pct_share")
                    if share is None:
                        share = round(100.0 * (val / tot_attr))
                    bar_w = min(100, max(5, int(share)))
                    expl = attr.get("explanation", "")
                    st.markdown(f"""
                        <div style="margin-top: 6px; font-size: 0.74rem;">
                            <div style="display: flex; justify-content: space-between; color: #334155; font-weight: 600; margin-bottom: 2px;">
                                <span>{feat}</span>
                                <span>+{val:.2f}°F ({share}%)</span>
                            </div>
                            <div style="width: 100%; height: 6px; background: #E2E8F0; border-radius: 3px; overflow: hidden;">
                                <div style="width: {bar_w}%; height: 100%; background: #EF4444; border-radius: 3px;"></div>
                            </div>
                            {f'<div style="font-size: 0.68rem; color: #64748B; margin-top: 2px;">{expl}</div>' if expl else ''}
                        </div>
                    """, unsafe_allow_html=True)

        # Card 2: Core Hackathon Principles & Grounding
        st.markdown(f"""
            <div style="background: #F1F5F9; border-radius: 10px; border: 1px solid #CBD5E1; padding: 16px 18px; margin-top: 20px;">
                <div style="font-size: 0.80rem; font-weight: 800; color: #0F172A; margin-bottom: 6px;">
                    🛡️ Core Architectural Principles Enforced:
                </div>
                <ul style="margin: 0; padding-left: 18px; font-size: 0.74rem; color: #334155; line-height: 1.5;">
                    <li><strong>Predicts Offset, NOT Weather:</strong> The model predicts temperature anomaly (+°F) based on built-environment thermodynamics.</li>
                    <li><strong>Strip the Variable, Audit the Outcome:</strong> Race and income are strictly excluded from model features; equity is audited downstream in Stage 3.</li>
                    <li><strong>Dataset Division of Labor:</strong> Single-day NOAA Heat Watch spatial snapshot explains <em>where</em> it is hot; 7-year GHCN weather stations explain <em>how often</em>.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
