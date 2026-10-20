"""
Screen 2: Agentic AI Planning Copilot
====================================
Demonstrates real Google ADK (Agent Development Kit) agent execution:
  - Conversational AI planning and suggested questions
  - Explicit execution of custom ADK tools (`predict_location_heat_offset`, `recommend_interventions`, etc.)
  - Step-by-step Execution Timeline & Tool Invocation Inspector
  - Unmeasured location microclimate heat predictions (`ML.EXPLAIN_PREDICT`)
  - Continuous priority scoring and multi-agency coordination
"""

from __future__ import annotations

import streamlit as st
from typing import Any, Dict, List

from agent.persona_agent import (
    execute_persona_agent_turn,
    PERSONA_REGISTRY,
    Persona
)
from components.copilot_timeline import render_copilot_timeline
from components.unmeasured_predictor import render_unmeasured_predictor


def render_screen_2() -> None:
    """Main render function for Screen 2."""
    active_persona = st.session_state.get("selected_persona", "City Planner")
    forecast_high_f = float(st.session_state.get("forecast_high_f", 98.0))
    city_key = st.session_state.get("selected_city", "brooklyn")
    city_label = "Brooklyn, New York" if "brooklyn" in city_key.lower() else "Dallas, Texas"

    # Header Card
    st.markdown(f'''
        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border-radius: 12px; padding: 22px 24px; margin-bottom: 20px; color: #FFFFFF; border: 1px solid #334155; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                        <span style="background: #0284C7; color: #FFFFFF; font-weight: 800; font-size: 0.70rem; padding: 3px 8px; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.05em;">
                            SCREEN 2 · AGENTIC COPILOT
                        </span>
                        <span style="background: #334155; color: #38BDF8; font-weight: 700; font-size: 0.70rem; padding: 3px 8px; border-radius: 4px;">
                            ADK Tools · BigQuery ML
                        </span>
                    </div>
                    <h1 style="margin: 0 0 6px 0; font-size: 1.7rem; font-weight: 800; color: #F8FAFC;">
                        Agentic AI Planning Copilot
                    </h1>
                    <p style="margin: 0; color: #94A3B8; font-size: 0.88rem; max-width: 780px;">
                        Interact with the Google ADK autonomous planning agent. Ask complex municipal siting questions,
                        predict offsets for unmeasured streets, and trace real-time tool execution timelines.
                    </p>
                </div>
                <div style="text-align: right; background: rgba(255,255,255,0.06); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Supervisor Persona</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #38BDF8;">{active_persona}</div>
                    <div style="font-size: 0.72rem; color: #CBD5E1;">Regional Forecast: <strong>{forecast_high_f:.0f}°F High</strong></div>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # 1. Initialize Copilot Session State
    if "copilot_chat_history" not in st.session_state:
        st.session_state["copilot_chat_history"] = [
            {
                "role": "assistant",
                "content": (
                    f"👋 Welcome to the **HeatShield AI Planning Copilot**! I am calibrated as your "
                    f"**{active_persona}** supervisor for **{city_label}**.\n\n"
                    f"I can invoke custom **ADK tools** and **BigQuery ML** models (`ML.EXPLAIN_PREDICT`) to:\n"
                    f"- Predict temperature offsets for any unmeasured street or coordinate\n"
                    f"- Optimize multi-criteria continuous priority rankings (Score 0–100)\n"
                    f"- Compare single-day NOAA traverses against 7-year GHCN climate station history\n"
                    f"- Coordinate multi-agency cooling center and mobile hydration dispatch\n\n"
                    f"Click one of the suggested questions below or enter a custom prompt."
                )
            }
        ]

    if "last_copilot_turn" not in st.session_state:
        st.session_state["last_copilot_turn"] = {
            "tools_called": ["recommend_interventions(track_mode='hybrid', num_sites=8, forecast_high_f=98.0)"],
            "tool_payload": {
                "action": "recommend_interventions",
                "sites_evaluated": 8,
                "top_site": "BK-01: Brownsville (Pitkin Ave & Thomas S. Boyland St)",
                "predicted_offset": "+5.4°F",
                "priority_score": 94.2
            }
        }

    # 2. Suggested Persona Questions
    persona_questions = {
        "City Planner": [
            "Where should we establish 3 cooling centers this summer?",
            "Run Title VI equity audit on recommended cooling sites",
            "How does single-day Heat Watch traverse compare to 7-year GHCN station history?"
        ],
        "Emergency Management Officer": [
            "Which intersections should receive hydration trucks tomorrow?",
            "Where should senior wellness checks be prioritized during peak heat?",
            "Which communities have high heat exposure and limited vehicle access?"
        ],
        "Sustainability / Urban Forestry Officer": [
            "Which locations should be evaluated for urban tree canopy expansion?",
            "Where does impervious surface contribute most to microclimate heat?",
            "Which unshaded commercial corridors are primary candidates for cool pavement?"
        ]
    }

    questions = persona_questions.get(active_persona, persona_questions["City Planner"])

    st.markdown('''
        <div style="margin-bottom: 8px;">
            <span style="font-size: 0.78rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;">
                💡 Suggested Questions for Active Persona:
            </span>
        </div>
    ''', unsafe_allow_html=True)

    q_cols = st.columns(len(questions))
    selected_question = None
    for i, q_text in enumerate(questions):
        with q_cols[i]:
            if st.button(f"👉 {q_text}", key=f"btn_q_{i}", use_container_width=True):
                selected_question = q_text

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 3. Two-Column Layout: Chat & Predictor on Left (1.15), Execution Timeline on Right (0.85)
    chat_col, timeline_col = st.columns([1.15, 0.85], gap="large")

    with chat_col:
        # A. Interactive Unmeasured Location Predictor
        render_unmeasured_predictor(forecast_high_f=forecast_high_f)

        # B. Chat Message History
        st.markdown('''
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
                <h3 style="margin: 0; font-size: 1.05rem; font-weight: 800; color: #0F172A;">
                    💬 Planning Copilot Conversation
                </h3>
                <span style="font-size: 0.72rem; color: #64748B;">
                    Context-Aware Multi-Turn Session
                </span>
            </div>
        ''', unsafe_allow_html=True)

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state["copilot_chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat Input
        user_input = st.chat_input("Ask HeatShield AI Copilot a municipal resilience question...")

        prompt_to_run = selected_question or user_input

        if prompt_to_run:
            # Append user message
            st.session_state["copilot_chat_history"].append({"role": "user", "content": prompt_to_run})
            
            with st.spinner("ADK Agent invoking custom tools and reasoning..."):
                turn_result = execute_persona_agent_turn(
                    query=prompt_to_run,
                    persona=active_persona,
                    forecast_high_f=forecast_high_f
                )
                
                reply_text = turn_result.get("reply", "No response generated.")
                st.session_state["copilot_chat_history"].append({"role": "assistant", "content": reply_text})
                
                # Update last turn telemetry for Timeline
                st.session_state["last_copilot_turn"] = {
                    "tools_called": turn_result.get("tools_called", []),
                    "tool_payload": {
                        "user_prompt": prompt_to_run,
                        "persona": active_persona,
                        "tools": turn_result.get("tools_called", []),
                        "highlight_sites": turn_result.get("highlight_sites", [])
                    }
                }
            st.rerun()

    with timeline_col:
        # Render the Agent Execution Timeline & Tool Inspector
        render_copilot_timeline(
            last_turn=st.session_state.get("last_copilot_turn"),
            active_persona=active_persona
        )

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        # Persona Mandate Card
        p_obj = PERSONA_REGISTRY.get(active_persona, PERSONA_REGISTRY.get("City Planner"))
        if p_obj:
            st.markdown(f'''
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                    <div style="font-size: 0.72rem; font-weight: 800; color: #64748B; text-transform: uppercase; margin-bottom: 4px;">
                        Active Persona Mandate
                    </div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 6px;">
                        {p_obj.icon} {p_obj.title}
                    </div>
                    <div style="font-size: 0.76rem; color: #475569; margin-bottom: 8px;">
                        <strong>Agency:</strong> {p_obj.agency}
                    </div>
                    <div style="font-size: 0.74rem; color: #64748B; line-height: 1.45; border-top: 1px solid #F1F5F9; padding-top: 8px;">
                        <strong>Primary Objective:</strong> {p_obj.primary_question}
                    </div>
                </div>
            ''', unsafe_allow_html=True)
