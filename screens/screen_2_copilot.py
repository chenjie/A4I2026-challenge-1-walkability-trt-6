"""Screen 2: focused ADK planning copilot with compact, real tool evidence."""

from __future__ import annotations

import html
import json
import uuid
from typing import Any

import pandas as pd
import streamlit as st

import backend
from agent.agent import AgentConfigurationError, run_planning_agent
from agent.tools import TOOL_DESCRIPTIONS

PLANNING_MODES = ("today", "budget", "long_term")
MODE_LABELS = {
    "today": "Today · immediate response",
    "budget": "Budget · this summer",
    "long_term": "Long term · climate resilience",
}
QUICK_PROMPTS = {
    "today": [
        "Where should hydration trucks go?",
        "Which location has the highest immediate risk?",
        "Where are vulnerable seniors most exposed?",
    ],
    "budget": [
        "Where should we fund 3 cooling centers?",
        "Compare the top candidate locations.",
        "Why is this location a priority?",
        "How often has extreme heat occurred here?",
    ],
    "long_term": [
        "What cooling intervention fits this site best?",
        "Can we audit Title VI equity for these choices?",
        "Draft a council memo explaining this choice.",
    ],
}


def _incoming_location() -> tuple[float | None, float | None, str, dict[str, Any] | None]:
    """Read Screen 1's selected location from the shared Streamlit session state."""
    candidate = st.session_state.get("selected_location") or st.session_state.get("screen1_selected_location")
    lat = st.session_state.get("selected_latitude", st.session_state.get("screen1_latitude"))
    lon = st.session_state.get("selected_longitude", st.session_state.get("screen1_longitude"))
    label = "Choose a Brooklyn area"
    if isinstance(candidate, dict):
        lat = candidate.get("latitude", candidate.get("lat", lat))
        lon = candidate.get("longitude", candidate.get("lon", candidate.get("lng", lon)))
        label = candidate.get("name", candidate.get("label", "Selected area from Screen 1"))
    try:
        if lat is not None and lon is not None:
            return float(lat), float(lon), str(label), candidate if isinstance(candidate, dict) else None
    except (ValueError, TypeError):
        pass
    return None, None, label, candidate if isinstance(candidate, dict) else None


def _is_unmeasured_location(candidate: dict[str, Any] | None) -> bool:
    if not isinstance(candidate, dict):
        return False
    return bool(
        candidate.get("is_unmeasured")
        or candidate.get("unmeasured")
        or candidate.get("source") == "unmeasured"
        or candidate.get("selection_type") == "unmeasured"
        or st.session_state.get("screen1_is_unmeasured")
    )


def _session_id() -> str:
    if "screen2_copilot_session_id" not in st.session_state:
        st.session_state["screen2_copilot_session_id"] = str(uuid.uuid4())
    return st.session_state["screen2_copilot_session_id"]


def _record_event(event: dict[str, Any]) -> None:
    events = st.session_state.setdefault("screen2_agent_timeline", [])
    events.append(dict(event, id=str(uuid.uuid4())))


def _run_turn(prompt: str, planning_mode: str, latitude: float, longitude: float) -> None:
    messages = st.session_state.setdefault("screen2_copilot_messages", [])
    messages.append({"role": "user", "content": prompt})
    _record_event({"kind": "agent", "label": "Question received", "detail": prompt})

    with st.status("Planning copilot is working…", expanded=False) as status:
        try:
            result = run_planning_agent(
                prompt=prompt,
                planning_mode=planning_mode,
                latitude=latitude,
                longitude=longitude,
                user_id=st.session_state.get("screen2_user_id", "brooklyn-planner"),
                session_id=_session_id(),
            )
            for event in result["timeline"]:
                _record_event(event)
            messages.append({"role": "assistant", "content": result["answer"]})
            status.update(label="Copilot response ready", state="complete", expanded=False)
        except AgentConfigurationError as exc:
            message = f"The ADK agent could not run: {exc}"
            messages.append({"role": "assistant", "content": message})
            _record_event({"kind": "error", "label": message, "status": "blocked"})
            status.update(label="Agent configuration required", state="error", expanded=True)
        except Exception as exc:
            message = f"Agent execution failed: {type(exc).__name__}: {exc}"
            messages.append({"role": "assistant", "content": message})
            _record_event({"kind": "error", "label": message, "status": "error"})
            status.update(label="Agent execution failed", state="error", expanded=True)


def _latest_prediction(latitude: float | None, longitude: float | None) -> dict[str, Any] | None:
    for event in reversed(st.session_state.get("screen2_agent_timeline", [])):
        if event.get("kind") != "tool_result" or "predict_location_heat_offset" not in event.get("label", ""):
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return None
        if latitude is not None and longitude is not None:
            if round(float(payload.get("latitude", 0)), 5) != round(latitude, 5):
                continue
            if round(float(payload.get("longitude", 0)), 5) != round(longitude, 5):
                continue
        return payload
    return None


def _latest_turn_events() -> list[dict[str, Any]]:
    events = st.session_state.get("screen2_agent_timeline", [])
    if not events:
        return []
    start_idx = 0
    for idx in range(len(events) - 1, -1, -1):
        if events[idx].get("label") == "Question received":
            start_idx = idx
            break
    return events[start_idx:]


@st.cache_data(ttl=300, show_spinner=False)
def _ranked_locations() -> tuple[list[dict[str, Any]], str | None]:
    try:
        return backend.rank_brooklyn_locations(limit=20), None
    except Exception as exc:
        return [], f"Ranked recommendations unavailable: {type(exc).__name__}: {exc}"


def _candidate_name(location: dict[str, Any], rank: int) -> tuple[str, str | None]:
    name = location.get("neighborhood") or location.get("name")
    tract_id = str(location.get("id", "")).removeprefix("tract-")
    if name and not str(name).lower().startswith("tract "):
        return str(name), tract_id or None
    if tract_id:
        return f"Tract {tract_id}", tract_id
    return f"Candidate area #{rank}", None


def _queue_recommendation_prompt(candidate_key: str, prompt: str) -> None:
    st.session_state["screen2_pending_prompt"] = prompt
    st.session_state["screen2_active_candidate_tab"] = candidate_key


def _rank_row_label(location: dict[str, Any], rank: int) -> str:
    name, tract_id = _candidate_name(location, rank)
    offset = location.get("offset_f")
    score = location.get("priority_score")
    seniors = location.get("age_65_plus")
    focus = str(location.get("recommended_intervention") or "Planning focus unavailable")
    first_focus_phrase = focus.split(";")[0].split("—")[0].strip()
    offset_text = f"{float(offset):+.2f}°F offset" if offset is not None else "offset pending"
    score_text = f"Index {float(score):.1f}" if score is not None else "score pending"
    senior_text = f"{int(seniors):,} age 65+" if seniors is not None else "seniors pending"
    tract_text = f" · {tract_id}" if tract_id else ""
    return f"#{rank} Recommended · {name}{tract_text} · {offset_text} · {score_text} · {senior_text} · {first_focus_phrase}"


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        .copilot-shell {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 14px;
            padding: 14px 18px;
            background: linear-gradient(180deg, rgba(248, 250, 252, 0.92), rgba(255, 255, 255, 0.96));
            margin-bottom: 12px;
        }
        .copilot-kicker {
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            color: #475569;
            margin-bottom: 4px;
        }
        .copilot-title {
            font-size: 1.18rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
        }
        .copilot-caption {
            font-size: 0.84rem;
            color: #475569;
            margin-top: 4px;
            margin-bottom: 0;
        }
        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            background: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
        }
        .status-chip-warn {
            background: #fffbeb;
            color: #b45309;
            border-color: #fde68a;
        }
        .candidate-summary {
            padding: 10px 14px;
            border-radius: 10px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            margin-bottom: 10px;
            font-size: 0.84rem;
            color: #334155;
        }
        .predicted-offset {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            margin-top: 8px;
            padding: 4px 10px;
            border-radius: 8px;
            background: #fef2f2;
            color: #991b1b;
            font-weight: 700;
            font-size: 0.8rem;
            border: 1px solid #fecaca;
        }
        .metric-inline {
            display: inline-block;
            margin-right: 14px;
            font-size: 0.8rem;
            color: #475569;
        }
        .metric-inline strong {
            color: #0f172a;
        }
        .tool-trace-card {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 8px 10px;
            margin-bottom: 8px;
            background: #ffffff;
        }
        .tool-trace-title {
            font-size: 0.78rem;
            font-weight: 700;
            color: #1e293b;
        }
        .tool-trace-desc {
            font-size: 0.74rem;
            color: #64748b;
            margin-top: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_agent_evidence(latitude: float | None, longitude: float | None) -> None:
    st.markdown("### Agent activity & evidence")
    st.caption("Real tool execution history emitted by the Google ADK runner.")

    events = _latest_turn_events()
    prediction = _latest_prediction(latitude, longitude)
    area_label = st.session_state.get("selected_location_name") or "Selected area"

    if prediction:
        offset = prediction.get("predicted_heat_offset_f")
        if offset is not None:
            st.markdown(
                f'<div class="predicted-offset">🌡️ {html.escape(area_label)} · predicted heat offset '
                f'{float(offset):+.2f}°F vs Brooklyn baseline</div>',
                unsafe_allow_html=True,
            )

    if not events:
        st.info("Run a planning prompt or ask about a candidate location to see the tool call trail.")
        return

    for event in events:
        kind = event.get("kind", "agent")
        label = event.get("label", "Agent event")
        detail = event.get("detail")
        payload = event.get("payload")

        if kind == "tool_call":
            tool_name = label.split("called ", 1)[-1].split("(", 1)[0]
            with st.container():
                st.markdown(
                    f'<div class="tool-trace-card"><div class="tool-trace-title">🛠️ Call: {html.escape(tool_name)}()</div>'
                    f'<div class="tool-trace-desc">{html.escape(TOOL_DESCRIPTIONS.get(tool_name, "ADK tool invocation"))}</div></div>',
                    unsafe_allow_html=True,
                )
                if isinstance(payload, dict) and payload:
                    with st.expander(f"View {tool_name}() input"):
                        st.code(json.dumps(payload, indent=2))
        elif kind == "tool_result":
            tool_name = label.split(" returned", 1)[0].removesuffix("()")
            with st.container():
                st.markdown(
                    f'<div class="tool-trace-card"><div class="tool-trace-title">✅ Result: {html.escape(tool_name)}()</div>'
                    f'<div class="tool-trace-desc">{html.escape(detail or "Tool returned successfully")}</div></div>',
                    unsafe_allow_html=True,
                )
                if isinstance(payload, (dict, list)) and payload:
                    with st.expander(f"View {tool_name}() result"):
                        st.code(json.dumps(payload, indent=2))
        elif kind == "agent":
            if label == "Question received":
                st.caption(f"Prompt: {detail}")
            elif label == "ADK runner started":
                st.caption("ADK session runner started.")
            elif label == "ADK returned its final response":
                st.caption("Agent finished reasoning and tool orchestration.")
            else:
                st.caption(label)
        elif kind == "error":
            st.error(label)


def _render_ranked_locations(recommendations: list[dict[str, Any]], error: str | None) -> None:
    st.markdown("### Top candidate locations")
    st.caption("Browse priority Brooklyn locations, inspect built-environment drivers, and ask follow-up questions.")

    if error:
        st.warning(error)
        return
    if not recommendations:
        st.info("No candidate recommendations returned.")
        return

    default_expanded_index = 0
    candidate_by_label: dict[str, tuple[int, dict[str, Any]]] = {}
    options: list[str] = []

    for rank, item in enumerate(recommendations, start=1):
        name, _ = _candidate_name(item, rank)
        label = f"{name} · {item.get('id', '')[-5:]}"
        candidate_by_label[label] = (rank, item)
        options.append(label)

    active_candidate_tab = st.session_state.get("screen2_active_candidate_tab")
    if active_candidate_tab in candidate_by_label:
        default_expanded_index = options.index(active_candidate_tab)

    selected_label = st.selectbox(
        "Highlight candidate location",
        options=options,
        index=default_expanded_index,
        key="screen2_candidate_select",
    )

    rank, expanded_location = candidate_by_label[selected_label]
    name, tract_id = _candidate_name(expanded_location, rank)
    offset = expanded_location.get("offset_f")
    score = expanded_location.get("priority_score")
    residents = expanded_location.get("population")
    seniors = expanded_location.get("age_65_plus")
    vehicle = expanded_location.get("no_vehicle_pct")
    focus = expanded_location.get("recommended_intervention")
    factors = expanded_location.get("drivers") or []

    st.markdown(
        f"""
        <div class="candidate-summary">
            <strong>#{rank} Recommended · {html.escape(name)}</strong>
            <div style="margin-top: 6px;">
                <span class="metric-inline">Offset: <strong>{f'{float(offset):+.2f}°F' if offset is not None else 'N/A'}</strong></span>
                <span class="metric-inline">Priority: <strong>{f'{float(score):.1f}' if score is not None else 'N/A'}</strong></span>
                <span class="metric-inline">Seniors 65+: <strong>{f'{int(seniors):,}' if seniors is not None else 'N/A'}</strong></span>
                <span class="metric-inline">No vehicle: <strong>{f'{float(vehicle):.1f}%' if vehicle is not None else 'N/A'}</strong></span>
            </div>
            <div style="margin-top: 6px; color: #0f172a;">
                Focus: {html.escape(str(focus or 'Planning focus unavailable'))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if factors:
        st.markdown("**Built-environment heat drivers:**")
        cols = st.columns(min(len(factors), 3))
        for idx, factor in enumerate(factors[:3]):
            with cols[idx]:
                if isinstance(factor, dict):
                    feat = factor.get("feature", "Factor")
                    contrib = factor.get("contribution", "")
                    st.metric(label=feat, value=str(contrib))
                elif isinstance(factor, tuple) and len(factor) == 2:
                    st.metric(label=factor[0], value=str(factor[1]))

    btn_c1, btn_c2, btn_c3 = st.columns(3)
    with btn_c1:
        if st.button("Why is this a priority?", key=f"q_priority_{rank}", use_container_width=True):
            _queue_recommendation_prompt(selected_label, f"Why is {name} ranked as a priority cooling location?")
            st.rerun()
    with btn_c2:
        if st.button("Extreme heat history", key=f"q_history_{rank}", use_container_width=True):
            _queue_recommendation_prompt(selected_label, f"How often has extreme heat occurred at {name} over the last 7 years?")
            st.rerun()
    with btn_c3:
        if st.button("Compare with top sites", key=f"q_compare_{rank}", use_container_width=True):
            _queue_recommendation_prompt(selected_label, f"Compare {name} with the other top priority Brooklyn cooling candidates.")
            st.rerun()


def _auto_predict_if_unmeasured(
    candidate: dict[str, Any] | None,
    planning_mode: str,
    latitude: float | None,
    longitude: float | None,
) -> bool:
    if latitude is None or longitude is None or not _is_unmeasured_location(candidate):
        return False
    unmeasured_key = f"{round(latitude, 5)}:{round(longitude, 5)}"
    if st.session_state.get("screen2_auto_predicted_location") == unmeasured_key:
        return False
    if _latest_prediction(latitude, longitude) is not None:
        return False

    prompt = (
        "Screen 1 selected an unmeasured street location. Call predict_location_heat_offset now for the selected coordinates. "
        "Report the returned value as a predicted heat offset relative to the Brooklyn campaign baseline, explain the strongest available attributions, "
        "and never present the offset as an absolute temperature."
    )
    _run_turn(prompt, planning_mode, latitude, longitude)
    st.session_state["screen2_auto_predicted_location"] = unmeasured_key
    return True


def render_screen_2(planning_mode: str = "budget") -> None:
    _render_styles()
    lat, lon, label, candidate = _incoming_location()
    is_unmeasured = _is_unmeasured_location(candidate)

    if lat is not None and lon is not None:
        st.session_state["selected_latitude"] = lat
        st.session_state["selected_longitude"] = lon
        st.session_state["selected_location_name"] = label

    try:
        from agent.agent import agent_status
        configured, status_message = agent_status()
    except Exception as exc:
        configured, status_message = False, str(exc)

    st.markdown(
        f"""
        <div class="copilot-shell">
            <div class="copilot-kicker">Screen 2 · Agentic AI Planning Copilot</div>
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <h2 class="copilot-title">Brooklyn Climate Resilience Copilot</h2>
                <div class="status-chip {'status-chip-warn' if not configured else ''}">
                    {'● ADK agent connected' if configured else '⚠️ Agent needs setup'}
                </div>
            </div>
            <p class="copilot-caption">
                Autonomous planning intelligence grounded in BigQuery ML microclimate inferences,
                7-year NOAA GHCN-Daily weather station records, and ACS demographic vulnerability.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_c1, top_c2 = st.columns([1, 1])
    with top_c1:
        selected_mode = st.radio(
            "Planning mode",
            options=PLANNING_MODES,
            index=PLANNING_MODES.index(planning_mode) if planning_mode in PLANNING_MODES else 1,
            format_func=lambda m: MODE_LABELS.get(m, m),
            horizontal=True,
            key="screen2_planning_mode_radio",
        )
    with top_c2:
        if is_unmeasured:
            st.info(f"📍 Context: Unmeasured street selection ({lat:.4f}, {lon:.4f}) carried over from Screen 1.")
        elif lat is not None and lon is not None:
            st.success(f"📍 Context: {label} ({lat:.4f}, {lon:.4f}) selected.")
        else:
            st.info("📍 Context: Borough-wide Brooklyn planning scope.")

    if lat is not None and lon is not None:
        if _auto_predict_if_unmeasured(candidate, selected_mode, lat, lon):
            st.rerun()

    recommendations, rec_error = _ranked_locations()

    col_chat, col_evidence = st.columns([1.1, 0.9], gap="medium")

    with col_chat:
        st.markdown("### Planning conversation")
        st.caption("Ask questions or click suggested prompts below.")

        prompts = QUICK_PROMPTS.get(selected_mode, QUICK_PROMPTS["budget"])
        p_cols = st.columns(len(prompts))
        for idx, prompt_text in enumerate(prompts):
            with p_cols[idx]:
                if st.button(prompt_text, key=f"quick_prompt_{selected_mode}_{idx}", use_container_width=True):
                    st.session_state["screen2_pending_prompt"] = prompt_text
                    st.rerun()

        messages = st.session_state.setdefault("screen2_copilot_messages", [])
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Ask about cooling locations, budget allocation, or heat drivers…")
        pending_prompt = st.session_state.pop("screen2_pending_prompt", None)
        active_prompt = user_input or pending_prompt

        if active_prompt:
            target_lat = lat if lat is not None else 40.6694
            target_lon = lon if lon is not None else -73.9085
            _run_turn(active_prompt, selected_mode, target_lat, target_lon)
            st.rerun()

    with col_evidence:
        _render_agent_evidence(lat, lon)
        st.markdown("<hr style='border-color: #e2e8f0; margin: 16px 0;'>", unsafe_allow_html=True)
        _render_ranked_locations(recommendations, rec_error)
