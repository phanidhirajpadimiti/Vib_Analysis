"""
Agentic vibration data labeling — dual-provider LangGraph agent.

Supports two modes controlled by USE_LOCAL_SLM in config:
  - Cloud (default): Gemini 2.5 Flash with multi-turn tool calling
  - Local: Llama 3.2 Vision 11B via Ollama with single-pass pre-computed tools

Graph topology (both modes share the same graph):

    prepare ──▶ agent ◀──▶ tools    (Gemini: multi-turn tool loop)
                  │
                  ▼
               finalize ──▶ END

    prepare ──▶ agent ──▶ finalize  (SLM: single-pass, no tool_calls)
"""

import json
import logging
import os
import re
import time
from typing import Annotated, TypedDict

import google.api_core.exceptions
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import (
    ENV_PATH,
    FEATURE_LABELS,
    GEMINI_MODEL,
    INITIAL_BACKOFF_SECONDS,
    ISO_MACHINE_CLASSES,
    ISO_THRESHOLDS,
    MAX_RETRIES,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    POSITION_ORDER,
    PROMPT_VERSION,
    RECENT_WINDOW_DAYS,
    SAMPLES_PER_DAY,
    USE_LOCAL_SLM,
)
from db import get_connection, load_machine_df
from plotting import render_scatter, to_base64

log = logging.getLogger(__name__)
load_dotenv(ENV_PATH)


# ═══════════════════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    machine_id: str
    machine_type: str
    use_local_slm: bool
    raw_context: str
    raw_response: str
    final_result: dict


# ═══════════════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════════════

def _compute_feature_stats(vals: np.ndarray) -> dict:
    """Shared stat computation for a single feature column."""
    slope = float(np.polyfit(np.arange(len(vals), dtype=float), vals, 1)[0])
    return {
        "mean": round(float(vals.mean()), 3),
        "std": round(float(vals.std()), 3),
        "min": round(float(vals.min()), 3),
        "max": round(float(vals.max()), 3),
        "trend_slope_per_day": round(slope * SAMPLES_PER_DAY, 5),
        "kurtosis": round(float(pd.Series(vals).kurtosis()), 3),
    }


@tool
def get_sensor_stats(sensor_id: str) -> str:
    """Get detailed statistics for a single sensor: mean, std, min, max,
    linear trend slope (per day), and kurtosis for accel_peak and vel_rms
    on both X and Y axes."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT sensor_position, sensor_axis, accel_peak, vel_rms "
            "FROM sensor_data WHERE sensor_id = ? ORDER BY time",
            conn,
            params=(sensor_id,),
        )
    if df.empty:
        return json.dumps({"error": f"No data for {sensor_id}"})

    result = {"sensor_id": sensor_id, "position": df["sensor_position"].iloc[0]}
    for axis in ("x", "y"):
        adf = df[df["sensor_axis"] == axis]
        for feature in ("accel_peak", "vel_rms"):
            vals = adf[feature].values
            if len(vals) < 2:
                continue
            result[f"{axis}_{feature}"] = _compute_feature_stats(vals)
    return json.dumps(result)


@tool
def compare_recent_vs_historical(sensor_id: str) -> str:
    """Compare the last 30 days against the first ~60 days for a sensor.
    Returns mean, std, and percentage change for each axis/feature.
    Useful for detecting recent degradation or improving trends."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT sensor_axis, time, accel_peak, vel_rms "
            "FROM sensor_data WHERE sensor_id = ? ORDER BY time",
            conn,
            params=(sensor_id,),
            parse_dates=["time"],
        )
    if df.empty:
        return json.dumps({"error": f"No data for {sensor_id}"})

    cutoff = df["time"].max() - pd.Timedelta(days=RECENT_WINDOW_DAYS)
    hist = df[df["time"] < cutoff]
    recent = df[df["time"] >= cutoff]
    total_days = (df["time"].max() - df["time"].min()).days

    result = {
        "sensor_id": sensor_id,
        "historical_days": total_days - RECENT_WINDOW_DAYS,
        "recent_days": RECENT_WINDOW_DAYS,
    }
    for axis in ("x", "y"):
        for feature in ("accel_peak", "vel_rms"):
            h = hist[hist["sensor_axis"] == axis][feature].values
            r = recent[recent["sensor_axis"] == axis][feature].values
            if len(h) == 0 or len(r) == 0:
                continue
            h_mean, r_mean = float(h.mean()), float(r.mean())
            change = ((r_mean - h_mean) / h_mean * 100) if h_mean else 0
            result[f"{axis}_{feature}"] = {
                "historical_mean": round(h_mean, 3),
                "recent_mean": round(r_mean, 3),
                "change_percent": round(change, 2),
                "historical_std": round(float(h.std()), 3),
                "recent_std": round(float(r.std()), 3),
            }
    return json.dumps(result)


@tool
def get_cross_sensor_comparison(machine_id: str) -> str:
    """Compare all 4 sensors on a machine side by side. Returns mean and std
    for each sensor/axis/feature. Useful for identifying which positions are
    anomalous relative to the others."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT sensor_id, sensor_position, sensor_axis, accel_peak, vel_rms "
            "FROM sensor_data WHERE machine_id = ?",
            conn,
            params=(machine_id,),
        )
    if df.empty:
        return json.dumps({"error": f"No data for {machine_id}"})

    sensors = []
    for sid in df["sensor_id"].unique():
        sdf = df[df["sensor_id"] == sid]
        entry = {"sensor_id": sid, "position": sdf["sensor_position"].iloc[0]}
        for axis in ("x", "y"):
            for feature in ("accel_peak", "vel_rms"):
                vals = sdf[sdf["sensor_axis"] == axis][feature].values
                if len(vals) == 0:
                    continue
                entry[f"{axis}_{feature}_mean"] = round(float(vals.mean()), 3)
                entry[f"{axis}_{feature}_std"] = round(float(vals.std()), 3)
        sensors.append(entry)

    sensors.sort(
        key=lambda s: POSITION_ORDER.index(s["position"])
        if s["position"] in POSITION_ORDER
        else 99
    )
    return json.dumps({"machine_id": machine_id, "sensors": sensors})


def _classify_iso_zone(vel_rms: float, thresholds: dict) -> str:
    if vel_rms <= thresholds["A_B"]:
        return "A"
    if vel_rms <= thresholds["B_C"]:
        return "B"
    if vel_rms <= thresholds["C_D"]:
        return "C"
    return "D"


@tool
def get_iso_assessment(sensor_id: str) -> str:
    """Assess a sensor against ISO 10816 vibration severity zones.
    Returns the current zone (A/B/C/D), the zone boundaries for the machine
    class, current velocity RMS, trend slope, and projected days until the
    next zone boundary is crossed. Essential for grounding labels in
    international standards."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT sensor_axis, time, vel_rms, machine_type "
            "FROM sensor_data WHERE sensor_id = ? ORDER BY time",
            conn,
            params=(sensor_id,),
            parse_dates=["time"],
        )
    if df.empty:
        return json.dumps({"error": f"No data for {sensor_id}"})

    machine_type = df["machine_type"].iloc[0]
    iso_class = ISO_MACHINE_CLASSES.get(machine_type, "II")
    thresholds = ISO_THRESHOLDS[iso_class]

    results = {"sensor_id": sensor_id, "iso_class": iso_class, "thresholds_mm_s": thresholds}
    for axis in ("x", "y"):
        adf = df[df["sensor_axis"] == axis].sort_values("time")
        vals = adf["vel_rms"].values
        if len(vals) < 2:
            continue

        current_mean = float(vals[-SAMPLES_PER_DAY:].mean()) if len(vals) >= SAMPLES_PER_DAY else float(vals.mean())
        zone = _classify_iso_zone(current_mean, thresholds)

        slope_per_sample = float(np.polyfit(np.arange(len(vals), dtype=float), vals, 1)[0])
        slope_per_day = slope_per_sample * SAMPLES_PER_DAY

        next_boundary = None
        days_to_next = None
        if zone == "A":
            next_boundary = thresholds["A_B"]
        elif zone == "B":
            next_boundary = thresholds["B_C"]
        elif zone == "C":
            next_boundary = thresholds["C_D"]

        if next_boundary and slope_per_day > 0.0001:
            gap = next_boundary - current_mean
            days_to_next = round(gap / slope_per_day, 1)

        results[f"{axis}_assessment"] = {
            "current_vel_rms": round(current_mean, 3),
            "zone": zone,
            "trend_slope_per_day": round(slope_per_day, 5),
            "next_zone_boundary": next_boundary,
            "days_to_next_zone": days_to_next,
        }
    return json.dumps(results)


TOOLS = [get_sensor_stats, compare_recent_vs_historical,
         get_cross_sensor_comparison, get_iso_assessment]


# ═══════════════════════════════════════════════════════════════════════════
# LLM (lazy-init, dual-provider cache)
# ═══════════════════════════════════════════════════════════════════════════

_llm_cache: dict = {}


def _get_llm(use_local: bool = False):
    """Return the LLM for the requested provider. Each provider is initialized
    once and cached. Gemini gets tool binding; Ollama does not (single-pass)."""
    key = "ollama" if use_local else "gemini"
    if key not in _llm_cache:
        if use_local:
            log.info("Initializing local SLM: %s at %s", OLLAMA_MODEL, OLLAMA_BASE_URL)
            _llm_cache[key] = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,
            )
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set. Check your .env file.")
            _llm_cache[key] = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=api_key,
                temperature=0,
            ).bind_tools(TOOLS)
    return _llm_cache[key]


# ═══════════════════════════════════════════════════════════════════════════
# Agent prompt — framed as a labeling task
# ═══════════════════════════════════════════════════════════════════════════

AGENT_PROMPT = """\
You are an expert vibration analyst acting as a DATA LABELER. Your job is to
assign quality labels to sensor data that will be used to train or validate
machine learning models for condition monitoring.

## Context
Machine: {machine_id} (type: {machine_type}, ISO 10816 class: {iso_class})
Sensors: {sensor_list}

I am providing you with 8 overview plots — the X-axis acceleration peak and
X-axis velocity RMS for each of the 4 sensor positions on this machine.

## Labeling guidelines
Use the following criteria (grounded in ISO 10816):
  - **healthy**: ISO Zone A or stable Zone B, no upward trend, consistent across positions
  - **monitor**: Zone B with upward trend, borderline B/C, or conflicting cross-sensor signals
  - **unhealthy**: Zone C or D, strong upward trend, or clear fault signature (bearing, misalignment, looseness)

## Your process
1. Visually inspect the 8 plots for anomalies
2. Use tools to quantify observations:
   - get_iso_assessment: assess each sensor against ISO 10816 zones (ALWAYS USE THIS)
   - get_sensor_stats: detailed statistics for one sensor
   - compare_recent_vs_historical: last 30 days vs baseline
   - get_cross_sensor_comparison: compare all 4 positions side-by-side
3. Cross-reference findings across sensor positions
4. Assign labels with confidence levels

## Confidence levels
  - **high**: clear evidence, all signals agree
  - **medium**: some ambiguity but balance of evidence favors the label
  - **low**: conflicting signals, borderline case — flag for human review

## Required output format (strict)
Reply with EXACTLY this JSON (no markdown fences):
{{
  "machine_label": "healthy" or "unhealthy" or "monitor",
  "machine_confidence": "high" or "medium" or "low",
  "machine_rationale": "one sentence grounded in ISO zones and tool results",
  "sensors": [
    {{
      "sensor_id": "...",
      "position": "...",
      "label": "healthy/unhealthy/monitor",
      "confidence": "high/medium/low",
      "finding": "one sentence with ISO zone reference",
      "iso_zone": "A/B/C/D"
    }},
    ...for all 4 sensors
  ],
  "recommended_action": "one sentence recommendation",
  "tools_reasoning": "brief summary of what the tool results revealed"
}}
"""

# ── SLM prompt (single-pass — all tool results pre-computed) ──────────────

SLM_AGENT_PROMPT = """\
You are an expert vibration analyst acting as a DATA LABELER.

## Context
Machine: {machine_id} (type: {machine_type}, ISO 10816 class: {iso_class})
Sensors: {sensor_list}

I am providing you with overview plots for each sensor position, plus
pre-computed analysis results from our diagnostic tools.

## Pre-computed tool results
{tool_results}

## Labeling criteria (ISO 10816)
- healthy: ISO Zone A or stable Zone B, no upward trend, consistent across positions
- monitor: Zone B with upward trend, borderline B/C, or conflicting cross-sensor signals
- unhealthy: Zone C or D, strong upward trend, or clear fault signature

## Confidence levels
- high: clear evidence, all signals agree
- medium: some ambiguity but balance of evidence favors the label
- low: conflicting signals, borderline case — flag for human review

## CRITICAL INSTRUCTIONS
You MUST output ONLY valid JSON. No explanation, no markdown, no text before or
after the JSON object. Your entire response must be parseable by json.loads().

Output this exact JSON structure:
{{
  "machine_label": "healthy" or "unhealthy" or "monitor",
  "machine_confidence": "high" or "medium" or "low",
  "machine_rationale": "one sentence grounded in ISO zones and data",
  "sensors": [
    {{
      "sensor_id": "...",
      "position": "...",
      "label": "healthy" or "unhealthy" or "monitor",
      "confidence": "high" or "medium" or "low",
      "finding": "one sentence with ISO zone reference",
      "iso_zone": "A" or "B" or "C" or "D"
    }}
  ],
  "recommended_action": "one sentence recommendation",
  "tools_reasoning": "brief summary of what the data revealed"
}}
"""


def _precompute_tool_results(machine_id: str, sensor_ids: list) -> str:
    """Call all tools programmatically and format results as text for SLM."""
    sections = []

    sections.append("=== Cross-Sensor Comparison ===")
    sections.append(get_cross_sensor_comparison.invoke(machine_id))

    for sid, pos in sensor_ids:
        sections.append(f"\n=== Sensor {sid} ({pos}) ===")
        sections.append(f"ISO Assessment: {get_iso_assessment.invoke(sid)}")
        sections.append(f"Statistics: {get_sensor_stats.invoke(sid)}")
        sections.append(
            f"Recent vs Historical: {compare_recent_vs_historical.invoke(sid)}"
        )

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════════
# Graph nodes
# ═══════════════════════════════════════════════════════════════════════════

def prepare(state: AgentState) -> dict:
    """Load machine data, generate overview plots, build initial message.

    In SLM mode, all tool results are pre-computed and injected into the prompt
    so the model can label in a single pass without tool calling.
    """
    machine_id = state["machine_id"]

    with get_connection() as conn:
        df = load_machine_df(conn, machine_id)

    if df.empty:
        return {"machine_type": "unknown", "messages": []}

    machine_type = df["machine_type"].iloc[0]
    iso_class = ISO_MACHINE_CLASSES.get(machine_type, "II")
    sensor_ids = (
        df.drop_duplicates("sensor_id")[["sensor_id", "sensor_position"]]
        .values.tolist()
    )
    sensor_list = ", ".join(f"{sid} ({pos})" for sid, pos in sensor_ids)
    sorted_sensors = sorted(
        sensor_ids,
        key=lambda s: POSITION_ORDER.index(s[1]) if s[1] in POSITION_ORDER else 99,
    )

    if state.get("use_local_slm", USE_LOCAL_SLM):
        tool_results = _precompute_tool_results(machine_id, sorted_sensors)
        prompt_text = SLM_AGENT_PROMPT.format(
            machine_id=machine_id,
            machine_type=machine_type,
            iso_class=iso_class,
            sensor_list=sensor_list,
            tool_results=tool_results,
        )
    else:
        prompt_text = AGENT_PROMPT.format(
            machine_id=machine_id,
            machine_type=machine_type,
            iso_class=iso_class,
            sensor_list=sensor_list,
        )

    content = [{"type": "text", "text": prompt_text}]

    for sid, pos in sorted_sensors:
        sdf = (
            df[(df["sensor_id"] == sid) & (df["sensor_axis"] == "x")]
            .sort_values("time")
        )
        for feature in ("accel_peak", "vel_rms"):
            ylabel = FEATURE_LABELS[feature]
            png = render_scatter(
                sdf["time"], sdf[feature],
                f"{sid} ({pos}) · X · {ylabel}", ylabel,
                figsize=(6, 3.5), dpi=90, point_size=0.6, alpha=0.4,
            )
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{to_base64(png)}"},
            })

    return {
        "messages": [HumanMessage(content=content)],
        "machine_type": machine_type,
        "raw_context": prompt_text,
    }


def agent(state: AgentState) -> dict:
    """Invoke the LLM with automatic retry on transient errors."""
    use_local = state.get("use_local_slm", USE_LOCAL_SLM)
    llm = _get_llm(use_local)
    retryable = (
        (ConnectionError, OSError)
        if use_local
        else (google.api_core.exceptions.ResourceExhausted,)
    )
    for attempt in range(MAX_RETRIES):
        try:
            response = llm.invoke(state["messages"])
            return {"messages": [response]}
        except retryable:
            wait = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
            log.warning(
                "Retryable error (attempt %d/%d), retrying in %ds",
                attempt + 1, MAX_RETRIES, wait,
            )
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(wait)


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "finalize"


def finalize(state: AgentState) -> dict:
    """Parse the agent's final text into the structured labeling result."""
    tool_calls = []
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({"tool": tc["name"], "args": tc["args"]})

    last = state["messages"][-1]
    raw = last.content if isinstance(last.content, str) else str(last.content)
    log.info("  Agent done: %d chars, %d tool calls total", len(raw), len(tool_calls))

    result = _parse_agent_response(raw)
    result["tool_calls"] = tool_calls
    result["machine_id"] = state["machine_id"]
    result["machine_type"] = state["machine_type"]
    result["raw_context"] = state.get("raw_context", "")
    result["raw_response"] = raw
    result["prompt_version"] = PROMPT_VERSION
    return {"final_result": result}


def _parse_agent_response(text: str) -> dict:
    """Extract the JSON labeling result from the agent's final text."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    return {
        "machine_label": "unknown",
        "machine_confidence": "low",
        "machine_rationale": text[:300],
        "sensors": [],
        "recommended_action": "Manual review needed — agent response could not be parsed.",
        "tools_reasoning": "",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Build and compile the graph
# ═══════════════════════════════════════════════════════════════════════════

def _build_graph():
    g = StateGraph(AgentState)

    g.add_node("prepare", prepare)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("finalize", finalize)

    g.set_entry_point("prepare")
    g.add_edge("prepare", "agent")
    g.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "finalize": "finalize",
    })
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)

    return g.compile()


_graph = _build_graph()


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def run_machine_analysis(
    machine_id: str,
    tags: list = None,
    metadata: dict = None,
    use_local_slm: bool = None,
) -> dict:
    """Run the full agentic labeling for a machine. Returns structured result.

    Args:
        use_local_slm: Override the global USE_LOCAL_SLM flag for this request.
                        None = use the global config default.
    """
    local = USE_LOCAL_SLM if use_local_slm is None else use_local_slm
    provider = "ollama" if local else "gemini"
    log.info("LangGraph agent labeling started for %s (provider=%s)", machine_id, provider)

    with get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM sensor_data WHERE machine_id = ?",
            (machine_id,),
        ).fetchone()[0]
    if count == 0:
        return {"error": f"No data for machine {machine_id}"}

    config = {
        "recursion_limit": 30,
        "run_name": f"label-{machine_id}",
        "tags": tags or [machine_id],
        "metadata": {**(metadata or {"machine_id": machine_id}), "provider": provider},
    }

    result = _graph.invoke(
        {
            "machine_id": machine_id,
            "machine_type": "",
            "messages": [],
            "use_local_slm": local,
            "raw_context": "",
            "raw_response": "",
            "final_result": {},
        },
        config,
    )

    return result["final_result"]
