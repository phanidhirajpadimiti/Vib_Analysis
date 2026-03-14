"""
Agentic vibration analysis — LangGraph + Gemini with tool use.

Graph topology:

    prepare ──▶ agent ◀──▶ tools
                  │
                  ▼
               finalize ──▶ END

The *prepare* node loads machine data and generates 8 overview plots.
The *agent* node invokes Gemini (with bound tools) to reason.
The *tools* node (LangGraph ToolNode) auto-executes any requested calls.
The *finalize* node parses the structured JSON diagnosis from the last message.
"""

import json
import logging
import os
import re
import sqlite3
import time
from typing import Annotated, TypedDict

import google.api_core.exceptions
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import (
    DB_PATH,
    ENV_PATH,
    FEATURE_LABELS,
    GEMINI_MODEL,
    INITIAL_BACKOFF_SECONDS,
    MAX_RETRIES,
    POSITION_ORDER,
    SAMPLES_PER_DAY,
    RECENT_WINDOW_DAYS,
)
from db import get_connection
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
    final_result: dict


# ═══════════════════════════════════════════════════════════════════════════
# Tools — each is self-contained and queries the DB directly
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

    result: dict = {"sensor_id": sensor_id, "position": df["sensor_position"].iloc[0]}
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

    result: dict = {
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
        entry: dict = {"sensor_id": sid, "position": sdf["sensor_position"].iloc[0]}
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


TOOLS = [get_sensor_stats, compare_recent_vs_historical, get_cross_sensor_comparison]


# ═══════════════════════════════════════════════════════════════════════════
# LLM (lazy-init so the env var is guaranteed to be loaded)
# ═══════════════════════════════════════════════════════════════════════════

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Check your .env file.")
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0,
        ).bind_tools(TOOLS)
    return _llm


# ═══════════════════════════════════════════════════════════════════════════
# Agent prompt
# ═══════════════════════════════════════════════════════════════════════════

AGENT_PROMPT = """\
You are an expert vibration analyst performing a MACHINE-LEVEL diagnosis.

## Context
Machine: {machine_id} (type: {machine_type})
Sensors: {sensor_list}

I am providing you with 8 overview plots — the X-axis acceleration peak and
X-axis velocity RMS for each of the 4 sensor positions on this machine.

## Your process
1. Visually inspect the 8 plots for anomalies (trends, spikes, elevated baselines, noise)
2. Use the available tools to quantify your observations:
   - get_sensor_stats: detailed statistics for one sensor
   - compare_recent_vs_historical: last 30 days vs first 60 days
   - get_cross_sensor_comparison: compare all 4 sensors side-by-side
3. Cross-reference findings across sensor positions to identify the fault mechanism
4. Produce your final diagnosis

## Required output format (strict)
When you are done investigating, reply with EXACTLY this JSON (no markdown fences):
{{
  "machine_label": "healthy" or "unhealthy" or "monitor",
  "machine_rationale": "one sentence overall diagnosis",
  "sensors": [
    {{"sensor_id": "...", "position": "...", "label": "healthy/unhealthy/monitor", "finding": "one sentence"}},
    ...for all 4 sensors
  ],
  "recommended_action": "one sentence recommendation",
  "tools_reasoning": "brief summary of what the tool results revealed"
}}
"""


# ═══════════════════════════════════════════════════════════════════════════
# Graph nodes
# ═══════════════════════════════════════════════════════════════════════════

def prepare(state: AgentState) -> dict:
    """Load machine data, generate 8 overview plots, build initial message."""
    machine_id = state["machine_id"]

    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT sensor_id, sensor_position, sensor_axis, "
            "time, accel_peak, vel_rms, machine_type "
            "FROM sensor_data WHERE machine_id = ?",
            conn,
            params=(machine_id,),
            parse_dates=["time"],
        )

    if df.empty:
        return {"machine_type": "unknown", "messages": []}

    machine_type = df["machine_type"].iloc[0]
    sensor_ids = (
        df.drop_duplicates("sensor_id")[["sensor_id", "sensor_position"]]
        .values.tolist()
    )
    sensor_list = ", ".join(f"{sid} ({pos})" for sid, pos in sensor_ids)
    sorted_sensors = sorted(
        sensor_ids,
        key=lambda s: POSITION_ORDER.index(s[1]) if s[1] in POSITION_ORDER else 99,
    )

    content: list[dict] = [
        {
            "type": "text",
            "text": AGENT_PROMPT.format(
                machine_id=machine_id,
                machine_type=machine_type,
                sensor_list=sensor_list,
            ),
        }
    ]

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
    }


def agent(state: AgentState) -> dict:
    """Invoke the LLM with automatic retry on rate-limit errors."""
    llm = _get_llm()
    for attempt in range(MAX_RETRIES):
        try:
            response = llm.invoke(state["messages"])
            return {"messages": [response]}
        except google.api_core.exceptions.ResourceExhausted:
            wait = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
            log.warning(
                "Rate limited (attempt %d/%d), retrying in %ds",
                attempt + 1, MAX_RETRIES, wait,
            )
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(wait)


def should_continue(state: AgentState) -> str:
    """Route: tools if the LLM requested tool calls, otherwise finalize."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "finalize"


def finalize(state: AgentState) -> dict:
    """Parse the agent's final text into the structured diagnosis dict."""
    tool_calls: list[dict] = []
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
    return {"final_result": result}


def _parse_agent_response(text: str) -> dict:
    """Extract the JSON diagnosis from the agent's final text."""
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

def run_machine_analysis(machine_id: str) -> dict:
    """Run the full agentic analysis for a machine. Returns structured result."""
    log.info("LangGraph agent analysis started for %s", machine_id)

    with get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM sensor_data WHERE machine_id = ?",
            (machine_id,),
        ).fetchone()[0]
    if count == 0:
        return {"error": f"No data for machine {machine_id}"}

    result = _graph.invoke(
        {
            "machine_id": machine_id,
            "machine_type": "",
            "messages": [],
            "final_result": {},
        },
        {"recursion_limit": 30},
    )

    return result["final_result"]
