"""
FastAPI application — REST endpoints for vibration data, plots, and AI analysis.
"""

import io
import logging
import os
import re
import time

import google.api_core.exceptions
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agent import run_machine_analysis
from config import (
    DB_PATH,
    ENV_PATH,
    FEATURE_LABELS,
    GEMINI_MODEL,
    INITIAL_BACKOFF_SECONDS,
    MAX_RETRIES,
    STATIC_DIR,
)
from db import fetch_machines, fetch_sensors, get_connection, load_sensor_df
from models import (
    MachineAnalysisResult,
    MachineInfo,
    SensorAssessment,
    SensorInfo,
    SingleAnalysisResult,
    ToolCallRecord,
)
from plotting import render_scatter, to_base64

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv(ENV_PATH)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set the GEMINI_API_KEY in your .env file.")

app = FastAPI(
    title="Vibration Analysis API",
    description="Analyse vibration sensor data with Gemini multimodal vision.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Gemini (single-sensor analysis uses a plain LLM, no tools)
# ---------------------------------------------------------------------------
_llm = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0,
        )
    return _llm


def _invoke_with_retry(content_parts: list) -> str:
    """Send a multimodal message to Gemini with rate-limit retry. Returns text."""
    llm = _get_llm()
    msg = HumanMessage(content=content_parts)
    for attempt in range(MAX_RETRIES):
        try:
            response = llm.invoke([msg])
            return response.content
        except google.api_core.exceptions.ResourceExhausted:
            wait = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
            log.warning("Rate limited (attempt %d/%d), retrying in %ds",
                        attempt + 1, MAX_RETRIES, wait)
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Single-sensor prompt + parser
# ---------------------------------------------------------------------------
ANALYSIS_PROMPT = """\
You are an expert vibration analyst reviewing 4 time-series scatter plots for a
single industrial sensor (acceleration peak and velocity RMS for both X and Y
axes over approximately 3 months).

Look for these fault signatures:
  • Gradual upward trend (bearing wear / degradation)
  • Sudden step-change in amplitude (fault onset)
  • Recurring high-amplitude bursts at regular intervals (gear-mesh / looseness)
  • Abnormally high baseline compared to typical machinery
  • Elevated noise floor (structural looseness)

Reply in EXACTLY this format (no markdown, no brackets):
LABEL | RATIONALE

LABEL must be one of: healthy, unhealthy
RATIONALE must be a single concise sentence (≤ 40 words).
"""

_STRIP_RE = re.compile(r"[\[\]*`]")


def _parse_gemini_response(text: str) -> tuple[str, str]:
    cleaned = _STRIP_RE.sub("", text).strip()
    first_line = cleaned.split("\n", 1)[0]

    if "|" in first_line:
        label_raw, rationale = first_line.split("|", 1)
    else:
        label_raw, rationale = first_line, cleaned

    label = label_raw.strip().lower()
    if label not in {"healthy", "unhealthy"}:
        label = "unknown"

    return label, rationale.strip()


# ---------------------------------------------------------------------------
# Helper: build 4 plots as base64 for a sensor
# ---------------------------------------------------------------------------
def _sensor_plot_parts(df, sensor_id: str, position: str) -> list[dict]:
    """Generate 4 base64 plot parts (x/y × accel/vel) for a single sensor."""
    parts: list[dict] = []
    for axis in ("x", "y"):
        axis_df = df[df["sensor_axis"] == axis]
        for feature in ("accel_peak", "vel_rms"):
            ylabel = FEATURE_LABELS[feature]
            title = f"{sensor_id} ({position})  ·  {axis.upper()} Axis  ·  {ylabel}"
            png = render_scatter(axis_df["time"], axis_df[feature], title, ylabel)
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{to_base64(png)}"},
            })
    return parts


# ---------------------------------------------------------------------------
# Endpoints — dashboard
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found.")
    return HTMLResponse(html_path.read_text())


# ---------------------------------------------------------------------------
# Endpoints — data
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "database_found": os.path.exists(DB_PATH)}


@app.get("/sensors", response_model=list[SensorInfo])
def list_sensors():
    with get_connection() as conn:
        return fetch_sensors(conn)


@app.get("/machines", response_model=list[MachineInfo])
def list_machines():
    with get_connection() as conn:
        return fetch_machines(conn)


@app.get("/machine/{machine_id}/sensors", response_model=list[SensorInfo])
def get_machine_sensors(machine_id: str):
    with get_connection() as conn:
        sensors = fetch_sensors(conn, machine_id=machine_id)
    if not sensors:
        raise HTTPException(status_code=404, detail=f"No machine {machine_id}")
    return sensors


# ---------------------------------------------------------------------------
# Endpoints — plots
# ---------------------------------------------------------------------------
@app.get("/sensor/{sensor_id}/plot/{axis}/{feature}")
def get_sensor_plot(sensor_id: str, axis: str, feature: str):
    if axis not in ("x", "y"):
        raise HTTPException(status_code=400, detail="axis must be 'x' or 'y'")
    if feature not in ("accel_peak", "vel_rms"):
        raise HTTPException(status_code=400, detail="feature must be 'accel_peak' or 'vel_rms'")

    with get_connection() as conn:
        df = load_sensor_df(conn, sensor_id)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for sensor {sensor_id}")

    pos = df["sensor_position"].iloc[0]
    axis_df = df[df["sensor_axis"] == axis]
    ylabel = FEATURE_LABELS.get(feature, feature)
    title = f"{sensor_id} ({pos})  ·  {axis.upper()} Axis  ·  {ylabel}"

    png = render_scatter(axis_df["time"], axis_df[feature], title, ylabel)
    return Response(content=png, media_type="image/png")


# ---------------------------------------------------------------------------
# Endpoints — single-sensor AI analysis
# ---------------------------------------------------------------------------
@app.get("/sensor/{sensor_id}/analyze", response_model=SingleAnalysisResult)
def analyze_single_sensor(sensor_id: str):
    log.info("Single-sensor analysis for %s", sensor_id)

    with get_connection() as conn:
        df = load_sensor_df(conn, sensor_id)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for sensor {sensor_id}")

    machine_id = df["machine_id"].iloc[0]
    machine_type = df["machine_type"].iloc[0]
    position = df["sensor_position"].iloc[0]

    try:
        parts: list[dict] = [{"type": "text", "text": ANALYSIS_PROMPT}]
        parts.extend(_sensor_plot_parts(df, sensor_id, position))
        text = _invoke_with_retry(parts)
        label, rationale = _parse_gemini_response(text)
        log.info("  %s → %s", sensor_id, label)
    except Exception:
        log.exception("Gemini call failed for %s", sensor_id)
        label, rationale = "error", "AI analysis failed. Check server logs."

    return SingleAnalysisResult(
        sensor_id=sensor_id,
        machine_id=machine_id,
        sensor_position=position,
        machine_type=machine_type,
        label=label,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Endpoints — agentic machine-level analysis
# ---------------------------------------------------------------------------
@app.get("/machine/{machine_id}/agent-analyze", response_model=MachineAnalysisResult)
def agent_analyze_machine(machine_id: str):
    log.info("Agent analysis started for %s", machine_id)
    try:
        result = run_machine_analysis(machine_id)
    except Exception:
        log.exception("Agent analysis failed for %s", machine_id)
        raise HTTPException(status_code=500, detail="Agent analysis failed. Check server logs.")

    if "error" in result and not result.get("machine_label"):
        raise HTTPException(status_code=404, detail=result["error"])

    log.info("Agent analysis complete for %s → %s", machine_id, result.get("machine_label"))
    return MachineAnalysisResult(
        machine_id=result.get("machine_id", machine_id),
        machine_type=result.get("machine_type", ""),
        machine_label=result.get("machine_label", "unknown"),
        machine_rationale=result.get("machine_rationale", ""),
        sensors=[
            SensorAssessment(
                sensor_id=s.get("sensor_id", ""),
                position=s.get("position", ""),
                label=s.get("label", "unknown"),
                finding=s.get("finding", ""),
            )
            for s in result.get("sensors", [])
        ],
        recommended_action=result.get("recommended_action", ""),
        tools_reasoning=result.get("tools_reasoning", ""),
        tool_calls=[
            ToolCallRecord(tool=tc["tool"], args=tc["args"])
            for tc in result.get("tool_calls", [])
        ],
    )


# ---------------------------------------------------------------------------
# Static files (must be last so it doesn't shadow API routes)
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
