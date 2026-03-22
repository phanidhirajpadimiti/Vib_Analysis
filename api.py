"""
FastAPI application — REST endpoints for vibration data, plots, and labeling.
"""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from agent import run_machine_analysis
from config import DB_PATH, ENV_PATH, FEATURE_LABELS, STATIC_DIR
from db import (
    export_labels_csv,
    fetch_machines,
    fetch_sensors,
    get_all_labels,
    get_connection,
    get_label_summary,
    get_machine_label,
    load_sensor_df,
    review_machine,
    save_labels,
)
from models import (
    LabelSummary,
    MachineAnalysisResult,
    MachineInfo,
    MachineReviewRequest,
    SensorAssessment,
    SensorInfo,
    ToolCallRecord,
)
from plotting import render_scatter

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
    title="VibLabel — Agentic Data Labeling for Vibration Analysis",
    description="LLM-powered labeling tool for industrial vibration sensor data.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
# Endpoints — labeling (agent)
# ---------------------------------------------------------------------------
@app.post("/label/machine/{machine_id}", response_model=MachineAnalysisResult)
def label_machine(
    machine_id: str,
    provider: str = Query("auto", pattern="^(auto|gemini|ollama)$"),
):
    """Run the agent labeler on a single machine. Persists labels to DB.

    Query params:
        provider: "gemini", "ollama", or "auto" (uses server default).
    """
    use_local = None if provider == "auto" else (provider == "ollama")
    log.info("Agent labeling started for %s (provider=%s)", machine_id, provider)
    try:
        result = run_machine_analysis(
            machine_id,
            tags=[machine_id, "single"],
            metadata={"machine_id": machine_id, "source": "single"},
            use_local_slm=use_local,
        )
    except Exception:
        log.exception("Agent labeling failed for %s", machine_id)
        raise HTTPException(status_code=500, detail="Agent labeling failed. Check server logs.")

    if "error" in result and not result.get("machine_label"):
        raise HTTPException(status_code=404, detail=result["error"])

    with get_connection() as conn:
        save_labels(conn, result)

    log.info("Agent labeling complete for %s → %s (%s confidence)",
             machine_id, result.get("machine_label"), result.get("machine_confidence"))
    return _result_to_response(result, machine_id)


@app.post("/label/batch")
def label_batch(
    provider: str = Query("auto", pattern="^(auto|gemini|ollama)$"),
):
    """Run the agent labeler on ALL unlabeled machines. Returns summary."""
    use_local = None if provider == "auto" else (provider == "ollama")
    with get_connection() as conn:
        machines = fetch_machines(conn)
        already_labeled = set()
        try:
            rows = conn.execute("SELECT machine_id FROM machine_labels").fetchall()
            already_labeled = {r[0] for r in rows}
        except Exception:
            pass

    to_label = [m for m in machines if m.machine_id not in already_labeled]
    results = {"total": len(to_label), "succeeded": 0, "failed": 0, "errors": []}

    for m in to_label:
        try:
            result = run_machine_analysis(
                m.machine_id,
                tags=[m.machine_id, "batch"],
                metadata={"machine_id": m.machine_id, "source": "batch"},
                use_local_slm=use_local,
            )
            if "error" in result and not result.get("machine_label"):
                results["failed"] += 1
                results["errors"].append({"machine_id": m.machine_id, "error": result["error"]})
                continue
            with get_connection() as conn:
                save_labels(conn, result)
            results["succeeded"] += 1
            log.info("Batch labeled %s → %s", m.machine_id, result.get("machine_label"))
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({"machine_id": m.machine_id, "error": str(e)})
            log.exception("Batch labeling failed for %s", m.machine_id)

    return results


# ---------------------------------------------------------------------------
# Endpoints — labels (read)
# ---------------------------------------------------------------------------
@app.get("/labels")
def get_labels():
    """Return all stored labels with review status."""
    with get_connection() as conn:
        return get_all_labels(conn)


@app.get("/labels/summary", response_model=LabelSummary)
def labels_summary():
    with get_connection() as conn:
        return get_label_summary(conn)


@app.get("/label/machine/{machine_id}")
def get_machine_labels(machine_id: str):
    """Return stored labels for a specific machine."""
    with get_connection() as conn:
        result = get_machine_label(conn, machine_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"No labels for {machine_id}")
    return result


# ---------------------------------------------------------------------------
# Endpoints — human review
# ---------------------------------------------------------------------------
@app.put("/label/machine/{machine_id}/review")
def submit_review(machine_id: str, body: MachineReviewRequest):
    """Accept or override agent labels for a machine."""
    with get_connection() as conn:
        existing = get_machine_label(conn, machine_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"No labels for {machine_id}")

    sensor_reviews = [s.dict() if hasattr(s, "dict") else s for s in body.sensors]
    with get_connection() as conn:
        review_machine(conn, machine_id, body.machine_label, body.machine_notes, sensor_reviews)

    log.info("Review saved for %s (override=%s)", machine_id, body.machine_label or "none")
    return {"status": "saved", "machine_id": machine_id}


# ---------------------------------------------------------------------------
# Endpoints — export
# ---------------------------------------------------------------------------
@app.get("/labels/export")
def export_labels():
    """Export all sensor labels as CSV. Human overrides take precedence
    as the final_label column — ready for ML pipeline ingestion."""
    with get_connection() as conn:
        csv_text = export_labels_csv(conn)
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vibration_labels.csv"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _result_to_response(result: dict, machine_id: str) -> MachineAnalysisResult:
    return MachineAnalysisResult(
        machine_id=result.get("machine_id", machine_id),
        machine_type=result.get("machine_type", ""),
        machine_label=result.get("machine_label", "unknown"),
        machine_confidence=result.get("machine_confidence", "medium"),
        machine_rationale=result.get("machine_rationale", ""),
        sensors=[
            SensorAssessment(
                sensor_id=s.get("sensor_id", ""),
                position=s.get("position", ""),
                label=s.get("label", "unknown"),
                confidence=s.get("confidence", "medium"),
                finding=s.get("finding", ""),
                iso_zone=s.get("iso_zone", ""),
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
