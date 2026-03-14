"""
Pydantic schemas shared across the API and agent.
"""

from pydantic import BaseModel, Field

from config import MAX_SENSORS_PER_REQUEST


# ── Request models ──

class AnalysisRequest(BaseModel):
    sensor_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_SENSORS_PER_REQUEST,
        description=f"1–{MAX_SENSORS_PER_REQUEST} sensor IDs to analyse.",
    )


# ── Sensor / machine listing ──

class SensorInfo(BaseModel):
    sensor_id: str
    machine_id: str
    sensor_position: str
    machine_type: str
    axes: list[str]
    row_count: int


class MachineInfo(BaseModel):
    machine_id: str
    machine_type: str
    sensors: list[SensorInfo]


# ── Single-sensor analysis ──

class SingleAnalysisResult(BaseModel):
    sensor_id: str
    machine_id: str
    sensor_position: str
    machine_type: str
    label: str
    rationale: str


# ── Agentic machine-level analysis ──

class SensorAssessment(BaseModel):
    sensor_id: str
    position: str
    label: str
    finding: str


class ToolCallRecord(BaseModel):
    tool: str
    args: dict


class MachineAnalysisResult(BaseModel):
    machine_id: str
    machine_type: str
    machine_label: str
    machine_rationale: str
    sensors: list[SensorAssessment]
    recommended_action: str
    tools_reasoning: str
    tool_calls: list[ToolCallRecord]
