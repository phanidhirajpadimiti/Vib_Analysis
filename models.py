"""
Pydantic schemas shared across the API, agent, and label storage.
"""

from typing import Optional

from pydantic import BaseModel


# ── Sensor / machine listing ──

class SensorInfo(BaseModel):
    sensor_id: str
    machine_id: str
    sensor_position: str
    machine_type: str
    axes: list
    row_count: int


class MachineInfo(BaseModel):
    machine_id: str
    machine_type: str
    sensors: list


# ── Agentic labeling result (returned by agent) ──

class SensorAssessment(BaseModel):
    sensor_id: str
    position: str
    label: str
    confidence: str  # high / medium / low
    finding: str
    iso_zone: str    # A / B / C / D


class ToolCallRecord(BaseModel):
    tool: str
    args: dict


class MachineAnalysisResult(BaseModel):
    machine_id: str
    machine_type: str
    machine_label: str
    machine_confidence: str
    machine_rationale: str
    sensors: list
    recommended_action: str
    tools_reasoning: str
    tool_calls: list


# ── Label review (human-in-the-loop) ──

class SensorReview(BaseModel):
    sensor_id: str
    human_label: str         # healthy / unhealthy / monitor
    human_notes: str = ""


class MachineReviewRequest(BaseModel):
    machine_label: Optional[str] = None
    machine_notes: str = ""
    sensors: list = []       # list of SensorReview dicts


# ── Stored labels (what the DB returns) ──

class StoredMachineLabel(BaseModel):
    machine_id: str
    machine_type: str
    iso_class: str
    agent_label: str
    agent_confidence: str
    agent_rationale: str
    recommended_action: str
    review_status: str       # pending / accepted / overridden
    human_label: Optional[str] = None
    human_notes: Optional[str] = None
    final_label: str         # human_label if set, else agent_label
    created_at: str
    reviewed_at: Optional[str] = None


class StoredSensorLabel(BaseModel):
    sensor_id: str
    machine_id: str
    sensor_position: str
    agent_label: str
    agent_confidence: str
    agent_finding: str
    iso_zone: str
    review_status: str
    human_label: Optional[str] = None
    human_notes: Optional[str] = None
    final_label: str
    created_at: str
    reviewed_at: Optional[str] = None


class LabelSummary(BaseModel):
    total_machines: int
    labeled: int
    pending_review: int
    accepted: int
    overridden: int
