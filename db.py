"""
Database helpers — connection management, data queries, and label storage.
"""

import csv
import io
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import pandas as pd

from config import DB_PATH, ISO_MACHINE_CLASSES
from models import SensorInfo, MachineInfo


# ═══════════════════════════════════════════════════════════════════════════
# Connection
# ═══════════════════════════════════════════════════════════════════════════

@contextmanager
def get_connection(db_path: Optional[str] = None):
    """Yield a SQLite connection, closing it on exit."""
    conn = sqlite3.connect(db_path or DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# Sensor data queries
# ═══════════════════════════════════════════════════════════════════════════

def load_sensor_df(conn: sqlite3.Connection, sensor_id: str) -> pd.DataFrame:
    """Load all rows for a single sensor as a DataFrame."""
    return pd.read_sql_query(
        "SELECT machine_id, sensor_position, sensor_axis, time, "
        "accel_peak, vel_rms, machine_type "
        "FROM sensor_data WHERE sensor_id = ? ORDER BY time",
        conn,
        params=(sensor_id,),
        parse_dates=["time"],
    )


def load_machine_df(conn: sqlite3.Connection, machine_id: str) -> pd.DataFrame:
    """Load all rows for every sensor on a machine."""
    return pd.read_sql_query(
        "SELECT sensor_id, sensor_position, sensor_axis, "
        "time, accel_peak, vel_rms, machine_type "
        "FROM sensor_data WHERE machine_id = ? ORDER BY time",
        conn,
        params=(machine_id,),
        parse_dates=["time"],
    )


_SENSOR_SUMMARY_SQL = """
    SELECT sensor_id, machine_id, sensor_position, machine_type,
           GROUP_CONCAT(DISTINCT sensor_axis) AS axes,
           COUNT(*) AS row_count
    FROM sensor_data
    {where}
    GROUP BY sensor_id
    ORDER BY machine_id, sensor_position
"""


def _rows_to_sensor_infos(rows: list) -> list:
    return [
        SensorInfo(
            sensor_id=r[0],
            machine_id=r[1],
            sensor_position=r[2],
            machine_type=r[3],
            axes=r[4].split(","),
            row_count=r[5],
        )
        for r in rows
    ]


def fetch_sensors(conn: sqlite3.Connection, machine_id: Optional[str] = None) -> list:
    """Return sensor summaries, optionally filtered by machine."""
    if machine_id:
        sql = _SENSOR_SUMMARY_SQL.format(where="WHERE machine_id = ?")
        rows = conn.execute(sql, (machine_id,)).fetchall()
    else:
        sql = _SENSOR_SUMMARY_SQL.format(where="")
        rows = conn.execute(sql).fetchall()
    return _rows_to_sensor_infos(rows)


def fetch_machines(conn: sqlite3.Connection) -> list:
    """Return all machines with their sensors."""
    sensors = fetch_sensors(conn)
    machines = {}
    for s in sensors:
        mid = s.machine_id
        if mid not in machines:
            machines[mid] = MachineInfo(machine_id=mid, machine_type=s.machine_type, sensors=[])
        machines[mid].sensors.append(s)
    return list(machines.values())


# ═══════════════════════════════════════════════════════════════════════════
# Label storage — schema
# ═══════════════════════════════════════════════════════════════════════════

_CREATE_MACHINE_LABELS = """
CREATE TABLE IF NOT EXISTS machine_labels (
    machine_id TEXT PRIMARY KEY,
    machine_type TEXT NOT NULL,
    iso_class TEXT NOT NULL,
    agent_label TEXT NOT NULL,
    agent_confidence TEXT NOT NULL,
    agent_rationale TEXT,
    recommended_action TEXT,
    tools_reasoning TEXT,
    tool_calls_json TEXT,
    review_status TEXT DEFAULT 'pending',
    human_label TEXT,
    human_notes TEXT,
    raw_context_fed_to_llm TEXT,
    raw_llm_response TEXT,
    prompt_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP
)
"""

_CREATE_SENSOR_LABELS = """
CREATE TABLE IF NOT EXISTS sensor_labels (
    sensor_id TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    sensor_position TEXT NOT NULL,
    agent_label TEXT NOT NULL,
    agent_confidence TEXT NOT NULL,
    agent_finding TEXT,
    iso_zone TEXT,
    review_status TEXT DEFAULT 'pending',
    human_label TEXT,
    human_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP,
    PRIMARY KEY (machine_id, sensor_id)
)
"""


def ensure_labels_tables(conn: sqlite3.Connection):
    """Create the label tables if they don't exist, and migrate schema for
    existing databases (non-destructive — adds nullable columns only)."""
    conn.execute(_CREATE_MACHINE_LABELS)
    conn.execute(_CREATE_SENSOR_LABELS)

    existing = {r[1] for r in conn.execute("PRAGMA table_info(machine_labels)").fetchall()}
    for col in ("raw_context_fed_to_llm", "raw_llm_response", "prompt_version"):
        if col not in existing:
            conn.execute(f"ALTER TABLE machine_labels ADD COLUMN {col} TEXT")

    conn.commit()


# ═══════════════════════════════════════════════════════════════════════════
# Label storage — write
# ═══════════════════════════════════════════════════════════════════════════

def save_labels(conn: sqlite3.Connection, result: dict):
    """Persist an agent's labeling result (machine + sensors) to the DB.
    Uses INSERT OR REPLACE so re-labeling overwrites previous labels."""
    ensure_labels_tables(conn)
    now = datetime.utcnow().isoformat()
    machine_id = result["machine_id"]
    machine_type = result.get("machine_type", "")
    iso_class = ISO_MACHINE_CLASSES.get(machine_type, "II")

    conn.execute(
        "INSERT OR REPLACE INTO machine_labels "
        "(machine_id, machine_type, iso_class, agent_label, agent_confidence, "
        " agent_rationale, recommended_action, tools_reasoning, tool_calls_json, "
        " review_status, human_label, human_notes, "
        " raw_context_fed_to_llm, raw_llm_response, prompt_version, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            machine_id, machine_type, iso_class,
            result.get("machine_label", "unknown"),
            result.get("machine_confidence", "medium"),
            result.get("machine_rationale", ""),
            result.get("recommended_action", ""),
            result.get("tools_reasoning", ""),
            json.dumps(result.get("tool_calls", [])),
            "pending", None, None,
            result.get("raw_context", ""),
            result.get("raw_response", ""),
            result.get("prompt_version", ""),
            now,
        ),
    )

    for s in result.get("sensors", []):
        conn.execute(
            "INSERT OR REPLACE INTO sensor_labels "
            "(sensor_id, machine_id, sensor_position, agent_label, agent_confidence, "
            " agent_finding, iso_zone, review_status, human_label, human_notes, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                s.get("sensor_id", ""),
                machine_id,
                s.get("position", ""),
                s.get("label", "unknown"),
                s.get("confidence", "medium"),
                s.get("finding", ""),
                s.get("iso_zone", ""),
                "pending", None, None, now,
            ),
        )
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════════
# Label storage — read
# ═══════════════════════════════════════════════════════════════════════════

def get_machine_label(conn: sqlite3.Connection, machine_id: str) -> Optional[dict]:
    """Return stored labels for one machine + its sensors, or None."""
    ensure_labels_tables(conn)
    row = conn.execute(
        "SELECT * FROM machine_labels WHERE machine_id = ?", (machine_id,)
    ).fetchone()
    if not row:
        return None

    cols = [d[0] for d in conn.execute("SELECT * FROM machine_labels LIMIT 0").description]
    ml = dict(zip(cols, row))
    ml["final_label"] = ml["human_label"] or ml["agent_label"]
    ml["tool_calls"] = json.loads(ml.pop("tool_calls_json", "[]"))

    s_rows = conn.execute(
        "SELECT * FROM sensor_labels WHERE machine_id = ? ORDER BY sensor_position",
        (machine_id,),
    ).fetchall()
    s_cols = [d[0] for d in conn.execute("SELECT * FROM sensor_labels LIMIT 0").description]
    sensors = []
    for sr in s_rows:
        sd = dict(zip(s_cols, sr))
        sd["final_label"] = sd["human_label"] or sd["agent_label"]
        sensors.append(sd)
    ml["sensors"] = sensors
    return ml


def get_all_labels(conn: sqlite3.Connection) -> list:
    """Return all stored machine labels with their sensors."""
    ensure_labels_tables(conn)
    rows = conn.execute(
        "SELECT machine_id FROM machine_labels ORDER BY machine_id"
    ).fetchall()
    return [get_machine_label(conn, r[0]) for r in rows]


def get_label_summary(conn: sqlite3.Connection) -> dict:
    """Return counts for the label overview."""
    ensure_labels_tables(conn)
    total_machines = conn.execute(
        "SELECT COUNT(DISTINCT machine_id) FROM sensor_data"
    ).fetchone()[0]
    labeled = conn.execute("SELECT COUNT(*) FROM machine_labels").fetchone()[0]
    pending = conn.execute(
        "SELECT COUNT(*) FROM machine_labels WHERE review_status = 'pending'"
    ).fetchone()[0]
    accepted = conn.execute(
        "SELECT COUNT(*) FROM machine_labels WHERE review_status = 'accepted'"
    ).fetchone()[0]
    overridden = conn.execute(
        "SELECT COUNT(*) FROM machine_labels WHERE review_status = 'overridden'"
    ).fetchone()[0]
    return {
        "total_machines": total_machines,
        "labeled": labeled,
        "pending_review": pending,
        "accepted": accepted,
        "overridden": overridden,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Label storage — review (human-in-the-loop)
# ═══════════════════════════════════════════════════════════════════════════

def review_machine(conn: sqlite3.Connection, machine_id: str,
                   machine_label: Optional[str], machine_notes: str,
                   sensor_reviews: list):
    """Save human review for a machine and its sensors."""
    ensure_labels_tables(conn)
    now = datetime.utcnow().isoformat()

    if machine_label:
        conn.execute(
            "UPDATE machine_labels SET human_label=?, human_notes=?, "
            "review_status='overridden', reviewed_at=? WHERE machine_id=?",
            (machine_label, machine_notes, now, machine_id),
        )
    else:
        conn.execute(
            "UPDATE machine_labels SET human_notes=?, "
            "review_status='accepted', reviewed_at=? WHERE machine_id=?",
            (machine_notes, now, machine_id),
        )

    for sr in sensor_reviews:
        sid = sr.get("sensor_id", "")
        h_label = sr.get("human_label")
        h_notes = sr.get("human_notes", "")
        status = "overridden" if h_label else "accepted"
        conn.execute(
            "UPDATE sensor_labels SET human_label=?, human_notes=?, "
            "review_status=?, reviewed_at=? WHERE machine_id=? AND sensor_id=?",
            (h_label, h_notes, status, now, machine_id, sid),
        )
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════════
# Export — CSV for ML pipeline consumption
# ═══════════════════════════════════════════════════════════════════════════

def export_labels_csv(conn: sqlite3.Connection) -> str:
    """Export all sensor labels as CSV text. Uses final_label (human override
    takes precedence over agent label)."""
    ensure_labels_tables(conn)
    rows = conn.execute("""
        SELECT s.sensor_id, s.machine_id, m.machine_type, m.iso_class,
               s.sensor_position, s.agent_label, s.agent_confidence,
               s.iso_zone, s.agent_finding,
               s.human_label, s.review_status,
               COALESCE(s.human_label, s.agent_label) AS final_label,
               s.created_at, s.reviewed_at
        FROM sensor_labels s
        JOIN machine_labels m ON s.machine_id = m.machine_id
        ORDER BY s.machine_id, s.sensor_position
    """).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "sensor_id", "machine_id", "machine_type", "iso_class",
        "sensor_position", "agent_label", "agent_confidence",
        "iso_zone", "agent_finding",
        "human_label", "review_status", "final_label",
        "created_at", "reviewed_at",
    ])
    writer.writerows(rows)
    return output.getvalue()
