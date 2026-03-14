"""
Database helpers — connection management and reusable queries.
"""

import sqlite3
from contextlib import contextmanager
from typing import Optional

import pandas as pd

from config import DB_PATH
from models import SensorInfo, MachineInfo


@contextmanager
def get_connection(db_path: Optional[str] = None):
    """Yield a SQLite connection, closing it on exit."""
    conn = sqlite3.connect(db_path or DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


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


_SENSOR_SUMMARY_SQL = """
    SELECT sensor_id, machine_id, sensor_position, machine_type,
           GROUP_CONCAT(DISTINCT sensor_axis) AS axes,
           COUNT(*) AS row_count
    FROM sensor_data
    {where}
    GROUP BY sensor_id
    ORDER BY machine_id, sensor_position
"""


def _rows_to_sensor_infos(rows: list[tuple]) -> list[SensorInfo]:
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


def fetch_sensors(conn: sqlite3.Connection, machine_id: Optional[str] = None) -> list[SensorInfo]:
    """Return sensor summaries, optionally filtered by machine."""
    if machine_id:
        sql = _SENSOR_SUMMARY_SQL.format(where="WHERE machine_id = ?")
        rows = conn.execute(sql, (machine_id,)).fetchall()
    else:
        sql = _SENSOR_SUMMARY_SQL.format(where="")
        rows = conn.execute(sql).fetchall()
    return _rows_to_sensor_infos(rows)


def fetch_machines(conn: sqlite3.Connection) -> list[MachineInfo]:
    """Return all machines with their sensors."""
    sensors = fetch_sensors(conn)
    machines: dict[str, MachineInfo] = {}
    for s in sensors:
        mid = s.machine_id
        if mid not in machines:
            machines[mid] = MachineInfo(machine_id=mid, machine_type=s.machine_type, sensors=[])
        machines[mid].sensors.append(s)
    return list(machines.values())
