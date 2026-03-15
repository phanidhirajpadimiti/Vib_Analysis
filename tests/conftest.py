"""
Shared test fixtures — creates a minimal deterministic test database.
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def test_db(tmp_path, monkeypatch):
    """Create a small test database and patch DB_PATH everywhere."""
    db_path = str(tmp_path / "test_vibration.db")

    rng = np.random.default_rng(99)
    n = 500
    time_index = pd.date_range("2025-01-01", periods=n, freq="12h")

    rows = []
    machines = [
        ("MACH-T1", "pump", "healthy"),
        ("MACH-T2", "motor", "faulty"),
    ]

    sensor_counter = 0
    for machine_id, machine_type, category in machines:
        for pos in ("drive_end", "non_drive_end", "gearbox", "base"):
            sensor_counter += 1
            sensor_id = f"SENS-T{sensor_counter:03d}"
            for axis in ("x", "y"):
                accel = rng.normal(5.0, 0.5, n)
                vel = rng.normal(2.0, 0.2, n)

                if category == "faulty" and pos == "drive_end":
                    accel += np.linspace(0, 4.0, n)
                    vel += np.linspace(0, 1.5, n)

                accel = np.clip(accel, 0, None)
                vel = np.clip(vel, 0, None)

                for i in range(n):
                    rows.append((
                        machine_id, sensor_id, pos, axis,
                        time_index[i].isoformat(),
                        float(accel[i]), float(vel[i]),
                        machine_type,
                    ))

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE sensor_data (
            machine_id TEXT, sensor_id TEXT, sensor_position TEXT,
            sensor_axis TEXT, time TIMESTAMP, accel_peak REAL,
            vel_rms REAL, machine_type TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO sensor_data VALUES (?,?,?,?,?,?,?,?)", rows
    )
    conn.execute("CREATE INDEX idx_sensor_id ON sensor_data(sensor_id)")
    conn.execute("CREATE INDEX idx_machine_id ON sensor_data(machine_id)")
    conn.commit()
    conn.close()

    monkeypatch.setattr("db.DB_PATH", db_path)
    monkeypatch.setattr("config.DB_PATH", db_path)

    return db_path
