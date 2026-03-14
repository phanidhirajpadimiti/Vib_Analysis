"""
Generate synthetic vibration data organised by machine → sensor → axis.

Machine layout:
  13 machines  ×  4 sensors each  =  52 sensors
  Each sensor has 2 axes (x, y) and 2 features (accel_peak, vel_rms).

Health categories:
  5 clearly healthy  ·  3 clearly faulty  ·  5 ambiguous
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import (
    BASE_DIR, DB_PATH, MACHINE_PROFILES, SENSOR_POSITIONS, AXES, SEED,
)

POSITIONS = list(SENSOR_POSITIONS.keys())  # fixed order for determinism


# ═══════════════════════════════════════════════════════════════════════════
# Healthy baseline generation
# ═══════════════════════════════════════════════════════════════════════════

def _baseline(
    profile: dict,
    position: str,
    n: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Return {accel_x, vel_x, accel_y, vel_y} for one sensor position."""
    pos = SENSOR_POSITIONS[position]
    a_mean = profile["accel_mean"] * pos["accel_mult"]
    a_std = profile["accel_std"] * pos["accel_mult"]
    v_mean = profile["vel_mean"] * pos["vel_mult"]
    v_std = profile["vel_std"] * pos["vel_mult"]

    return {
        "accel_x": rng.normal(a_mean, a_std, n),
        "vel_x":   rng.normal(v_mean, v_std, n),
        "accel_y": rng.normal(a_mean, a_std, n),
        "vel_y":   rng.normal(v_mean, v_std, n),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Clear fault patterns — applied per-machine, correlated across sensors
# ═══════════════════════════════════════════════════════════════════════════

# How strongly each position reacts to each fault mechanism (0 → 1).
FAULT_SENSITIVITY = {
    "bearing_fault_de": {
        "drive_end": 1.0, "non_drive_end": 0.35, "gearbox": 0.10, "base": 0.15,
    },
    "misalignment": {
        "drive_end": 1.0, "non_drive_end": 0.90, "gearbox": 0.20, "base": 0.30,
    },
    "gear_fault": {
        "drive_end": 0.25, "non_drive_end": 0.15, "gearbox": 1.0, "base": 0.20,
    },
    "looseness": {
        "drive_end": 0.50, "non_drive_end": 0.45, "gearbox": 0.35, "base": 1.0,
    },
}

CLEAR_FAULT_TYPES = list(FAULT_SENSITIVITY.keys())


def _inject_clear_fault(
    signals: dict[str, np.ndarray],
    position: str,
    fault_type: str,
    rng: np.random.Generator,
) -> None:
    """Mutate signals in-place with a clear fault scaled by position sensitivity."""
    sensitivity = FAULT_SENSITIVITY[fault_type][position]
    if sensitivity < 0.05:
        return

    n = len(signals["accel_x"])
    magnitude = rng.uniform(3.0, 6.0) * sensitivity

    if fault_type == "bearing_fault_de":
        ramp = np.linspace(0, 1, n) ** 2 * magnitude
        signals["accel_x"] += ramp
        signals["vel_x"] += ramp * 0.45
        signals["accel_y"] += ramp * rng.uniform(0.5, 0.8)
        signals["vel_y"] += ramp * rng.uniform(0.2, 0.4)

    elif fault_type == "misalignment":
        start = rng.integers(int(n * 0.3), int(n * 0.5))
        signals["accel_x"][start:] += magnitude
        signals["vel_x"][start:] += magnitude * 0.4
        signals["accel_y"][start:] += magnitude * rng.uniform(0.7, 1.0)
        signals["vel_y"][start:] += magnitude * rng.uniform(0.3, 0.6)

    elif fault_type == "gear_fault":
        period = rng.integers(200, 500)
        for idx in range(0, n, period):
            w = min(12, n - idx)
            burst = rng.uniform(0.6, 1.0) * magnitude
            signals["accel_x"][idx:idx + w] += burst
            signals["vel_x"][idx:idx + w] += burst * 0.5
            signals["accel_y"][idx:idx + w] += burst * rng.uniform(0.4, 0.8)

    elif fault_type == "looseness":
        extra_std = magnitude * 0.6
        signals["accel_x"] += rng.normal(0, extra_std, n)
        signals["accel_y"] += rng.normal(0, extra_std, n)
        signals["vel_x"] += rng.normal(0, extra_std * 0.4, n)
        signals["vel_y"] += rng.normal(0, extra_std * 0.4, n)
        signals["accel_x"] += magnitude * 0.3
        signals["accel_y"] += magnitude * 0.25


# ═══════════════════════════════════════════════════════════════════════════
# Ambiguous fault patterns — subtler, cross-sensor correlation is weaker
# ═══════════════════════════════════════════════════════════════════════════

AMBIGUOUS_TYPES = [
    "one_sensor_anomaly",
    "thermal_shift",
    "conflicting_trend",
    "intermittent_spikes",
    "early_stage_bearing",
    "noisy_base",
    "late_micro_trend",
]


def _inject_ambiguous(
    all_signals: dict[str, dict[str, np.ndarray]],
    kind: str,
    rng: np.random.Generator,
) -> None:
    """Mutate signals across all 4 positions in-place with subtle patterns."""
    n = len(next(iter(all_signals.values()))["accel_x"])

    if kind == "one_sensor_anomaly":
        # Only one position shows a mild anomaly — rest are clean
        pos = rng.choice(POSITIONS)
        sig = all_signals[pos]
        start = rng.integers(int(n * 0.5), int(n * 0.7))
        bump = rng.uniform(1.0, 2.2)
        sig["accel_x"][start:] += bump
        sig["vel_x"][start:] += bump * 0.35

    elif kind == "thermal_shift":
        # All sensors shift up slightly (environmental change)
        offset = rng.uniform(0.5, 1.2)
        for sig in all_signals.values():
            sig["accel_x"] += offset * rng.uniform(0.7, 1.0)
            sig["accel_y"] += offset * rng.uniform(0.6, 0.9)
            sig["vel_x"] += offset * 0.3 * rng.uniform(0.7, 1.0)
            sig["vel_y"] += offset * 0.3 * rng.uniform(0.6, 0.9)

    elif kind == "conflicting_trend":
        # Drive end trending up, non-drive end trending down
        ramp_up = np.linspace(0, rng.uniform(0.8, 1.5), n)
        ramp_dn = np.linspace(0, rng.uniform(0.4, 0.9), n)
        de = all_signals["drive_end"]
        nde = all_signals["non_drive_end"]
        de["accel_x"] += ramp_up
        de["vel_x"] += ramp_up * 0.35
        nde["accel_x"] -= ramp_dn
        nde["vel_x"] -= ramp_dn * 0.3

    elif kind == "intermittent_spikes":
        # Random mild spikes across multiple sensors
        for sig in all_signals.values():
            n_sp = rng.integers(5, 15)
            idxs = rng.choice(n, size=n_sp, replace=False)
            for p in idxs:
                mag = rng.uniform(1.5, 3.5)
                sig["accel_x"][p] += mag
                if rng.random() < 0.4:
                    sig["accel_y"][p] += mag * rng.uniform(0.3, 0.6)

    elif kind == "early_stage_bearing":
        # Very subtle ramp only on drive_end — barely above noise
        de = all_signals["drive_end"]
        gain = rng.uniform(0.5, 1.0)
        ramp = np.linspace(0, gain, n) ** 2
        de["accel_x"] += ramp
        de["vel_x"] += ramp * 0.3
        de["accel_y"] += ramp * rng.uniform(0.2, 0.5)

    elif kind == "noisy_base":
        # Base sensor is noisier than expected — others are fine
        sig = all_signals["base"]
        extra = rng.uniform(0.5, 1.0)
        sig["accel_x"] += rng.normal(0, extra, n)
        sig["accel_y"] += rng.normal(0, extra, n)
        sig["vel_x"] += rng.normal(0, extra * 0.35, n)
        sig["vel_y"] += rng.normal(0, extra * 0.35, n)

    elif kind == "late_micro_trend":
        # Flat for ~80 %, then a small uptick on drive_end + gearbox
        knee = int(n * rng.uniform(0.75, 0.85))
        rise = rng.uniform(0.6, 1.4)
        ramp = np.linspace(0, rise, n - knee)
        for pos in ("drive_end", "gearbox"):
            sig = all_signals[pos]
            scale = 1.0 if pos == "drive_end" else rng.uniform(0.4, 0.7)
            sig["accel_x"][knee:] += ramp * scale
            sig["vel_x"][knee:] += ramp * scale * 0.4


# ═══════════════════════════════════════════════════════════════════════════
# Main generation
# ═══════════════════════════════════════════════════════════════════════════

def seed_database(
    n_healthy: int = 5,
    n_faulty: int = 3,
    n_ambiguous: int = 5,
) -> None:
    n_machines = n_healthy + n_faulty + n_ambiguous
    os.makedirs(str(BASE_DIR), exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"Generating {n_machines} machines × 4 sensors = {n_machines * 4} sensors")
    print(f"  {n_healthy} healthy  ·  {n_faulty} faulty  ·  {n_ambiguous} ambiguous\n")

    end_time = datetime.now()
    start_time = end_time - timedelta(days=90)
    time_array = pd.date_range(start=start_time, end=end_time, freq="10min")
    n_samples = len(time_array)

    machine_type_names = list(MACHINE_PROFILES.keys())
    sensor_counter = 0
    all_frames: list[pd.DataFrame] = []

    for m_idx in range(n_machines):
        machine_id = f"MACH-{str(m_idx + 1).zfill(2)}"
        machine_type = machine_type_names[m_idx % len(machine_type_names)]
        profile = MACHINE_PROFILES[machine_type]

        # Determine health category
        if m_idx < n_healthy:
            category = "healthy"
        elif m_idx < n_healthy + n_faulty:
            category = "faulty"
        else:
            category = "ambiguous"

        # Generate baseline signals for all 4 positions
        machine_signals: dict[str, dict[str, np.ndarray]] = {}
        for pos in POSITIONS:
            machine_signals[pos] = _baseline(profile, pos, n_samples, rng)

        # Apply fault patterns
        fault_label = ""
        if category == "faulty":
            fault_type = rng.choice(CLEAR_FAULT_TYPES)
            fault_label = fault_type
            for pos in POSITIONS:
                _inject_clear_fault(machine_signals[pos], pos, fault_type, rng)

        elif category == "ambiguous":
            amb_idx = m_idx - (n_healthy + n_faulty)
            amb_type = AMBIGUOUS_TYPES[amb_idx % len(AMBIGUOUS_TYPES)]
            fault_label = amb_type
            _inject_ambiguous(machine_signals, amb_type, rng)

        # Convert to DataFrames
        for pos in POSITIONS:
            sensor_counter += 1
            sensor_id = f"SENS-{str(sensor_counter).zfill(3)}"
            sig = machine_signals[pos]

            for axis in AXES:
                accel_key = f"accel_{axis}"
                vel_key = f"vel_{axis}"
                accel = np.clip(sig[accel_key], 0, None)
                vel = np.clip(sig[vel_key], 0, None)

                all_frames.append(
                    pd.DataFrame({
                        "machine_id": machine_id,
                        "sensor_id": sensor_id,
                        "sensor_position": pos,
                        "sensor_axis": axis,
                        "time": time_array,
                        "accel_peak": accel,
                        "vel_rms": vel,
                        "machine_type": machine_type,
                    })
                )

            pos_tag = f"{sensor_id} ({pos})"
            if category == "healthy":
                print(f"  {machine_id} {pos_tag:>30}  — healthy")
            else:
                print(f"  {machine_id} {pos_tag:>30}  — {category} ({fault_label})")

        print()  # blank line between machines

    final_df = pd.concat(all_frames, ignore_index=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DROP TABLE IF EXISTS sensor_data")
        conn.execute(
            """
            CREATE TABLE sensor_data (
                machine_id       TEXT,
                sensor_id        TEXT,
                sensor_position  TEXT,
                sensor_axis      TEXT,
                time             TIMESTAMP,
                accel_peak       REAL,
                vel_rms          REAL,
                machine_type     TEXT
            )
            """
        )
        final_df.to_sql("sensor_data", conn, if_exists="append", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sensor_id  ON sensor_data (sensor_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_machine_id ON sensor_data (machine_id)")
        conn.commit()

    print(f"Done — {len(final_df):,} rows across {n_machines} machines written to {DB_PATH}")


if __name__ == "__main__":
    seed_database(n_healthy=5, n_faulty=3, n_ambiguous=5)
