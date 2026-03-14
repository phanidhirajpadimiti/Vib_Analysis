"""
Quick explorer for the synthetic vibration database.
Run:  python3 query_data.py
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from config import DB_PATH, POSITION_ORDER


def overview(conn: sqlite3.Connection) -> None:
    """Print high-level stats about the database."""
    total = pd.read_sql("SELECT COUNT(*) AS n FROM sensor_data", conn).iloc[0, 0]
    machines = pd.read_sql(
        """
        SELECT machine_id, machine_type,
               COUNT(DISTINCT sensor_id) AS sensors,
               COUNT(*) AS rows
        FROM sensor_data
        GROUP BY machine_id
        ORDER BY machine_id
        """,
        conn,
    )
    sensors = pd.read_sql(
        """
        SELECT machine_id, sensor_id, sensor_position, machine_type,
               GROUP_CONCAT(DISTINCT sensor_axis) AS axes,
               COUNT(*) AS rows
        FROM sensor_data
        GROUP BY sensor_id
        ORDER BY machine_id, sensor_position
        """,
        conn,
    )
    print("=" * 80)
    print(f"  Database : {DB_PATH}")
    print(f"  Total rows : {total:,}")
    print(f"  Machines   : {len(machines)}")
    print(f"  Sensors    : {len(sensors)}")
    print("=" * 80)
    print("\nMachines:")
    print(machines.to_string(index=False))
    print("\nSensors:")
    print(sensors.to_string(index=False))
    print()


def sample_rows(conn: sqlite3.Connection, sensor_id: str = "SENS-001", n: int = 10) -> None:
    """Show a few sample rows for a given sensor."""
    df = pd.read_sql(
        "SELECT * FROM sensor_data WHERE sensor_id = ? ORDER BY time LIMIT ?",
        conn,
        params=(sensor_id, n),
    )
    print(f"Sample rows for {sensor_id}:")
    print(df.to_string(index=False))
    print()


def statistics(conn: sqlite3.Connection) -> None:
    """Print per-sensor descriptive statistics for accel_peak and vel_rms."""
    df = pd.read_sql(
        """
        SELECT machine_id, sensor_id, sensor_position, sensor_axis,
               ROUND(AVG(accel_peak), 3)  AS accel_mean,
               ROUND(MIN(accel_peak), 3)  AS accel_min,
               ROUND(MAX(accel_peak), 3)  AS accel_max,
               ROUND(AVG(vel_rms), 3)     AS vel_mean,
               ROUND(MIN(vel_rms), 3)     AS vel_min,
               ROUND(MAX(vel_rms), 3)     AS vel_max
        FROM sensor_data
        GROUP BY sensor_id, sensor_axis
        ORDER BY machine_id, sensor_position, sensor_axis
        """,
        conn,
    )
    print("Per-sensor / per-axis statistics:")
    print(df.to_string(index=False))
    print()


def plot_machine(conn: sqlite3.Connection, machine_id: str = "MACH-01") -> None:
    """Generate a 4×4 grid: rows = sensor positions, cols = features × axes."""
    df = pd.read_sql(
        """
        SELECT sensor_id, sensor_position, sensor_axis, time, accel_peak, vel_rms
        FROM sensor_data
        WHERE machine_id = ?
        """,
        conn,
        params=(machine_id,),
        parse_dates=["time"],
    )

    if df.empty:
        print(f"No data for {machine_id}")
        return

    positions = POSITION_ORDER
    combos = [("x", "accel_peak"), ("y", "accel_peak"), ("x", "vel_rms"), ("y", "vel_rms")]

    fig, axes = plt.subplots(4, 4, figsize=(22, 16), sharex=True)
    fig.suptitle(f"Machine {machine_id} — All Sensors", fontsize=16, fontweight="bold")

    for row, pos in enumerate(positions):
        pos_df = df[df["sensor_position"] == pos]
        sid = pos_df["sensor_id"].iloc[0] if not pos_df.empty else "?"
        for col, (axis_label, feature) in enumerate(combos):
            ax = axes[row][col]
            subset = pos_df[pos_df["sensor_axis"] == axis_label].sort_values("time")
            ax.scatter(subset["time"], subset[feature], s=0.5, alpha=0.4, color="black")
            if row == 0:
                ax.set_title(f"{axis_label.upper()} — {feature}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{pos}\n({sid})", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)

    for ax in axes[-1]:
        ax.set_xlabel("Time", fontsize=8)

    fig.autofmt_xdate()
    fig.tight_layout()

    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{machine_id}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Plot saved → {out_path}")


def main() -> None:
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}. Run generate_data.py first.")
        return

    with sqlite3.connect(DB_PATH) as conn:
        overview(conn)
        sample_rows(conn, "SENS-001")
        statistics(conn)

        plot_machine(conn, "MACH-01")  # healthy
        plot_machine(conn, "MACH-06")  # faulty
        plot_machine(conn, "MACH-09")  # ambiguous


if __name__ == "__main__":
    main()
