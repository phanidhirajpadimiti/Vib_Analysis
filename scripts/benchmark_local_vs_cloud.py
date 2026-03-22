#!/usr/bin/env python3
"""
Benchmark: Local SLM (Ollama) vs Cloud (Gemini) labeling comparison.

Runs the same machines through both providers and prints a side-by-side
comparison of latency, labels, and confidence levels.

Prerequisites:
    - Gemini: GEMINI_API_KEY set in .env
    - Ollama: `ollama serve` running with the configured model pulled

Usage:
    python scripts/benchmark_local_vs_cloud.py
    python scripts/benchmark_local_vs_cloud.py --machines MACH-01 MACH-05
"""

import argparse
import importlib
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import get_connection


def _run_with_provider(machine_id: str, use_local: bool) -> dict:
    """Run labeling for one machine with the specified provider.

    Reloads agent + config modules to pick up the changed flag, and resets
    the cached LLM so the correct provider is initialized.
    """
    os.environ["USE_LOCAL_SLM"] = "true" if use_local else "false"

    import agent
    import config

    importlib.reload(config)
    importlib.reload(agent)
    agent._llm = None

    start = time.perf_counter()
    result = agent.run_machine_analysis(
        machine_id,
        tags=[machine_id, "benchmark"],
        metadata={"machine_id": machine_id, "source": "benchmark"},
    )
    elapsed = time.perf_counter() - start
    return {"result": result, "elapsed": round(elapsed, 2)}


def _print_comparison(machine_id: str, cloud: dict, local: dict):
    w = 40
    print(f"\n{'=' * (w * 2 + 3)}")
    print(f"  Machine: {machine_id}")
    print(f"{'=' * (w * 2 + 3)}")
    print(f"  {'Gemini (cloud)':<{w}} {'Ollama (local)':<{w}}")
    print(f"  {'-' * w} {'-' * w}")

    cr, lr = cloud["result"], local["result"]

    rows = [
        ("Latency", f"{cloud['elapsed']}s", f"{local['elapsed']}s"),
        ("Machine label", cr.get("machine_label", "?"), lr.get("machine_label", "?")),
        ("Confidence", cr.get("machine_confidence", "?"), lr.get("machine_confidence", "?")),
    ]

    cloud_sensors = {s["sensor_id"]: s for s in cr.get("sensors", [])}
    local_sensors = {s["sensor_id"]: s for s in lr.get("sensors", [])}
    all_sids = sorted(set(list(cloud_sensors.keys()) + list(local_sensors.keys())))

    for sid in all_sids:
        cs = cloud_sensors.get(sid, {})
        ls = local_sensors.get(sid, {})
        c_label = f"{cs.get('label', '?')} ({cs.get('confidence', '?')})"
        l_label = f"{ls.get('label', '?')} ({ls.get('confidence', '?')})"
        rows.append((f"  {sid}", c_label, l_label))

    for label, c_val, l_val in rows:
        match = " " if c_val.split()[0] == l_val.split()[0] else "*"
        print(f"{match} {label:<{w}} {c_val:<{w}} {l_val:<{w}}")

    print(f"\n  Cloud rationale: {cr.get('machine_rationale', 'N/A')}")
    print(f"  Local rationale: {lr.get('machine_rationale', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark local vs cloud labeling")
    parser.add_argument(
        "--machines", nargs="+", default=None,
        help="Machine IDs to benchmark (default: first 2 in DB)",
    )
    args = parser.parse_args()

    if args.machines:
        machine_ids = args.machines
    else:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT machine_id FROM sensor_data LIMIT 2"
            ).fetchall()
        machine_ids = [r[0] for r in rows]

    if not machine_ids:
        print("No machines found in database. Run generate_data.py first.")
        sys.exit(1)

    print(f"Benchmarking {len(machine_ids)} machine(s): {', '.join(machine_ids)}")
    print(f"Cloud model: Gemini | Local model: {os.getenv('OLLAMA_MODEL', 'llama3.2-vision:11b')}")

    for mid in machine_ids:
        print(f"\n--- Running Gemini for {mid} ---")
        try:
            cloud = _run_with_provider(mid, use_local=False)
        except Exception as e:
            print(f"  Gemini failed: {e}")
            cloud = {"result": {"error": str(e)}, "elapsed": 0}

        print(f"--- Running Ollama for {mid} ---")
        try:
            local = _run_with_provider(mid, use_local=True)
        except Exception as e:
            print(f"  Ollama failed: {e}")
            local = {"result": {"error": str(e)}, "elapsed": 0}

        _print_comparison(mid, cloud, local)

    print(f"\n{'=' * 83}")
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
