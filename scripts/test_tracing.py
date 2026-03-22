#!/usr/bin/env python3
"""
Quick verification that LangSmith tracing is wired up correctly.

Usage:
    python scripts/test_tracing.py

Prerequisites:
    - LANGCHAIN_TRACING_V2=true in .env or environment
    - LANGCHAIN_API_KEY set to a valid LangSmith key
    - LANGCHAIN_PROJECT set (defaults to "viblabel")
"""

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()


def check_env():
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "").lower()
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    project = os.getenv("LANGCHAIN_PROJECT", "viblabel")

    if tracing != "true":
        print("LANGCHAIN_TRACING_V2 is not set to 'true'. Tracing is disabled.")
        print("Set it in your .env file to enable LangSmith tracing.")
        sys.exit(1)

    if not api_key or api_key.startswith("your-"):
        print("LANGCHAIN_API_KEY is not configured.")
        print("Get your key at https://smith.langchain.com and add it to .env")
        sys.exit(1)

    print(f"Tracing enabled — project: {project}")
    return project


def run_single_trace():
    from agent import run_machine_analysis
    from db import get_connection

    with get_connection() as conn:
        row = conn.execute(
            "SELECT DISTINCT machine_id FROM sensor_data LIMIT 1"
        ).fetchone()

    if not row:
        print("No machines in database. Run generate_data.py first.")
        sys.exit(1)

    machine_id = row[0]
    print(f"Running agent labeling for {machine_id} ...")
    result = run_machine_analysis(
        machine_id,
        tags=[machine_id, "tracing-test"],
        metadata={"machine_id": machine_id, "source": "test_tracing"},
    )
    print(f"Result: machine_label={result.get('machine_label')}, "
          f"confidence={result.get('machine_confidence')}")
    return machine_id


def verify_trace(project: str):
    try:
        from langsmith import Client

        client = Client()
        time.sleep(3)

        runs = list(client.list_runs(project_name=project, limit=5))
        if runs:
            print(f"\nFound {len(runs)} recent trace(s) in project '{project}':")
            for run in runs[:3]:
                print(f"  - {run.name} | status={run.status} | "
                      f"latency={run.total_tokens}tok")
            print("\nLangSmith tracing is working correctly!")
        else:
            print(f"\nNo traces found in project '{project}' yet.")
            print("Traces may take a few seconds to appear. "
                  "Check https://smith.langchain.com")
    except ImportError:
        print("\nlangsmith package not installed — skipping trace verification.")
        print("Install with: pip install langsmith")
    except Exception as e:
        print(f"\nCould not verify traces: {e}")
        print("Check your LANGCHAIN_API_KEY and visit https://smith.langchain.com")


if __name__ == "__main__":
    project = check_env()
    run_single_trace()
    verify_trace(project)
