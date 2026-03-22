# VibLabel — Agentic Data Labeling for Vibration Analysis

An LLM-powered labeling tool that uses a **LangGraph agent** backed by Google Gemini to generate quality labels for industrial vibration sensor data. The labeled data is designed for training or validating machine learning models for predictive maintenance.

Instead of replacing ML inference, this tool **produces the training and validation data** that ML models need — automating what traditionally requires expensive domain experts annotating sensor readings by hand.

## Architecture

### System Overview

```mermaid
flowchart TB
    subgraph FRONTEND["<b>Labeling Dashboard</b> &nbsp;·&nbsp; static/index.html"]
        direction LR
        F1["Machine Selector"]
        F2["Sensor Plot Grid"]
        F3["Label + Review UI"]
        F4["Batch Label &amp; Export"]
    end

    subgraph API["<b>FastAPI Backend</b> &nbsp;·&nbsp; api.py"]
        direction LR
        R1["POST /label/machine/{id}"]
        R2["POST /label/batch"]
        R3["PUT /label/.../review"]
        R4["GET /labels/export"]
    end

    subgraph AGENT["<b>LangGraph Agent</b> &nbsp;·&nbsp; agent.py"]
        direction TB
        subgraph GRAPH[" "]
            direction LR
            P["Prepare<br/><i>load data + plots</i>"]
            LLM["Agent Node<br/><i>Gemini 2.5 Flash</i>"]
            TN["Tool Node<br/><i>auto-execute</i>"]
            FIN["Finalize<br/><i>parse labels</i>"]
        end
    end

    subgraph TOOLS["<b>Agent Tools</b>"]
        direction LR
        T1["get_iso_assessment<br/><i>ISO 10816 zone + trend projection</i>"]
        T2["get_sensor_stats<br/><i>mean · std · trend · kurtosis</i>"]
        T3["compare_recent_vs_historical<br/><i>30-day vs baseline shift</i>"]
        T4["get_cross_sensor_comparison<br/><i>4 positions side-by-side</i>"]
    end

    subgraph STORE["<b>Label Storage</b> &nbsp;·&nbsp; SQLite"]
        direction LR
        DB[("machine_labels<br/>sensor_labels<br/>+ human overrides")]
    end

    subgraph EXPORT["<b>ML Pipeline</b>"]
        CSV["vibration_labels.csv<br/><i>final_label = human override ?? agent label</i>"]
    end

    subgraph OBS["<b>Observability</b> &nbsp;·&nbsp; LangSmith"]
        direction LR
        LS["Traces · Latency · Token Usage · Cost"]
    end

    FRONTEND -- "HTTP" --> API
    R1 & R2 -- "invoke" --> AGENT
    AGENT -. "traces" .-> OBS
    P --> LLM
    LLM <-- "tool calls" --> TN
    LLM --> FIN
    TN --> TOOLS
    TOOLS -- "SQL" --> STORE
    FIN -- "save labels" --> STORE
    R3 -- "human review" --> STORE
    R4 -- "export" --> CSV
    STORE --> CSV

    style FRONTEND fill:#e8f4fd,stroke:#2196f3,stroke-width:2px,color:#0d47a1
    style API fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#e65100
    style AGENT fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px,color:#4a148c
    style GRAPH fill:#f3e5f5,stroke:none
    style TOOLS fill:#e8f5e9,stroke:#4caf50,stroke-width:2px,color:#1b5e20
    style STORE fill:#fce4ec,stroke:#e91e63,stroke-width:2px,color:#880e4f
    style EXPORT fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534
    style DB fill:#fce4ec,stroke:#e91e63,color:#880e4f
    style CSV fill:#f0fdf4,stroke:#16a34a,color:#166534

    style P fill:#ce93d8,stroke:#7b1fa2,color:#fff
    style LLM fill:#ba68c8,stroke:#7b1fa2,color:#fff
    style TN fill:#ce93d8,stroke:#7b1fa2,color:#fff
    style FIN fill:#ce93d8,stroke:#7b1fa2,color:#fff

    style T1 fill:#a5d6a7,stroke:#388e3c,color:#1b5e20
    style T2 fill:#a5d6a7,stroke:#388e3c,color:#1b5e20
    style T3 fill:#a5d6a7,stroke:#388e3c,color:#1b5e20
    style T4 fill:#a5d6a7,stroke:#388e3c,color:#1b5e20

    style OBS fill:#fff9c4,stroke:#f9a825,stroke-width:2px,color:#f57f17
    style LS fill:#fff9c4,stroke:#f9a825,color:#f57f17
```

### Labeling Flow

```mermaid
flowchart LR
    START(("START")) --> PREP

    PREP["<b>prepare</b><br/>Load machine data<br/>Generate 8 overview plots<br/>Include ISO 10816 context"]
    PREP --> AGT

    AGT["<b>agent</b><br/>Gemini 2.5 Flash<br/>Reads plots + tool results<br/>Applies labeling guidelines"]

    AGT -->|"tool_calls"| TOOL["<b>tools</b><br/>get_iso_assessment<br/>get_sensor_stats<br/>compare_recent_vs_historical<br/>get_cross_sensor_comparison"]
    TOOL -->|"results"| AGT

    AGT -->|"done"| FIN["<b>finalize</b><br/>Parse structured labels<br/>Assign confidence levels<br/>Save to label store"]
    FIN --> REVIEW["<b>human review</b><br/>Accept / Override<br/>per-sensor labels"]
    REVIEW --> OUT(("EXPORT<br/>CSV"))

    style START fill:#4caf50,stroke:#2e7d32,color:#fff
    style OUT fill:#16a34a,stroke:#166534,color:#fff
    style PREP fill:#90caf9,stroke:#1565c0,color:#0d47a1
    style AGT fill:#ba68c8,stroke:#7b1fa2,color:#fff
    style TOOL fill:#a5d6a7,stroke:#388e3c,color:#1b5e20
    style FIN fill:#ffcc80,stroke:#ef6c00,color:#e65100
    style REVIEW fill:#e8f4fd,stroke:#2196f3,color:#0d47a1
```

## Why This Approach

| Challenge | How VibLabel solves it |
|-----------|----------------------|
| Labeled industrial data is expensive | LLM agent automates the domain-expert annotation workflow |
| Labels need to be grounded, not subjective | ISO 10816 zones provide objective severity criteria |
| Edge cases need human judgment | Human-in-the-loop review for low-confidence labels |
| Labels need to be explainable | Every label includes rationale, tool evidence, and ISO zone reference |
| ML models need structured training data | One-click CSV export with `final_label` column |

## Key Features

- **Agentic labeling** — the LLM uses tools to gather quantitative evidence before assigning labels, not just pattern-matching on plots
- **ISO 10816 grounded** — labels reference international vibration severity zones (A/B/C/D) with trend projections
- **Confidence scoring** — each label has a confidence level (high/medium/low) that prioritizes human review effort
- **Human-in-the-loop** — reviewers can accept or override agent labels per-sensor, with overrides taking precedence in exports
- **Batch processing** — label all unlabeled machines in one click
- **ML-ready export** — CSV with `final_label` column (human override ?? agent label) for direct pipeline ingestion
- **LangSmith observability** — opt-in tracing of every agent run, tool call, and LLM invocation with latency, token usage, and cost tracking

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | LangGraph (state graph, tool nodes, conditional routing) |
| LLM | Google Gemini 2.5 Flash via `langchain-google-genai` |
| Backend | FastAPI + Uvicorn |
| Database | SQLite (sensor data + label storage) |
| Observability | LangSmith (opt-in tracing, token/cost tracking) |
| Plotting | Matplotlib (thread-safe, Figure API) |
| Frontend | Vanilla HTML/CSS/JS |

## Project Structure

```
├── config.py            # Central configuration (ISO thresholds, model, constants)
├── models.py            # Pydantic schemas (labeling, review, export)
├── db.py                # DB helpers + label storage (save, review, export)
├── plotting.py          # Shared scatter-plot renderer
│
├── generate_data.py     # Synthetic data generator (13 machines × 4 sensors)
├── query_data.py        # CLI database explorer
│
├── agent.py             # LangGraph agent (4 tools including ISO assessment)
├── api.py               # FastAPI endpoints (label, review, export, data, plots)
│
├── static/index.html    # Labeling dashboard with human review
├── tests/               # pytest suite (tools + API endpoints)
├── scripts/             # Utility scripts (tracing verification)
│
├── .env.example         # Template for API keys (Gemini + LangSmith)
├── requirements.txt     # Dependencies
└── README.md
```

## Setup

```bash
# 1. Clone and enter the directory
git clone <repo-url>
cd Vib_Analysis

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — add your Gemini key (https://aistudio.google.com/app/apikey)
# Optionally add your LangSmith key for observability (https://smith.langchain.com)

# 5. Generate synthetic data
python generate_data.py

# 6. Start the server
uvicorn api:app --host 0.0.0.0 --port 8000

# 7. Open the dashboard
open http://localhost:8000
```

## Running Tests

```bash
pytest tests/ -v
```

## How the Labeling Works

1. **Select a machine** (or click "Label All Unlabeled" for batch mode)
2. **Agent labels** — the LLM inspects 8 sensor plots, calls ISO assessment + statistical tools, and assigns per-sensor labels with confidence
3. **Review** — low-confidence labels are flagged; reviewers can accept or override any label
4. **Export** — download `vibration_labels.csv` where `final_label` = human override when present, otherwise agent label

### Labeling Guidelines (embedded in agent prompt)

| Condition | Label | Typical Confidence |
|-----------|-------|--------------------|
| ISO Zone A or stable B, no trend | healthy | high |
| Zone B with upward trend, borderline B/C | monitor | medium |
| Zone C or D, strong trend, clear fault | unhealthy | high |
| Conflicting cross-sensor signals | varies | low (flagged for review) |

## Observability (LangSmith)

Tracing is **opt-in**. When `LANGCHAIN_TRACING_V2=true` is set in `.env`, every agent invocation is traced to [LangSmith](https://smith.langchain.com) with:

- Full graph execution path (prepare → agent → tools → finalize)
- Per-tool inputs/outputs as child spans
- Gemini LLM token usage and cost
- `run_name`, `tags`, and `metadata` for filtering (single vs batch, by machine ID)

```bash
# Verify tracing is working
python scripts/test_tracing.py
```

When tracing is off (no env vars set), there is zero overhead — the agent runs identically.

## Data Model

Each machine has **4 sensors** at standardized positions:

| Position | Accel Multiplier | ISO Relevance |
|----------|-----------------|---------------|
| Drive End | 1.00× | Bearing faults appear strongest here |
| Non-Drive End | 0.85× | Misalignment shows on both ends |
| Gearbox | 0.90× | Gear-mesh faults are localized here |
| Base | 0.65× | Structural looseness elevates base noise |

ISO 10816 machine class assignments:

| Machine Type | ISO Class | Zone B/C Boundary (vel RMS, mm/s) |
|-------------|-----------|-----------------------------------|
| Pump | Class II | 2.80 |
| Fan | Class I | 1.80 |
| Motor | Class III | 4.50 |

## License

MIT
