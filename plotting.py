"""
Shared scatter-plot renderer — one implementation, multiple output formats.
"""

import base64
import io
from typing import Tuple

import matplotlib
import matplotlib.dates as mdates
import matplotlib.figure as mfig

matplotlib.use("Agg")


def render_scatter(
    time_data,
    feature_data,
    title: str,
    ylabel: str,
    *,
    figsize: Tuple[float, float] = (7, 4),
    dpi: int = 110,
    point_size: float = 1.0,
    alpha: float = 0.45,
) -> bytes:
    """Render a time-series scatter plot and return raw PNG bytes."""
    fig = mfig.Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    ax.scatter(time_data, feature_data, s=point_size, color="#1a1a2e", alpha=alpha)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Time", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    del fig
    return buf.getvalue()


def to_base64(png_bytes: bytes) -> str:
    """Encode PNG bytes as a base64 string (for multimodal LLM messages)."""
    return base64.b64encode(png_bytes).decode()
