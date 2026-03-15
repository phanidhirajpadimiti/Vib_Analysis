"""
Central configuration — single source of truth for all shared constants.
"""

from pathlib import Path

# ── Paths ──
BASE_DIR = Path("synthetic_data")
DB_PATH = str(BASE_DIR / "vibration_data.db")
STATIC_DIR = Path(__file__).parent / "static"
ENV_PATH = Path(__file__).parent / ".env"

# ── Gemini ──
GEMINI_MODEL = "gemini-2.5-flash"

# ── Retry / rate-limit ──
MAX_RETRIES = 4
INITIAL_BACKOFF_SECONDS = 15

# ── Machine profiles (used by generate_data.py) ──
MACHINE_PROFILES = {
    "pump":  {"accel_mean": 6.0, "accel_std": 0.6, "vel_mean": 2.5, "vel_std": 0.25},
    "fan":   {"accel_mean": 4.0, "accel_std": 0.4, "vel_mean": 1.8, "vel_std": 0.18},
    "motor": {"accel_mean": 5.5, "accel_std": 0.5, "vel_mean": 2.2, "vel_std": 0.22},
}

# Each machine has 4 sensors at these positions.
# The multiplier scales the baseline amplitude relative to the machine profile.
SENSOR_POSITIONS = {
    "drive_end":     {"accel_mult": 1.00, "vel_mult": 1.00},
    "non_drive_end": {"accel_mult": 0.85, "vel_mult": 0.85},
    "gearbox":       {"accel_mult": 0.90, "vel_mult": 0.90},
    "base":          {"accel_mult": 0.65, "vel_mult": 0.65},
}

POSITION_ORDER = list(SENSOR_POSITIONS.keys())

# ── Signal / analysis constants ──
AXES = ["x", "y"]
FEATURES = ["accel_peak", "vel_rms"]
SEED = 42
SAMPLES_PER_DAY = 144            # 10-min intervals
RECENT_WINDOW_DAYS = 30
MAX_SENSORS_PER_REQUEST = 50

FEATURE_LABELS = {
    "accel_peak": "Acceleration Peak (g)",
    "vel_rms": "Velocity RMS (mm/s)",
}

POSITION_LABELS = {
    "drive_end": "Drive End",
    "non_drive_end": "Non-Drive End",
    "gearbox": "Gearbox",
    "base": "Base",
}

# ── ISO 10816 vibration severity ──
# Zone boundaries are velocity RMS in mm/s.
# A = Good, B = Acceptable, C = Tolerable (plan maintenance), D = Dangerous.
ISO_MACHINE_CLASSES = {
    "pump":  "II",
    "fan":   "I",
    "motor": "III",
}

ISO_THRESHOLDS = {
    "I":   {"A_B": 0.71, "B_C": 1.80, "C_D": 4.50},
    "II":  {"A_B": 1.12, "B_C": 2.80, "C_D": 7.10},
    "III": {"A_B": 1.80, "B_C": 4.50, "C_D": 11.20},
    "IV":  {"A_B": 2.80, "B_C": 7.10, "C_D": 18.00},
}

LABEL_TAXONOMY = ["healthy", "unhealthy", "monitor"]
