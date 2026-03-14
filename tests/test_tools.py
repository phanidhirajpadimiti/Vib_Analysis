"""
Unit tests for agent tool functions — no LLM calls, just stat computation.
"""

import json

from agent import (
    _compute_feature_stats,
    get_cross_sensor_comparison,
    get_sensor_stats,
    compare_recent_vs_historical,
)
import numpy as np


class TestComputeFeatureStats:
    def test_returns_expected_keys(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_feature_stats(vals)
        assert set(result.keys()) == {"mean", "std", "min", "max", "trend_slope_per_day", "kurtosis"}

    def test_mean_is_correct(self):
        vals = np.array([10.0, 10.0, 10.0])
        result = _compute_feature_stats(vals)
        assert result["mean"] == 10.0

    def test_trend_positive_for_increasing(self):
        vals = np.linspace(0, 10, 100)
        result = _compute_feature_stats(vals)
        assert result["trend_slope_per_day"] > 0

    def test_trend_near_zero_for_flat(self):
        vals = np.full(100, 5.0)
        result = _compute_feature_stats(vals)
        assert abs(result["trend_slope_per_day"]) < 0.01


class TestGetSensorStats:
    def test_healthy_sensor(self):
        raw = get_sensor_stats.invoke("SENS-T001")
        result = json.loads(raw)
        assert result["sensor_id"] == "SENS-T001"
        assert result["position"] == "drive_end"
        assert "x_accel_peak" in result

    def test_faulty_sensor_has_positive_trend(self):
        raw = get_sensor_stats.invoke("SENS-T005")
        result = json.loads(raw)
        assert result["position"] == "drive_end"
        assert result["x_accel_peak"]["trend_slope_per_day"] > 0

    def test_unknown_sensor_returns_error(self):
        raw = get_sensor_stats.invoke("SENS-NOPE")
        result = json.loads(raw)
        assert "error" in result


class TestCompareRecentVsHistorical:
    def test_returns_change_percent(self):
        raw = compare_recent_vs_historical.invoke("SENS-T005")
        result = json.loads(raw)
        assert "x_accel_peak" in result
        assert "change_percent" in result["x_accel_peak"]

    def test_faulty_sensor_shows_increase(self):
        raw = compare_recent_vs_historical.invoke("SENS-T005")
        result = json.loads(raw)
        assert result["x_accel_peak"]["change_percent"] > 0

    def test_unknown_sensor_returns_error(self):
        raw = compare_recent_vs_historical.invoke("SENS-NOPE")
        result = json.loads(raw)
        assert "error" in result


class TestCrossSensorComparison:
    def test_returns_all_four_sensors(self):
        raw = get_cross_sensor_comparison.invoke("MACH-T1")
        result = json.loads(raw)
        assert len(result["sensors"]) == 4

    def test_positions_are_sorted(self):
        raw = get_cross_sensor_comparison.invoke("MACH-T1")
        result = json.loads(raw)
        positions = [s["position"] for s in result["sensors"]]
        assert positions == ["drive_end", "non_drive_end", "gearbox", "base"]

    def test_unknown_machine_returns_error(self):
        raw = get_cross_sensor_comparison.invoke("MACH-NOPE")
        result = json.loads(raw)
        assert "error" in result
