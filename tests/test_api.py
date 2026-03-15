"""
Integration tests for FastAPI endpoints — uses TestClient, no live server needed.
"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")

import pytest
from fastapi.testclient import TestClient

from api import app


@pytest.fixture()
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestSensorsEndpoint:
    def test_returns_list(self, client):
        resp = client.get("/sensors")
        assert resp.status_code == 200
        sensors = resp.json()
        assert len(sensors) == 8

    def test_sensor_has_required_fields(self, client):
        resp = client.get("/sensors")
        sensor = resp.json()[0]
        assert "sensor_id" in sensor
        assert "machine_id" in sensor
        assert "sensor_position" in sensor
        assert "axes" in sensor


class TestMachinesEndpoint:
    def test_returns_two_machines(self, client):
        resp = client.get("/machines")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_machine_has_four_sensors(self, client):
        resp = client.get("/machines")
        machine = resp.json()[0]
        assert len(machine["sensors"]) == 4


class TestMachineSensorsEndpoint:
    def test_returns_sensors_for_machine(self, client):
        resp = client.get("/machine/MACH-T1/sensors")
        assert resp.status_code == 200
        assert len(resp.json()) == 4

    def test_unknown_machine_404(self, client):
        resp = client.get("/machine/MACH-NOPE/sensors")
        assert resp.status_code == 404


class TestPlotEndpoint:
    def test_returns_png(self, client):
        resp = client.get("/sensor/SENS-T001/plot/x/accel_peak")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert len(resp.content) > 1000

    def test_invalid_axis_400(self, client):
        resp = client.get("/sensor/SENS-T001/plot/z/accel_peak")
        assert resp.status_code == 400

    def test_invalid_feature_400(self, client):
        resp = client.get("/sensor/SENS-T001/plot/x/invalid")
        assert resp.status_code == 400

    def test_unknown_sensor_404(self, client):
        resp = client.get("/sensor/SENS-NOPE/plot/x/accel_peak")
        assert resp.status_code == 404


class TestDashboard:
    def test_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "VibLabel" in resp.text


class TestLabelSummary:
    def test_returns_summary(self, client):
        resp = client.get("/labels/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_machines" in data
        assert data["total_machines"] == 2
        assert data["labeled"] == 0


class TestLabelExport:
    def test_returns_csv_header(self, client):
        resp = client.get("/labels/export")
        assert resp.status_code == 200
        assert "sensor_id" in resp.text
        assert "final_label" in resp.text


class TestGetLabels:
    def test_no_labels_returns_empty(self, client):
        resp = client.get("/labels")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_unknown_machine_label_404(self, client):
        resp = client.get("/label/machine/MACH-NOPE")
        assert resp.status_code == 404
