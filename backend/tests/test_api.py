"""End-to-end API tests using FastAPI's TestClient.

These tests run only when the real depth model is ready (see /api/health):
there is no synthetic depth mode, so a full pipeline run requires the actual
Depth Anything V2 weights to be loadable.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services import depth_estimation as depth_service

client = TestClient(app)


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


class TestHealth:
    def test_health(self):
        res = client.get("/api/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        # "fallback" here means the real model could not be loaded
        # (diagnostic); no synthetic depth mode exists any more.
        assert body["depth_mode"] in ("ai", "fallback")
        assert body["device"] in ("cpu", "cuda")
        assert "demo_file" not in body


def _model_ready() -> bool:
    return depth_service.get_depth_mode()[0] == "ai"


def _skip_without_model():
    if not _model_ready():
        pytest.skip("Depth model not loadable in this environment")


class TestUpload:
    def test_upload_png(self):
        res = client.post(
            "/api/upload",
            files={"file": ("scene.png", _png_bytes(), "image/png")},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["job_id"]
        assert body["georeferenced"] is False
        assert body["input_type"] == "image"

    def test_upload_invalid_extension(self):
        res = client.post(
            "/api/upload",
            files={"file": ("scene.gif", _png_bytes(), "image/gif")},
        )
        assert res.status_code == 400

    def test_upload_corrupted(self):
        res = client.post(
            "/api/upload",
            files={"file": ("bad.png", b"\x89PNG broken", "image/png")},
        )
        assert res.status_code == 400


class TestPipeline:
    def test_full_pipeline(self):
        _skip_without_model()
        up = client.post(
            "/api/upload",
            files={"file": ("scene.png", _png_bytes(), "image/png")},
        ).json()
        job_id = up["job_id"]

        analyze = client.post("/api/analyze", json={"job_id": job_id, "mode": "relative"})
        assert analyze.status_code == 200

        # poll until completed
        status = None
        for _ in range(120):
            r = client.get(f"/api/statistics/{job_id}")
            assert r.status_code == 200
            status = r.json()
            if status["status"] in ("completed", "error"):
                break
            import time

            time.sleep(0.1)
        assert status["status"] == "completed", status.get("error")

        result = client.get(f"/api/result/{job_id}").json()
        assert result["statistics"]["dsm"]["units"] == "relative"
        assert result["statistics"]["dsm"]["calibrated"] is False
        assert "depth_color" in result["files"]
        assert "mesh" in result["files"]

    def test_statistics_unknown_job(self):
        res = client.get("/api/statistics/does-not-exist")
        assert res.status_code == 404


class TestValidation:
    def test_validation_without_reference(self):
        res = client.post(
            "/api/validate", json={"job_id": None, "reference": ""}
        )
        assert res.status_code == 200
        body = res.json()
        assert body["available"] is False
        assert "not available" in body["message"].lower()


class TestDemoEndpointsRemoved:
    def test_demo_endpoint_gone(self):
        res = client.post("/api/demo")
        assert res.status_code == 404

    def test_validate_demo_endpoint_gone(self):
        res = client.post("/api/validate-demo", json={"job_id": "anything"})
        assert res.status_code == 404