"""Tests for the FastAPI demo app, using a fake model so no checkpoint or
GPU is needed. Exercises the actual request/response plumbing (routing,
file-type validation, threshold handling) rather than detection quality.
"""

import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app import main as app_main


class _FakeDetectionModel(torch.nn.Module):
    def eval(self):
        return self

    def __call__(self, images):
        return [
            {
                "boxes": torch.tensor([[1.0, 2.0, 10.0, 12.0], [5.0, 5.0, 20.0, 20.0]]),
                "scores": torch.tensor([0.9, 0.2]),
                "labels": torch.tensor([1, 1]),
            }
        ]


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(app_main, "_model", _FakeDetectionModel())
    monkeypatch.setattr(app_main, "get_model", lambda: app_main._model)
    yield
    app_main._model = None


@pytest.fixture
def client():
    return TestClient(app_main.app)


def _image_bytes():
    buf = io.BytesIO()
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_default_threshold_filters_low_confidence(client):
    response = client.post("/predict", files={"file": ("img.jpeg", _image_bytes(), "image/jpeg")})
    assert response.status_code == 200
    detections = response.json()["detections"]
    assert len(detections) == 1
    assert detections[0]["score"] == pytest.approx(0.9)


def test_predict_zero_threshold_is_not_treated_as_falsy(client):
    """Regression test: `score_threshold or SCORE_THRESHOLD` used to silently
    ignore an explicit 0.0 threshold since 0.0 is falsy in Python."""
    response = client.post(
        "/predict",
        files={"file": ("img.jpeg", _image_bytes(), "image/jpeg")},
        params={"score_threshold": 0.0},
    )
    assert response.status_code == 200
    assert len(response.json()["detections"]) == 2


def test_predict_rejects_non_image_upload(client):
    response = client.post("/predict", files={"file": ("note.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_predict_without_a_model_returns_503(client, monkeypatch):
    def _raise():
        raise RuntimeError("No checkpoint at outputs/missing.pth.")

    monkeypatch.setattr(app_main, "get_model", _raise)
    response = client.post("/predict", files={"file": ("img.jpeg", _image_bytes(), "image/jpeg")})
    assert response.status_code == 503
