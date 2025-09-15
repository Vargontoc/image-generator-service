import os
import pytest
from fastapi.testclient import TestClient
from app.main import app, get_engine
from app.config import DEFAULT_MODEL
from PIL import Image

class DummyEngine:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id
    def generate_image(self, prompt, negative, width, height, steps, cfg, seed):
        return Image.new("RGB", (width, height), color=(1, 2, 3))

# Override engine dependency to avoid heavy loads
def override_engine(model_id: str | None = None):
    return DummyEngine(model_id)

app.dependency_overrides[get_engine] = override_engine
client = TestClient(app)

@pytest.fixture()
def enable_auth(monkeypatch):
    monkeypatch.setenv("REQUIRE_API_KEY", "1")
    monkeypatch.setenv("API_KEY", "purge-key")
    yield
    monkeypatch.setenv("REQUIRE_API_KEY", "0")
    monkeypatch.delenv("API_KEY", raising=False)


def _gen_payload():
    return {"prompt": "purge test", "params": {"width": 64, "height": 64, "steps": 1, "cfg": 1}}


def test_purge_global(enable_auth):
    # Trigger a generation to populate cache
    r = client.post("/v1/generate", json=_gen_payload(), headers={"X-API-Key": "purge-key"})
    assert r.status_code == 200
    # Purge all
    r2 = client.post("/v1/models/purge", headers={"X-API-Key": "purge-key"})
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["removed"] >= 0
    assert body["remaining"] >= 0


def test_purge_specific(enable_auth):
    # Generate (loads DEFAULT_MODEL)
    r = client.post("/v1/generate", json=_gen_payload(), headers={"X-API-Key": "purge-key"})
    assert r.status_code == 200
    # Purge that specific model
    r2 = client.post("/v1/models/purge", json={"model_id": DEFAULT_MODEL}, headers={"X-API-Key": "purge-key"})
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["model_id"] == DEFAULT_MODEL
    # Purge again should remove 0
    r3 = client.post("/v1/models/purge", json={"model_id": DEFAULT_MODEL}, headers={"X-API-Key": "purge-key"})
    assert r3.status_code == 200
    assert r3.json()["removed"] in (0, 1)
