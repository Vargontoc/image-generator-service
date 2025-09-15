import os
from fastapi.testclient import TestClient
from app.main import app, get_engine
from app.config import DEFAULT_MODEL
from PIL import Image


class DummyEngine:
    def generate_image(self, prompt, negative, width, height, steps, cfg, seed):
        return Image.new("RGB", (width, height), color=(1, 2, 3))


def override_engine(model_id: str | None = None):
    return DummyEngine()


app.dependency_overrides[get_engine] = override_engine


def setup_module(module):
    # Force auth enabled with a known key
    os.environ["REQUIRE_API_KEY"] = "1"
    os.environ["API_KEY"] = "test-key"


client = TestClient(app)


def _payload():
    return {"prompt": "auth test", "params": {"width": 64, "height": 64, "steps": 1, "cfg": 1}}


def test_missing_key():
    r = client.post("/v1/generate", json=_payload())
    assert r.status_code == 401
    assert r.json()["detail"].startswith("Missing")


def test_invalid_key():
    r = client.post("/v1/generate", json=_payload(), headers={"X-API-Key": "wrong"})
    assert r.status_code == 401
    assert r.json()["detail"].startswith("Invalid")


def test_valid_key():
    r = client.post("/v1/generate", json=_payload(), headers={"X-API-Key": "test-key"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body["audit"]["model"] == DEFAULT_MODEL