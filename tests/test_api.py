from fastapi.testclient import TestClient
from PIL import Image
from app.main import app, get_engine
import os

# Ensure auth disabled for these tests
os.environ["REQUIRE_API_KEY"] = "0"

class DummyEngine:
    def generate_image(self, prompt, negative, width, height, steps, cfg, seed):
        return Image.new("RGB", (width, height), color=(0, 0, 0))

def override_engine():
    return DummyEngine()

app.dependency_overrides[get_engine] = override_engine
client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_generate_success():
    payload = {"prompt": "mountain landscape", "params": {"width": 64, "height": 64, "steps": 1, "cfg": 1}}
    r = client.post("/v1/generate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["images"][0]["url"].endswith('.png')


def test_generate_invalid_size():
    payload = {"prompt": "x", "params": {"width": 66, "height": 65, "steps": 1, "cfg": 1}}
    r = client.post("/v1/generate", json=payload)
    assert r.status_code == 400


def test_generate_empty_prompt():
    payload = {"prompt": "   ", "params": {"width": 64, "height": 64, "steps": 1, "cfg": 1}}
    r = client.post("/v1/generate", json=payload)
    assert r.status_code == 400


def test_jobs_deprecated():
    r = client.get("/v1/jobs/any")
    assert r.status_code == 410
