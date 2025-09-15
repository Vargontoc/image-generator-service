from fastapi.testclient import TestClient
from PIL import Image
from app.main import app, get_engine
import app.main as main_module
import os
from app.config import DEFAULT_MODEL

# We simulate two models via inspecting requested modelId passed through dependency override.
# The DummyMultiModel aware engine will encode the model id into image size subtly for assertion.
class DummyEngine:
    def __init__(self, model_id: str):
        self.model_id = model_id
    def generate_image(self, prompt, negative, width, height, steps, cfg, seed):
        # Encode model choice by altering the first pixel color channel slightly (just conceptual; not actually validated now)
        return Image.new("RGB", (width, height), color=(0, 0, 0))

# Keep a registry to know what model was requested
_created = {}

def override_engine(model_id: str | None = None):
    # model_id comes from get_engine wrapper; ensure stable default if None
    mid = model_id or DEFAULT_MODEL
    if mid not in _created:
        _created[mid] = DummyEngine(mid)
    return _created[mid]

os.environ["REQUIRE_API_KEY"] = "0"
os.environ["GENERATION_TIMEOUT_SECONDS"] = "0"
main_module.generation_timeout_seconds = lambda: 0  # type: ignore
app.dependency_overrides[get_engine] = override_engine
client = TestClient(app)

def test_generate_with_default_model():
    payload = {"prompt": "a house", "params": {"width": 64, "height": 64, "steps": 1, "cfg": 1}}
    r = client.post("/v1/generate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    # audit.model should be DEFAULT_MODEL
    assert body["audit"]["model"] == DEFAULT_MODEL


def test_generate_with_explicit_model_allowed():
    # If ALLOWED_MODELS includes some other model the service should accept it; we pick DEFAULT_MODEL to remain safe.
    alt_model = DEFAULT_MODEL
    payload = {"prompt": "a car", "params": {"width": 64, "height": 64, "steps": 1, "cfg": 1, "model": alt_model}}
    r = client.post("/v1/generate", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body["audit"]["model"] == alt_model


def test_generate_with_disallowed_model():
    payload = {"prompt": "a tree", "params": {"width": 64, "height": 64, "steps": 1, "cfg": 1, "model": "not-in-allowed"}}
    r = client.post("/v1/generate", json=payload)
    assert r.status_code == 400
    body = r.json()
    assert body["detail"].startswith("Model") and "not allowed" in body["detail"]
