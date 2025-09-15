import os
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app, get_engine
import app.main as main_module

# Enable metrics, disable auth & timeout
os.environ["METRICS_ENABLED"] = "1"
os.environ["REQUIRE_API_KEY"] = "0"
os.environ["GENERATION_TIMEOUT_SECONDS"] = "0"
main_module.generation_timeout_seconds = lambda: 0  # type: ignore

class DummyEngine:
    def generate_image(self, prompt, negative, width, height, steps, cfg, seed):
        return Image.new("RGB", (width, height), color=(123, 123, 123))

def override_engine(model_id: str | None = None):
    return DummyEngine()

app.dependency_overrides[get_engine] = override_engine
client = TestClient(app)

def test_metrics_endpoint_after_generation():
    payload = {"prompt": "metric test", "params": {"width": 64, "height": 64, "steps": 1, "cfg": 1}}
    r = client.post("/v1/generate", json=payload)
    assert r.status_code == 200, r.text
    m = client.get("/metrics")
    assert m.status_code == 200
    body = m.text
    assert "image_generations_total" in body
    assert "image_generation_seconds" in body