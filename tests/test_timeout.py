import os
import time
from fastapi.testclient import TestClient
from PIL import Image

# Set env BEFORE importing app.main so dynamic timeout function picks it up
os.environ["REQUIRE_API_KEY"] = "0"
os.environ["GENERATION_TIMEOUT_SECONDS"] = "0.3"

import app.main as main_module  # noqa: E402
from app.main import app  # noqa: E402

class SlowEngine:
    def generate_image(self, prompt, negative, width, height, steps, cfg, seed):
        # Sleep longer than timeout (0.3s)
        time.sleep(0.6)
        return Image.new("RGB", (width, height), color=(10, 20, 30))

def override_engine(model_id: str | None = None):
    return SlowEngine()

# Monkeypatch the module-level get_engine so generate() uses our slow engine
main_module.get_engine = override_engine
client = TestClient(app)

def test_generation_timeout():
    payload = {"prompt": "slow", "params": {"width": 64, "height": 64, "steps": 1, "cfg": 1}}
    r = client.post("/v1/generate", json=payload)
    assert r.status_code == 504, r.text
    assert r.json()["detail"].startswith("Generation timeout")
