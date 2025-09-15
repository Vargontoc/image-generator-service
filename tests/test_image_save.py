import os
import io
from PIL import Image
import os
from fastapi.testclient import TestClient
from app.main import app, get_engine

class DummyEngine:
    def generate_image(self, prompt, negative, width, height, steps, cfg, seed):
        # Return a simple 1x1 white image
        return Image.new("RGB", (width, height), color=(255, 255, 255))

def override_engine():
    return DummyEngine()

os.environ["REQUIRE_API_KEY"] = "0"
app.dependency_overrides[get_engine] = override_engine
client = TestClient(app)

def test_image_generation_and_save(tmp_path):
    payload = {
        "prompt": "a cat",
        "params": {"width": 64, "height": 64, "steps": 1, "cfg": 1.0},
    }
    resp = client.post("/v1/generate", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "completed"
    assert len(data["images"]) == 1
    img_url = data["images"][0]["url"]
    assert img_url.endswith('.png')
    # Extract image id
    image_id = data["images"][0]["image_id"]
    # Compute path (same logic as storage.path_for)
    from app.config import IMAGES_DIR
    img_path = os.path.join(IMAGES_DIR, f"{image_id}.png")
    assert os.path.exists(img_path)
    # Validate PNG
    with Image.open(img_path) as im:
        assert im.format == "PNG"
        assert im.size == (64, 64)


def test_invalid_dimensions():
    payload = {"prompt": "x", "params": {"width": 65, "height": 64, "steps": 1, "cfg": 1}}
    resp = client.post("/v1/generate", json=payload)
    assert resp.status_code == 400
    assert "multiples of 8" in resp.text


def test_empty_prompt():
    payload = {"prompt": "   ", "params": {"width": 64, "height": 64, "steps": 1, "cfg": 1}}
    resp = client.post("/v1/generate", json=payload)
    assert resp.status_code == 400
    assert "Prompt cannot be empty" in resp.text
