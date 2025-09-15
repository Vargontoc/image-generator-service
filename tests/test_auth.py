import os
from fastapi.testclient import TestClient
from app.main import app, get_engine
from app.auth import require_api_key
from app.config import DEFAULT_MODEL
from PIL import Image


class DummyEngine:
    def generate_image(self, prompt, negative, width, height, steps, cfg, seed):
        return Image.new("RGB", (width, height), color=(1, 2, 3))


def override_engine(model_id: str | None = None):
    import os
    import pytest
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
    client = TestClient(app)


    @pytest.fixture()
    def enable_auth(monkeypatch):
        monkeypatch.setenv("REQUIRE_API_KEY", "1")
        monkeypatch.setenv("API_KEY", "test-key")
        yield
        # restore disabled state
        monkeypatch.setenv("REQUIRE_API_KEY", "0")
        monkeypatch.delenv("API_KEY", raising=False)


    def _payload():
        return {"prompt": "auth test", "params": {"width": 64, "height": 64, "steps": 1, "cfg": 1}}


    def test_missing_key(enable_auth):
        # No header provided
        r = client.post("/v1/generate", json=_payload())
        assert r.status_code == 401
        assert r.json()["detail"].startswith("Missing")


    def test_invalid_key(enable_auth):
        r = client.post("/v1/generate", json=_payload(), headers={"X-API-Key": "wrong"})
        assert r.status_code == 401
        assert r.json()["detail"].startswith("Invalid")


    def test_valid_key(enable_auth):
        r = client.post("/v1/generate", json=_payload(), headers={"X-API-Key": "test-key"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "completed"
        assert body["audit"]["model"] == DEFAULT_MODEL