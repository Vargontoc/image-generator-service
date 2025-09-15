import os

APP_PORT = int(os.getenv("APP_PORT", "8001"))
DATA_DIR = os.getenv("DATA_DIR", "./data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
# Optional external base URL (e.g., https://cdn.example.com) without trailing slash
BASE_URL = os.getenv("BASE_URL", "")

# Multi-model support
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "stabilityai/sdxl-turbo")
ALLOWED_MODELS = [m.strip() for m in os.getenv("ALLOWED_MODELS", DEFAULT_MODEL).split(",") if m.strip()]
MAX_MODELS_CACHE = int(os.getenv("MAX_MODELS_CACHE", "2"))

# Generation timeout (seconds). 0 or negative disables.
def generation_timeout_seconds() -> float:
	"""Return current generation timeout in seconds (0 or less disables)."""
	try:
		return float(os.getenv("GENERATION_TIMEOUT_SECONDS", "0"))
	except ValueError:
		return 0.0