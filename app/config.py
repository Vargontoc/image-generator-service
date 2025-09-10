import os

APP_PORT = int(os.getenv("APP_PORT", "8001"))
DATA_DIR = os.getenv("DATA_DIR", "./data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")