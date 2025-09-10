import os
import uuid
from PIL import Image
from app.config import IMAGES_DIR


os.makedirs(IMAGES_DIR, exist_ok=True)

def new_image_id() -> str:
    return f"im_{uuid.uuid4().hex[:8]}"

def save_placeholder(image_id: str, width: int = 512, height: int = 512) -> str:    
    path = os.path.join(IMAGES_DIR, f"{image_id}.png")
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    img.save(path, format="PNG")
    return path

def url_for(image_id: str) -> str:
    return f"/files/{image_id}.png"