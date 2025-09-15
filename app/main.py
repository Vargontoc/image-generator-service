from typing import Dict
from fastapi import FastAPI, HTTPException
import time
import logging
import random
from fastapi.staticfiles import StaticFiles

from app.engines.diffuser_engine import DiffusersEngine

from .models import GenerateRequest, HealthStatus, ImageItem, ImageModelInfo, JobStatus
from .storage import new_image_id, save_placeholder, url_for
from .config import IMAGES_DIR

app = FastAPI(title="Image Generation Service", version="0.1.0")
# Static files: align mount path with url_for() helper returning /files/<id>.png
app.mount("/files", StaticFiles(directory=IMAGES_DIR), name="files")
_ENGINE: DiffusersEngine | None = None  # lazy singleton

def get_engine() -> DiffusersEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = DiffusersEngine("stabilityai/sdxl-turbo")
    return _ENGINE


@app.get("/health", response_model=HealthStatus)
def health():
    return HealthStatus()

@app.get("/v1/models")
def image_models():
    return {
        "default_model" : "stabilityai/sdxl-turbo",
        "models": [
            ImageModelInfo(name="stabilityai/sdxl-turbo", family="sdxl", min_vram_gb=8.0, resolution="best@1024", tag=["fast", "mvp"]).model_dump()
        ]
    }

@app.post("/v1/generate", response_model=JobStatus)
def generate(req: GenerateRequest):
    # Validaciones básicas
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(400, "Prompt cannot be empty")
    if(req.params.width > 2048 or req.params.height > 2048):
        raise HTTPException(400, "Max size is 2048x2048")
    if req.params.steps > 100 or req.params.cfg > 20:
        raise HTTPException(400, "Steps/CFG exceed allowed limits")
    if req.params.width % 8 != 0 or req.params.height % 8 != 0:
        raise HTTPException(400, "Width and height must be multiples of 8")
    
    # Negative prompt por defecto (saneado)
    negative = req.negative_prompt or "low quality, bad anatomy, nsfw, watermark"

    # Seed: generar si no se especifica
    seed = req.params.seed if req.params.seed is not None else random.randint(0, 2**32 - 1)
    
    # Generar imagen
    logger = logging.getLogger("uvicorn.error")
    start = time.time()
    try:
        image = get_engine().generate_image(
            prompt=req.prompt.strip(),
            negative=negative,
            width=req.params.width,
            height=req.params.height,
            steps=req.params.steps,
            cfg=req.params.cfg,
            seed=seed)

        # Guardar imagen
        image_id = new_image_id()
        path = f"{IMAGES_DIR}/{image_id}.png"
        image.save(path, format="PNG")

        item = ImageItem(image_id=image_id, url=url_for(image_id), seed=seed)
        duration = round(time.time() - start, 3)
        logger.info("generation.completed", extra={
            "prompt_len": len(req.prompt.strip()),
            "width": req.params.width,
            "height": req.params.height,
            "steps": req.params.steps,
            "cfg": req.params.cfg,
            "seed": seed,
            "duration_sec": duration,
            "status": "completed"
        })
        return JobStatus(status="completed", images=[item], audit={"policy":"standard", "duration_sec": str(duration)})
    except Exception as e:  # broad catch to return structured error
        duration = round(time.time() - start, 3)
        logger.error("generation.failed", extra={
            "error": str(e),
            "duration_sec": duration,
            "seed": seed
        })
        return JobStatus(status="failed", images=[], error={"message": str(e)})

@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str):
    # Deprecated endpoint: previously used for async jobs
    raise HTTPException(410, detail="Endpoint deprecated: generation is synchronous now")