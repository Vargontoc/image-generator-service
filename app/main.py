from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.engines.diffuser_engine import DiffusersEngine

from .models import GenerateRequest, HealthStatus, ImageItem, ImageModelInfo, JobAccepted, JobStatus
from .storage import new_image_id, save_placeholder, url_for
from .config import IMAGES_DIR

app = FastAPI(title="Image Generation Service", version="0.1.0")
app.mount("/file", StaticFiles(directory=IMAGES_DIR), name="files")
JOBS: Dict[str, JobStatus] = {}
ENGINE = DiffusersEngine("stabilityai/sdxl-turbo")


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

@app.post("/v1/generate", response_model=JobAccepted)
def generate(req: GenerateRequest):
    # Validaciones básicas
    if(req.params.width > 2048 or req.params.height > 2048):
        raise HTTPException(400, "Max size is 2048x2048")
    if req.params.steps > 100 or req.params.cfg > 20:
        raise HTTPException(400, "Steps/CFG exceed allowed limits")
    
    # Negative prompt por defecto
    negative = req.negative_prompt or "low quality, bad anatomy, sexual minors"
    
    # Generar imagen
    image = ENGINE.generate_image(
        prompt=req.prompt.strip(),
        negative=negative,
        width=req.params.width,
        height=req.params.height,
        steps=req.params.steps,
        cfg=req.params.cfg,
        seed=req.params.seed)
    
    # Guardar imagen
    image_id = new_image_id()
    path = f"{IMAGES_DIR}/{image_id}.png"
    image.save(path, format="PNG")

    # Crear entrada de trabajo
    job_id = f"jb_{image_id[3:]}"
    item = ImageItem(image_id=image_id, url=url_for(image_id), seed=req.params.seed)
    status = JobStatus(status="completed", images=[item], audit={"policy":"standard"})
    JOBS[job_id] = status
    
    # Responder
    return JobAccepted(job_id=job_id, status="completed")

@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str):
    st = JOBS.get(job_id)
    if not st:
        raise HTTPException(404, "Job not found")
    return st