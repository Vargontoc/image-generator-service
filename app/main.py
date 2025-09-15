from typing import Dict
from fastapi import FastAPI, HTTPException, Depends
from fastapi import Body
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import logging
import random
from fastapi.staticfiles import StaticFiles

from app.engines.diffuser_engine import DiffusersEngine
from app.engines.multi_model_engine import MultiModelEngine

from .models import GenerateRequest, HealthStatus, ImageItem, ImageModelInfo, JobStatus
from .storage import new_image_id, save_placeholder, url_for, path_for
from .config import IMAGES_DIR, DEFAULT_MODEL, ALLOWED_MODELS, generation_timeout_seconds
from .auth import AuthDependency
from .metrics import record_generation, prometheus_exposition_body, prometheus_content_type, metrics_enabled

app = FastAPI(title="Image Generation Service", version="0.1.0")
# Static files: align mount path with url_for() helper returning /files/<id>.png
app.mount("/files", StaticFiles(directory=IMAGES_DIR), name="files")
_MULTI_ENGINE: MultiModelEngine | None = None

def get_engine(model_id: str | None = None) -> DiffusersEngine:
    global _MULTI_ENGINE
    if _MULTI_ENGINE is None:
        _MULTI_ENGINE = MultiModelEngine()
    return _MULTI_ENGINE.get(model_id)


@app.get("/health", response_model=HealthStatus)
def health():
    return HealthStatus()

@app.get("/v1/models")
def image_models(_: None = AuthDependency):
    models = [
        ImageModelInfo(name=m, family="sdxl", min_vram_gb=8.0, resolution="best@1024", tag=["multi-model"]).model_dump()
        for m in ALLOWED_MODELS
    ]
    return {"default_model": DEFAULT_MODEL, "models": models}

@app.post("/v1/generate", response_model=JobStatus)
def generate(req: GenerateRequest, _: None = AuthDependency):
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
        # Selección de modelo
        selected_model = req.params.model or DEFAULT_MODEL
        if selected_model not in ALLOWED_MODELS:
            raise HTTPException(400, f"Model '{selected_model}' not allowed")
        def _do_generate():
            return get_engine(selected_model).generate_image(
                prompt=req.prompt.strip(),
                negative=negative,
                width=req.params.width,
                height=req.params.height,
                steps=req.params.steps,
                cfg=req.params.cfg,
                seed=seed)

        timeout_sec = generation_timeout_seconds()
        if timeout_sec and timeout_sec > 0:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_generate)
                try:
                    image = future.result(timeout=timeout_sec)
                except TimeoutError:
                    logger.error("generation.timeout", extra={
                        "model": selected_model,
                        "seed": seed,
                        "timeout_sec": timeout_sec
                    })
                    raise HTTPException(status_code=504, detail="Generation timeout exceeded")
        else:
            image = _do_generate()

        # Guardar imagen
        image_id = new_image_id()
        img_path = path_for(image_id)
        image.save(img_path, format="PNG")

        item = ImageItem(image_id=image_id, url=url_for(image_id), seed=seed)
        duration = round(time.time() - start, 3)
        logger.info("generation.completed", extra={
            "prompt_len": len(req.prompt.strip()),
            "width": req.params.width,
            "height": req.params.height,
            "steps": req.params.steps,
            "cfg": req.params.cfg,
            "seed": seed,
            "model": selected_model,
            "duration_sec": duration,
            "status": "completed"
        })
        record_generation("completed", selected_model, duration)
        return JobStatus(status="completed", images=[item], audit={"policy":"standard", "model": selected_model, "duration_sec": str(duration)})
    except HTTPException:
        # Re-lanzar para que FastAPI maneje correctamente el código de estado
        raise
    except Exception as e:  # broad catch to return structured error (500)
        duration = round(time.time() - start, 3)
        logger.error("generation.failed", extra={
            "error": str(e),
            "duration_sec": duration,
            "seed": seed,
            "model": req.params.model or DEFAULT_MODEL
        })
        record_generation("failed", req.params.model or DEFAULT_MODEL, duration)
        raise HTTPException(status_code=500, detail="Internal generation error")

@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str):
    # Deprecated endpoint: previously used for async jobs
    raise HTTPException(410, detail="Endpoint deprecated: generation is synchronous now")

@app.get("/metrics")
def metrics():
    if not metrics_enabled():
        return ""  # métricas deshabilitadas
    body = prometheus_exposition_body()
    from fastapi.responses import Response
    return Response(content=body, media_type=prometheus_content_type())


@app.post("/v1/models/purge")
def purge_models(payload: Dict[str, str] | None = Body(default=None), _: None = AuthDependency):
    """Purga el caché completo de modelos o un modelo específico.
    Body opcional: {"model_id": "nombre"}
    """
    global _MULTI_ENGINE
    if _MULTI_ENGINE is None:
        # Nada cargado aún; devolver estructura consistente
        mid = None
        if payload and isinstance(payload, dict):
            mid = payload.get("model_id")
        return {"model_id": mid, "removed": 0, "remaining": 0, "note": "cache empty"}
    model_id = None
    if payload and isinstance(payload, dict):
        model_id = payload.get("model_id")
    try:
        result = _MULTI_ENGINE.purge(model_id=model_id)
        # Incluir siempre la clave model_id (puede ser None)
        result = {"model_id": model_id, **result}
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))