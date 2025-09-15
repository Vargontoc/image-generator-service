import os
from typing import Optional
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

# Environment variable: METRICS_ENABLED (default 0)

def metrics_enabled() -> bool:
    return os.getenv("METRICS_ENABLED", "0").lower() not in ("0", "false", "no")

_registry: Optional[CollectorRegistry] = None
_generation_counter: Optional[Counter] = None
_generation_hist: Optional[Histogram] = None


def _ensure_metrics():
    global _registry, _generation_counter, _generation_hist
    if _registry is None:
        _registry = CollectorRegistry()
        _generation_counter = Counter(
            "image_generations_total",
            "Total de solicitudes de generación de imágenes",
            ["status", "model"],
            registry=_registry,
        )
        _generation_hist = Histogram(
            "image_generation_seconds",
            "Duración de generación de imágenes",
            ["model"],
            registry=_registry,
            buckets=(0.1,0.25,0.5,1,2,4,8,16,32,64)
        )


def record_generation(status: str, model: str, duration_sec: float):
    if not metrics_enabled():
        return
    _ensure_metrics()
    assert _generation_counter and _generation_hist
    _generation_counter.labels(status=status, model=model).inc()
    # Usar observe de histogram
    _generation_hist.labels(model=model).observe(duration_sec)


def prometheus_exposition_body() -> bytes:
    if not metrics_enabled():
        return b""  # vacío
    _ensure_metrics()
    return generate_latest(_registry)  # type: ignore[arg-type]


def prometheus_content_type() -> str:
    return CONTENT_TYPE_LATEST
