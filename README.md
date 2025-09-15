# Image Generation Service (SDXL-Turbo)

Servicio FastAPI para generar imágenes usando el modelo `stabilityai/sdxl-turbo` (vía diffusers), con carga perezosa del pipeline, validaciones de parámetros, soporte multi-model y logging estructurado.

## Características
- Generación síncrona (endpoint `/v1/generate` devuelve el resultado directamente)
- Modelo(s) cargado(s) de forma lazy al primer uso (multi-model con caché LRU configurable)
- Validaciones: prompt no vacío, dimensiones <= 2048 y múltiplos de 8, límites de steps/CFG
- Seed reproducible (se genera uno si no se envía)
- Negative prompt seguro por defecto
- Logging de duración y metadatos de generación
- Tests unitarios con motor simulado

## Requisitos
- Python 3.11+ recomendado
- GPU opcional (CUDA) para mejor rendimiento
- Dependencias listadas en `requirements.txt`

## Instalación
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Para instalación específica de PyTorch (CPU / CUDA) consultar: https://pytorch.org/get-started/locally/

Ejemplo (CUDA 12.x):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Variables de Entorno
| Variable | Descripción | Default |
|----------|-------------|---------|
| `APP_PORT` | Puerto HTTP | 8001 |
| `DATA_DIR` | Directorio raíz de datos | ./data |
| `BASE_URL` | Prefijo absoluto para URLs de imágenes (sin slash final) | (vacío) |
| `DEFAULT_MODEL` | Modelo por defecto al generar si no se especifica | stabilityai/sdxl-turbo |
| `ALLOWED_MODELS` | Lista separada por comas de modelos permitidos | (igual a DEFAULT_MODEL) |
| `MAX_MODELS_CACHE` | Cuántos modelos mantener en memoria (LRU) | 2 |
| `MODEL_ID` | (Obsoleto) Anterior bandera única de modelo | stabilityai/sdxl-turbo |

Ejemplo para habilitar dos modelos y caché de 2:
```bash
export DEFAULT_MODEL="stabilityai/sdxl-turbo"
export ALLOWED_MODELS="stabilityai/sdxl-turbo,stabilityai/sdxl-lightning"
export MAX_MODELS_CACHE=2
```

## Ejecución
```bash
uvicorn app.main:app --reload --port 8001
```

## Endpoints
### GET /health
Estado simple del servicio.

### GET /v1/models
Lista dinámica de modelos soportados definida por `ALLOWED_MODELS`. Devuelve también el `default_model`.

### POST /v1/generate
Genera una imagen. Se puede especificar un modelo alternativo (si está permitido) dentro de `params.model`.

Ejemplo JSON:
```json
{
  "prompt": "studio photo of a corgi astronaut on the moon",
  "negative_prompt": null,
  "params": {"width": 1024, "height": 1024, "steps": 25, "cfg": 7.5, "model": "stabilityai/sdxl-lightning"}
}
```

Respuesta (ejemplo):
```json
{
  "status": "completed",
  "images": [
    {"image_id": "im_ab12cd34", "url": "/files/im_ab12cd34.png", "seed": 123456789}
  ],
  "audit": {"policy": "standard", "duration_sec": "0.823"},
  "error": null
}
```

### GET /v1/jobs/{job_id}
Obsoleto: devuelve 410 porque la generación ahora es síncrona.

## Ejemplos curl
```bash
curl -X POST http://localhost:8001/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"hyperrealistic portrait of a cyberpunk samurai","params":{"width":512,"height":512,"steps":15,"cfg":6.5}}'
```

Con selección explícita de modelo:
```bash
curl -X POST http://localhost:8001/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"dramatic cinematic landscape","params":{"width":512,"height":512,"steps":10,"cfg":5.5,"model":"stabilityai/sdxl-lightning"}}'
```


## Tests
Ejecutar (requiere dependencia `pytest` si no está):
```bash
pip install pytest
pytest -q
```
Los tests mockean el motor de difusión para ser rápidos y deterministas.

## Notas de Rendimiento (Futuro)
- Activar `torch.autocast` para GPU/CPU Ampere+.
- Offloading / attention slicing.
- Warmup inicial opcional.

## Seguridad y Uso Responsable
No uses el servicio para generar contenido prohibido. 
## Licencia
MIT (añadir archivo LICENSE si se requiere).

---
_Mantén las dependencias actualizadas y revisa los límites de VRAM según el hardware._
