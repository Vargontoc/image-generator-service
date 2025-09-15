from collections import OrderedDict
from typing import Dict
from app.engines.diffuser_engine import DiffusersEngine
from app.config import DEFAULT_MODEL, ALLOWED_MODELS, MAX_MODELS_CACHE

class MultiModelEngine:
    """Mantiene un caché LRU de pipelines DiffusersEngine por model_id."""
    def __init__(self):
        self._cache: OrderedDict[str, DiffusersEngine] = OrderedDict()

    def get(self, model_id: str | None) -> DiffusersEngine:
        mid = model_id or DEFAULT_MODEL
        if mid not in ALLOWED_MODELS:
            raise ValueError(f"Model '{mid}' not allowed")
        # LRU hit
        if mid in self._cache:
            eng = self._cache.pop(mid)
            self._cache[mid] = eng
            return eng
        # Load new
        eng = DiffusersEngine(mid)
        self._cache[mid] = eng
        # Evict if over capacity
        if len(self._cache) > MAX_MODELS_CACHE:
            self._cache.popitem(last=False)
        return eng

    def list_models(self) -> Dict[str, Dict]:
        return {m: {"loaded": m in self._cache} for m in ALLOWED_MODELS}
