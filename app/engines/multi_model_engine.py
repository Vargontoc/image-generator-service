from collections import OrderedDict
from typing import Dict
import threading
from app.engines.diffuser_engine import DiffusersEngine
from app.config import DEFAULT_MODEL, ALLOWED_MODELS, MAX_MODELS_CACHE

class MultiModelEngine:
    """Mantiene un caché LRU de pipelines DiffusersEngine por model_id."""
    def __init__(self):
        self._cache: OrderedDict[str, DiffusersEngine] = OrderedDict()
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def get(self, model_id: str | None) -> DiffusersEngine:
        mid = model_id or DEFAULT_MODEL
        if mid not in ALLOWED_MODELS:
            raise ValueError(f"Model '{mid}' not allowed")
        # LRU hit
        if mid in self._cache:
            eng = self._cache.pop(mid)
            self._cache[mid] = eng
            return eng
        # Double-checked locking per model
        # Acquire (or create) a lock specific for this model id
        with self._global_lock:
            lock = self._locks.get(mid)
            if lock is None:
                lock = threading.Lock()
                self._locks[mid] = lock
        with lock:
            # Re-check cache under lock (another thread may have loaded it)
            if mid in self._cache:
                eng2 = self._cache.pop(mid)
                self._cache[mid] = eng2
                return eng2
            eng = DiffusersEngine(mid)
            self._cache[mid] = eng
            if len(self._cache) > MAX_MODELS_CACHE:
                self._cache.popitem(last=False)
            return eng

    def list_models(self) -> Dict[str, Dict]:
        return {m: {"loaded": m in self._cache} for m in ALLOWED_MODELS}
