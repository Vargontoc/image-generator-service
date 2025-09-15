import os
from fastapi import Header, HTTPException, status, Depends
from typing import List, Optional

# Environment variables:
# REQUIRE_API_KEY (default: 0 -> disabled)
# API_KEY (single) or API_KEYS (comma separated)

def _auth_enabled() -> bool:
    return os.getenv("REQUIRE_API_KEY", "0").lower() not in ("0", "false", "no")


def _allowed_keys() -> List[str]:
    multi = os.getenv("API_KEYS")
    single = os.getenv("API_KEY")
    if multi:
        return [k.strip() for k in multi.split(",") if k.strip()]
    if single:
        return [single.strip()]
    return []


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    if not _auth_enabled():
        return None
    allowed = _allowed_keys()
    if not allowed:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="API key auth enabled but no keys configured")
    if x_api_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-API-Key header")
    if x_api_key not in allowed:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return None

# Dependency alias for readability
AuthDependency = Depends(require_api_key)
