from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

class HealthStatus(BaseModel):
    status: str = "ok"
    
class GenerateParams(BaseModel):
    width: int = Field(1024, ge=64, le=2048)
    height: int = Field(1024, ge=64, le=2048)
    steps: int = Field(25, ge=1, le=100)
    cfg: float = Field(7.5, ge=0, le=20)
    seed: Optional[int] = None
    model: Optional[str] = None
    
class SafetyConfig(BaseModel):
    # Correct typo: allow_ature_implicit -> allow_mature_implicit
    allow_mature_implicit: bool = Field(False, alias="allow_ature_implicit")

class Metadata(BaseModel):
    project_id: Optional[str] = None
    agent_id: Optional[str] = None

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    params: GenerateParams = GenerateParams()
    safety: SafetyConfig = SafetyConfig()
    # Correct typo: metada -> metadata (keep alias for backward compatibility)
    metadata: Metadata = Field(default_factory=Metadata, alias="metada")
    # Usar ConfigDict (Pydantic v2) para evitar deprecation warning
    model_config = ConfigDict(populate_by_name=True)
    
class JobAccepted(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed", "rejected"] = "completed"
    
class ImageItem(BaseModel):
    image_id: str
    url: str
    seed: Optional[int] = None
    
class JobStatus(BaseModel):
    status: Literal["queued", "running", "completed", "failed", "rejected"]
    images: List[ImageItem] = Field(default_factory=list)
    audit: Dict[str, str] | None = None
    error: Dict[str, str] | None = None
    
class ImageModelInfo(BaseModel):
    name: str
    family: str
    min_vram_gb: float | None = None
    resolution: str | None = None
    tag: List[str] = Field(default_factory=list)
    