from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

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
    allow_ature_implicit: bool = False

class Metadata(BaseModel):
    project_id: Optional[str] = None
    agent_id: Optional[str] = None

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    params: GenerateParams = GenerateParams()
    safety: SafetyConfig = SafetyConfig()
    metada: Metadata =  Metadata()
    
class JobAccepted(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed", "rejected"] = "completed"
    
class ImageItem(BaseModel):
    image_id: str
    url: str
    seed: Optional[int] = None
    
class JobStatus(BaseModel):
    status: Literal["queued", "running", "completed", "failed", "rejected"]
    images: List[ImageItem] = []
    audit: Dict[str, str] | None = None
    error: Dict[str, str] | None = None
    
class ImageModelInfo(BaseModel):
    name: str
    family: str
    min_vram_gb: float | None = None
    resolution: str | None = None
    tag: List[str] = []
    