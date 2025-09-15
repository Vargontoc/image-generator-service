from typing import Optional
import torch
from diffusers import AutoPipelineForText2Image

class DiffusersEngine:
    def __init__(self, model_id: str = "stabilityai/sdxl-turbo"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = None

    def _ensure_pipeline(self):
        if self.pipe is None:
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype
            )
            if self.device == "cuda":
                pipe = pipe.to(self.device)
            self.pipe = pipe
        
    def generate_image(self, prompt: str, negative: Optional[str], width: int, height: int, steps: int, cfg: float, seed: Optional[int]):
        g = None
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(seed)
        
        self._ensure_pipeline()
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=g
        )
        return result.images[0]