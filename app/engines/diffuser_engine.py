from typing import Optional
import torch
from diffusers import AutoPipelineForText2Image

class DiffusersEngine:
    def __init__(self, model_id: str = "stabilityai/sdxl-turbo"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Prefer bfloat16 if available (new GPUs) else float16 for CUDA
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        elif self.device == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        # Enable TF32 on Ampere+ for speed (safe for diffusion)
        if self.device == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
        self.pipe = None

    def _ensure_pipeline(self):
        if self.pipe is None:
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype
            )
            if self.device == "cuda":
                pipe = pipe.to(self.device)
                # Memory / speed tweaks
                try:
                    pipe.enable_attention_slicing()
                    pipe.enable_vae_slicing()
                except Exception:
                    pass
            else:
                # On CPU attention slicing still helps peak memory
                try:
                    pipe.enable_attention_slicing()
                except Exception:
                    pass
            self.pipe = pipe
        
    def generate_image(self, prompt: str, negative: Optional[str], width: int, height: int, steps: int, cfg: float, seed: Optional[int]):
        g = None
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(seed)
        
        self._ensure_pipeline()
        # Autocast for performance (half / bf16) where it makes sense
        if self.device == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
            autocast_dtype = torch.bfloat16 if self.dtype == torch.bfloat16 else torch.float16
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=g
                )
        else:
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