from typing import Optional

from pydantic import BaseModel, Field


class PreprocessRequest(BaseModel):
    image_base64: str
    low_threshold: int = Field(default=100, ge=1, le=255)
    high_threshold: int = Field(default=200, ge=1, le=255)
    blur_kernel: int = Field(default=3, ge=1, le=15)
    method: str = Field(default="canny")  # canny | adaptive_threshold | lineart


class SpecRenderRequest(BaseModel):
    category: str = "Chair"
    material: str = "Wood"
    color: str = "Warm Walnut"
    style: str = "Modern"
    finish: str = "Matte"
    custom_prompt: Optional[str] = ""


class GenerateRequest(BaseModel):
    # Exactly one of image_base64 / sample_id must be given.
    image_base64: Optional[str] = None
    sample_id: Optional[str] = None

    category: str = "Chair"     # Chair | Table | Sofa | Bed | Cabinet
    material: str = "Wood"      # Wood | Metal | Glass | Leather | Fabric
    color: str = "Warm Walnut"
    style: str = "Modern"       # Modern | Minimalist | Traditional | Luxury | Scandinavian | Industrial
    finish: str = "Matte"       # Matte | Glossy | Polished | Brushed
    custom_prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""

    # Generation settings
    mode: str = Field(default="controlnet", pattern="^(controlnet|img2img)$")
    strength: float = Field(default=0.85, ge=0.3, le=1.0)      # img2img only
    fast: bool = True                                           # LCM-LoRA
    num_variations: int = Field(default=1, ge=1, le=4)
    guidance_scale: float = Field(default=1.6, ge=1.0, le=20.0)
    controlnet_conditioning_scale: float = Field(default=0.70, ge=0.1, le=2.0)
    num_inference_steps: int = Field(default=6, ge=4, le=50)
    seed: Optional[int] = None

    # OpenCV edge extraction
    low_threshold: int = Field(default=100, ge=1, le=255)
    high_threshold: int = Field(default=200, ge=1, le=255)


class Generate3DRequest(BaseModel):
    generation_id: str = Field(..., min_length=1)
    image_url: Optional[str] = None
    variation_index: Optional[int] = Field(default=0, ge=0, le=3)
    model_preference: str = Field(
        default="auto",
        pattern="^(auto|triposr|hunyuan3d)$",
    )
    mc_resolution: int = Field(default=160, ge=96, le=256)
    bake_texture: bool = True
    seed: int = Field(default=42, ge=0, le=2147483647)
