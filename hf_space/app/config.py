"""
FurnForge AI - central configuration.

All values can be overridden through environment variables or a `backend/.env` file
(see `.env.example`).
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # python-dotenv is optional
    pass


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
# Hugging Face & Torch caches on D: drive to keep C: drive clean and avoid disk limits
HF_CACHE_DIR = BASE_DIR / ".hf_cache"
TORCH_CACHE_DIR = BASE_DIR / ".torch_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Images uploaded by users (model INPUT).
UPLOAD_DIR = BASE_DIR / "uploads"
# Images written by the diffusion model + edge maps (model OUTPUT storage, used by
# the gallery and download buttons). Nothing in here is ever used as a result source.
OUTPUT_DIR = BASE_DIR / "outputs"
# 3D meshes reconstructed by TripoSR / Hunyuan3D-2mini (.glb / .gltf / .obj + textures).
MODEL_3D_DIR = OUTPUT_DIR / "models3d"
# Optional example sketches (model INPUT) so the app can be tried without drawing.
SAMPLE_DIR = BASE_DIR / "sample_sketches"
DB_PATH = BASE_DIR / "app" / "database" / "furnforge.db"

for _d in (UPLOAD_DIR, OUTPUT_DIR, MODEL_3D_DIR, SAMPLE_DIR, DB_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
APP_NAME = "FurnForge AI"
APP_VERSION = "2.0.0"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

# ---------------------------------------------------------------------------
# Generative models (Hugging Face Hub ids)
# ---------------------------------------------------------------------------
# Stable Diffusion 1.5 base model (community mirror of runwayml/stable-diffusion-v1-5).
SD_MODEL = os.getenv("SD_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
# ControlNet checkpoints trained for SD 1.5 ("scribble" understands freehand pen/pencil sketches cleanly).
CONTROLNET_MODELS = {
    "scribble": os.getenv("CONTROLNET_SCRIBBLE", "lllyasviel/control_v11p_sd15_scribble"),
    "canny": os.getenv("CONTROLNET_CANNY", "lllyasviel/control_v11p_sd15_canny"),
}
DEFAULT_CONTROLNET = os.getenv("DEFAULT_CONTROLNET", "scribble")
# CPU inference resolution (384x384 -> crisp Lanczos 512x512 runs ~3x faster on CPU than 512x512).
CPU_INFERENCE_SIZE = int(os.getenv("CPU_INFERENCE_SIZE", "384"))
# LCM-LoRA distils SD 1.5 so it produces good images in 4-8 denoising steps.
# This is what makes CPU-only generation practical (~1 min instead of ~5 min).
LCM_LORA = os.getenv("LCM_LORA", "latent-consistency/lcm-lora-sdv1-5")
ENABLE_LCM = _env_bool("ENABLE_LCM", True)

# Load the models in a background thread when the server starts.
PRELOAD_MODELS = _env_bool("PRELOAD_MODELS", True)

# Pretrained CNN used for feature extraction / Grad-CAM.
CNN_BACKBONE = os.getenv("CNN_BACKBONE", "resnet50")

# ---------------------------------------------------------------------------
# Image-to-3D Reconstruction (Priority 1: TripoSR, Priority 2 Fallback: Hunyuan3D-2mini)
# ---------------------------------------------------------------------------
TRIPOSR_MODEL = os.getenv("TRIPOSR_MODEL", "stabilityai/TripoSR")
TRIPOSR_MC_RESOLUTION = int(os.getenv("TRIPOSR_MC_RESOLUTION", "160"))
TRIPOSR_CHUNK_SIZE = int(os.getenv("TRIPOSR_CHUNK_SIZE", "4096"))
TRIPOSR_FOREGROUND_RATIO = float(os.getenv("TRIPOSR_FOREGROUND_RATIO", "0.85"))
TRIPOSR_BAKE_TEXTURE = _env_bool("TRIPOSR_BAKE_TEXTURE", True)
TRIPOSR_TEXTURE_RESOLUTION = int(os.getenv("TRIPOSR_TEXTURE_RESOLUTION", "1024"))

HUNYUAN3D_MODEL = os.getenv("HUNYUAN3D_MODEL", "tencent/Hunyuan3D-2mini")
HUNYUAN3D_SUBFOLDER = os.getenv("HUNYUAN3D_SUBFOLDER", "hunyuan3d-dit-v2-mini")
HUNYUAN3D_STEPS = int(os.getenv("HUNYUAN3D_STEPS", "30"))
HUNYUAN3D_OCTREE_RES = int(os.getenv("HUNYUAN3D_OCTREE_RES", "380"))
HUNYUAN3D_MIN_VRAM_GB = float(os.getenv("HUNYUAN3D_MIN_VRAM_GB", "6.0"))
HUNYUAN3D_ENABLE_TEXTURE = _env_bool("HUNYUAN3D_ENABLE_TEXTURE", True)


# ---------------------------------------------------------------------------
# Compute device
# ---------------------------------------------------------------------------
def get_torch_device() -> str:
    forced = os.getenv("TORCH_DEVICE")
    if forced:
        return forced
    # In Hugging Face ZeroGPU environments, default to CPU to avoid fake zero-weight allocation
    # and daily GPU quota lockout, enabling unlimited fast 4-step LCM-LoRA generations.
    if os.getenv("SPACES_ZERO_GPU") or "spaces" in sys.modules:
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel Arc
            return "xpu"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


DEVICE = get_torch_device()
