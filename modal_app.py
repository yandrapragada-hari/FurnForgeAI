"""
FurnForge AI — Modal Serverless GPU Deployment

Deploys the FastAPI backend with:
- Dedicated NVIDIA T4 GPU for ControlNet & TripoSR 3D reconstruction
- Persistent Modal Volume for uploads, outputs, and model caches
- Auto-scaling (scales to 0 when idle)
- Permanent HTTPS URL for Vercel integration
"""
import os
from pathlib import Path
import modal

app = modal.App("furnforge-ai")

# Base container image with system C++ / GL / OpenCV dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libegl1",
    )
    .pip_install(
        "fastapi>=0.110.0",
        "uvicorn>=0.28.0",
        "pydantic>=2.6.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "diffusers>=0.26.0",
        "transformers>=4.38.0",
        "accelerate>=0.27.0",
        "peft>=0.9.0",
        "python-multipart>=0.0.9",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "trimesh>=4.4.0",
        "einops>=0.7.0",
        "omegaconf>=2.3.0",
        "rembg>=2.0.50",
        "onnxruntime>=1.17.0",
        "scikit-image>=0.22.0",
        "moderngl>=5.10.0",
    )
    .add_local_dir("backend", remote_path="/root/backend")
)

# Persistent volume for HuggingFace model cache and generated 2D/3D assets
assets_volume = modal.Volume.from_name("furnforge-assets", create_if_missing=True)
model_cache_volume = modal.Volume.from_name("furnforge-model-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    container_idle_timeout=120,
    volumes={
        "/root/backend/outputs": assets_volume,
        "/root/backend/uploads": assets_volume,
        "/root/.hf_cache": model_cache_volume,
        "/root/.torch_cache": model_cache_volume,
    },
    env={
        "TORCH_DEVICE": "cuda",
        "PRELOAD_MODELS": "true",
        "ENABLE_LCM": "true",
        "HF_HOME": "/root/.hf_cache",
        "TORCH_HOME": "/root/.torch_cache",
    },
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/root/backend")
    from app.main import app as web_app
    return web_app
