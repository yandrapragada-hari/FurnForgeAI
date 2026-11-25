"""
FurnForge AI - FastAPI backend.

Pipeline for every generation request:
  1. Receive sketch (upload or example sketch)          -> uploads/
  2. OpenCV preprocessing + Canny edge map              -> outputs/edge_*.png
  3. Pretrained CNN (ResNet) analysis + Grad-CAM
  4. Stable Diffusion 1.5 + ControlNet image-to-image   -> outputs/render_*.png
  5. Result stored in the SQLite gallery
"""
import base64
import binascii
import json
import logging
import random
import shutil
import time
import uuid
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.config import (
    APP_NAME,
    APP_VERSION,
    DEVICE,
    ENABLE_LCM,
    MAX_UPLOAD_MB,
    MODEL_3D_DIR,
    OUTPUT_DIR,
    PRELOAD_MODELS,
    SAMPLE_DIR,
    SD_MODEL,
    CONTROLNET_MODELS,
    DEFAULT_CONTROLNET,
    TRIPOSR_MODEL,
    HUNYUAN3D_MODEL,
    UPLOAD_DIR,
)
from app.database.db import db
from app.jobs import Job, JobManager
from app.models.schemas import Generate3DRequest, GenerateRequest, PreprocessRequest, SpecRenderRequest
from app.pipeline.cnn_extractor import CNNFeatureExtractor
from app.pipeline.cv_preprocessor import OpenCVPreprocessor
from app.pipeline.diffusion_engine import DiffusionEngine
from app.pipeline.photoreal_renderer import render_specification
from app.pipeline.prompt_builder import build_prompt
from app.pipeline.mesh3d_engine import (
    Mesh3DEngine,
    PhotorealInputError,
    check_3d_capabilities,
    resolve_photoreal_render_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("FurnForgeAI")

try:
    import spaces
    gpu_decorator = spaces.GPU(duration=30)
    IS_ZEROGPU = True
except Exception:
    def gpu_decorator(fn):
        return fn
    IS_ZEROGPU = bool(os.getenv("SPACES_ZERO_GPU"))

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Sketch-to-photoreal furniture rendering with OpenCV, a pretrained CNN, "
                "Stable Diffusion 1.5 and ControlNet.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/sample_sketches", StaticFiles(directory=str(SAMPLE_DIR)), name="sample_sketches")

cnn = CNNFeatureExtractor()
engine = DiffusionEngine()
jobs = JobManager(default_message="Starting diffusion iterations...")
mesh3d = Mesh3DEngine()
jobs_3d = JobManager(default_message="Initializing TripoSR 3D reconstruction...")

SAMPLES = [
    {"id": "chair", "name": "Lounge chair", "category": "Chair"},
    {"id": "sofa", "name": "Three-seater sofa", "category": "Sofa"},
    {"id": "table", "name": "Pedestal table", "category": "Table"},
    {"id": "bed", "name": "Platform bed", "category": "Bed"},
    {"id": "cabinet", "name": "Credenza", "category": "Cabinet"},
]
ALLOWED_FORMATS = {".png", ".jpg", ".jpeg", ".webp"}


@app.on_event("startup")
def _startup() -> None:
    if PRELOAD_MODELS:
        engine.load_async()


# --------------------------------------------------------------------------- helpers
def _decode_upload(data_url: str) -> np.ndarray:
    """Validate size/format of a base64 upload or file path and decode it to BGR."""
    if data_url.startswith("/") or not data_url.startswith("data:"):
        clean_name = data_url.split("?")[0].split("/")[-1]
        for candidate_dir in (SAMPLE_DIR, UPLOAD_DIR, OUTPUT_DIR):
            cand_path = candidate_dir / clean_name
            if cand_path.exists():
                img = cv2.imread(str(cand_path))
                if img is not None:
                    return img
    payload = data_url.split(",", 1)[1] if "," in data_url else data_url
    if len(payload) * 3 / 4 > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds the {MAX_UPLOAD_MB} MB limit.")
    if data_url.startswith("data:") and not any(
        t in data_url[:40] for t in ("image/png", "image/jpeg", "image/jpg", "image/webp")
    ):
        raise HTTPException(415, "Unsupported format. Upload PNG, JPG or JPEG.")
    try:
        return OpenCVPreprocessor.decode_image_base64(data_url)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(400, f"Could not decode image: {exc}")


def _load_input(req_image: Optional[str], sample_id: Optional[str]):
    if sample_id:
        path = SAMPLE_DIR / f"{sample_id}_sketch.png"
        if not path.exists():
            raise HTTPException(404, f"Example sketch '{sample_id}' not found.")
        return cv2.imread(str(path)), f"/sample_sketches/{path.name}"
    if req_image:
        bgr = _decode_upload(req_image)
        name = f"upload_{uuid.uuid4().hex[:12]}.png"
        cv2.imwrite(str(UPLOAD_DIR / name), bgr)
        return bgr, f"/uploads/{name}"
    raise HTTPException(400, "Provide either 'image_base64' or 'sample_id'.")


def _model_status() -> dict:
    return {
        "status": engine.status,
        "detail": engine.status_detail,
        "error": engine.error,
        "engine": engine.engine_label,
        "sd_model": SD_MODEL,
        "controlnet_model": CONTROLNET_MODELS[DEFAULT_CONTROLNET],
        "lcm_enabled": ENABLE_LCM,
        "lcm_available": engine.lcm_available,
        "load_seconds": engine.load_seconds,
    }


# --------------------------------------------------------------------------- routes
@app.get("/")
def _root():
    return {
        "status": "online",
        "app": APP_NAME,
        "version": APP_VERSION,
        "health": "/api/health",
        "docs": "/docs"
    }



@app.get("/api/health")
def health():
    caps_3d = check_3d_capabilities()
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
        "device": DEVICE,
        "model": _model_status(),
        "cnn": {"ready": cnn.ready, "backbone": getattr(cnn, "backbone_name", None), "error": cnn.error},
        "mesh3d": {
            "primary_model": TRIPOSR_MODEL,
            "fallback_model": HUNYUAN3D_MODEL,
            "ready": caps_3d["ready"],
            "cuda_available": caps_3d["cuda_available"],
            "gpu_name": caps_3d["gpu_name"],
            "vram_gb": caps_3d["vram_gb"],
        },
        "queue_length": jobs.pending,
    }


@app.post("/api/model/load")
def load_model():
    """Start loading the diffusion models (useful when PRELOAD_MODELS=false)."""
    engine.load_async()
    return _model_status()


@app.get("/api/samples")
def samples():
    return {"samples": [
        {**s, "url": f"/sample_sketches/{s['id']}_sketch.png"}
        for s in SAMPLES if (SAMPLE_DIR / f"{s['id']}_sketch.png").exists()
    ]}


@app.post("/api/preprocess")
def preprocess(req: PreprocessRequest):
    """OpenCV preprocessing + CNN analysis, used for the live preview in the studio."""
    try:
        bgr = _decode_upload(req.image_base64)
        prep = OpenCVPreprocessor.process_sketch(
            bgr, low_threshold=req.low_threshold, high_threshold=req.high_threshold,
            blur_kernel=req.blur_kernel if req.blur_kernel % 2 else req.blur_kernel + 1,
            method=req.method,
        )
        features = None
        if cnn.ready:
            try:
                features = cnn.analyze(prep["padded_bgr"], prep["edge_map"])
            except Exception as cnn_err:
                logger.warning("CNN analyze non-fatal error: %s", cnn_err)
        cond_b64 = prep["base64_scribble_map"] if DEFAULT_CONTROLNET == "scribble" else prep["base64_edge_map"]
        return {
            "edge_map_base64": cond_b64,
            "scribble_map_base64": prep["base64_scribble_map"],
            "display_sketch_base64": prep["base64_display_sketch"],
            "edge_density_percent": prep["edge_density_percent"],
            "contour_count": prep["contour_count"],
            "cnn_features": features,
            "cnn_heatmap_base64": features.get("b64_heatmap") if features else None,
            "pixel_metrics": features.get("pixel_metrics") if features else None,
            "cnn_error": None if cnn.ready else cnn.error,
        }
    except Exception as exc:
        logger.exception("Preprocess critical error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(exc)}")


@app.post("/api/spec-render")
def spec_render(req: SpecRenderRequest):
    """
    Instantly render a photoreal furniture response image strictly adhering
    to the student's selected specification (Material, Palette Color, Style, Surface Finish).
    """
    res = render_specification(
        category=req.category,
        material=req.material,
        color=req.color,
        style=req.style,
        finish=req.finish,
        custom_prompt=req.custom_prompt or "",
    )
    # Also attach CNN features for the category if available
    try:
        cat_key = req.category.strip().lower()
        sketch_p = SAMPLE_DIR / f"{cat_key}_sketch.png"
        if sketch_p.exists() and cnn.ready:
            bgr = cv2.imread(str(sketch_p))
            prep = OpenCVPreprocessor.process_sketch(bgr)
            features = cnn.analyze(prep["padded_bgr"], prep["edge_map"])
            res["cnn_features"] = features
    except Exception as e:
        logger.warning("CNN analysis during spec render: %s", e)
    return res


@app.post("/api/generate", status_code=202)
def generate(req: GenerateRequest):
    """
    Validate input, run OpenCV + CNN synchronously, then queue the diffusion job.
    Returns a job id; poll GET /api/jobs/{job_id} for progress and the result.
    """
    if engine.status == "error":
        raise HTTPException(503, f"Generative model error: {engine.error or engine.status_detail}")

    raw_bgr, original_url = _load_input(req.image_base64, req.sample_id)
    if req.low_threshold >= req.high_threshold:
        raise HTTPException(422, "low_threshold must be smaller than high_threshold.")

    gen_id = f"furn_{uuid.uuid4().hex[:10]}"
    prep = OpenCVPreprocessor.process_sketch(
        raw_bgr, low_threshold=req.low_threshold, high_threshold=req.high_threshold
    )
    cond_map = prep["scribble_map"] if DEFAULT_CONTROLNET == "scribble" else prep["edge_map"]
    edge_name = f"edge_{gen_id}.png"
    cv2.imwrite(str(OUTPUT_DIR / edge_name), cond_map)

    cnn_features = cnn.analyze(prep["padded_bgr"], prep["edge_map"]) if cnn.ready else {}

    prompt = build_prompt(
        req.category, req.material, req.color, req.style, req.finish,
        req.custom_prompt or "", cnn_features=cnn_features
    )
    negative = (req.negative_prompt or "").strip() or prompt["negative_prompt"]
    base_seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)

    # Diffusers inputs: ControlNet receives the clean scribble/edge map; img2img starts from cleaned sketch.
    control_image = Image.fromarray(cv2.cvtColor(cond_map, cv2.COLOR_GRAY2RGB))
    init_image = Image.fromarray(cv2.cvtColor(prep["padded_bgr"], cv2.COLOR_BGR2RGB))

    @gpu_decorator
    def _execute_sync() -> dict:
        if engine.status != "ready":
            engine.load()
            if engine.status != "ready":
                raise RuntimeError(engine.error or engine.status_detail)

        t0 = time.time()
        variations = []
        for i in range(req.num_variations):
            seed = base_seed + i
            image = engine.generate(
                control_image=control_image, init_image=init_image,
                prompt=prompt["positive_prompt"], negative_prompt=negative,
                mode=req.mode, strength=req.strength, fast=req.fast,
                steps=req.num_inference_steps, guidance_scale=req.guidance_scale,
                controlnet_scale=req.controlnet_conditioning_scale, seed=seed,
            )
            name = f"render_{gen_id}_{i}.png"
            image.save(OUTPUT_DIR / name)
            variations.append({"variation_index": i, "image_url": f"/outputs/{name}", "seed": seed})

        elapsed = round(time.time() - t0, 2)
        mode_label = engine.engine_label + (" + LCM-LoRA" if req.fast and engine.lcm_available else "")
        mode_label += " | img2img" if req.mode == "img2img" else " | ControlNet txt2img"
        record = {
            "id": gen_id,
            "category": req.category, "material": req.material, "color": req.color,
            "style": req.style, "finish": req.finish,
            "prompt": prompt["positive_prompt"],
            "original_image_url": original_url,
            "preprocessed_edge_url": f"/outputs/{edge_name}",
            "result_image_url": variations[0]["image_url"],
            "variations": variations,
            "cnn_features": {k: v for k, v in cnn_features.items() if k != "b64_heatmap"},
            "pipeline_mode": mode_label,
            "execution_time": elapsed,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        db.save_generation(record)
        return {**record, "cnn_features": cnn_features, "negative_prompt": negative, "device": DEVICE,
                "settings": req.model_dump(exclude={"image_base64"})}

    def _execute_cpu_fallback() -> dict:
        logger.info("Executing generation on CPU fallback...")
        t0 = time.time()
        variations = []
        for i in range(req.num_variations):
            seed = base_seed + i
            orig_device = engine.device
            engine.device = "cpu"
            try:
                if hasattr(engine, "_pipe") and engine._pipe is not None:
                    engine._pipe = engine._pipe.to("cpu")
                if hasattr(engine, "_img2img") and engine._img2img is not None:
                    engine._img2img = engine._img2img.to("cpu")
                image = engine.generate(
                    control_image=control_image, init_image=init_image,
                    prompt=prompt["positive_prompt"], negative_prompt=negative,
                    mode=req.mode, strength=req.strength, fast=True,
                    steps=5, guidance_scale=1.5,
                    controlnet_scale=req.controlnet_conditioning_scale, seed=seed,
                )
            finally:
                pass
            name = f"render_{gen_id}_{i}.png"
            image.save(OUTPUT_DIR / name)
            variations.append({"variation_index": i, "image_url": f"/outputs/{name}", "seed": seed})

        elapsed = round(time.time() - t0, 2)
        record = {
            "id": gen_id,
            "category": req.category, "material": req.material, "color": req.color,
            "style": req.style, "finish": req.finish,
            "prompt": prompt["positive_prompt"],
            "original_image_url": original_url,
            "preprocessed_edge_url": f"/outputs/{edge_name}",
            "result_image_url": variations[0]["image_url"],
            "variations": variations,
            "cnn_features": {k: v for k, v in cnn_features.items() if k != "b64_heatmap"},
            "pipeline_mode": "Stable Diffusion 1.5 + ControlNet (CPU Fallback)",
            "execution_time": elapsed,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        db.save_generation(record)
        return {**record, "cnn_features": cnn_features, "negative_prompt": negative, "device": "cpu",
                "settings": req.model_dump(exclude={"image_base64"})}

    if IS_ZEROGPU:
        try:
            record = _execute_sync()
        except Exception as exc:
            logger.warning("ZeroGPU execution note (%s); switching to CPU fallback", exc)
            record = _execute_cpu_fallback()

        job = Job(lambda j: record, total_units=req.num_variations)
        job.id = f"job_{uuid.uuid4().hex[:12]}"
        job.status = "completed"
        job.result = record
        jobs._jobs[job.id] = job
        return {
            "status": "completed",
            "job_id": job.id,
            "generation_id": gen_id,
            "result": record,
            "primary_image_url": record["result_image_url"],
            "result_image_url": record["result_image_url"],
            "original_image_url": original_url,
            "preprocessed_edge_url": f"/outputs/{edge_name}",
            "cnn_features": cnn_features,
            "variations": record["variations"],
            "queue_position": 0,
        }

    def run(job: Job) -> dict:
        if engine.status != "ready":
            job.report(0, 0, 1, "Loading Stable Diffusion 1.5 + ControlNet weights")
            engine.load()
            if engine.status != "ready":
                raise RuntimeError(engine.error or engine.status_detail)

        t0 = time.time()
        variations = []
        for i in range(req.num_variations):
            seed = base_seed + i
            job.report(i, 0, 0, f"Denoising variation {i + 1}/{req.num_variations}")

            def on_step(step: int, total: int, _i=i):
                job.report(_i, step, total,
                           f"Variation {_i + 1}/{req.num_variations}: denoising step {step}/{total}")

            image = engine.generate(
                control_image=control_image, init_image=init_image,
                prompt=prompt["positive_prompt"], negative_prompt=negative,
                mode=req.mode, strength=req.strength, fast=req.fast,
                steps=req.num_inference_steps, guidance_scale=req.guidance_scale,
                controlnet_scale=req.controlnet_conditioning_scale, seed=seed, progress=on_step,
            )
            name = f"render_{gen_id}_{i}.png"
            image.save(OUTPUT_DIR / name)
            variations.append({"variation_index": i, "image_url": f"/outputs/{name}", "seed": seed})

        elapsed = round(time.time() - t0, 2)
        mode_label = engine.engine_label + (" + LCM-LoRA" if req.fast and engine.lcm_available else "")
        mode_label += " | img2img" if req.mode == "img2img" else " | ControlNet txt2img"
        record = {
            "id": gen_id,
            "category": req.category, "material": req.material, "color": req.color,
            "style": req.style, "finish": req.finish,
            "prompt": prompt["positive_prompt"],
            "original_image_url": original_url,
            "preprocessed_edge_url": f"/outputs/{edge_name}",
            "result_image_url": variations[0]["image_url"],
            "variations": variations,
            "cnn_features": {k: v for k, v in cnn_features.items() if k != "b64_heatmap"},
            "pipeline_mode": mode_label,
            "execution_time": elapsed,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        db.save_generation(record)
        return {**record, "cnn_features": cnn_features, "negative_prompt": negative, "device": DEVICE,
                "settings": req.model_dump(exclude={"image_base64"})}

    job = jobs.submit(run, total_units=req.num_variations)
    return {
        "job_id": job.id,
        "generation_id": gen_id,
        "original_image_url": original_url,
        "preprocessed_edge_url": f"/outputs/{edge_name}",
        "cnn_features": cnn_features,
        "queue_position": jobs.pending,
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id) or jobs_3d.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job.to_dict()


# --------------------------------------------------------------------------- 3D reconstruction (TripoSR -> Hunyuan3D-2mini)
@app.get("/api/3d/requirements")
def mesh3d_requirements():
    """Report hardware and package readiness for TripoSR (Priority 1) and Hunyuan3D-2mini (Fallback)."""
    return check_3d_capabilities()


def _load_cached_3d_model(generation_id: str) -> Optional[dict]:
    asset_dir = (MODEL_3D_DIR / generation_id).resolve()
    if asset_dir.parent != MODEL_3D_DIR.resolve() or not asset_dir.exists():
        return None
    glb_path = asset_dir / "model.glb"
    if not glb_path.exists():
        return None
    meta_path = asset_dir / "meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    formats = ["glb"]
    if (asset_dir / "gltf" / "model.gltf").exists() or (asset_dir / "model-gltf.zip").exists():
        formats.append("gltf")
    if (asset_dir / "obj" / "model.obj").exists() or (asset_dir / "model-obj.zip").exists():
        formats.append("obj")
    return {
        "status": "ready",
        "engine_used": "TripoSR",
        "glb_url": f"/outputs/models3d/{generation_id}/model.glb",
        "gltf_url": f"/api/3d/download/{generation_id}/gltf" if "gltf" in formats else None,
        "obj_url": f"/api/3d/download/{generation_id}/obj" if "obj" in formats else None,
        "supported_formats": formats,
    }


@app.get("/api/3d/status/{generation_id}")
def mesh3d_status(generation_id: str):
    item = db.get_by_id(generation_id)
    if not item and not generation_id.startswith("spec_"):
        raise HTTPException(404, "Design generation not found.")
    model_3d = item.get("model_3d") if item else None
    if not model_3d:
        model_3d = _load_cached_3d_model(generation_id)
    return {
        "generation_id": generation_id,
        "model_3d": model_3d,
        "compatibility": check_3d_capabilities(),
    }


@app.post("/api/3d/generate", status_code=202)
def generate_3d_model(req: Generate3DRequest):
    """
    Reconstruct an interactive 3D mesh from an existing FurnForge AI photorealistic 2D render.
    Priority 1: stabilityai/TripoSR
    Priority 2 Fallback: tencent/Hunyuan3D-2mini
    """
    item = db.get_by_id(req.generation_id)
    img_path = req.image_url
    if img_path and "://" in img_path:
        from urllib.parse import urlparse
        img_path = urlparse(img_path).path

    if not item:
        if req.generation_id.startswith("spec_") and img_path and img_path.startswith("/outputs/spec_"):
            item = {
                "id": req.generation_id,
                "result_image_url": img_path,
                "variations": [{"variation_index": 0, "image_url": img_path}],
                "original_image_url": None,
                "preprocessed_edge_url": None,
            }
        else:
            raise HTTPException(404, f"Generation '{req.generation_id}' not found in gallery.")

    try:
        image_path, source_url = resolve_photoreal_render_path(
            item,
            requested_image_url=req.image_url,
            variation_index=req.variation_index,
        )
    except PhotorealInputError as exc:
        raise HTTPException(422, str(exc)) from exc

    caps = check_3d_capabilities()
    if req.model_preference == "hunyuan3d" and not caps["fallback"]["available"]:
        raise HTTPException(
            503,
            detail={
                "message": " ".join(caps["fallback"]["blocking_reasons"]),
                "compatibility": caps,
            },
        )
    if req.model_preference == "triposr" and not caps["primary"]["available"]:
        raise HTTPException(
            503,
            detail={
                "message": " ".join(caps["primary"]["blocking_reasons"]),
                "compatibility": caps,
            },
        )
    if not caps["ready"]:
        raise HTTPException(
            503,
            detail={
                "message": " ".join(caps["blocking_reasons"]),
                "compatibility": caps,
            },
        )

    @gpu_decorator
    def _execute_3d_sync() -> dict:
        unloaded_2d = False
        if engine.status in ("loading", "ready"):
            engine.unload()
            unloaded_2d = True

        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

        try:
            model_3d = mesh3d.reconstruct_from_image(
                image_path=image_path,
                generation_id=req.generation_id,
                source_image_url=source_url,
                model_preference=req.model_preference,
                mc_resolution=req.mc_resolution,
                bake_texture=req.bake_texture,
                seed=req.seed,
            )
            db.update_3d_model(req.generation_id, model_3d)
            return {"generation_id": req.generation_id, "model_3d": model_3d}
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            if unloaded_2d and PRELOAD_MODELS:
                engine.load_async()

    if IS_ZEROGPU:
        res_3d = _execute_3d_sync()
        job = Job(lambda j: res_3d, total_units=1)
        job.id = f"job_3d_{uuid.uuid4().hex[:12]}"
        job.status = "completed"
        job.result = res_3d
        jobs_3d._jobs[job.id] = job
        return {
            "status": "completed",
            "job_id": job.id,
            "generation_id": req.generation_id,
            "source_image_url": source_url,
            "primary_model": TRIPOSR_MODEL,
            "fallback_model": HUNYUAN3D_MODEL,
            "model_3d": res_3d["model_3d"],
            "queue_position": 0,
        }

    def run_3d(job: Job) -> dict:
        job.report(0, 5, 100, "Downloading & preparing TripoSR 3D AI weights...")

        def on_progress(step: int, total: int, message: str) -> None:
            job.report(0, step, total, message)

        unloaded_2d = False
        if engine.status in ("loading", "ready"):
            job.report(0, 3, 100, "Optimizing memory: releasing 2D diffusion pipeline for 3D...")
            engine.unload()
            unloaded_2d = True

        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

        try:
            model_3d = mesh3d.reconstruct_from_image(
                image_path=image_path,
                generation_id=req.generation_id,
                source_image_url=source_url,
                model_preference=req.model_preference,
                mc_resolution=req.mc_resolution,
                bake_texture=req.bake_texture,
                seed=req.seed,
                progress=on_progress,
            )
            db.update_3d_model(req.generation_id, model_3d)
            return {"generation_id": req.generation_id, "model_3d": model_3d}
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            import gc
            gc.collect()
            if unloaded_2d and PRELOAD_MODELS:
                engine.load_async()

    job = jobs_3d.submit(run_3d, total_units=1)
    return {
        "job_id": job.id,
        "generation_id": req.generation_id,
        "source_image_url": source_url,
        "primary_model": TRIPOSR_MODEL,
        "fallback_model": HUNYUAN3D_MODEL,
        "queue_position": jobs_3d.pending,
    }


@app.get("/api/3d/jobs/{job_id}")
def mesh3d_job_status(job_id: str):
    job = jobs_3d.get(job_id)
    if not job:
        raise HTTPException(404, "3D reconstruction job not found.")
    return job.to_dict()


@app.get("/api/3d/download/{generation_id}/{fmt}")
def download_3d_model(generation_id: str, fmt: str, raw: bool = False):
    fmt_clean = fmt.lower().strip()
    if fmt_clean not in ("glb", "gltf", "obj"):
        raise HTTPException(400, "Unsupported 3D export format. Choose glb, gltf, or obj.")

    asset_dir = (MODEL_3D_DIR / generation_id).resolve()
    if asset_dir.parent != MODEL_3D_DIR.resolve() or not asset_dir.exists():
        raise HTTPException(404, "No 3D model generated for this design yet.")

    if fmt_clean == "glb":
        target = asset_dir / "model.glb"
        if not target.exists():
            raise HTTPException(404, "GLB asset not found.")
        return FileResponse(
            path=str(target),
            media_type="model/gltf-binary",
            filename=f"furnforge-{generation_id}.glb",
        )

    if fmt_clean == "gltf":
        bundle = asset_dir / "model-gltf.zip"
        single = asset_dir / "gltf" / "model.gltf"
        if not raw and bundle.exists():
            return FileResponse(
                path=str(bundle),
                media_type="application/zip",
                filename=f"furnforge-{generation_id}-gltf.zip",
            )
        if single.exists():
            return FileResponse(
                path=str(single),
                media_type="model/gltf+json",
                filename=f"furnforge-{generation_id}.gltf",
            )
        raise HTTPException(404, "GLTF asset not found.")

    bundle = asset_dir / "model-obj.zip"
    single = asset_dir / "obj" / "model.obj"
    if not raw and bundle.exists():
        return FileResponse(
            path=str(bundle),
            media_type="application/zip",
            filename=f"furnforge-{generation_id}-obj.zip",
        )
    if single.exists():
        return FileResponse(
            path=str(single),
            media_type="text/plain",
            filename=f"furnforge-{generation_id}.obj",
        )
    raise HTTPException(404, "OBJ asset not found.")


@app.get("/api/gallery")
def gallery(category: Optional[str] = None, limit: int = 60):
    items = db.get_all(category=category, limit=limit)
    return {"count": len(items), "items": items}


@app.get("/api/gallery/{item_id}")
def gallery_item(item_id: str):
    item = db.get_by_id(item_id)
    if not item:
        raise HTTPException(404, "Item not found.")
    return item


@app.delete("/api/gallery/{item_id}")
def delete_gallery_item(item_id: str):
    item = db.get_by_id(item_id)
    if not item or not db.delete_by_id(item_id):
        raise HTTPException(404, "Item not found.")
    # Remove the generated files belonging to this item.
    urls = [v["image_url"] for v in item.get("variations", [])] + [item["preprocessed_edge_url"]]
    if item["original_image_url"].startswith("/uploads/"):
        urls.append(item["original_image_url"])
    for url in urls:
        folder = OUTPUT_DIR if url.startswith("/outputs/") else UPLOAD_DIR
        (folder / url.rsplit("/", 1)[-1]).unlink(missing_ok=True)
    asset_dir = MODEL_3D_DIR / item_id
    if asset_dir.exists():
        shutil.rmtree(asset_dir, ignore_errors=True)
    return {"status": "deleted", "id": item_id}
