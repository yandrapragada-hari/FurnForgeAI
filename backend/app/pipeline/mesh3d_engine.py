"""
FurnForge AI - Image-to-3D Reconstruction Engine.

Model priority chain:
  1. Primary (Priority 1): stabilityai/TripoSR (supports CUDA GPU and CPU)
  2. Fallback (Priority 2): tencent/Hunyuan3D-2mini (requires CUDA GPU + hy3dgen)

Design guarantees:
- Accepts ONLY the final photorealistic 2D render produced by FurnForge AI.
- Rejects sketches, ControlNet/OpenCV edge maps, and unverified image paths.
- Never generates procedural/random placeholder meshes or fake 3D views.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import logging
import platform
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from app.config import (
    HUNYUAN3D_ENABLE_TEXTURE,
    HUNYUAN3D_MIN_VRAM_GB,
    HUNYUAN3D_MODEL,
    HUNYUAN3D_OCTREE_RES,
    HUNYUAN3D_STEPS,
    HUNYUAN3D_SUBFOLDER,
    MODEL_3D_DIR,
    OUTPUT_DIR,
    TRIPOSR_BAKE_TEXTURE,
    TRIPOSR_CHUNK_SIZE,
    TRIPOSR_FOREGROUND_RATIO,
    TRIPOSR_MC_RESOLUTION,
    TRIPOSR_MODEL,
    TRIPOSR_TEXTURE_RESOLUTION,
)

logger = logging.getLogger("FurnForgeAI.Mesh3D")

ProgressCallback = Callable[[int, int, str], None]


class PhotorealInputError(ValueError):
    """Raised when the requested 3D input is not a valid generated photoreal render."""


class Mesh3DIncompatibleError(RuntimeError):
    """Raised when neither Priority 1 (TripoSR) nor Priority 2 (Hunyuan3D-2mini) can run."""


class Mesh3DGenerationError(RuntimeError):
    """Raised when 3D reconstruction or mesh export fails."""


def check_3d_capabilities() -> Dict[str, Any]:
    """
    Inspect hardware & library support for Priority 1 (TripoSR) and
    Priority 2 Fallback (tencent/Hunyuan3D-2mini).
    """
    os_name = platform.system()
    torch_version = None
    cuda_available = False
    cuda_version = None
    gpu_name = None
    vram_gb = 0.0

    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = getattr(torch.version, "cuda", None)
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            vram_gb = round(props.total_memory / (1024 ** 3), 2)
    except ImportError as exc:
        return {
            "ready": False,
            "os": os_name,
            "torch_version": None,
            "cuda_available": False,
            "device": "cpu",
            "gpu_name": None,
            "vram_gb": 0.0,
            "primary": {
                "name": "TripoSR",
                "model_id": TRIPOSR_MODEL,
                "available": False,
                "blocking_reasons": [f"PyTorch is not installed: {exc}"],
            },
            "fallback": {
                "name": "Hunyuan3D-2mini",
                "model_id": HUNYUAN3D_MODEL,
                "available": False,
                "blocking_reasons": [f"PyTorch is not installed: {exc}"],
            },
            "blocking_reasons": [f"PyTorch is not installed: {exc}"],
        }

    # Check Priority 1: TripoSR (supports CUDA GPU and CPU)
    triposr_missing = [
        pkg for pkg in ("einops", "omegaconf", "trimesh", "rembg")
        if importlib.util.find_spec(pkg) is None
    ]
    has_torchmcubes = importlib.util.find_spec("torchmcubes") is not None
    has_skimage = importlib.util.find_spec("skimage") is not None
    if not has_torchmcubes and not has_skimage:
        triposr_missing.append("torchmcubes or scikit-image")

    has_uv_bake = False
    if (
        importlib.util.find_spec("xatlas") is not None
        and importlib.util.find_spec("moderngl") is not None
    ):
        try:
            import moderngl

            # Test whether a headless OpenGL context (EGL, standalone, or OSMesa) can be instantiated
            for backend in ("egl", None, "osmesa"):
                try:
                    if backend:
                        test_ctx = moderngl.create_context(standalone=True, backend=backend)
                    else:
                        test_ctx = moderngl.create_context(standalone=True)
                    if test_ctx is not None:
                        try:
                            test_ctx.release()
                        except Exception:
                            pass
                        has_uv_bake = True
                        break
                except Exception:
                    continue
        except Exception:
            has_uv_bake = False
    triposr_reasons: List[str] = []
    if triposr_missing:
        triposr_reasons.append(
            "Missing TripoSR Python packages: " + ", ".join(triposr_missing)
        )
    triposr_available = len(triposr_reasons) == 0

    # Check Priority 2 Fallback: tencent/Hunyuan3D-2mini (requires CUDA GPU + hy3dgen)
    hy3d_installed = importlib.util.find_spec("hy3dgen") is not None
    hunyuan_reasons: List[str] = []
    if not cuda_available:
        hunyuan_reasons.append(
            f"{HUNYUAN3D_MODEL} requires an NVIDIA CUDA GPU (detected CPU-only PyTorch {torch_version})."
        )
    elif vram_gb < HUNYUAN3D_MIN_VRAM_GB:
        hunyuan_reasons.append(
            f"{HUNYUAN3D_MODEL} requires >= {HUNYUAN3D_MIN_VRAM_GB:.0f} GB VRAM (detected {vram_gb} GB on {gpu_name})."
        )
    if not hy3d_installed:
        hunyuan_reasons.append(
            "Package 'hy3dgen' is not installed (install from https://github.com/Tencent-Hunyuan/Hunyuan3D-2)."
        )
    hunyuan_available = len(hunyuan_reasons) == 0

    blocking_reasons = []
    if not triposr_available and not hunyuan_available:
        blocking_reasons = triposr_reasons + hunyuan_reasons

    return {
        "ready": triposr_available or hunyuan_available,
        "os": os_name,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device": "cuda" if cuda_available else "cpu",
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "priority_order": ["triposr", "hunyuan3d"],
        "primary": {
            "key": "triposr",
            "name": "TripoSR",
            "model_id": TRIPOSR_MODEL,
            "model_url": f"https://huggingface.co/{TRIPOSR_MODEL}",
            "available": triposr_available,
            "device": "cuda" if cuda_available else "cpu",
            "marching_cubes_backend": "torchmcubes" if has_torchmcubes else "scikit-image",
            "uv_texture_baking": has_uv_bake,
            "blocking_reasons": triposr_reasons,
        },
        "fallback": {
            "key": "hunyuan3d",
            "name": "Hunyuan3D-2mini",
            "model_id": HUNYUAN3D_MODEL,
            "model_url": f"https://huggingface.co/{HUNYUAN3D_MODEL}",
            "available": hunyuan_available,
            "device": "cuda" if cuda_available else "unavailable",
            "min_vram_gb": HUNYUAN3D_MIN_VRAM_GB,
            "blocking_reasons": hunyuan_reasons,
        },
        "blocking_reasons": blocking_reasons,
    }


def resolve_photoreal_render_path(
    generation_record: Dict[str, Any],
    requested_image_url: Optional[str] = None,
    variation_index: Optional[int] = 0,
) -> tuple[Path, str]:
    """
    Resolve and strictly validate the generated 2D photorealistic image on disk.

    Rejects sketches, ControlNet/OpenCV edge maps, and any path outside OUTPUT_DIR.
    Accepts both relative paths (/outputs/...) and full public URLs (https://.../outputs/...).
    """
    def _clean_path(u: Optional[str]) -> Optional[str]:
        if not u or not isinstance(u, str):
            return None
        cleaned = u.split("?", 1)[0].strip()
        if "://" in cleaned:
            from urllib.parse import urlparse
            cleaned = urlparse(cleaned).path
        return cleaned

    candidate_url = requested_image_url
    if not candidate_url:
        variations = generation_record.get("variations") or []
        if variation_index is not None and 0 <= variation_index < len(variations):
            candidate_url = variations[variation_index].get("image_url")
        candidate_url = candidate_url or generation_record.get("result_image_url")

    if not candidate_url:
        raise PhotorealInputError("Generation record does not contain a photoreal render URL.")

    normalized = _clean_path(candidate_url)
    forbidden_urls = {
        _clean_path(generation_record.get("original_image_url")),
        _clean_path(generation_record.get("preprocessed_edge_url")),
    }

    if not normalized or normalized in forbidden_urls or "/sample_sketches/" in normalized or "/uploads/" in normalized:
        raise PhotorealInputError(
            "3D reconstruction requires the final generated photorealistic render, not a sketch or ControlNet edge map."
        )

    filename = Path(normalized).name
    if filename.startswith("edge_") or filename.startswith("mask_") or filename.startswith("upload_"):
        raise PhotorealInputError(
            f"Refusing non-photoreal input '{filename}'. Only generated photorealistic furniture renders are allowed."
        )

    allowed_urls = {_clean_path(generation_record.get("result_image_url"))}
    for var in generation_record.get("variations") or []:
        if var.get("image_url"):
            allowed_urls.add(_clean_path(var["image_url"]))

    if normalized not in allowed_urls or not normalized.startswith("/outputs/"):
        raise PhotorealInputError(
            f"Image URL '{normalized}' does not match a verified photoreal render of generation '{generation_record.get('id')}'."
        )

    output_root = OUTPUT_DIR.resolve()
    image_path = (output_root / filename).resolve()
    if image_path.parent != output_root or not image_path.exists():
        raise PhotorealInputError(f"Generated photoreal image file not found on disk: {normalized}")

    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as exc:
        raise PhotorealInputError(f"Photoreal image '{filename}' is unreadable or corrupted: {exc}") from exc

    return image_path, normalized


def _extract_mesh_stats(scene_or_mesh: Any, has_uv: bool, pbr_materials: bool) -> Dict[str, Any]:
    import trimesh

    if isinstance(scene_or_mesh, trimesh.Scene):
        geometries = list(scene_or_mesh.geometry.values())
        vertices = sum(len(g.vertices) for g in geometries if hasattr(g, "vertices"))
        faces = sum(len(g.faces) for g in geometries if hasattr(g, "faces"))
        bounds = scene_or_mesh.bounds.tolist() if scene_or_mesh.bounds is not None else None
    else:
        vertices = int(len(getattr(scene_or_mesh, "vertices", [])))
        faces = int(len(getattr(scene_or_mesh, "faces", [])))
        bounds_arr = getattr(scene_or_mesh, "bounds", None)
        bounds = bounds_arr.tolist() if bounds_arr is not None else None

    return {
        "vertex_count": int(vertices),
        "face_count": int(faces),
        "uv_mapped": bool(has_uv),
        "pbr_materials": bool(pbr_materials),
        "bounding_box": bounds,
    }


def _package_sidecars_if_needed(target_dir: Path, primary_file: Path, bundle_zip_path: Path) -> Path:
    files = [p for p in target_dir.iterdir() if p.is_file()]
    if len(files) <= 1:
        return primary_file
    with zipfile.ZipFile(bundle_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(files):
            zf.write(file_path, arcname=file_path.name)
    return bundle_zip_path


def _srgb_to_linear_rgba(rgba_float: np.ndarray) -> np.ndarray:
    """Convert [0..1] sRGB RGBA vertex colors to glTF-2.0 spec Linear RGBA uint8."""
    rgb = np.clip(rgba_float[:, :3], 0.0, 1.0)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    out = np.ones((len(rgb), 4), dtype=np.uint8) * 255
    out[:, :3] = np.clip(np.round(linear * 255.0), 0, 255).astype(np.uint8)
    if rgba_float.shape[1] == 4:
        out[:, 3] = np.clip(np.round(rgba_float[:, 3] * 255.0), 0, 255).astype(np.uint8)
    return out


def _export_trimesh_formats(
    mesh: Any,
    generation_id: str,
    srgb_vertex_colors: Optional[np.ndarray] = None,
) -> List[str]:
    """Export a smooth trimesh Mesh/Scene to real GLB, GLTF, and OBJ files."""
    import trimesh

    asset_dir = MODEL_3D_DIR / generation_id
    gltf_dir = asset_dir / "gltf"
    obj_dir = asset_dir / "obj"
    for folder in (asset_dir, gltf_dir, obj_dir):
        folder.mkdir(parents=True, exist_ok=True)

    for old_zip in (asset_dir / "model-gltf.zip", asset_dir / "model-obj.zip"):
        old_zip.unlink(missing_ok=True)

    # Ensure vertex normals exist so Three.js & glTF viewers use smooth shading
    _ = mesh.vertex_normals

    glb_path = asset_dir / "model.glb"
    try:
        glb_path.write_bytes(trimesh.exchange.gltf.export_glb(mesh, include_normals=True))
    except TypeError:
        mesh.export(str(glb_path))

    supported = ["glb"]
    try:
        gltf_files = trimesh.exchange.gltf.export_gltf(mesh, include_normals=True)
        for name, data in gltf_files.items():
            out_name = "model.gltf" if name.endswith(".gltf") else name
            content = data.encode("utf-8") if isinstance(data, str) else data
            if out_name == "model.gltf" and isinstance(content, bytes):
                # Fix internal buffer URI if renamed
                (gltf_dir / out_name).write_bytes(content)
            else:
                (gltf_dir / name).write_bytes(content)
        _package_sidecars_if_needed(gltf_dir, gltf_dir / "model.gltf", asset_dir / "model-gltf.zip")
        supported.append("gltf")
    except Exception as exc:
        logger.warning("GLTF export skipped for %s: %s", generation_id, exc)

    try:
        obj_mesh = mesh
        if srgb_vertex_colors is not None:
            obj_mesh = mesh.copy()
            obj_mesh.visual = trimesh.visual.ColorVisuals(mesh=obj_mesh, vertex_colors=srgb_vertex_colors)
        obj_path = obj_dir / "model.obj"
        obj_mesh.export(str(obj_path), include_normals=True)
        _package_sidecars_if_needed(obj_dir, obj_path, asset_dir / "model-obj.zip")
        supported.append("obj")
    except Exception as exc:
        logger.warning("OBJ export skipped for %s: %s", generation_id, exc)

    return supported


def _clean_furniture_foreground(pil_rgb: Image.Image, rembg_session: Any) -> Image.Image:
    """
    Remove studio backdrop cleanly, pruning any border-connected wall vignette or
    drop shadow that u2net occasionally merges with rectangular furniture (e.g. cabinets).
    """
    from app.pipeline.tsr.utils import remove_background

    rgba = np.array(remove_background(pil_rgb, rembg_session))
    alpha = rgba[:, :, 3]
    bgr = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
    h, w = alpha.shape

    touches_border = (
        np.mean(alpha[:8, :] > 20) > 0.03
        or np.mean(alpha[-8:, :] > 20) > 0.03
        or np.mean(alpha[:, :8] > 20) > 0.03
        or np.mean(alpha[:, -8:] > 20) > 0.03
    )
    if not touches_border:
        return Image.fromarray(rgba)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)

    barrier = cv2.dilate((grad > 28.0).astype(np.uint8), np.ones((3, 3), np.uint8))
    passable = (barrier == 0).astype(np.uint8)
    _num_cc, labels = cv2.connectedComponents(passable, connectivity=4)

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:4, :] = True
    border_mask[-4:, :] = True
    border_mask[:, :4] = True
    border_mask[:, -4:] = True
    border_labels = set(np.unique(labels[border_mask])) - {0}
    border_wall = np.isin(labels, list(border_labels))

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[alpha >= 160] = cv2.GC_PR_FGD
    core_fg = (
        cv2.erode(((alpha >= 230) & (~border_wall)).astype(np.uint8), np.ones((11, 11), np.uint8))
        > 0
    )
    gc_mask[core_fg] = cv2.GC_FGD
    gc_mask[(alpha < 20) | border_wall] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc_mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
        refined = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    except Exception:
        refined = np.where((alpha > 127) & (~border_wall), 255, 0).astype(np.uint8)

    n_fg, fg_labels, stats, _ = cv2.connectedComponentsWithStats(refined)
    if n_fg > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = areas.max()
        keep = np.zeros_like(refined)
        for idx, area in enumerate(areas, start=1):
            if area >= max_area * 0.15:
                keep[fg_labels == idx] = 255
        refined = keep

    alpha = np.where(refined > 0, np.maximum(alpha, 235), 0).astype(np.uint8)
    rgba[:, :, 3] = cv2.GaussianBlur(alpha, (3, 3), 0)
    return Image.fromarray(rgba)


class Mesh3DEngine:
    """
    Manages Image-to-3D reconstruction with:
      Priority 1: stabilityai/TripoSR (CUDA GPU or CPU)
      Priority 2 (Fallback): tencent/Hunyuan3D-2mini (CUDA GPU)
    """

    def __init__(self) -> None:
        self._triposr = None
        self._rembg_session = None
        self._hy_shapegen = None
        self._hy_texgen = None
        self._hy_rembg = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()

    def _ensure_triposr(self, device: str, progress: Optional[ProgressCallback] = None):
        with self._load_lock:
            if self._triposr is not None:
                return self._triposr

            if progress:
                progress(12, 100, f"Loading {TRIPOSR_MODEL} (Priority 1) on {device.upper()}")

            import rembg
            from app.pipeline.tsr.system import TSR

            gc.collect()
            model = TSR.from_pretrained(
                TRIPOSR_MODEL,
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            chunk_size = min(TRIPOSR_CHUNK_SIZE, 4096) if device == "cpu" else TRIPOSR_CHUNK_SIZE
            model.renderer.set_chunk_size(chunk_size)
            model.to(device)
            self._triposr = model
            if self._rembg_session is None:
                self._rembg_session = rembg.new_session("u2net")
            return self._triposr

    def _run_triposr(
        self,
        image_path: Path,
        generation_id: str,
        source_image_url: str,
        mc_resolution: int,
        bake_texture: bool,
        progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        import torch
        import trimesh
        from app.pipeline.tsr.utils import (
            remove_background,
            resize_foreground,
            to_gradio_3d_orientation,
        )

        caps = check_3d_capabilities()
        if not caps["primary"]["available"]:
            raise Mesh3DIncompatibleError(" ".join(caps["primary"]["blocking_reasons"]))

        device = caps["primary"]["device"]
        effective_mc_res = min(int(mc_resolution), 160) if device == "cpu" else int(mc_resolution)
        t0 = time.time()
        if progress:
            progress(8, 100, f"Initializing {TRIPOSR_MODEL} ({device.upper()})")
        model = self._ensure_triposr(device, progress=progress)

        try:
            if progress:
                progress(25, 100, "Isolating furniture subject from photoreal render")
            with Image.open(image_path) as src_img:
                fg = _clean_furniture_foreground(src_img.convert("RGB"), self._rembg_session)
                fg = resize_foreground(fg, TRIPOSR_FOREGROUND_RATIO)
                fg_arr = np.array(fg).astype(np.float32) / 255.0
                fg_rgb = fg_arr[:, :, :3] * fg_arr[:, :, 3:4] + (1.0 - fg_arr[:, :, 3:4]) * 0.5
                cond_image = Image.fromarray((fg_rgb * 255.0).astype(np.uint8))

            if progress:
                progress(42, 100, f"Inferring 3D triplane representation with {TRIPOSR_MODEL} ({device.upper()})")
            with torch.inference_mode():
                scene_codes = model([cond_image], device=device)

            if progress:
                progress(68, 100, f"Extracting 3D mesh via marching cubes ({effective_mc_res}³ grid)")

            can_bake = bake_texture and TRIPOSR_BAKE_TEXTURE and caps["primary"]["uv_texture_baking"]
            meshes = model.extract_mesh(
                scene_codes,
                has_vertex_color=False,
                resolution=effective_mc_res,
            )
            raw_mesh = meshes[0]

            if progress:
                progress(78, 100, "Smoothing mesh topology and computing surface normals")
            trimesh.smoothing.filter_taubin(raw_mesh, lamb=0.5, nu=-0.53, iterations=10)
            raw_mesh.fix_normals()

            srgb_colors_uint8 = None
            uv_mapped = False
            if can_bake:
                mesh = to_gradio_3d_orientation(raw_mesh)
                if progress:
                    progress(86, 100, "Baking UV texture atlas")
                try:
                    from app.pipeline.tsr.bake_texture import bake_texture as tsr_bake_texture

                    bake_out = tsr_bake_texture(mesh, model, scene_codes[0], TRIPOSR_TEXTURE_RESOLUTION)
                    tex_img = Image.fromarray((bake_out["colors"] * 255.0).astype(np.uint8)).transpose(
                        Image.FLIP_TOP_BOTTOM
                    )
                    material = trimesh.visual.material.PBRMaterial(
                        baseColorTexture=tex_img,
                        metallicFactor=0.05,
                        roughnessFactor=0.72,
                    )
                    visual = trimesh.visual.TextureVisuals(uv=bake_out["uvs"], material=material)
                    mesh = trimesh.Trimesh(
                        vertices=mesh.vertices[bake_out["vmapping"]],
                        faces=mesh.faces if len(bake_out["indices"]) == 0 else bake_out["indices"],
                        vertex_normals=mesh.vertex_normals[bake_out["vmapping"]],
                        visual=visual,
                        process=False,
                    )
                    uv_mapped = True
                except Exception as bake_err:
                    logger.warning(
                        "UV texture atlas baking failed (%s); seamlessly falling back to high-fidelity triplane vertex coloring.",
                        bake_err,
                    )
                    uv_mapped = False

            if not uv_mapped:
                if progress:
                    progress(86, 100, "Sampling high-clarity surface colors & smooth vertex normals")
                with torch.inference_mode():
                    verts_t = torch.from_numpy(
                        np.ascontiguousarray(raw_mesh.vertices, dtype=np.float32)
                    ).to(device)
                    decoded_rgb = (
                        model.renderer.query_triplane(model.decoder, verts_t, scene_codes[0])["color"]
                        .cpu()
                        .numpy()
                    )
                srgb_colors_uint8 = np.clip(np.round(decoded_rgb * 255.0), 0, 255).astype(np.uint8)
                linear_rgba = _srgb_to_linear_rgba(decoded_rgb)
                raw_mesh.visual = trimesh.visual.ColorVisuals(mesh=raw_mesh, vertex_colors=linear_rgba)
                mesh = to_gradio_3d_orientation(raw_mesh)
                _ = mesh.vertex_normals

            if progress:
                progress(94, 100, "Exporting real 3D mesh assets (GLB, GLTF, OBJ)")
            supported_formats = _export_trimesh_formats(
                mesh, generation_id, srgb_vertex_colors=srgb_colors_uint8
            )
            stats = _extract_mesh_stats(mesh, has_uv=uv_mapped, pbr_materials=uv_mapped)
            elapsed = round(time.time() - t0, 2)
        finally:
            if device == "cpu":
                self._triposr = None
                self._rembg_session = None
            gc.collect()

        return {
            "status": "ready",
            "engine_used": "TripoSR",
            "model_id": TRIPOSR_MODEL,
            "priority_tier": "primary",
            "device": device,
            "gpu_name": caps.get("gpu_name"),
            "vram_gb": caps.get("vram_gb"),
            "source_image_url": source_image_url,
            "glb_url": f"/outputs/models3d/{generation_id}/model.glb",
            "gltf_url": f"/api/3d/download/{generation_id}/gltf" if "gltf" in supported_formats else None,
            "obj_url": f"/api/3d/download/{generation_id}/obj" if "obj" in supported_formats else None,
            "supported_formats": supported_formats,
            "mc_resolution": effective_mc_res,
            "execution_time": elapsed,
            "proportion_note": (
                "Single-view AI geometry reconstruction estimates hidden back/underside surfaces "
                "and relative furniture proportions without claiming exact physical measurements."
            ),
            **stats,
        }

    def _run_hunyuan3d_fallback(
        self,
        image_path: Path,
        generation_id: str,
        source_image_url: str,
        seed: int,
        progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        caps = check_3d_capabilities()
        if not caps["fallback"]["available"]:
            raise Mesh3DIncompatibleError(" ".join(caps["fallback"]["blocking_reasons"]))

        import torch
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        t0 = time.time()
        if progress:
            progress(1, 5, f"Loading fallback {HUNYUAN3D_MODEL} on CUDA ({caps['gpu_name']})")

        if self._hy_shapegen is None:
            self._hy_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                HUNYUAN3D_MODEL,
                subfolder=HUNYUAN3D_SUBFOLDER,
                use_safetensors=True,
                device="cuda",
            )
        if self._hy_rembg is None:
            self._hy_rembg = BackgroundRemover()

        if progress:
            progress(2, 5, "Removing background from photoreal render")
        with Image.open(image_path) as src_img:
            rgba_img = src_img.convert("RGBA")
        rgba_img = self._hy_rembg(rgba_img)

        if progress:
            progress(3, 5, f"Generating 3D shape mesh with {HUNYUAN3D_MODEL}")
        mesh = self._hy_shapegen(
            image=rgba_img,
            num_inference_steps=HUNYUAN3D_STEPS,
            octree_resolution=HUNYUAN3D_OCTREE_RES,
            num_chunks=20000,
            generator=torch.manual_seed(int(seed)),
            output_type="trimesh",
        )[0]

        uv_mapped = False
        if HUNYUAN3D_ENABLE_TEXTURE and caps["vram_gb"] >= 10.0:
            try:
                if progress:
                    progress(4, 5, "Synthesizing UV texture with Hunyuan3D Paint")
                if self._hy_texgen is None:
                    from hy3dgen.texgen import Hunyuan3DPaintPipeline

                    self._hy_texgen = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
                mesh = self._hy_texgen(mesh, image=rgba_img)
                uv_mapped = True
            except Exception as tex_exc:
                logger.warning("Hunyuan3D texture pass skipped (shape preserved): %s", tex_exc)

        if progress:
            progress(5, 5, "Exporting real 3D mesh assets (GLB, GLTF, OBJ)")
        supported_formats = _export_trimesh_formats(mesh, generation_id)
        stats = _extract_mesh_stats(mesh, has_uv=uv_mapped, pbr_materials=uv_mapped)
        elapsed = round(time.time() - t0, 2)

        return {
            "status": "ready",
            "engine_used": "Hunyuan3D-2mini",
            "model_id": HUNYUAN3D_MODEL,
            "priority_tier": "fallback",
            "device": "cuda",
            "gpu_name": caps.get("gpu_name"),
            "vram_gb": caps.get("vram_gb"),
            "source_image_url": source_image_url,
            "glb_url": f"/outputs/models3d/{generation_id}/model.glb",
            "gltf_url": f"/api/3d/download/{generation_id}/gltf" if "gltf" in supported_formats else None,
            "obj_url": f"/api/3d/download/{generation_id}/obj" if "obj" in supported_formats else None,
            "supported_formats": supported_formats,
            "execution_time": elapsed,
            "proportion_note": (
                "Single-view AI geometry reconstruction estimates hidden back/underside surfaces "
                "and relative furniture proportions without claiming exact physical measurements."
            ),
            **stats,
        }

    @staticmethod
    def _save_metadata(generation_id: str, payload: Dict[str, Any]) -> None:
        """Persist reconstructed 3D metadata beside model.glb so spec_* items survive reloads."""
        try:
            meta_path = MODEL_3D_DIR / generation_id / "meta.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not write 3D metadata for %s: %s", generation_id, exc)

    def reconstruct_from_image(
        self,
        image_path: Path,
        generation_id: str,
        source_image_url: str,
        model_preference: str = "auto",
        mc_resolution: int = TRIPOSR_MC_RESOLUTION,
        bake_texture: bool = TRIPOSR_BAKE_TEXTURE,
        seed: int = 42,
        progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """
        Reconstruct a 3D mesh from a generated 2D photoreal render.

        Order for `auto`:
          1. TripoSR (`stabilityai/TripoSR`)
          2. Fallback to Hunyuan3D-2mini (`tencent/Hunyuan3D-2mini`) if TripoSR fails
        """
        with self._lock:
            if model_preference == "hunyuan3d":
                result = self._run_hunyuan3d_fallback(
                    image_path=image_path,
                    generation_id=generation_id,
                    source_image_url=source_image_url,
                    seed=seed,
                    progress=progress,
                )
                self._save_metadata(generation_id, result)
                return result

            primary_error = None
            try:
                result = self._run_triposr(
                    image_path=image_path,
                    generation_id=generation_id,
                    source_image_url=source_image_url,
                    mc_resolution=mc_resolution,
                    bake_texture=bake_texture,
                    progress=progress,
                )
                self._save_metadata(generation_id, result)
                return result
            except Exception as exc:
                primary_error = f"TripoSR ({TRIPOSR_MODEL}) failed: {exc}"
                logger.warning("%s", primary_error)
                if model_preference == "triposr":
                    raise Mesh3DGenerationError(primary_error) from exc

            if progress:
                progress(1, 5, f"TripoSR unavailable; falling back to {HUNYUAN3D_MODEL}")

            try:
                result = self._run_hunyuan3d_fallback(
                    image_path=image_path,
                    generation_id=generation_id,
                    source_image_url=source_image_url,
                    seed=seed,
                    progress=progress,
                )
                result["fallback_reason"] = primary_error
                self._save_metadata(generation_id, result)
                return result
            except Exception as fb_exc:
                raise Mesh3DGenerationError(
                    f"{primary_error} | Fallback {HUNYUAN3D_MODEL} failed: {fb_exc}"
                ) from fb_exc

