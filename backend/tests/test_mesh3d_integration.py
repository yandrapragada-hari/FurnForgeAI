"""Tests for TripoSR (Priority 1) + Hunyuan3D-2mini (Fallback) 3D reconstruction."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import OUTPUT_DIR
from app.database.db import db
from app.main import app
from app.pipeline.mesh3d_engine import (
    Mesh3DEngine,
    PhotorealInputError,
    check_3d_capabilities,
    resolve_photoreal_render_path,
)

client = TestClient(app)


def _find_existing_generated_record():
    for item in db.get_all(limit=60):
        rel = (item.get("result_image_url") or "").lstrip("/")
        if rel.startswith("outputs/"):
            fname = rel.split("/", 1)[1]
            if (OUTPUT_DIR / fname).exists():
                return item
    return None


def test_3d_requirements_reports_priority_chain():
    res = client.get("/api/3d/requirements")
    assert res.status_code == 200
    data = res.json()
    assert data["priority_order"] == ["triposr", "hunyuan3d"]
    assert data["primary"]["model_id"] == "stabilityai/TripoSR"
    assert data["fallback"]["model_id"] == "tencent/Hunyuan3D-2mini"
    if data["ready"]:
        assert data["primary"]["available"] is True


def test_photoreal_input_validation_rejects_sketch_and_edge_map():
    record = _find_existing_generated_record()
    if record is None:
        pytest.skip("No existing generated furniture render in gallery DB.")

    path, url = resolve_photoreal_render_path(record, requested_image_url=record["result_image_url"])
    assert path.exists()
    assert url == record["result_image_url"]

    # Support full public URLs (e.g. from Cloudflare tunnels or custom backend domains)
    full_tunnel_url = f"https://requires-accurately-makeup-backup.trycloudflare.com{record['result_image_url']}"
    path_full, url_full = resolve_photoreal_render_path(record, requested_image_url=full_tunnel_url)
    assert path_full.exists()
    assert url_full == record["result_image_url"]

    for forbidden in (
        record["original_image_url"],
        record["preprocessed_edge_url"],
        "/sample_sketches/sofa_sketch.png",
        "/outputs/edge_sofa.png",
        "/outputs/../config.py",
    ):
        try:
            resolve_photoreal_render_path(record, requested_image_url=forbidden)
            raise AssertionError(f"Expected PhotorealInputError for {forbidden}")
        except PhotorealInputError:
            pass


def test_priority_order_falls_back_from_triposr_to_hunyuan3d(tmp_path):
    dummy_img = tmp_path / "render.png"
    dummy_img.write_bytes(b"fake")
    engine = Mesh3DEngine()

    calls = []

    def fail_triposr(*args, **kwargs):
        calls.append("triposr")
        raise RuntimeError("Simulated TripoSR error")

    def ok_hunyuan(*args, **kwargs):
        calls.append("hunyuan3d")
        return {"status": "ready", "engine_used": "Hunyuan3D-2mini", "priority_tier": "fallback"}

    with patch.object(engine, "_run_triposr", side_effect=fail_triposr), patch.object(
        engine, "_run_hunyuan3d_fallback", side_effect=ok_hunyuan
    ):
        res = engine.reconstruct_from_image(
            image_path=dummy_img,
            generation_id="test_fb",
            source_image_url="/outputs/render.png",
            model_preference="auto",
        )

    assert calls == ["triposr", "hunyuan3d"]
    assert res["engine_used"] == "Hunyuan3D-2mini"
    assert "Simulated TripoSR error" in res["fallback_reason"]


def test_3d_endpoints_input_and_error_handling():
    record = _find_existing_generated_record()
    if record is None:
        pytest.skip("No existing generated furniture render in gallery DB.")

    bad = client.post(
        "/api/3d/generate",
        json={
            "generation_id": record["id"],
            "image_url": record["preprocessed_edge_url"],
        },
    )
    assert bad.status_code == 422

    # Explicitly requesting CUDA-only fallback on CPU-only machine returns clear 503
    caps = check_3d_capabilities()
    if not caps["fallback"]["available"]:
        fb = client.post(
            "/api/3d/generate",
            json={
                "generation_id": record["id"],
                "image_url": record["result_image_url"],
                "model_preference": "hunyuan3d",
            },
        )
        assert fb.status_code == 503

    assert client.get("/api/3d/status/nonexistent_id").status_code == 404
    assert client.get("/api/3d/download/furn_demo/fbx").status_code == 400
    assert client.get("/api/3d/download/nonexistent_id/glb").status_code == 404


def test_clean_furniture_foreground_prunes_border_wall_bleed():
    cupboard_path = OUTPUT_DIR / "render_furn_3567855e57_0.png"
    if not cupboard_path.exists():
        pytest.skip("Test render image not present")

    try:
        import rembg
    except ImportError:
        pytest.skip("rembg not installed")

    import numpy as np
    from app.pipeline.mesh3d_engine import _clean_furniture_foreground
    from PIL import Image

    with Image.open(cupboard_path) as im:
        cleaned = _clean_furniture_foreground(im.convert("RGB"), rembg.new_session("u2net"))
    alpha = np.array(cleaned)[:, :, 3]
    assert np.mean(alpha[:, -6:] > 20) < 0.02
    assert np.mean(alpha[:, :6] > 20) < 0.02

