import base64

import cv2
import numpy as np
from app.config import APP_NAME
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def _generate_mock_canvas_base64(size: int = 256) -> str:
    """Generate a mock base64 canvas image with a drawn furniture sketch."""
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    # Draw simple chair outline
    cv2.rectangle(canvas, (size // 4, size // 3), (3 * size // 4, 2 * size // 3), (0, 0, 0), 3)
    cv2.line(canvas, (size // 4, 2 * size // 3), (size // 4, size - 20), (0, 0, 0), 3)
    cv2.line(canvas, (3 * size // 4, 2 * size // 3), (3 * size // 4, size - 20), (0, 0, 0), 3)
    _, buf = cv2.imencode(".png", canvas)
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


class TestAPIEndpoints:
    def test_get_health(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "ok"
        assert data["app"] == APP_NAME
        assert "version" in data
        assert "device" in data
        assert "model" in data
        assert "cnn" in data
        assert "mesh3d" in data
        assert isinstance(data["queue_length"], int)

        # 3D mesh metadata
        mesh3d = data["mesh3d"]
        assert "primary_model" in mesh3d
        assert "fallback_model" in mesh3d
        assert "ready" in mesh3d
        assert "cuda_available" in mesh3d

    def test_get_samples(self):
        response = client.get("/api/samples")
        assert response.status_code == 200
        data = response.json()

        assert "samples" in data
        assert isinstance(data["samples"], list)
        for sample in data["samples"]:
            assert "id" in sample
            assert "name" in sample
            assert "category" in sample
            assert "url" in sample
            assert sample["url"].startswith("/sample_sketches/")

    def test_post_preprocess_valid_sketch(self):
        b64 = _generate_mock_canvas_base64(256)
        payload = {
            "image_base64": b64,
            "low_threshold": 100,
            "high_threshold": 200,
            "blur_kernel": 3,
            "method": "canny",
        }
        response = client.post("/api/preprocess", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "edge_map_base64" in data
        assert "scribble_map_base64" in data
        assert "display_sketch_base64" in data
        assert "edge_density_percent" in data
        assert "contour_count" in data
        assert data["edge_map_base64"].startswith("data:image/png;base64,")
        assert data["scribble_map_base64"].startswith("data:image/png;base64,")
        assert data["display_sketch_base64"].startswith("data:image/png;base64,")
        assert data["edge_density_percent"] >= 0.0
        assert data["contour_count"] >= 0

    def test_post_preprocess_invalid_base64_returns_error(self):
        payload = {"image_base64": "invalid_not_base64_data"}
        response = client.post("/api/preprocess", json=payload)
        assert response.status_code in (400, 422, 500)

    def test_get_3d_requirements(self):
        response = client.get("/api/3d/requirements")
        assert response.status_code == 200
        data = response.json()

        assert "ready" in data
        assert "cuda_available" in data
        assert "priority_order" in data
        assert data["priority_order"] == ["triposr", "hunyuan3d"]
        assert "primary" in data
        assert data["primary"]["name"] == "TripoSR"
        assert data["primary"]["model_id"] == "stabilityai/TripoSR"
        assert "fallback" in data
        assert data["fallback"]["name"] == "Hunyuan3D-2mini"
        assert data["fallback"]["model_id"] == "tencent/Hunyuan3D-2mini"

    def test_nonexistent_endpoints_return_404(self):
        res_get = client.get("/api/nonexistent_route_test_12345")
        assert res_get.status_code == 404

        res_post = client.post("/api/another_missing_endpoint", json={"foo": "bar"})
        assert res_post.status_code == 404
