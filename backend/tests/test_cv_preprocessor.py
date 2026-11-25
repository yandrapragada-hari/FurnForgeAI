import base64

import cv2
import numpy as np
import pytest
from app.pipeline.cv_preprocessor import OpenCVPreprocessor


def _create_synthetic_image(width: int = 512, height: int = 512, channels: int = 3, bg_color: int = 255) -> np.ndarray:
    """Create a synthetic test image with a simple drawn shape."""
    if channels == 1:
        img = np.full((height, width), bg_color, dtype=np.uint8)
        # Draw a black rectangle (stroke)
        cv2.rectangle(img, (width // 4, height // 4), (3 * width // 4, 3 * height // 4), 0, thickness=4)
        return img
    elif channels == 4:
        img = np.full((height, width, 4), [bg_color, bg_color, bg_color, 255], dtype=np.uint8)
        # Transparent border or partially transparent
        cv2.rectangle(img, (width // 4, height // 4), (3 * width // 4, 3 * height // 4), (0, 0, 0, 255), thickness=4)
        return img
    else:
        img = np.full((height, width, 3), bg_color, dtype=np.uint8)
        # Draw a black rectangle (stroke) simulating a furniture sketch
        cv2.rectangle(img, (width // 4, height // 4), (3 * width // 4, 3 * height // 4), (0, 0, 0), thickness=4)
        return img


def _to_base64_str(img: np.ndarray, ext: str = ".png", with_prefix: bool = True) -> str:
    """Encode an OpenCV image to base64 string with optional data URI prefix."""
    success, buffer = cv2.imencode(ext, img)
    assert success
    b64_str = base64.b64encode(buffer).decode("utf-8")
    if with_prefix:
        mime = "image/png" if ext == ".png" else "image/jpeg"
        return f"data:{mime};base64,{b64_str}"
    return b64_str


class TestOpenCVPreprocessorDecode:
    def test_decode_valid_png_base64_with_data_prefix(self):
        img_orig = _create_synthetic_image(200, 150)
        b64 = _to_base64_str(img_orig, ".png", with_prefix=True)
        decoded = OpenCVPreprocessor.decode_image_base64(b64)

        assert isinstance(decoded, np.ndarray)
        assert decoded.shape == (150, 200, 3)
        assert decoded.dtype == np.uint8

    def test_decode_valid_png_raw_base64(self):
        img_orig = _create_synthetic_image(120, 80)
        b64 = _to_base64_str(img_orig, ".png", with_prefix=False)
        decoded = OpenCVPreprocessor.decode_image_base64(b64)

        assert isinstance(decoded, np.ndarray)
        assert decoded.shape == (80, 120, 3)
        assert decoded.dtype == np.uint8

    def test_decode_valid_jpg_base64(self):
        img_orig = _create_synthetic_image(180, 180)
        b64 = _to_base64_str(img_orig, ".jpg", with_prefix=True)
        decoded = OpenCVPreprocessor.decode_image_base64(b64)

        assert isinstance(decoded, np.ndarray)
        assert decoded.shape == (180, 180, 3)
        assert decoded.dtype == np.uint8

    def test_decode_transparent_png(self):
        """Alpha channel should be composited onto white paper background (255)."""
        img_rgba = _create_synthetic_image(100, 100, channels=4)
        b64 = _to_base64_str(img_rgba, ".png", with_prefix=True)
        decoded = OpenCVPreprocessor.decode_image_base64(b64)

        assert decoded.ndim == 3
        assert decoded.shape == (100, 100, 3)
        assert decoded.dtype == np.uint8
        # Corners should be pure white due to alpha compositing
        assert np.all(decoded[0, 0] == [255, 255, 255])

    def test_decode_invalid_base64_raises_exception(self):
        with pytest.raises(ValueError):
            OpenCVPreprocessor.decode_image_base64("this_is_not_base64!@#$%^&*()")

    def test_decode_corrupted_payload_raises_value_error(self):
        corrupt_b64 = base64.b64encode(b"not a valid image byte stream").decode("utf-8")
        with pytest.raises(ValueError):
            OpenCVPreprocessor.decode_image_base64(corrupt_b64)

    def test_encode_and_decode_cycle(self):
        img_orig = _create_synthetic_image(100, 100)
        encoded = OpenCVPreprocessor.encode_image_base64(img_orig, format=".png")
        assert encoded.startswith("data:image/png;base64,")
        decoded = OpenCVPreprocessor.decode_image_base64(encoded)
        assert decoded.shape == img_orig.shape
        assert np.array_equal(decoded, img_orig)


class TestProcessSketchPipeline:
    def test_process_sketch_with_synthetic_512x512_ndarray(self):
        sketch = _create_synthetic_image(512, 512, bg_color=255)
        result = OpenCVPreprocessor.process_sketch(sketch, target_size=512)

        assert isinstance(result, dict)
        assert "padded_bgr" in result
        assert "edge_map" in result
        assert "scribble_map" in result
        assert "display_sketch" in result
        assert "base64_edge_map" in result
        assert "pad_metadata" in result

        # Verify shapes
        assert result["padded_bgr"].shape == (512, 512, 3)
        assert result["edge_map"].shape[:2] == (512, 512)
        assert result["scribble_map"].shape[:2] == (512, 512)
        assert result["display_sketch"].shape[:2] == (512, 512)

        # Verify 3D compatible representation shape
        edge_bgr = cv2.cvtColor(result["edge_map"], cv2.COLOR_GRAY2BGR)
        assert edge_bgr.shape == (512, 512, 3)

        # Verify data type and value range
        assert result["edge_map"].dtype == np.uint8
        assert result["padded_bgr"].dtype == np.uint8
        assert 0 <= result["edge_map"].min() <= 255
        assert 0 <= result["edge_map"].max() <= 255
        assert 0 <= result["padded_bgr"].min() <= 255
        assert 0 <= result["padded_bgr"].max() <= 255

        # Verify density and contours
        assert isinstance(result["edge_density_percent"], (int, float))
        assert result["edge_density_percent"] >= 0.0
        assert isinstance(result["contour_count"], int)
        assert result["contour_count"] >= 0

    def test_process_sketch_with_base64_input(self):
        sketch = _create_synthetic_image(512, 512, bg_color=255)
        b64 = _to_base64_str(sketch)
        result = OpenCVPreprocessor.process_sketch(b64, target_size=512)

        assert result["padded_bgr"].shape == (512, 512, 3)
        assert result["edge_map"].shape[:2] == (512, 512)
        assert result["base64_edge_map"].startswith("data:image/png;base64,")

    def test_process_sketch_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError):
            OpenCVPreprocessor.process_sketch(12345)

    def test_process_sketch_methods(self):
        sketch = _create_synthetic_image(512, 512, bg_color=255)
        for method in ("canny", "adaptive_threshold", "lineart"):
            res = OpenCVPreprocessor.process_sketch(sketch, target_size=512, method=method)
            assert res["edge_map"].shape[:2] == (512, 512)
            assert res["edge_map"].dtype == np.uint8


class TestLetterboxingAndPadding:
    def test_horizontal_aspect_ratio_padding(self):
        """Wide image (800x400) should be letterboxed with top and bottom padding."""
        wide_img = _create_synthetic_image(width=800, height=400)
        padded, meta = OpenCVPreprocessor.resize_and_pad(wide_img, target_size=512, pad_color=255)

        assert padded.shape == (512, 512, 3)
        assert meta["original_width"] == 800
        assert meta["original_height"] == 400
        assert meta["scaled_width"] == 512
        assert meta["scaled_height"] == 256
        assert meta["pad_top"] > 0
        assert meta["pad_bottom"] > 0
        assert meta["pad_left"] == 0
        assert meta["pad_right"] == 0
        assert meta["scaled_height"] + meta["pad_top"] + meta["pad_bottom"] == 512

        # Verify aspect ratio preservation
        orig_ar = 800 / 400
        scaled_ar = meta["scaled_width"] / meta["scaled_height"]
        assert scaled_ar == pytest.approx(orig_ar, abs=0.02)

        # Check padding color at top/bottom border
        assert np.all(padded[0, 256] == [255, 255, 255])
        assert np.all(padded[511, 256] == [255, 255, 255])

    def test_vertical_aspect_ratio_padding(self):
        """Tall image (300x600) should be pillarboxed with left and right padding."""
        tall_img = _create_synthetic_image(width=300, height=600)
        padded, meta = OpenCVPreprocessor.resize_and_pad(tall_img, target_size=512, pad_color=0)

        assert padded.shape == (512, 512, 3)
        assert meta["original_width"] == 300
        assert meta["original_height"] == 600
        assert meta["scaled_height"] == 512
        assert meta["scaled_width"] == 256
        assert meta["pad_left"] > 0
        assert meta["pad_right"] > 0
        assert meta["pad_top"] == 0
        assert meta["pad_bottom"] == 0
        assert meta["scaled_width"] + meta["pad_left"] + meta["pad_right"] == 512

        # Verify aspect ratio preservation
        orig_ar = 300 / 600
        scaled_ar = meta["scaled_width"] / meta["scaled_height"]
        assert scaled_ar == pytest.approx(orig_ar, abs=0.02)

        # Check black padding color at left/right border
        assert np.all(padded[256, 0] == [0, 0, 0])
        assert np.all(padded[256, 511] == [0, 0, 0])

    def test_square_image_padding(self):
        """Square image (256x256) scaled to 512 should have zero padding."""
        sq_img = _create_synthetic_image(width=256, height=256)
        padded, meta = OpenCVPreprocessor.resize_and_pad(sq_img, target_size=512)

        assert padded.shape == (512, 512, 3)
        assert meta["scaled_width"] == 512
        assert meta["scaled_height"] == 512
        assert meta["pad_top"] == 0
        assert meta["pad_bottom"] == 0
        assert meta["pad_left"] == 0
        assert meta["pad_right"] == 0
