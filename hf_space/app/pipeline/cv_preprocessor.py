import base64
from typing import Any, Dict, Tuple

import cv2
import numpy as np


class OpenCVPreprocessor:
    """
    OpenCV Computer Vision Preprocessing Pipeline for FurnForge AI.
    Performs flat-field paper illumination cleanup, outer frame/border suppression,
    aspect-ratio preserved padding, bilateral denoising, Canny edge detection,
    and clean scribble stroke extraction for ControlNet conditioning.
    """

    @staticmethod
    def decode_image_base64(base64_str: str) -> np.ndarray:
        """Decode a base64 string to a BGR OpenCV numpy array."""
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError("Failed to decode image from base64 string.")

        # Composite transparent PNGs onto pure white paper background
        if len(img.shape) == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3] / 255.0
            bgr = img[:, :, :3]
            white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
            img = (bgr * alpha[:, :, None] + white_bg * (1.0 - alpha[:, :, None])).astype(np.uint8)
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img

    @staticmethod
    def encode_image_base64(img_bgr_or_gray: np.ndarray, format: str = ".png") -> str:
        """Encode OpenCV image array to base64 data URL."""
        success, buffer = cv2.imencode(format, img_bgr_or_gray)
        if not success:
            raise ValueError("Failed to encode image to base64.")
        b64_str = base64.b64encode(buffer).decode("utf-8")
        mime = "image/png" if format == ".png" else "image/jpeg"
        return f"data:{mime};base64,{b64_str}"

    @staticmethod
    def _clean_light_paper_before_pad(bgr_img: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Normalize uneven paper illumination (e.g. smartphone photos of paper sketches)
        to pure white (255) before canvas padding so grey paper edges never create
        fake rectangular border lines.
        """
        h0, w0 = bgr_img.shape[:2]
        gray0 = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        is_light_bg = float(np.median(gray0)) > 100.0

        if not is_light_bg:
            return bgr_img, False

        k = max(31, (min(h0, w0) // 10) | 1)
        bg = cv2.GaussianBlur(gray0, (k, k), 0)
        flat = cv2.divide(gray0, bg, scale=255)
        flat[flat > 228] = 255

        # Remove outer 2% camera/screenshot rim artifacts
        my, mx = max(4, int(h0 * 0.02)), max(4, int(w0 * 0.02))
        flat[:my, :] = 255
        flat[-my:, :] = 255
        flat[:, :mx] = 255
        flat[:, -mx:] = 255

        return cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR), True

    @staticmethod
    def _strip_margin_frame_lines(mask: np.ndarray, pad_meta: Dict[str, Any]) -> np.ndarray:
        """
        Remove long straight page-border / UI-card border lines sitting in the outer
        16% frame margins or right along padding seams without affecting interior furniture.
        """
        cleaned = mask.copy()
        h, w = cleaned.shape[:2]

        # Clear thin seam band around padding borders
        for seam, axis in [
            (pad_meta["pad_top"], "h"),
            (h - 1 - pad_meta["pad_bottom"], "h"),
            (pad_meta["pad_left"], "v"),
            (w - 1 - pad_meta["pad_right"], "v"),
        ]:
            if 2 < seam < (h - 2 if axis == "h" else w - 2):
                if axis == "h":
                    cleaned[max(0, seam - 3) : min(h, seam + 4), :] = 0
                else:
                    cleaned[:, max(0, seam - 3) : min(w, seam + 4)] = 0

        # Detect long straight frame/notebook edges near outer 16% rim
        lines = cv2.HoughLinesP(
            cleaned, 1, np.pi / 180, threshold=100,
            minLineLength=int(0.55 * min(h, w)), maxLineGap=24
        )
        if lines is not None:
            mx, my = int(0.16 * w), int(0.16 * h)
            for line in lines.reshape(-1, 4):
                x1, y1, x2, y2 = line
                is_vert_margin = abs(x1 - x2) < 12 and (x1 < mx or x1 > w - mx)
                is_horiz_margin = abs(y1 - y2) < 12 and (y1 < my or y1 > h - my)
                if is_vert_margin or is_horiz_margin:
                    cv2.line(cleaned, (x1, y1), (x2, y2), 0, thickness=9)

        # Despeckle tiny isolated dust dots (< 18 px)
        num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
        despeckled = np.zeros_like(cleaned)
        for lbl in range(1, num_lbl):
            if stats[lbl, cv2.CC_STAT_AREA] >= 18:
                despeckled[labels == lbl] = 255
        return despeckled

    @staticmethod
    def resize_and_pad(
        img: np.ndarray, target_size: int = 512, pad_color: int = 255
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resize image maintaining aspect ratio and pad onto a square canvas."""
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        pad_top = (target_size - new_h) // 2
        pad_bottom = target_size - new_h - pad_top
        pad_left = (target_size - new_w) // 2
        pad_right = target_size - new_w - pad_left

        color_val = [pad_color, pad_color, pad_color] if len(img.shape) == 3 else pad_color
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=color_val
        )

        meta = {
            "original_width": w,
            "original_height": h,
            "scaled_width": new_w,
            "scaled_height": new_h,
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
            "scale_factor": round(scale, 4),
        }
        return padded, meta

    @classmethod
    def process_sketch(
        cls,
        image_input,
        target_size: int = 512,
        low_threshold: int = 100,
        high_threshold: int = 200,
        blur_kernel: int = 3,
        method: str = "canny",
        invert_if_dark: bool = True,
    ) -> Dict[str, Any]:
        """
        Full OpenCV preprocessing pipeline returning:
        - padded_bgr: illumination-normalized 512x512 sketch for CNN / img2img
        - edge_map: despeckled Canny edge map (white on black)
        - scribble_map: clean single-line ink scribble map (ControlNet Scribble ready)
        - display_sketch: crisp dark lines on white canvas for UI display
        """
        if isinstance(image_input, str):
            bgr_img = cls.decode_image_base64(image_input)
        elif isinstance(image_input, np.ndarray):
            bgr_img = image_input
        else:
            raise TypeError("Expected base64 string or numpy ndarray.")

        cleaned_raw, is_light_bg = cls._clean_light_paper_before_pad(bgr_img)
        pad_col = 255 if is_light_bg else 0
        padded_bgr, pad_meta = cls.resize_and_pad(cleaned_raw, target_size=target_size, pad_color=pad_col)

        gray = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=45, sigmaSpace=45)

        # 1) Extract clean single-stroke ink mask for ControlNet Scribble
        ink_diff = np.clip((242.0 - denoised.astype(np.float32)) if is_light_bg else denoised.astype(np.float32), 0, 255)
        fg = ink_diff[ink_diff > 18]
        scale_top = float(np.percentile(fg, 90)) if fg.size > 10 else 60.0
        ink_norm = np.clip((ink_diff - 18.0) * (255.0 / max(40.0, scale_top)), 0, 255).astype(np.uint8)
        _, scribble_bin = cv2.threshold(ink_norm, 45, 255, cv2.THRESH_BINARY)
        scribble_bin = cls._strip_margin_frame_lines(scribble_bin, pad_meta)

        scribble_map = cv2.dilate(scribble_bin, np.ones((2, 2), np.uint8), iterations=1)
        scribble_map = cv2.GaussianBlur(scribble_map, (3, 3), 0)

        # 2) Also extract Canny / adaptive / lineart edge map cleanly
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)
        if method == "adaptive_threshold":
            flag = cv2.THRESH_BINARY_INV if is_light_bg else cv2.THRESH_BINARY
            edge_map = cv2.adaptiveThreshold(
                contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, 15, 4
            )
        elif method == "lineart":
            edge_map = scribble_bin.copy()
        else:
            k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            blur = cv2.GaussianBlur(contrast_enhanced, (k, k), 0)
            edge_map = cv2.Canny(blur, low_threshold, high_threshold)

        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edge_map = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, morph_kernel)
        edge_map = cls._strip_margin_frame_lines(edge_map, pad_meta)

        # Ensure padded_bgr matches the border-stripped ink for clean CNN/Grad-CAM view
        if is_light_bg:
            clean_canvas = np.where(scribble_bin > 0, gray, 255).astype(np.uint8)
            padded_bgr = cv2.cvtColor(clean_canvas, cv2.COLOR_GRAY2BGR)

        total_pixels = target_size * target_size
        white_pixels = int(np.count_nonzero(scribble_bin))
        edge_density = round((white_pixels / total_pixels) * 100, 2)

        contours, _ = cv2.findContours(scribble_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)
        display_sketch = 255 - scribble_map

        return {
            "padded_bgr": padded_bgr,
            "edge_map": edge_map,
            "scribble_map": scribble_map,
            "display_sketch": display_sketch,
            "base64_edge_map": cls.encode_image_base64(edge_map),
            "base64_scribble_map": cls.encode_image_base64(scribble_map),
            "base64_display_sketch": cls.encode_image_base64(display_sketch),
            "base64_original_padded": cls.encode_image_base64(padded_bgr),
            "edge_density_percent": edge_density,
            "contour_count": contour_count,
            "pad_metadata": pad_meta,
            "detected_bg": "light (paper)" if is_light_bg else "dark",
        }
