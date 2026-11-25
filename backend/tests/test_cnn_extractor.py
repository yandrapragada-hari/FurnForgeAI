from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from app.pipeline.cnn_extractor import (
    ACCESSORY_CUES,
    FURNITURE_CLASS_GROUPS,
    CNNFeatureExtractor,
)


def _create_symmetric_shape(size: int = 512) -> np.ndarray:
    """Create a laterally symmetric synthetic binary edge map."""
    img = np.zeros((size, size), dtype=np.uint8)
    # Centered rectangle
    cv2.rectangle(img, (size // 4, size // 4), (3 * size // 4, 3 * size // 4), 255, thickness=4)
    # Centered circle
    cv2.circle(img, (size // 2, size // 2), 60, 255, thickness=4)
    return img


def _create_asymmetric_shape(size: int = 512) -> np.ndarray:
    """Create an asymmetric edge map with features only on the left side."""
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(img, (40, 100), (140, 400), 255, thickness=4)
    cv2.circle(img, (90, 250), 30, 255, thickness=4)
    return img


class TestCNNFeatureExtractorClassMapping:
    def test_furniture_class_groups_completeness(self):
        expected_categories = {"Chair", "Sofa", "Table", "Bed", "Cabinet"}
        assert set(FURNITURE_CLASS_GROUPS.keys()) == expected_categories

        for cat, classes in FURNITURE_CLASS_GROUPS.items():
            assert isinstance(classes, list)
            assert len(classes) >= 2, f"Category {cat} should have at least 2 ImageNet class mappings"
            for cls_name in classes:
                assert isinstance(cls_name, str)
                assert len(cls_name.strip()) > 0

    def test_accessory_cues_mapping(self):
        assert "pillow" in ACCESSORY_CUES
        assert "table lamp" in ACCESSORY_CUES
        assert "pot" in ACCESSORY_CUES
        assert "vase" in ACCESSORY_CUES
        for phrase in ACCESSORY_CUES.values():
            assert isinstance(phrase, str)
            assert len(phrase) > 0


class TestCNNSymmetryAndHuMoments:
    def test_compute_symmetry_index_symmetric_vs_asymmetric(self):
        sym_img = _create_symmetric_shape(512)
        asym_img = _create_asymmetric_shape(512)

        sym_score = CNNFeatureExtractor.compute_symmetry_index(sym_img)
        asym_score = CNNFeatureExtractor.compute_symmetry_index(asym_img)

        assert sym_score >= 0.85, f"Expected high symmetry index for symmetric shape, got {sym_score}"
        assert asym_score <= 0.20, f"Expected low symmetry index for asymmetric shape, got {asym_score}"
        assert sym_score > asym_score

    def test_compute_symmetry_index_empty_and_small(self):
        empty_img = np.zeros((512, 512), dtype=np.uint8)
        assert CNNFeatureExtractor.compute_symmetry_index(empty_img) == 1.0

        tiny_img = np.zeros((1, 1), dtype=np.uint8)
        assert CNNFeatureExtractor.compute_symmetry_index(tiny_img) == 0.0

    def test_compute_hu_moments_shape_and_rotation_invariance(self):
        shape = _create_symmetric_shape(256)
        hu = CNNFeatureExtractor.compute_hu_moments(shape)

        assert isinstance(hu, np.ndarray)
        assert hu.shape == (7,)
        assert np.all(np.isfinite(hu))

        # Test rotation invariance (90 degree rotation)
        rotated = cv2.rotate(shape, cv2.ROTATE_90_CLOCKWISE)
        hu_rot = CNNFeatureExtractor.compute_hu_moments(rotated)

        # First Hu moment is rotation invariant
        assert hu[0] == pytest.approx(hu_rot[0], rel=0.1)

    def test_geometry_helper_outputs(self):
        sym_img = _create_symmetric_shape(512)
        geo = CNNFeatureExtractor._geometry(sym_img)

        assert "edge_density_percent" in geo
        assert "symmetry_score" in geo
        assert "aspect_ratio" in geo
        assert "contour_count" in geo
        assert "hu_moments" in geo

        assert geo["symmetry_score"] >= 0.85
        assert geo["contour_count"] >= 1
        assert len(geo["hu_moments"]) == 7


class TestCNNFeatureExtractorFallbackAndMocking:
    def test_graceful_handling_when_torch_unavailable(self):
        with patch.dict("sys.modules", {"torch": None, "torchvision": None, "torchvision.models": None}):
            extractor = CNNFeatureExtractor()
            # If torch is not available, extractor handles it gracefully
            assert extractor.ready is False
            assert extractor.error is not None
            with pytest.raises(RuntimeError, match="CNN not available"):
                extractor.analyze(np.zeros((512, 512, 3), dtype=np.uint8), np.zeros((512, 512), dtype=np.uint8))

    def test_mock_cpu_execution(self):
        """Test full analyze pipeline with mocked PyTorch model on CPU."""
        extractor = CNNFeatureExtractor.__new__(CNNFeatureExtractor)
        extractor.ready = True
        extractor.error = None
        extractor._lock = MagicMock()
        extractor._lock.__enter__ = MagicMock(return_value=None)
        extractor._lock.__exit__ = MagicMock(return_value=None)
        extractor.backbone_name = "resnet50 (Mock CPU)"
        extractor.class_names = [f"class_{i}" for i in range(1000)]
        extractor.class_names[0] = "rocking chair"
        extractor.class_names[1] = "studio couch"
        extractor.class_names[2] = "dining table"

        extractor.group_indices = {
            "Chair": [0],
            "Sofa": [1],
            "Table": [2],
            "Bed": [],
            "Cabinet": [],
        }

        # Mock torch
        mock_torch = MagicMock()
        mock_torch.enable_grad = MagicMock()

        # Simulated logits where 'rocking chair' (index 0) has the highest logit
        mock_logits = MagicMock()

        # Create mock tensor objects that support detach, sum, cpu, numpy
        probs_array = np.zeros(1000, dtype=np.float32)
        probs_array[0] = 0.8
        probs_array[1] = 0.1
        probs_array[2] = 0.1

        class MockTensor:
            def __init__(self, arr):
                self._arr = np.array(arr)

            def detach(self):
                return self

            def __getitem__(self, idx):
                if isinstance(idx, list):
                    return MockTensor(self._arr[idx])
                return float(self._arr[idx])

            def sum(self, dim=None):
                return float(self._arr.sum())

            def mean(self, dim=None):
                return MockTensor(np.mean(self._arr, axis=dim))

            def cpu(self):
                return self

            def numpy(self):
                return self._arr

        mock_torch.softmax.return_value = [MockTensor(probs_array)]

        # Mock topk
        class MockTopK:
            values = MagicMock(tolist=lambda: [0.8, 0.1, 0.1, 0.0, 0.0])
            indices = MagicMock(tolist=lambda: [0, 1, 2, 3, 4])

        mock_torch.topk.return_value = MockTopK()
        mock_torch.relu = lambda x: x

        extractor.torch = mock_torch
        extractor.preprocess = MagicMock(return_value=MagicMock(unsqueeze=lambda dim: MagicMock(requires_grad_=lambda b: None)))

        mock_acts = MockTensor(np.ones((2048, 7, 7), dtype=np.float32))

        # Mock forward pass populating _activations hook
        def model_forward(t):
            extractor._activations = MagicMock(detach=lambda: [mock_acts])
            extractor._gradients = None
            return mock_logits

        mock_model = MagicMock(side_effect=model_forward)
        mock_model.zero_grad = MagicMock()
        extractor.model = mock_model

        mock_logits[0, [0]].sum.return_value.backward = MagicMock()
        mock_torch.zeros.return_value = mock_acts

        sketch = np.full((512, 512, 3), 255, dtype=np.uint8)
        edge = np.zeros((512, 512), dtype=np.uint8)
        cv2.rectangle(edge, (100, 100), (400, 400), 255, 2)

        res = extractor.analyze(sketch, edge)

        assert isinstance(res, dict)
        assert res["detected_category"] == "Chair"
        assert "all_category_scores" in res
        assert "symmetry_score" in res
        assert "aspect_ratio" in res
        assert "gradcam" in res
        assert "pixel_metrics" in res
        assert res["pixel_metrics"]["total_pixels_scanned"] == 512 * 512
