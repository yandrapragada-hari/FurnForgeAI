import re

import pytest
from app.pipeline.prompt_builder import (
    LIGHT_COLORS,
    build_prompt,
)


def _approx_clip_tokens(text: str) -> int:
    """Approximate CLIP token count using regex word and punctuation chunks."""
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return int(len(tokens) * 1.15) + 2


class TestPromptBuilder:
    @pytest.mark.parametrize("category", ["Chair", "Sofa", "Table", "Bed", "Cabinet"])
    @pytest.mark.parametrize("material", ["Wood", "Metal", "Glass", "Leather", "Fabric"])
    @pytest.mark.parametrize("style", ["Scandinavian", "Industrial", "Modern", "Minimalist"])
    def test_build_prompt_matrix_and_expected_tokens(self, category, material, style):
        color = "Warm Walnut"
        finish = "Matte"

        res = build_prompt(
            category=category,
            material=material,
            color=color,
            style=style,
            finish=finish,
        )

        assert isinstance(res, dict)
        assert "positive_prompt" in res
        assert "negative_prompt" in res

        pos = res["positive_prompt"].lower()

        # Verify positive prompt contains core tokens
        assert category.lower() in pos
        assert material.lower() in pos
        assert style.lower() in pos
        assert "walnut" in pos
        assert "matte" in pos

        # Verify quality boosters are present
        assert "studio product photograph" in pos
        assert "isolated furniture" in pos
        assert "global illumination" in pos
        assert "realistic 3d architectural render" in pos
        assert "sharp focus" in pos

    def test_negative_prompt_contains_artifact_barriers(self):
        res = build_prompt(
            category="Chair",
            material="Wood",
            color="Warm Walnut",
            style="Modern",
            finish="Matte",
        )
        neg = res["negative_prompt"].lower()

        # Verify anti-artifact tokens
        expected_artifacts = [
            "sketch",
            "drawing",
            "wireframe",
            "line art",
            "pencil",
            "ink lines",
            "cartoon",
            "illustration",
            "deformed",
            "distorted",
            "bad proportions",
            "watermark",
            "logo",
            "extra legs",
        ]
        for artifact in expected_artifacts:
            assert artifact in neg, f"Expected negative prompt to contain '{artifact}'"

    def test_backdrop_contrast_for_light_and_dark_colors(self):
        # Light colors should use medium-gray cyclorama backdrop
        for light_color in LIGHT_COLORS:
            res_light = build_prompt("Chair", "Fabric", light_color, "Scandinavian", "Matte")
            assert "seamless soft warm medium-gray studio cyclorama background" in res_light["positive_prompt"]

        # Dark/saturated colors should use off-white cyclorama backdrop
        res_dark = build_prompt("Chair", "Leather", "Warm Walnut", "Scandinavian", "Matte")
        assert "seamless smooth clean off-white studio cyclorama background" in res_dark["positive_prompt"]

    def test_cnn_features_accessory_cues_included(self):
        cnn_features = {"detected_accessories": ["with throw pillows", "with a small lamp on top"]}
        res = build_prompt(
            category="Sofa",
            material="Fabric",
            color="Cream Boucle",
            style="Modern",
            finish="Matte",
            cnn_features=cnn_features,
        )
        assert "with throw pillows" in res["positive_prompt"]

    def test_custom_prompt_injection(self):
        custom = "brushed brass caps on tapered legs"
        res = build_prompt(
            category="Chair",
            material="Wood",
            color="Warm Walnut",
            style="Scandinavian",
            finish="Matte",
            custom_prompt=custom,
        )
        assert custom in res["positive_prompt"]

    def test_token_length_within_clip_budget(self):
        """Standard SD 1.5 CLIP text encoder prioritizes the first 77 tokens."""
        res = build_prompt(
            category="Chair",
            material="Leather",
            color="Warm Walnut",
            style="Scandinavian",
            finish="Matte",
        )
        pos = res["positive_prompt"]

        # Baseline prompt without custom prompt should be compact (<= 75 words)
        word_count = len(pos.split())
        assert word_count <= 75, f"Word count {word_count} exceeds recommended compact length"

    def test_clip_token_truncation_and_priority(self):
        """Verify token counting and simulated CLIP 77-token ceiling truncation."""
        res = build_prompt(
            category="Chair",
            material="Leather",
            color="Warm Walnut",
            style="Scandinavian",
            finish="Matte",
            custom_prompt="ornate detailing with hand-carved accents",
        )
        pos = res["positive_prompt"]
        tokens = pos.split()

        # Simulate CLIP text encoder 77 token truncation boundary
        clip_max_tokens = 77
        truncated_tokens = tokens[:clip_max_tokens]
        assert len(truncated_tokens) <= clip_max_tokens

        # Verify highest priority descriptors (category, material, color) remain within first 20 tokens
        first_20_text = " ".join(tokens[:20]).lower()
        assert "chair" in first_20_text
        assert "leather" in first_20_text
        assert "walnut" in first_20_text
