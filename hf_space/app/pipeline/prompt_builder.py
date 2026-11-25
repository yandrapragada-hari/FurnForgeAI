"""
Prompt construction for the diffusion model.

CLIP (SD 1.5's text encoder) prioritizes the first 77 tokens, so exact color,
material, furniture category, finish, and style come first with strong unambiguous
color descriptors and background contrast matching the selected palette swatch.
"""
from typing import Any, Dict, Optional

STYLE = {
    "Modern": "modern contemporary design, clean geometric form",
    "Minimalist": "minimalist japandi design, simple refined silhouette",
    "Traditional": "traditional classic craftsmanship, sculpted detailing",
    "Luxury": "luxury high-end designer piece, bespoke tailoring",
    "Scandinavian": "scandinavian nordic design, warm functional form",
    "Industrial": "industrial loft design, structural hardware",
}

MATERIAL = {
    "Wood": "crafted from solid natural wood timber",
    "Metal": "crafted with solid metal construction",
    "Glass": "crafted with clear tempered architectural glass",
    "Leather": "upholstered in genuine stitched leather",
    "Fabric": "upholstered in woven designer fabric textile",
}

FINISH = {
    "Matte": "smooth matte finish",
    "Glossy": "glossy lacquered finish with soft reflections",
    "Polished": "polished satin sheen",
    "Brushed": "fine brushed surface texture",
}

COLOR_PROMPTS = {
    "Warm Walnut": "rich dark walnut brown colored",
    "Charcoal Gray": "dark charcoal slate gray colored",
    "Deep Emerald": "deep emerald forest green colored",
    "Cognac Brown": "warm cognac tan caramel brown colored",
    "Cream Boucle": "warm cream ivory beige colored",
    "Matte Black": "jet black colored",
    "Champagne Gold": "warm champagne metallic gold colored",
    "Navy Blue": "deep dark navy blue colored",
    "Terracotta": "warm terracotta rust orange colored",
    "Scandinavian Birch": "pale blonde light birch beige colored",
    "Sage Green": "muted sage olive green colored",
    "White": "pure crisp bright white colored",
}

LIGHT_COLORS = {"White", "Cream Boucle", "Scandinavian Birch"}

NEGATIVE = (
    "sketch, drawing, wireframe, line art, chalk outline, white outline, glowing outline, "
    "pencil, ink lines, sticker, cutout, 2d, flat, cartoon, illustration, wallpaper, "
    "wood wall background, textured wall, speckled noise, grainy wall, extra legs, "
    "wood floor, parquet floor, reflective floor, mirror reflections, dark floor puddles, glossy flooring, "
    "deformed, distorted, bad proportions, cropped, text, watermark, logo, signature, people, clutter"
)


def build_prompt(
    category: str,
    material: str,
    color: str,
    style: str,
    finish: str,
    custom_prompt: str = "",
    cnn_features: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Build CLIP-compact prompts strictly honoring the user's selected category, material, and color."""
    color_phrase = COLOR_PROMPTS.get(color, f"{color.lower()} colored")
    backdrop = (
        "seamless soft warm medium-gray studio cyclorama background"
        if color in LIGHT_COLORS
        else "seamless smooth clean off-white studio cyclorama background"
    )

    parts = [
        f"studio product photograph of a {color_phrase} {material.lower()} {category.lower()}",
        f"{category.lower()} surface in {color_phrase} tone",
        MATERIAL.get(material, material.lower()),
        FINISH.get(finish, finish.lower()),
        STYLE.get(style, style.lower()),
    ]

    if cnn_features:
        accessories = cnn_features.get("detected_accessories") or []
        for acc in accessories[:1]:
            parts.append(acc)

    if custom_prompt and custom_prompt.strip():
        parts.append(custom_prompt.strip())

    parts.append(
        f"isolated furniture centered on {backdrop}, clean non-reflective matte studio floor, "
        "subtle soft ambient contact shadow only, no floor reflection, "
        "global illumination, realistic 3d architectural render, sharp focus"
    )
    return {"positive_prompt": ", ".join(parts), "negative_prompt": NEGATIVE}
