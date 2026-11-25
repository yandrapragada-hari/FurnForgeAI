"""
FurnForge AI - Evaluation & Ablation Study Suite
================================================

Comprehensive academic evaluation metrics, ablation benchmarking,
and hardware profiling tools for sketch-to-3D furniture synthesis.
"""

from .metrics import (
    calculate_edge_iou,
    calculate_chamfer_edge_distance,
    evaluate_3d_mesh,
    calculate_clip_similarity,
    extract_contour_points,
)
from .ablation_study import AblationStudy, run_ablation_experiment
from .benchmark import BenchmarkRunner, run_benchmark

__all__ = [
    "calculate_edge_iou",
    "calculate_chamfer_edge_distance",
    "evaluate_3d_mesh",
    "calculate_clip_similarity",
    "extract_contour_points",
    "AblationStudy",
    "run_ablation_experiment",
    "BenchmarkRunner",
    "run_benchmark",
]
