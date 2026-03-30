"""Programmatic evaluation metrics for the MAS sycophancy experiment."""
from src.metrics.flip_metrics import compute_nof, compute_tof
from src.metrics.linguistic import count_deference_markers, measure_semantic_compression
from src.metrics.sycophancy_effect import compute_delta_squared
from src.metrics.trail import TrailCategory, categorize_failure

__all__ = [
    "compute_delta_squared",
    "compute_tof",
    "compute_nof",
    "TrailCategory",
    "categorize_failure",
    "count_deference_markers",
    "measure_semantic_compression",
]
