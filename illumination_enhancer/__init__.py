"""
Zero-Reference Illumination Enhancement Pipeline

A PyTorch-based package for intelligent image illumination enhancement
using region-aware processing and zero-reference training.
"""

__version__ = "1.0.0"
__author__ = "Computer Vision Team"

from .io import read_image, save_image
from .stats import global_hist_stats
from .day_night import detect_day_night
from .light_gate import is_well_lit
from .illumination_repr import make_dual_repr
from .illum_map import estimate_map
from .region_classifier import classify_patches
from .curve_estim import adaptive_curve
from .overexposed import handle_overexposed
from .fusion import fuse_regions

__all__ = [
    "read_image",
    "save_image", 
    "global_hist_stats",
    "detect_day_night",
    "is_well_lit",
    "make_dual_repr",
    "estimate_map",
    "classify_patches", 
    "adaptive_curve",
    "handle_overexposed",
    "fuse_regions"
]
