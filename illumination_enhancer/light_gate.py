"""
Light gate module for early exit decisions.

Determines if an image is already well-lit and only needs minor adjustments,
allowing for early exit from the full enhancement pipeline.
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2


def is_well_lit(stats: Dict[str, float], flag_day: bool, cfg: Dict[str, Any]) -> bool:
    """
    Determine if image is well-lit and needs minimal processing.
    
    Args:
        stats: Dictionary from global_hist_stats() 
        flag_day: True if daytime, False if nighttime
        cfg: Configuration dictionary with light_gate parameters
        
    Returns:
        True if image is well-lit (early exit), False if needs enhancement
        
    Example:
        >>> is_good = is_well_lit(stats, True, config['light_gate'])
        >>> if is_good:
        ...     return apply_minimal_enhancement(img)
    """
    # Extract thresholds from config
    mean_thr = cfg.get('well_lit_mean_threshold', 0.6)
    dark_ratio_thr = cfg.get('well_lit_dark_ratio_threshold', 0.2)
    
    # More lenient thresholds for nighttime
    if not flag_day:
        mean_thr *= 0.7  # Lower threshold for night
        dark_ratio_thr *= 1.5  # Allow more dark pixels at night
    
    # Check if image meets well-lit criteria
    is_bright_enough = stats['mean'] > mean_thr
    has_few_dark_pixels = stats['dark_ratio'] < dark_ratio_thr
    has_good_contrast = stats['contrast'] > 0.1  # Avoid flat images
    
    return is_bright_enough and has_few_dark_pixels and has_good_contrast


def apply_minimal_enhancement(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Apply minimal enhancement for well-lit images.
    
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) with
    conservative parameters to slightly improve contrast without artifacts.
    
    Args:
        img: RGB image with shape (H, W, 3) and dtype uint8
        cfg: Configuration dictionary with light_gate parameters
        
    Returns:
        Lightly enhanced RGB image
    """
    # Extract CLAHE parameters
    tile_grid = cfg.get('clahe_tile_grid_size', 8)
    clip_limit = cfg.get('clahe_clip_limit', 2.0)
    
    # Convert to LAB color space for better perceptual processing
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel = img_lab[:, :, 0]
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_enhanced = clahe.apply(l_channel)
    
    # Reconstruct LAB image
    img_lab_enhanced = img_lab.copy()
    img_lab_enhanced[:, :, 0] = l_enhanced
    
    # Convert back to RGB
    img_enhanced = cv2.cvtColor(img_lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return img_enhanced


def compute_enhancement_strength(stats: Dict[str, float], flag_day: bool) -> float:
    """
    Compute how much enhancement is needed based on lighting conditions.
    
    Args:
        stats: Dictionary from global_hist_stats()
        flag_day: True if daytime, False if nighttime
        
    Returns:
        Enhancement strength factor (0.0 = no enhancement, 1.0 = full enhancement)
    """
    mean_brightness = stats['mean']
    dark_ratio = stats['dark_ratio']
    
    # Base enhancement need
    brightness_factor = 1.0 - mean_brightness  # Darker images need more enhancement
    dark_factor = dark_ratio  # More dark pixels = more enhancement needed
    
    # Combine factors
    enhancement_need = (brightness_factor + dark_factor) / 2.0
    
    # Adjust for day/night
    if flag_day:
        # Day images with low brightness might be shadows/indoor
        enhancement_need *= 0.8
    else:
        # Night images naturally need more enhancement
        enhancement_need *= 1.2
    
    return min(enhancement_need, 1.0)


def should_skip_processing(img: np.ndarray, cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Comprehensive check if image processing should be skipped.
    
    Args:
        img: RGB image with shape (H, W, 3) and dtype uint8
        cfg: Full configuration dictionary
        
    Returns:
        Tuple of (should_skip, reason)
    """
    from .stats import global_hist_stats
    
    # Get image statistics
    stats = global_hist_stats(img)
    
    # Detect day/night
    flag_day = detect_day_night(stats, cfg.get('day_night', {}))
    
    # Check if well-lit
    well_lit = is_well_lit(stats, flag_day, cfg.get('light_gate', {}))
    
    if well_lit:
        return True, "Image is already well-lit"
    
    # Check for edge cases
    if stats['mean'] > 0.95:
        return True, "Image is already very bright"
        
    if stats['contrast'] < 0.05:
        return True, "Image has very low contrast (possibly corrupted)"
    
    return False, "Image needs enhancement"


# Import here to avoid circular dependency
def detect_day_night(stats: Dict[str, float], cfg: Dict[str, Any]) -> bool:
    """Import from day_night module to avoid circular dependency."""
    from .day_night import detect_day_night as _detect_day_night
    return _detect_day_night(stats, cfg)
