"""
Day/Night detection module for automatic parameter adjustment.

Determines whether an image was captured during day or night conditions
based on histogram statistics and brightness distribution.
"""

from typing import Dict, Any


def detect_day_night(stats: Dict[str, float], cfg: Dict[str, Any]) -> bool:
    """
    Detect if image was captured during day or night conditions.
    
    Uses a rule-based approach combining average brightness and dark pixel ratio
    to classify lighting conditions.
    
    Args:
        stats: Dictionary from global_hist_stats() containing image statistics
        cfg: Configuration dictionary with day_night parameters
        
    Returns:
        True if day time, False if night time
        
    Example:
        >>> stats = global_hist_stats(img)
        >>> is_day = detect_day_night(stats, config['day_night'])
        >>> print(f"Lighting condition: {'Day' if is_day else 'Night'}")
    """
    # Extract thresholds from config
    night_mean_thr = cfg.get('night_threshold_mean', 0.25)
    dark_ratio_thr = cfg.get('dark_ratio_threshold', 0.45)
    
    # Extract statistics
    mean_brightness = stats['mean']
    dark_ratio = stats['dark_ratio']
    
    # Day/night classification logic
    # Night if: low mean brightness OR high dark pixel ratio
    is_night = (mean_brightness < night_mean_thr) or (dark_ratio > dark_ratio_thr)
    
    return not is_night


def get_lighting_confidence(stats: Dict[str, float], cfg: Dict[str, Any]) -> float:
    """
    Get confidence score for day/night classification.
    
    Args:
        stats: Dictionary from global_hist_stats()
        cfg: Configuration dictionary with day_night parameters
        
    Returns:
        Confidence score between 0 and 1 (higher = more confident)
    """
    night_mean_thr = cfg.get('night_threshold_mean', 0.25)
    dark_ratio_thr = cfg.get('dark_ratio_threshold', 0.45)
    
    mean_brightness = stats['mean']
    dark_ratio = stats['dark_ratio']
    
    # Distance from decision boundary
    mean_dist = abs(mean_brightness - night_mean_thr) / night_mean_thr
    ratio_dist = abs(dark_ratio - dark_ratio_thr) / dark_ratio_thr
    
    # Combine distances (closer to boundary = lower confidence)
    confidence = min(mean_dist, ratio_dist)
    confidence = min(confidence, 1.0)
    
    return confidence


def analyze_lighting_conditions(stats: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive lighting condition analysis.
    
    Args:
        stats: Dictionary from global_hist_stats()
        cfg: Configuration dictionary with day_night parameters
        
    Returns:
        Dictionary with lighting analysis results
    """
    is_day = detect_day_night(stats, cfg)
    confidence = get_lighting_confidence(stats, cfg)
    
    # Additional lighting characteristics
    dynamic_range = stats['p99'] - stats['p01']
    is_low_contrast = dynamic_range < 0.3
    is_high_contrast = dynamic_range > 0.8
    
    # Exposure assessment
    if stats['mean'] < 0.2:
        exposure_type = "underexposed"
    elif stats['mean'] > 0.8:
        exposure_type = "overexposed"
    else:
        exposure_type = "normal"
    
    return {
        "is_day": is_day,
        "confidence": confidence,
        "lighting_type": "day" if is_day else "night",
        "exposure_type": exposure_type,
        "dynamic_range": dynamic_range,
        "is_low_contrast": is_low_contrast,
        "is_high_contrast": is_high_contrast,
        "mean_brightness": stats['mean'],
        "dark_pixel_ratio": stats['dark_ratio']
    }
