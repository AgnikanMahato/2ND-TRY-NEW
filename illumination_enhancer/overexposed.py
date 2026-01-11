"""
Overexposed region handling module.

Provides multiple strategies for handling overexposed regions including
clipping, tone mapping, and detail recovery using guided filtering.
"""

from typing import Dict, Any, Optional
import numpy as np
import cv2


def handle_overexposed(img: np.ndarray, mask_over: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Handle overexposed regions using configured method.
    
    Args:
        img: RGB image with shape (H, W, 3) and dtype uint8 (0-255)
        mask_over: Binary mask for overexposed regions (H, W) with values [0, 1]
        cfg: Configuration dictionary with overexposed parameters
        
    Returns:
        Image with corrected overexposed regions
        
    Example:
        >>> corrected = handle_overexposed(img, over_mask, config['overexposed'])
    """
    method = cfg.get('method', 'guided_filter')
    
    if method == 'clip':
        return _clip_highlights(img, mask_over, cfg)
    elif method == 'tonemap':
        return _tonemap_highlights(img, mask_over, cfg)
    elif method == 'guided_filter':
        return _guided_filter_recovery(img, mask_over, cfg)
    else:
        raise ValueError(f"Unknown method: {method}. Supported: 'clip', 'tonemap', 'guided_filter'")


def _clip_highlights(img: np.ndarray, mask_over: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Simple highlight clipping to specified percentile."""
    clip_percentile = cfg.get('clip_percentile', 95)
    
    # Convert to float
    img_float = img.astype(np.float32) / 255.0
    
    # Compute clipping threshold from non-overexposed regions
    normal_regions = mask_over < 0.5
    if np.sum(normal_regions) > 0:
        normal_pixels = img_float[normal_regions]
        clip_value = np.percentile(normal_pixels, clip_percentile)
    else:
        clip_value = 0.95
    
    # Apply clipping only to overexposed regions
    mask_smooth = cv2.GaussianBlur(mask_over, (15, 15), 5.0)
    mask_smooth = mask_smooth[:, :, np.newaxis]
    
    img_clipped = np.clip(img_float, 0, clip_value)
    
    # Blend clipped and original
    result = mask_smooth * img_clipped + (1 - mask_smooth) * img_float
    
    return (result * 255).clip(0, 255).astype(np.uint8)


def _tonemap_highlights(img: np.ndarray, mask_over: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Apply tone mapping to compress highlights."""
    # Convert to float
    img_float = img.astype(np.float32) / 255.0
    
    # Apply Reinhard tone mapping to overexposed regions
    tonemapped = img_float / (1 + img_float)
    
    # Smooth the mask for gradual transition
    mask_smooth = cv2.GaussianBlur(mask_over, (21, 21), 7.0)
    mask_smooth = mask_smooth[:, :, np.newaxis]
    
    # Blend tonemapped and original
    result = mask_smooth * tonemapped + (1 - mask_smooth) * img_float
    
    return (result * 255).clip(0, 255).astype(np.uint8)


def _guided_filter_recovery(img: np.ndarray, mask_over: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Recover highlight details using guided filtering with inverted representation."""
    # Extract parameters
    radius = cfg.get('guided_filter_radius', 8)
    eps = cfg.get('guided_filter_eps', 0.01)
    
    # Convert to float
    img_float = img.astype(np.float32) / 255.0
    
    # Create inverted representation for detail recovery
    img_inv = 1.0 - img_float
    
    # Use original image as guide
    guide = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # Apply guided filter to each channel of inverted image
    recovered_channels = []
    for c in range(3):
        # Guided filter for detail recovery
        recovered_channel = cv2.ximgproc.guidedFilter(
            guide, img_inv[:, :, c], radius, eps
        )
        recovered_channels.append(recovered_channel)
    
    # Stack channels
    img_recovered = np.stack(recovered_channels, axis=2)
    
    # Invert back to get recovered details
    img_recovered = 1.0 - img_recovered
    
    # Smooth the mask for gradual transition
    mask_smooth = cv2.GaussianBlur(mask_over, (21, 21), 7.0)
    mask_smooth = mask_smooth[:, :, np.newaxis]
    
    # Blend recovered and original
    result = mask_smooth * img_recovered + (1 - mask_smooth) * img_float
    
    return (result * 255).clip(0, 255).astype(np.uint8)


def detect_blown_highlights(img: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    """
    Detect completely blown-out highlights that can't be recovered.
    
    Args:
        img: RGB image with shape (H, W, 3) and dtype uint8
        threshold: Threshold for detecting blown highlights (0-1)
        
    Returns:
        Binary mask of blown highlights
    """
    img_float = img.astype(np.float32) / 255.0
    
    # Detect pixels that are blown in all channels
    blown_mask = np.all(img_float > threshold, axis=2)
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blown_mask = cv2.morphologyEx(blown_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    return blown_mask.astype(np.float32)


def progressive_highlight_compression(img: np.ndarray, mask_over: np.ndarray, 
                                    num_steps: int = 5) -> np.ndarray:
    """
    Apply progressive highlight compression in multiple steps.
    
    Args:
        img: RGB image uint8
        mask_over: Overexposed region mask
        num_steps: Number of compression steps
        
    Returns:
        Progressively compressed image
    """
    img_float = img.astype(np.float32) / 255.0
    result = img_float.copy()
    
    # Smooth mask
    mask_smooth = cv2.GaussianBlur(mask_over, (15, 15), 5.0)
    
    for step in range(num_steps):
        # Progressive compression strength
        compression = 0.1 + (0.4 * step / num_steps)  # 0.1 to 0.5
        
        # Apply tone mapping
        compressed = result / (1 + compression * result)
        
        # Blend with mask
        step_weight = mask_smooth[:, :, np.newaxis] * (1.0 / num_steps)
        result = step_weight * compressed + (1 - step_weight) * result
    
    return (result * 255).clip(0, 255).astype(np.uint8)


def enhance_highlight_details(img: np.ndarray, mask_over: np.ndarray) -> np.ndarray:
    """
    Enhance details in highlight regions using unsharp masking.
    
    Args:
        img: RGB image uint8
        mask_over: Overexposed region mask
        
    Returns:
        Image with enhanced highlight details
    """
    img_float = img.astype(np.float32) / 255.0
    
    # Convert to LAB for better perceptual processing
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel = img_lab[:, :, 0] / 255.0
    
    # Create unsharp mask
    blurred = cv2.GaussianBlur(l_channel, (9, 9), 2.0)
    unsharp_mask = l_channel - blurred
    
    # Apply unsharp masking only to overexposed regions
    mask_smooth = cv2.GaussianBlur(mask_over, (11, 11), 3.0)
    enhanced_l = l_channel + 0.3 * mask_smooth * unsharp_mask
    enhanced_l = np.clip(enhanced_l, 0, 1)
    
    # Update L channel
    img_lab[:, :, 0] = enhanced_l * 255.0
    
    # Convert back to RGB
    result = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    return result


def analyze_overexposure_severity(img: np.ndarray, mask_over: np.ndarray) -> Dict[str, float]:
    """
    Analyze severity of overexposure for adaptive correction.
    
    Args:
        img: RGB image uint8
        mask_over: Overexposed region mask
        
    Returns:
        Dictionary with overexposure analysis
    """
    img_float = img.astype(np.float32) / 255.0
    
    if np.sum(mask_over) == 0:
        return {"severity": 0.0, "recoverable_ratio": 1.0}
    
    # Analyze overexposed regions
    over_indices = mask_over > 0.5
    over_pixels = img_float[over_indices]
    
    # Severity based on how close to complete saturation
    severity = np.mean(over_pixels)
    
    # Estimate how much is recoverable (not completely blown)
    blown_threshold = 0.99
    recoverable = np.mean(over_pixels < blown_threshold)
    
    return {
        "severity": float(severity),
        "recoverable_ratio": float(recoverable),
        "avg_brightness": float(np.mean(over_pixels)),
        "overexposed_pixel_count": int(np.sum(over_indices))
    }
