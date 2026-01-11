"""
Region classification module for patch-based illumination analysis.

Classifies image patches into Dark, Normal, and Overexposed regions
using sliding-window analysis of illumination maps.
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2
from enum import Enum


class RegionType(Enum):
    """Enumeration for region types."""
    DARK = 0
    NORMAL = 1 
    OVEREXPOSED = 2


def classify_patches(illum_map: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Classify image patches based on illumination map statistics.
    
    Args:
        illum_map: Illumination map (H, W) with values in [0, 1]
        cfg: Configuration dictionary with region_classifier parameters
        
    Returns:
        Tuple of:
        - Region mask (H, W) with values {0: Dark, 1: Normal, 2: Overexposed}
        - Debug info dictionary with patch statistics
        
    Example:
        >>> region_mask, debug_info = classify_patches(illum_map, config['region_classifier'])
        >>> dark_mask = (region_mask == RegionType.DARK.value)
    """
    # Extract parameters
    patch_size = cfg.get('patch_size', 32)
    stride_ratio = cfg.get('stride_ratio', 0.5)
    dark_threshold = cfg.get('dark_threshold', 0.3)
    normal_threshold = cfg.get('normal_threshold', 0.7)
    
    h, w = illum_map.shape
    stride = int(patch_size * stride_ratio)
    
    # Initialize region mask
    region_mask = np.ones((h, w), dtype=np.uint8) * RegionType.NORMAL.value
    
    # Storage for debug information
    patch_stats = []
    
    # Sliding window analysis
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract patch
            patch = illum_map[y:y+patch_size, x:x+patch_size]
            
            # Compute patch statistics
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)
            patch_min = np.min(patch)
            patch_max = np.max(patch)
            
            # Classify patch
            if patch_mean < dark_threshold:
                region_type = RegionType.DARK
            elif patch_mean > normal_threshold:
                region_type = RegionType.OVEREXPOSED
            else:
                region_type = RegionType.NORMAL
            
            # Update region mask
            region_mask[y:y+patch_size, x:x+patch_size] = region_type.value
            
            # Store debug info
            patch_stats.append({
                'x': x, 'y': y,
                'mean': patch_mean,
                'std': patch_std,
                'min': patch_min,
                'max': patch_max,
                'type': region_type.name
            })
    
    # Post-process mask with morphological operations
    region_mask = post_process_region_mask(region_mask)
    
    # Compile debug information
    debug_info = compile_debug_info(patch_stats, region_mask)
    
    return region_mask, debug_info


def post_process_region_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean up region mask using morphological operations.
    
    Args:
        mask: Raw region mask with patch-based classifications
        
    Returns:
        Cleaned region mask
    """
    # Apply median filter to remove noise
    mask_filtered = cv2.medianBlur(mask.astype(np.uint8), 5)
    
    # Apply morphological opening to each region type
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    for region_type in [RegionType.DARK.value, RegionType.NORMAL.value, RegionType.OVEREXPOSED.value]:
        # Extract binary mask for this region
        binary_mask = (mask_filtered == region_type).astype(np.uint8)
        
        # Apply morphological opening
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Update filtered mask
        mask_filtered[binary_mask == 1] = region_type
    
    return mask_filtered


def compile_debug_info(patch_stats: list, region_mask: np.ndarray) -> Dict[str, Any]:
    """
    Compile debug information about region classification.
    
    Args:
        patch_stats: List of patch statistics dictionaries
        region_mask: Final region classification mask
        
    Returns:
        Debug information dictionary
    """
    # Count pixels in each region
    total_pixels = region_mask.size
    dark_pixels = np.sum(region_mask == RegionType.DARK.value)
    normal_pixels = np.sum(region_mask == RegionType.NORMAL.value)
    over_pixels = np.sum(region_mask == RegionType.OVEREXPOSED.value)
    
    # Compute region statistics
    patch_means = [p['mean'] for p in patch_stats]
    
    return {
        'total_patches': len(patch_stats),
        'region_pixel_counts': {
            'dark': int(dark_pixels),
            'normal': int(normal_pixels), 
            'overexposed': int(over_pixels)
        },
        'region_pixel_ratios': {
            'dark': float(dark_pixels / total_pixels),
            'normal': float(normal_pixels / total_pixels),
            'overexposed': float(over_pixels / total_pixels)
        },
        'patch_mean_stats': {
            'min': float(np.min(patch_means)),
            'max': float(np.max(patch_means)),
            'avg': float(np.mean(patch_means)),
            'std': float(np.std(patch_means))
        }
    }


def create_region_masks(region_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Create separate binary masks for each region type.
    
    Args:
        region_mask: Region classification mask
        
    Returns:
        Dictionary with binary masks for each region type
    """
    return {
        'dark': (region_mask == RegionType.DARK.value).astype(np.float32),
        'normal': (region_mask == RegionType.NORMAL.value).astype(np.float32),
        'overexposed': (region_mask == RegionType.OVEREXPOSED.value).astype(np.float32)
    }


def adaptive_threshold_classification(illum_map: np.ndarray) -> np.ndarray:
    """
    Adaptive threshold-based region classification.
    
    Uses Otsu's method to automatically determine thresholds for
    dark and overexposed regions.
    
    Args:
        illum_map: Illumination map (H, W) with values in [0, 1]
        
    Returns:
        Region mask with adaptive thresholds
    """
    # Convert to 8-bit for Otsu thresholding
    illum_8bit = (illum_map * 255).astype(np.uint8)
    
    # Apply Otsu's thresholding to find two main thresholds
    _, binary_dark = cv2.threshold(illum_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find second threshold for overexposed regions
    mask_bright = illum_8bit > 127
    if np.sum(mask_bright) > 0:
        bright_values = illum_8bit[mask_bright]
        bright_threshold, _ = cv2.threshold(bright_values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright_threshold += 127  # Offset since we only considered bright pixels
    else:
        bright_threshold = 200
    
    # Create region mask
    region_mask = np.ones_like(illum_8bit, dtype=np.uint8) * RegionType.NORMAL.value
    region_mask[illum_8bit < 255 - binary_dark] = RegionType.DARK.value
    region_mask[illum_8bit > bright_threshold] = RegionType.OVEREXPOSED.value
    
    return region_mask


def get_region_boundaries(region_mask: np.ndarray) -> np.ndarray:
    """
    Find boundaries between different regions for seamless blending.
    
    Args:
        region_mask: Region classification mask
        
    Returns:
        Binary boundary mask
    """
    # Create boundary detection kernels
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    # Detect boundaries using gradient
    grad_x = cv2.filter2D(region_mask.astype(np.float32), -1, kernel_x)
    grad_y = cv2.filter2D(region_mask.astype(np.float32), -1, kernel_y)
    
    # Combine gradients
    boundaries = np.sqrt(grad_x**2 + grad_y**2) > 0.1
    
    # Dilate boundaries slightly for smoother blending
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundaries = cv2.dilate(boundaries.astype(np.uint8), kernel, iterations=1)
    
    return boundaries.astype(np.float32)
