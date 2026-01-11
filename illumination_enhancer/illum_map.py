"""
Illumination map estimation module.

Converts extracted features to pixel-level illumination maps with
proper upsampling and smoothness constraints.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
import cv2


def estimate_map(features: torch.Tensor, original_size: tuple, 
                 cfg: Dict[str, Any]) -> np.ndarray:
    """
    Estimate illumination map from extracted features.
    
    Args:
        features: Feature tensor from LightFeatNet (B, C, H/4, W/4)
        original_size: Target size (H, W) for upsampling
        cfg: Configuration dictionary with illum_map parameters
        
    Returns:
        Illumination map as numpy array (H, W) with values in [0, 1]
        
    Example:
        >>> illum_map = estimate_map(features, (480, 640), config['illum_map'])
        >>> # Use map for region classification
    """
    from .networks.feature_net import IlluminationHead
    
    # Create illumination head if not provided
    if not hasattr(estimate_map, '_illum_head'):
        estimate_map._illum_head = IlluminationHead(features.shape[1])
        if torch.cuda.is_available() and features.is_cuda:
            estimate_map._illum_head = estimate_map._illum_head.cuda()
    
    illum_head = estimate_map._illum_head
    
    # Generate illumination map
    with torch.no_grad():
        illum_head.eval()
        illum_map_tensor = illum_head(features)  # (B, 1, H/4, W/4)
    
    # Upsample to original resolution
    upsample_mode = cfg.get('upsample_mode', 'bilinear')
    illum_map_upsampled = F.interpolate(
        illum_map_tensor, 
        size=original_size, 
        mode=upsample_mode, 
        align_corners=False
    )
    
    # Convert to numpy
    illum_map = illum_map_upsampled[0, 0].cpu().numpy()  # Remove batch and channel dims
    
    # Apply smoothness constraint
    smoothness_weight = cfg.get('smoothness_weight', 0.1)
    if smoothness_weight > 0:
        illum_map = apply_smoothness_constraint(illum_map, smoothness_weight)
    
    return illum_map


def apply_smoothness_constraint(illum_map: np.ndarray, weight: float) -> np.ndarray:
    """
    Apply smoothness constraint to illumination map using bilateral filtering.
    
    Args:
        illum_map: Illumination map (H, W) with values in [0, 1]
        weight: Smoothness weight (higher = smoother)
        
    Returns:
        Smoothed illumination map
    """
    # Convert to 8-bit for bilateral filter
    illum_8bit = (illum_map * 255).astype(np.uint8)
    
    # Apply bilateral filter for edge-preserving smoothing
    diameter = int(weight * 20) + 5  # Scale weight to filter diameter
    sigma_color = weight * 50 + 10
    sigma_space = weight * 50 + 10
    
    smoothed_8bit = cv2.bilateralFilter(illum_8bit, diameter, sigma_color, sigma_space)
    smoothed = smoothed_8bit.astype(np.float32) / 255.0
    
    # Blend with original based on weight
    alpha = min(weight, 0.8)  # Limit smoothing effect
    final_map = alpha * smoothed + (1 - alpha) * illum_map
    
    return final_map


def create_illumination_map_from_image(img: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Create illumination map directly from image using classical methods.
    
    This serves as a fallback when neural network features are not available.
    
    Args:
        img: RGB image with shape (H, W, 3) and dtype uint8
        method: Method for map estimation ('adaptive', 'retinex', 'gray_world')
        
    Returns:
        Illumination map (H, W) with values in [0, 1]
    """
    if method == "adaptive":
        return _adaptive_illumination_map(img)
    elif method == "retinex":
        return _retinex_illumination_map(img)
    elif method == "gray_world":
        return _gray_world_illumination_map(img)
    else:
        raise ValueError(f"Unknown method: {method}")


def _adaptive_illumination_map(img: np.ndarray) -> np.ndarray:
    """Adaptive illumination estimation using local statistics."""
    # Convert to grayscale for illumination estimation
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # Apply Gaussian blur to estimate local illumination
    kernel_size = max(img.shape[0], img.shape[1]) // 20  # Adaptive kernel size
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # Ensure odd
    kernel_size = max(kernel_size, 15)  # Minimum size
    
    illumination = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    
    # Normalize to [0, 1] range
    illumination = (illumination - illumination.min()) / (illumination.max() - illumination.min() + 1e-6)
    
    return illumination


def _retinex_illumination_map(img: np.ndarray) -> np.ndarray:
    """Single-scale Retinex for illumination estimation."""
    img_float = img.astype(np.float32) / 255.0 + 1e-6
    
    # Convert to log domain
    log_img = np.log(img_float)
    
    # Average across channels for illumination
    log_illum = np.mean(log_img, axis=2)
    
    # Apply Gaussian blur in log domain
    sigma = min(img.shape[0], img.shape[1]) / 10
    log_illum_blurred = cv2.GaussianBlur(log_illum, (0, 0), sigma)
    
    # Convert back to linear domain and normalize
    illumination = np.exp(log_illum_blurred)
    illumination = (illumination - illumination.min()) / (illumination.max() - illumination.min() + 1e-6)
    
    return illumination


def _gray_world_illumination_map(img: np.ndarray) -> np.ndarray:
    """Gray world assumption for illumination estimation."""
    img_float = img.astype(np.float32) / 255.0
    
    # Compute channel means
    channel_means = np.mean(img_float, axis=(0, 1))
    gray_mean = np.mean(channel_means)
    
    # Create illumination map based on deviation from gray world
    illumination = np.mean(img_float, axis=2) / (gray_mean + 1e-6)
    illumination = np.clip(illumination, 0, 2)  # Reasonable range
    illumination = illumination / 2.0  # Normalize to [0, 1]
    
    return illumination


def refine_illumination_map(illum_map: np.ndarray, img: np.ndarray, 
                           iterations: int = 3) -> np.ndarray:
    """
    Refine illumination map using iterative guided filtering.
    
    Args:
        illum_map: Initial illumination map (H, W) in [0, 1]
        img: Original RGB image (H, W, 3) uint8
        iterations: Number of refinement iterations
        
    Returns:
        Refined illumination map
    """
    # Convert image to grayscale guide
    guide = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    refined_map = illum_map.copy()
    
    for _ in range(iterations):
        # Apply guided filter
        refined_map = cv2.ximgproc.guidedFilter(
            guide, refined_map.astype(np.float32), 
            radius=8, eps=0.01
        )
        
        # Ensure valid range
        refined_map = np.clip(refined_map, 0, 1)
    
    return refined_map
