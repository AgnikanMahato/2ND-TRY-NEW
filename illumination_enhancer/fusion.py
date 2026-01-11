"""
Region fusion module for seamless blending of enhanced regions.

Combines enhanced dark regions, normal regions, and corrected overexposed
regions using advanced blending techniques like Poisson blending.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import cv2


def fuse_regions(orig: np.ndarray, dark_enh: np.ndarray, 
                 norm_mask: np.ndarray, over_corr: np.ndarray, 
                 region_masks: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> np.ndarray:
    """
    Fuse different region enhancements into final output image.
    
    Args:
        orig: Original RGB image (H, W, 3) uint8
        dark_enh: Enhanced dark regions image uint8  
        norm_mask: Normal region preservation mask [0, 1]
        over_corr: Corrected overexposed regions image uint8
        region_masks: Dictionary with 'dark', 'normal', 'overexposed' masks
        cfg: Configuration dictionary with fusion parameters
        
    Returns:
        Final fused RGB image uint8
        
    Example:
        >>> fused = fuse_regions(orig, dark_enhanced, normal_mask, 
        ...                     over_corrected, masks, config['fusion'])
    """
    blend_mode = cfg.get('blend_mode', 'poisson')
    mask_blur_sigma = cfg.get('mask_blur_sigma', 5.0)
    
    if blend_mode == 'alpha':
        return _alpha_blend_fusion(orig, dark_enh, over_corr, region_masks, mask_blur_sigma)
    elif blend_mode == 'poisson':
        return _poisson_blend_fusion(orig, dark_enh, over_corr, region_masks, cfg)
    else:
        raise ValueError(f"Unknown blend_mode: {blend_mode}. Supported: 'alpha', 'poisson'")


def _alpha_blend_fusion(orig: np.ndarray, dark_enh: np.ndarray, over_corr: np.ndarray,
                       region_masks: Dict[str, np.ndarray], blur_sigma: float) -> np.ndarray:
    """Fusion using alpha blending with smooth transitions."""
    
    # Smooth all masks for gradual transitions
    dark_mask = cv2.GaussianBlur(region_masks['dark'], (0, 0), blur_sigma)
    over_mask = cv2.GaussianBlur(region_masks['overexposed'], (0, 0), blur_sigma)
    
    # Ensure masks sum to 1 (normalization)
    total_mask = dark_mask + over_mask
    normal_mask = 1.0 - np.clip(total_mask, 0, 1)
    
    # Renormalize to ensure sum = 1
    mask_sum = dark_mask + normal_mask + over_mask + 1e-6
    dark_mask = dark_mask / mask_sum
    normal_mask = normal_mask / mask_sum  
    over_mask = over_mask / mask_sum
    
    # Convert to float for blending
    orig_f = orig.astype(np.float32)
    dark_f = dark_enh.astype(np.float32)
    over_f = over_corr.astype(np.float32)
    
    # Weighted blend
    result = (dark_mask[:, :, np.newaxis] * dark_f + 
              normal_mask[:, :, np.newaxis] * orig_f + 
              over_mask[:, :, np.newaxis] * over_f)
    
    return result.clip(0, 255).astype(np.uint8)


def _poisson_blend_fusion(orig: np.ndarray, dark_enh: np.ndarray, over_corr: np.ndarray,
                         region_masks: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> np.ndarray:
    """Fusion using Poisson blending for seamless transitions."""
    
    result = orig.copy()
    
    # Blur masks slightly for smoother boundaries
    blur_sigma = cfg.get('mask_blur_sigma', 5.0)
    
    # Process dark regions with Poisson blending
    dark_mask = region_masks['dark']
    if np.sum(dark_mask) > 0:
        result = _poisson_blend_region(result, dark_enh, dark_mask, blur_sigma)
    
    # Process overexposed regions  
    over_mask = region_masks['overexposed']
    if np.sum(over_mask) > 0:
        result = _poisson_blend_region(result, over_corr, over_mask, blur_sigma)
    
    return result


def _poisson_blend_region(base: np.ndarray, source: np.ndarray, 
                         mask: np.ndarray, blur_sigma: float) -> np.ndarray:
    """Apply Poisson blending to a specific region."""
    
    # Create binary mask for cv2.seamlessClone
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Smooth mask edges
    if blur_sigma > 0:
        binary_mask = cv2.GaussianBlur(binary_mask, (0, 0), blur_sigma / 2)
        binary_mask = (binary_mask > 127).astype(np.uint8) * 255
    
    # Find center of region for seamless clone
    moments = cv2.moments(binary_mask)
    if moments['m00'] > 0:
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
        center = (center_x, center_y)
    else:
        # Fallback to image center
        center = (base.shape[1] // 2, base.shape[0] // 2)
    
    try:
        # Apply seamless cloning
        result = cv2.seamlessClone(source, base, binary_mask, center, cv2.NORMAL_CLONE)
        return result
    except cv2.error:
        # Fallback to alpha blending if Poisson fails
        mask_smooth = cv2.GaussianBlur(mask, (0, 0), blur_sigma)
        mask_3d = mask_smooth[:, :, np.newaxis]
        
        base_f = base.astype(np.float32)
        source_f = source.astype(np.float32)
        
        blended = mask_3d * source_f + (1 - mask_3d) * base_f
        return blended.clip(0, 255).astype(np.uint8)


def create_transition_masks(region_masks: Dict[str, np.ndarray], 
                           transition_width: int = 20) -> Dict[str, np.ndarray]:
    """
    Create smooth transition masks between regions.
    
    Args:
        region_masks: Dictionary with region masks
        transition_width: Width of transition zone in pixels
        
    Returns:
        Dictionary with smooth transition masks
    """
    smooth_masks = {}
    
    for region_name, mask in region_masks.items():
        # Create distance transform for smooth transitions
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Distance from boundary
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        
        # Create smooth transition based on distance
        smooth_mask = np.clip(dist_transform / transition_width, 0, 1)
        
        smooth_masks[region_name] = smooth_mask
    
    return smooth_masks


def multi_scale_blending(images: List[np.ndarray], masks: List[np.ndarray], 
                        num_levels: int = 4) -> np.ndarray:
    """
    Multi-scale blending using Laplacian pyramids.
    
    Args:
        images: List of images to blend
        masks: List of corresponding masks  
        num_levels: Number of pyramid levels
        
    Returns:
        Blended image
    """
    if len(images) != len(masks):
        raise ValueError("Number of images must match number of masks")
    
    if not images:
        raise ValueError("At least one image required")
    
    h, w = images[0].shape[:2]
    
    # Build Laplacian pyramids for images
    image_pyramids = []
    for img in images:
        pyramid = build_laplacian_pyramid(img, num_levels)
        image_pyramids.append(pyramid)
    
    # Build Gaussian pyramids for masks
    mask_pyramids = []
    for mask in masks:
        # Ensure mask is 3-channel
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        if mask.shape[2] == 1:
            mask = np.repeat(mask, 3, axis=2)
            
        pyramid = build_gaussian_pyramid(mask, num_levels)
        mask_pyramids.append(pyramid)
    
    # Blend at each pyramid level
    blended_pyramid = []
    for level in range(num_levels):
        level_result = np.zeros_like(image_pyramids[0][level])
        total_weight = np.zeros_like(mask_pyramids[0][level])
        
        for i in range(len(images)):
            weight = mask_pyramids[i][level]
            level_result += weight * image_pyramids[i][level]
            total_weight += weight
        
        # Normalize by total weight
        total_weight[total_weight == 0] = 1  # Avoid division by zero
        level_result = level_result / total_weight
        blended_pyramid.append(level_result)
    
    # Reconstruct from pyramid
    result = reconstruct_from_laplacian_pyramid(blended_pyramid)
    
    return result.clip(0, 255).astype(np.uint8)


def build_gaussian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    """Build Gaussian pyramid."""
    pyramid = [img.astype(np.float32)]
    
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        pyramid.append(img.astype(np.float32))
    
    return pyramid


def build_laplacian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    """Build Laplacian pyramid."""
    gaussian_pyramid = build_gaussian_pyramid(img, levels)
    laplacian_pyramid = []
    
    for i in range(levels - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        upsampled = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian = gaussian_pyramid[i] - upsampled
        laplacian_pyramid.append(laplacian)
    
    # Add the smallest level
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return laplacian_pyramid


def reconstruct_from_laplacian_pyramid(pyramid: List[np.ndarray]) -> np.ndarray:
    """Reconstruct image from Laplacian pyramid."""
    result = pyramid[-1]  # Start with smallest level
    
    for i in range(len(pyramid) - 2, -1, -1):
        size = (pyramid[i].shape[1], pyramid[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + pyramid[i]
    
    return result
