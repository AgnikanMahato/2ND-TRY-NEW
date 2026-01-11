"""
Dual illumination representation module.

Creates complementary representations of the input image to capture
different aspects of illumination for robust feature extraction.
"""

from typing import Tuple, Union
import numpy as np
import cv2


def make_dual_repr(img: np.ndarray, mode: str = "hsv_v") -> Tuple[np.ndarray, np.ndarray]:
    """
    Create dual illumination representation of input image.
    
    Args:
        img: RGB image with shape (H, W, 3) and dtype uint8 (0-255)
        mode: Type of dual representation
              - "hsv_v": Invert HSV V channel
              - "lab_l": Invert LAB L channel  
              - "rgb": Invert RGB channels
              
    Returns:
        Tuple of (original_image, inverted_image) both in RGB uint8 format
        
    Example:
        >>> orig, inv = make_dual_repr(img, "hsv_v")
        >>> # Use both representations for feature extraction
    """
    if mode == "hsv_v":
        return _invert_hsv_v(img)
    elif mode == "lab_l":
        return _invert_lab_l(img)
    elif mode == "rgb":
        return _invert_rgb(img)
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported: 'hsv_v', 'lab_l', 'rgb'")


def _invert_hsv_v(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Invert HSV V channel while preserving hue and saturation."""
    # Convert to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Create inverted version
    img_hsv_inv = img_hsv.copy()
    img_hsv_inv[:, :, 2] = 255.0 - img_hsv_inv[:, :, 2]  # Invert V channel
    
    # Convert back to RGB
    img_inv = cv2.cvtColor(img_hsv_inv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return img, img_inv


def _invert_lab_l(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Invert LAB L channel while preserving color information."""
    # Convert to LAB
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Create inverted version
    img_lab_inv = img_lab.copy()
    img_lab_inv[:, :, 0] = 255.0 - img_lab_inv[:, :, 0]  # Invert L channel
    
    # Convert back to RGB
    img_inv = cv2.cvtColor(img_lab_inv.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    return img, img_inv


def _invert_rgb(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simple RGB channel inversion."""
    img_inv = 255 - img
    return img, img_inv


def create_illumination_invariant(img: np.ndarray) -> np.ndarray:
    """
    Create illumination-invariant representation using log-chromaticity.
    
    Args:
        img: RGB image with shape (H, W, 3) and dtype uint8
        
    Returns:
        Illumination-invariant image as float32 array
    """
    # Convert to float and add small epsilon to avoid log(0)
    img_float = img.astype(np.float32) + 1e-6
    
    # Compute log chromaticity
    log_img = np.log(img_float)
    
    # Geometric mean across channels
    geom_mean = np.exp(np.mean(log_img, axis=2, keepdims=True))
    
    # Normalize by geometric mean
    invariant = log_img - np.log(geom_mean)
    
    # Normalize to [0, 1] range
    invariant = (invariant - invariant.min()) / (invariant.max() - invariant.min() + 1e-6)
    
    return invariant.astype(np.float32)


def enhance_contrast_selectively(img: np.ndarray, mask: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Apply selective contrast enhancement based on region mask.
    
    Args:
        img: RGB image with shape (H, W, 3) and dtype uint8
        mask: Binary mask indicating regions to enhance (0-1, float32)
        strength: Enhancement strength factor (0-2, default 1.0)
        
    Returns:
        Enhanced RGB image
    """
    # Convert to LAB for perceptual processing
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Apply contrast enhancement to L channel
    l_channel = img_lab[:, :, 0]
    l_mean = np.mean(l_channel)
    
    # Selective contrast stretching
    l_enhanced = l_mean + (l_channel - l_mean) * (1.0 + strength * mask)
    l_enhanced = np.clip(l_enhanced, 0, 255)
    
    # Update L channel
    img_lab[:, :, 0] = l_enhanced
    
    # Convert back to RGB
    img_enhanced = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    return img_enhanced


def compute_illumination_gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute illumination gradients for structure preservation.
    
    Args:
        img: RGB image with shape (H, W, 3) and dtype uint8
        
    Returns:
        Tuple of (grad_x, grad_y) gradient magnitude maps
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Compute gradients using Sobel operators
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    return grad_x, grad_y
