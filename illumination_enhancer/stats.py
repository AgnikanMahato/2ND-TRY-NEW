"""
Global histogram analysis for illumination assessment.

Computes comprehensive histogram statistics to characterize image brightness
distribution and lighting conditions.
"""

from typing import Dict, Tuple
import numpy as np
import cv2


def global_hist_stats(img: np.ndarray) -> Dict[str, float]:
    """
    Compute global histogram statistics for illumination analysis.
    
    Args:
        img: RGB image with shape (H, W, 3) and dtype uint8 (0-255)
        
    Returns:
        Dictionary containing:
        - mean: Average brightness value (0-1)
        - p01, p25, p50, p75, p99: Percentile values (0-1)
        - dark_ratio: Fraction of pixels with V < 30/255
        - bright_ratio: Fraction of pixels with V > 200/255
        - contrast: Standard deviation of V channel
        
    Example:
        >>> stats = global_hist_stats(img)
        >>> print(f"Mean brightness: {stats['mean']:.3f}")
        >>> print(f"Dark pixel ratio: {stats['dark_ratio']:.3f}")
    """
    # Convert to HSV to get value (brightness) channel
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = img_hsv[:, :, 2].astype(np.float32) / 255.0
    
    # Flatten for histogram computation
    v_flat = v_channel.flatten()
    
    # Compute basic statistics
    mean_val = float(np.mean(v_flat))
    std_val = float(np.std(v_flat))
    
    # Compute percentiles
    percentiles = np.percentile(v_flat, [1, 25, 50, 75, 99])
    p01, p25, p50, p75, p99 = percentiles
    
    # Compute dark and bright ratios
    dark_ratio = float(np.mean(v_flat < (30.0 / 255.0)))
    bright_ratio = float(np.mean(v_flat > (200.0 / 255.0)))
    
    return {
        "mean": mean_val,
        "p01": float(p01),
        "p25": float(p25), 
        "p50": float(p50),
        "p75": float(p75),
        "p99": float(p99),
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "contrast": std_val
    }


def compute_rgb_histogram(img: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute RGB channel histograms.
    
    Args:
        img: RGB image with shape (H, W, 3) and dtype uint8
        bins: Number of histogram bins
        
    Returns:
        Tuple of (hist_r, hist_g, hist_b) normalized histograms
    """
    hist_r = cv2.calcHist([img], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([img], [2], None, [bins], [0, 256])
    
    # Normalize histograms
    total_pixels = img.shape[0] * img.shape[1]
    hist_r = hist_r.flatten() / total_pixels
    hist_g = hist_g.flatten() / total_pixels
    hist_b = hist_b.flatten() / total_pixels
    
    return hist_r, hist_g, hist_b


def compute_local_contrast(img: np.ndarray, window_size: int = 9) -> float:
    """
    Compute local contrast measure using standard deviation.
    
    Args:
        img: RGB image with shape (H, W, 3) and dtype uint8
        window_size: Size of local window for contrast computation
        
    Returns:
        Average local contrast value
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Compute local standard deviation using morphological operations
    kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
    
    # Local mean
    local_mean = cv2.filter2D(gray, -1, kernel)
    
    # Local variance
    local_var = cv2.filter2D(gray * gray, -1, kernel) - local_mean * local_mean
    local_std = np.sqrt(np.maximum(local_var, 0))
    
    return float(np.mean(local_std))


def analyze_color_distribution(img: np.ndarray) -> Dict[str, float]:
    """
    Analyze color distribution characteristics.
    
    Args:
        img: RGB image with shape (H, W, 3) and dtype uint8
        
    Returns:
        Dictionary with color distribution metrics
    """
    img_float = img.astype(np.float32) / 255.0
    
    # Compute color moments
    r_mean, g_mean, b_mean = np.mean(img_float, axis=(0, 1))
    r_std, g_std, b_std = np.std(img_float, axis=(0, 1))
    
    # Color cast detection (deviation from gray)
    gray_mean = (r_mean + g_mean + b_mean) / 3
    color_cast = float(np.sqrt((r_mean - gray_mean)**2 + 
                              (g_mean - gray_mean)**2 + 
                              (b_mean - gray_mean)**2))
    
    # Saturation analysis
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    saturation = img_hsv[:, :, 1].astype(np.float32) / 255.0
    avg_saturation = float(np.mean(saturation))
    
    return {
        "r_mean": float(r_mean),
        "g_mean": float(g_mean), 
        "b_mean": float(b_mean),
        "r_std": float(r_std),
        "g_std": float(g_std),
        "b_std": float(b_std),
        "color_cast": color_cast,
        "avg_saturation": avg_saturation
    }
