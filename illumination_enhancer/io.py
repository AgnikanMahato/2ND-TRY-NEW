"""
Image I/O utilities for the illumination enhancement pipeline.

Handles reading, writing, and basic color space conversions while maintaining
proper data types and value ranges.
"""

from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np
import cv2


def read_image(path: Union[str, Path]) -> np.ndarray:
    """
    Read an RGB image from file path.
    
    Args:
        path: Path to image file
        
    Returns:
        RGB image as numpy array with shape (H, W, 3) and dtype uint8 (0-255)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be read or is not valid
        
    Example:
        >>> img = read_image("sample.jpg")
        >>> print(img.shape, img.dtype)
        (480, 640, 3) uint8
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    # Read image using OpenCV (BGR format)
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise ValueError(f"Could not read image from: {path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Ensure uint8 dtype
    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)
    
    return img_rgb


def save_image(img: np.ndarray, path: Union[str, Path], quality: int = 95) -> None:
    """
    Save an RGB image to file path.
    
    Args:
        img: RGB image array with shape (H, W, 3) and dtype uint8 (0-255)
        path: Output file path
        quality: JPEG quality (0-100), only used for JPEG files
        
    Raises:
        ValueError: If image array is not valid format
        
    Example:
        >>> save_image(enhanced_img, "output.jpg")
    """
    path = Path(path)
    
    # Validate input image
    if not isinstance(img, np.ndarray):
        raise ValueError("Image must be a numpy array")
    
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Image must have shape (H, W, 3)")
    
    # Convert to proper dtype if needed
    if img.dtype != np.uint8:
        if img.dtype == np.float32 or img.dtype == np.float64:
            # Assume float images are in range [0, 1]
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # Create output directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Set encoding parameters based on file extension
    ext = path.suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == '.png':
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    else:
        encode_params = []
    
    # Save image
    success = cv2.imwrite(str(path), img_bgr, encode_params)
    
    if not success:
        raise ValueError(f"Failed to save image to: {path}")


def convert_to_float(img: np.ndarray) -> np.ndarray:
    """
    Convert uint8 image to float32 in range [0, 1].
    
    Args:
        img: Image array with dtype uint8
        
    Returns:
        Image array with dtype float32 in range [0, 1]
    """
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert float image to uint8 in range [0, 255].
    
    Args:
        img: Image array with dtype float32/float64 in range [0, 1]
        
    Returns:
        Image array with dtype uint8 in range [0, 255]
    """
    if img.dtype in [np.float32, np.float64]:
        return (img * 255).clip(0, 255).astype(np.uint8)
    return img.astype(np.uint8)


def get_image_stats(img: np.ndarray) -> dict:
    """
    Get basic statistics about an image.
    
    Args:
        img: Input image array
        
    Returns:
        Dictionary with image statistics
    """
    return {
        "shape": img.shape,
        "dtype": str(img.dtype),
        "min": float(img.min()),
        "max": float(img.max()),
        "mean": float(img.mean()),
        "std": float(img.std())
    }
