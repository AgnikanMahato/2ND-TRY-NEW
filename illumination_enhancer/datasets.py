"""
Dataset creation and data loading utilities.

Provides data loaders for training with dual image representations
and various augmentation strategies.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from .illumination_repr import make_dual_repr


class IlluminationDataset(Dataset):
    """Dataset for illumination enhancement training."""
    
    def __init__(self, data_root: str, split: str = 'train', 
                 augment: bool = True, dual_mode: str = 'hsv_v'):
        """
        Initialize dataset.
        
        Args:
            data_root: Root directory containing images
            split: Dataset split ('train', 'val', 'test')
            augment: Whether to apply data augmentation
            dual_mode: Mode for dual representation ('hsv_v', 'lab_l', 'rgb')
        """
        self.data_root = Path(data_root)
        self.split = split
        self.augment = augment
        self.dual_mode = dual_mode
        
        # Find all image files
        self.image_paths = self._find_images()
        
        # Setup transforms
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1] range
        ])
        
        if augment and split == 'train':
            self.augmentation = self._get_augmentations()
        else:
            self.augmentation = None
    
    def _find_images(self) -> list:
        """Find all image files in the dataset directory."""
        image_dir = self.data_root / self.split
        if not image_dir.exists():
            # Fallback: use data_root directly if split directory doesn't exist
            image_dir = self.data_root
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for ext in valid_extensions:
            image_paths.extend(image_dir.glob(f'*{ext}'))
            image_paths.extend(image_dir.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_paths)} images in {image_dir}")
        return sorted(image_paths)
    
    def _get_augmentations(self) -> transforms.Compose:
        """Get data augmentation transforms."""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomRotation(degrees=15),
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if too large (for memory efficiency)
        h, w = image.shape[:2]
        max_size = 512
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create dual representation
        orig_img, inv_img = make_dual_repr(image, self.dual_mode)
        
        # Convert to tensors
        orig_tensor = self.base_transform(orig_img)  # (3, H, W)
        inv_tensor = self.base_transform(inv_img)    # (3, H, W)
        
        # Apply augmentation if enabled
        if self.augmentation is not None:
            # Apply same augmentation to both images
            seed = torch.randint(0, 2**32, (1,)).item()
            
            torch.manual_seed(seed)
            orig_tensor = self.augmentation(orig_tensor)
            
            torch.manual_seed(seed)
            inv_tensor = self.augmentation(inv_tensor)
        
        # Concatenate for dual input
        dual_tensor = torch.cat([orig_tensor, inv_tensor], dim=0)  # (6, H, W)
        
        return {
            'image': orig_tensor,           # Original image (3, H, W)
            'dual_image': dual_tensor,      # Dual representation (6, H, W)
            'image_path': str(img_path)
        }


class SyntheticLowLightDataset(Dataset):
    """
    Synthetic low-light dataset created by applying random degradations
    to normal-light images. Useful when real paired data is not available.
    """
    
    def __init__(self, normal_images_dir: str, degradation_params: Dict[str, Any]):
        """
        Initialize synthetic dataset.
        
        Args:
            normal_images_dir: Directory with normal-light images
            degradation_params: Parameters for synthetic degradation
        """
        self.normal_dir = Path(normal_images_dir)
        self.degradation_params = degradation_params
        
        # Find normal-light images
        self.image_paths = self._find_images()
        
        self.base_transform = transforms.ToTensor()
    
    def _find_images(self) -> list:
        """Find all normal-light images."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = []
        
        for ext in valid_extensions:
            image_paths.extend(self.normal_dir.glob(f'*{ext}'))
            image_paths.extend(self.normal_dir.glob(f'*{ext.upper()}'))
        
        return sorted(image_paths)
    
    def _apply_synthetic_degradation(self, image: np.ndarray) -> np.ndarray:
        """Apply synthetic low-light degradation."""
        img_float = image.astype(np.float32) / 255.0
        
        # Random gamma correction to simulate underexposure
        gamma = np.random.uniform(1.5, 3.0)  # Higher gamma = darker
        degraded = np.power(img_float, gamma)
        
        # Add slight noise
        noise_level = np.random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_level, degraded.shape)
        degraded = degraded + noise
        
        # Random brightness reduction
        brightness_factor = np.random.uniform(0.3, 0.8)
        degraded = degraded * brightness_factor
        
        # Clip and convert back
        degraded = np.clip(degraded, 0, 1)
        return (degraded * 255).astype(np.uint8)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get synthetic low-light sample."""
        # Load normal image
        img_path = self.image_paths[idx]
        normal_image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        normal_image = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
        
        # Apply synthetic degradation
        low_light_image = self._apply_synthetic_degradation(normal_image)
        
        # Create dual representation
        orig_img, inv_img = make_dual_repr(low_light_image, 'hsv_v')
        
        # Convert to tensors
        orig_tensor = self.base_transform(orig_img)
        inv_tensor = self.base_transform(inv_img)
        dual_tensor = torch.cat([orig_tensor, inv_tensor], dim=0)
        
        return {
            'image': orig_tensor,
            'dual_image': dual_tensor,
            'normal_image': self.base_transform(normal_image),  # Ground truth
            'image_path': str(img_path)
        }


def create_dataloader(data_root: str, split: str = 'train', 
                     batch_size: int = 8, shuffle: bool = True,
                     num_workers: int = 4, augment: bool = True) -> DataLoader:
    """
    Create data loader for training/validation.
    
    Args:
        data_root: Root directory of dataset
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        augment: Whether to apply data augmentation
        
    Returns:
        DataLoader instance
    """
    dataset = IlluminationDataset(
        data_root=data_root,
        split=split,
        augment=augment and split == 'train'
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=split == 'train'  # Drop last incomplete batch for training
    )


def create_synthetic_dataloader(normal_images_dir: str, batch_size: int = 8,
                              shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """Create data loader for synthetic low-light data."""
    degradation_params = {
        'gamma_range': (1.5, 3.0),
        'noise_range': (0.01, 0.05),
        'brightness_range': (0.3, 0.8)
    }
    
    dataset = SyntheticLowLightDataset(normal_images_dir, degradation_params)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
