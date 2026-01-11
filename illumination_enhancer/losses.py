"""
Zero-reference loss functions for illumination enhancement training.

Implements spatial consistency, exposure control, color constancy,
illumination smoothness, and region-wise non-reference losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class ZeroReferenceLoss(nn.Module):
    """
    Combined zero-reference loss for illumination enhancement.
    
    Combines multiple loss terms:
    - Spatial consistency: Preserves local structures
    - Exposure control: Prevents over/under exposure
    - Color constancy: Maintains natural colors
    - Illumination smoothness: Ensures gradual transitions
    - Region-wise non-reference: Quality assessment without reference
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        
        # Loss weights
        self.spatial_weight = cfg.get('spatial_consistency_weight', 1.0)
        self.exposure_weight = cfg.get('exposure_control_weight', 10.0)
        self.color_weight = cfg.get('color_constancy_weight', 5.0)
        self.smoothness_weight = cfg.get('illumination_smoothness_weight', 0.5)
        self.region_weight = cfg.get('region_wise_nonref_weight', 1.0)
        
        print(f"Loss weights - Spatial: {self.spatial_weight}, Exposure: {self.exposure_weight}, "
              f"Color: {self.color_weight}, Smoothness: {self.smoothness_weight}, "
              f"Region: {self.region_weight}")
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                original: torch.Tensor, dual_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and individual components.
        
        Args:
            outputs: Model outputs containing illumination maps and features
            original: Original images (B, 3, H, W)
            dual_input: Dual representation input (B, 6, H, W)
            
        Returns:
            Dictionary with loss components
        """
        illum_map = outputs['illumination_map']  # (B, 1, H, W)
        
        # Extract original image from dual input
        orig_img = dual_input[:, :3, :, :]  # First 3 channels
        
        # Compute individual losses
        spatial_loss = self._spatial_consistency_loss(illum_map, orig_img)
        exposure_loss = self._exposure_control_loss(illum_map)
        color_loss = self._color_constancy_loss(illum_map, orig_img)
        smoothness_loss = self._illumination_smoothness_loss(illum_map)
        region_loss = self._region_wise_loss(illum_map, orig_img)
        
        # Combine losses
        total_loss = (self.spatial_weight * spatial_loss +
                     self.exposure_weight * exposure_loss +
                     self.color_weight * color_loss +
                     self.smoothness_weight * smoothness_loss +
                     self.region_weight * region_loss)
        
        return {
            'total_loss': total_loss,
            'spatial_loss': spatial_loss,
            'exposure_loss': exposure_loss,
            'color_loss': color_loss,
            'smoothness_loss': smoothness_loss,
            'region_loss': region_loss
        }
    
    def _spatial_consistency_loss(self, illum_map: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Spatial consistency loss to preserve local structures.
        
        Ensures that the illumination map respects image structure boundaries.
        """
        # Convert image to grayscale
        gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        gray = gray.unsqueeze(1)  # (B, 1, H, W)
        
        # Compute gradients
        def compute_gradients(x):
            grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
            grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
            return grad_x, grad_y
        
        # Image gradients
        img_grad_x, img_grad_y = compute_gradients(gray)
        
        # Illumination map gradients
        illum_grad_x, illum_grad_y = compute_gradients(illum_map)
        
        # Spatial consistency: illumination gradients should be smaller where image gradients are large
        consistency_x = torch.abs(illum_grad_x) * torch.exp(torch.abs(img_grad_x))
        consistency_y = torch.abs(illum_grad_y) * torch.exp(torch.abs(img_grad_y))
        
        return torch.mean(consistency_x) + torch.mean(consistency_y)
    
    def _exposure_control_loss(self, illum_map: torch.Tensor) -> torch.Tensor:
        """
        Exposure control loss to prevent extreme illumination values.
        
        Encourages illumination values to stay within reasonable range.
        """
        # Target illumination range [0.2, 0.8]
        target_min, target_max = 0.2, 0.8
        
        # Penalty for values outside target range
        under_penalty = F.relu(target_min - illum_map)
        over_penalty = F.relu(illum_map - target_max)
        
        exposure_loss = torch.mean(under_penalty**2) + torch.mean(over_penalty**2)
        
        # Additional penalty for extreme values
        extreme_penalty = F.relu(illum_map - 0.95)**2 + F.relu(0.05 - illum_map)**2
        
        return exposure_loss + torch.mean(extreme_penalty)
    
    def _color_constancy_loss(self, illum_map: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Color constancy loss to maintain natural color balance.
        
        Assumes that the average reflectance should be achromatic (gray world assumption).
        """
        # Apply illumination correction to get reflectance
        eps = 1e-6
        corrected = image / (illum_map + eps)
        
        # Gray world assumption: average reflectance should be gray
        mean_r = torch.mean(corrected[:, 0])
        mean_g = torch.mean(corrected[:, 1]) 
        mean_b = torch.mean(corrected[:, 2])
        
        # Color constancy: RGB means should be similar
        color_diff = (mean_r - mean_g)**2 + (mean_g - mean_b)**2 + (mean_b - mean_r)**2
        
        return color_diff
    
    def _illumination_smoothness_loss(self, illum_map: torch.Tensor) -> torch.Tensor:
        """
        Illumination smoothness loss for gradual transitions.
        
        Encourages smooth illumination changes to avoid artifacts.
        """
        # First-order smoothness (gradient magnitude)
        grad_x = illum_map[:, :, :, 1:] - illum_map[:, :, :, :-1]
        grad_y = illum_map[:, :, 1:, :] - illum_map[:, :, :-1, :]
        
        first_order = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))
        
        # Second-order smoothness (curvature)
        second_x = grad_x[:, :, :, 1:] - grad_x[:, :, :, :-1]
        second_y = grad_y[:, :, 1:, :] - grad_y[:, :, :-1, :]
        
        second_order = torch.mean(torch.abs(second_x)) + torch.mean(torch.abs(second_y))
        
        return first_order + 0.5 * second_order
    
    def _region_wise_loss(self, illum_map: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Region-wise non-reference quality loss.
        
        Evaluates enhancement quality in different brightness regions.
        """
        # Convert to grayscale for region analysis
        gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        
        # Define brightness regions
        dark_mask = (gray < 0.3).float()
        normal_mask = ((gray >= 0.3) & (gray <= 0.7)).float()
        bright_mask = (gray > 0.7).float()
        
        # Region-specific losses
        dark_loss = self._dark_region_loss(illum_map, dark_mask)
        normal_loss = self._normal_region_loss(illum_map, normal_mask)
        bright_loss = self._bright_region_loss(illum_map, bright_mask)
        
        return dark_loss + normal_loss + bright_loss
    
    def _dark_region_loss(self, illum_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Loss for dark regions - should be enhanced (higher illumination)."""
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, device=illum_map.device)
        
        # Dark regions should have higher illumination values
        masked_illum = illum_map.squeeze(1) * mask
        avg_illum = torch.sum(masked_illum) / (torch.sum(mask) + 1e-6)
        
        # Penalty if average illumination in dark regions is too low
        target_illum = 0.6
        loss = F.relu(target_illum - avg_illum)**2
        
        return loss
    
    def _normal_region_loss(self, illum_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Loss for normal regions - should be minimally changed."""
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, device=illum_map.device)
        
        # Normal regions should have illumination close to 0.5 (minimal change)
        masked_illum = illum_map.squeeze(1) * mask
        avg_illum = torch.sum(masked_illum) / (torch.sum(mask) + 1e-6)
        
        target_illum = 0.5
        loss = (avg_illum - target_illum)**2
        
        return loss
    
    def _bright_region_loss(self, illum_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Loss for bright regions - should be slightly compressed."""
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, device=illum_map.device)
        
        # Bright regions should have slightly lower illumination
        masked_illum = illum_map.squeeze(1) * mask
        avg_illum = torch.sum(masked_illum) / (torch.sum(mask) + 1e-6)
        
        target_illum = 0.4
        loss = F.relu(avg_illum - target_illum)**2
        
        return loss
