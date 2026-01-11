"""
Lightweight feature extraction network for illumination analysis.

Implements LightFeatNet - a multi-scale CNN with Global Channel Attention (GCA)
for extracting illumination-aware features from dual image representations.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalChannelAttention(nn.Module):
    """Global Channel Attention module for feature recalibration."""
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize GCA module.
        
        Args:
            channels: Number of input channels
            reduction: Channel reduction factor for bottleneck
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GCA module.
        
        Args:
            x: Input feature tensor (B, C, H, W)
            
        Returns:
            Attention-weighted feature tensor
        """
        # Average and max pooling
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # Combine and apply attention
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class ConvBlock(nn.Module):
    """Convolution block with normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, dilation: int = 1, 
                 use_attention: bool = False):
        """
        Initialize convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            dilation: Dilation factor
            use_attention: Whether to include GCA module
        """
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             padding=padding, dilation=dilation, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.attention = GlobalChannelAttention(out_channels) if use_attention else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of convolution block."""
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        
        if self.attention is not None:
            x = self.attention(x)
            
        return x


class LightFeatNet(nn.Module):
    """
    Lightweight CNN for illumination feature extraction.
    
    Architecture:
    - 3-stage conv blocks with dilations (1, 2, 4)
    - Multi-scale feature fusion
    - Global Channel Attention
    - Output: 1/4 resolution feature maps
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize LightFeatNet.
        
        Args:
            cfg: Configuration dictionary with feature_net parameters
        """
        super().__init__()
        
        # Extract config parameters
        input_channels = cfg.get('input_channels', 6)  # RGB + inverted RGB
        base_channels = cfg.get('base_channels', 32)
        dilations = cfg.get('dilations', [1, 2, 4])
        use_gca = cfg.get('use_gca', True)
        
        # Input projection
        self.input_conv = ConvBlock(input_channels, base_channels, kernel_size=3)
        
        # Multi-scale feature extraction stages
        self.stages = nn.ModuleList()
        in_ch = base_channels
        
        for i, dilation in enumerate(dilations):
            out_ch = base_channels * (2 ** i)
            use_attn = use_gca and (i == len(dilations) - 1)  # Only last stage
            
            stage = nn.Sequential(
                ConvBlock(in_ch, out_ch, kernel_size=3, dilation=dilation),
                ConvBlock(out_ch, out_ch, kernel_size=3, dilation=1, use_attention=use_attn)
            )
            self.stages.append(stage)
            in_ch = out_ch
        
        # Feature fusion
        total_channels = sum(base_channels * (2 ** i) for i in range(len(dilations)))
        self.fusion_conv = ConvBlock(total_channels, base_channels * 4, kernel_size=1)
        
        # Downsampling to 1/4 resolution
        self.downsample = nn.AvgPool2d(kernel_size=4, stride=4)
        
        self.output_channels = base_channels * 4
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LightFeatNet.
        
        Args:
            x: Input tensor (B, C, H, W) where C=6 for dual representation
            
        Returns:
            Feature tensor (B, C_out, H/4, W/4)
        """
        # Input projection
        x = self.input_conv(x)
        
        # Multi-scale feature extraction
        features = []
        current = x
        
        for stage in self.stages:
            current = stage(current)
            features.append(current)
        
        # Concatenate multi-scale features
        fused = torch.cat(features, dim=1)
        
        # Final fusion
        fused = self.fusion_conv(fused)
        
        # Downsample to 1/4 resolution
        output = self.downsample(fused)
        
        return output


class IlluminationHead(nn.Module):
    """Head network for illumination map estimation."""
    
    def __init__(self, input_channels: int):
        """
        Initialize illumination head.
        
        Args:
            input_channels: Number of input feature channels
        """
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 2, input_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 4, 1, 1),  # 1x1 conv to single channel
            nn.Sigmoid()  # Constrain output to [0, 1]
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of illumination head.
        
        Args:
            features: Input feature tensor (B, C, H/4, W/4)
            
        Returns:
            Illumination map (B, 1, H/4, W/4) with values in [0, 1]
        """
        return self.conv_layers(features)


def create_network(cfg: Dict[str, Any]) -> nn.Module:
    """
    Create complete feature extraction network.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Complete network module
    """
    return LightFeatNet(cfg.get('feature_net', {}))


def load_pretrained_weights(model: nn.Module, weights_path: str) -> nn.Module:
    """
    Load pretrained weights into model.
    
    Args:
        model: PyTorch model
        weights_path: Path to weights file (.pth)
        
    Returns:
        Model with loaded weights
    """
    try:
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {weights_path}")
    except FileNotFoundError:
        print(f"Weights file not found: {weights_path}. Using random initialization.")
    except Exception as e:
        print(f"Error loading weights: {e}. Using random initialization.")
    
    return model


def initialize_weights(model: nn.Module) -> None:
    """Initialize network weights using Xavier/He initialization."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
