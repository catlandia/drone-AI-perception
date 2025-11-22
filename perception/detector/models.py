"""
Neural Network Architectures for Object Detection

Provides backbone and detection head networks optimized for
aerial/drone imagery obstacle detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class DetectorBackbone(nn.Module):
    """
    Lightweight backbone network for object detection.

    Designed for real-time performance on drone hardware.
    Input: RGB image (B, 3, H, W)
    Output: Feature maps at multiple scales
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        # Initial convolution
        self.stem = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=2, padding=1),  # /2
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
        )

        # Stage 1: 1/4 resolution
        self.stage1 = nn.Sequential(
            ConvBlock(64, 64, stride=2),  # /4
            ResidualBlock(64),
            ResidualBlock(64),
        )

        # Stage 2: 1/8 resolution
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, stride=2),  # /8
            ResidualBlock(128),
            ResidualBlock(128),
        )

        # Stage 3: 1/16 resolution
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, stride=2),  # /16
            ResidualBlock(256),
            ResidualBlock(256),
        )

        # Stage 4: 1/32 resolution
        self.stage4 = nn.Sequential(
            ConvBlock(256, 512, stride=2),  # /32
            ResidualBlock(512),
            ResidualBlock(512),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.

        Returns:
            List of feature maps at scales [1/4, 1/8, 1/16, 1/32]
        """
        x = self.stem(x)
        c1 = self.stage1(x)   # 1/4
        c2 = self.stage2(c1)  # 1/8
        c3 = self.stage3(c2)  # 1/16
        c4 = self.stage4(c3)  # 1/32
        return [c1, c2, c3, c4]


class FPN(nn.Module):
    """
    Feature Pyramid Network for multi-scale detection.

    Combines features from different scales for detecting
    objects at various sizes and distances.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels_list
        ])

        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Build feature pyramid."""
        # Lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )

        # Output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs


class DetectionHead(nn.Module):
    """
    Detection head for predicting bounding boxes and classes.

    Outputs:
        - Box coordinates (4 values per anchor)
        - Class scores (num_classes per anchor)
        - Objectness score (1 value per anchor)
    """

    # Object types for aerial/drone detection
    CLASS_NAMES = ["tree", "building", "person", "vehicle", "pole", "wire", "bird", "unknown"]
    NUM_CLASSES = len(CLASS_NAMES)

    def __init__(
        self,
        in_channels: int = 256,
        num_anchors: int = 3,
        num_classes: int = NUM_CLASSES
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Shared convolutions
        self.shared_conv = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            ConvBlock(in_channels, in_channels),
        )

        # Box regression head
        self.box_head = nn.Conv2d(
            in_channels,
            num_anchors * 4,  # x, y, w, h
            kernel_size=1
        )

        # Classification head
        self.cls_head = nn.Conv2d(
            in_channels,
            num_anchors * num_classes,
            kernel_size=1
        )

        # Objectness head
        self.obj_head = nn.Conv2d(
            in_channels,
            num_anchors,
            kernel_size=1
        )

        # Size estimation head (for 3D size prediction)
        self.size_head = nn.Conv2d(
            in_channels,
            num_anchors * 3,  # width, depth, height
            kernel_size=1
        )

    def forward(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict detections from feature map.

        Returns:
            boxes: (B, H*W*A, 4) - bounding box coordinates
            classes: (B, H*W*A, num_classes) - class scores
            objectness: (B, H*W*A, 1) - objectness scores
            sizes: (B, H*W*A, 3) - estimated 3D sizes
        """
        B = features.shape[0]
        shared = self.shared_conv(features)

        # Predictions
        boxes = self.box_head(shared)
        classes = self.cls_head(shared)
        objectness = self.obj_head(shared)
        sizes = self.size_head(shared)

        # Reshape to (B, H*W*A, C)
        boxes = boxes.permute(0, 2, 3, 1).reshape(B, -1, 4)
        classes = classes.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
        objectness = objectness.permute(0, 2, 3, 1).reshape(B, -1, 1)
        sizes = sizes.permute(0, 2, 3, 1).reshape(B, -1, 3)

        return boxes, classes, objectness, sizes


class PerceptionDetector(nn.Module):
    """
    Complete detection model combining backbone, FPN, and detection heads.

    Optimized for aerial obstacle detection with:
    - Multi-scale feature extraction
    - Class-specific detection (trees, buildings, people, etc.)
    - 3D size estimation
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        self.backbone = DetectorBackbone(pretrained=pretrained)
        self.fpn = FPN(
            in_channels_list=[64, 128, 256, 512],
            out_channels=256
        )
        self.heads = nn.ModuleList([
            DetectionHead(in_channels=256)
            for _ in range(4)  # One head per FPN level
        ])

        # Anchor sizes for different FPN levels (in pixels)
        self.anchor_sizes = [
            [16, 32, 64],      # Small objects (close)
            [32, 64, 128],     # Medium objects
            [64, 128, 256],    # Large objects
            [128, 256, 512],   # Very large objects (far)
        ]

    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass.

        Args:
            x: Input image tensor (B, 3, H, W)

        Returns:
            Dict with 'boxes', 'classes', 'objectness', 'sizes' for each level
        """
        # Extract multi-scale features
        backbone_features = self.backbone(x)
        fpn_features = self.fpn(backbone_features)

        # Detect at each scale
        all_boxes = []
        all_classes = []
        all_objectness = []
        all_sizes = []

        for i, (feat, head) in enumerate(zip(fpn_features, self.heads)):
            boxes, classes, objectness, sizes = head(feat)
            all_boxes.append(boxes)
            all_classes.append(classes)
            all_objectness.append(objectness)
            all_sizes.append(sizes)

        return {
            'boxes': torch.cat(all_boxes, dim=1),
            'classes': torch.cat(all_classes, dim=1),
            'objectness': torch.cat(all_objectness, dim=1),
            'sizes': torch.cat(all_sizes, dim=1),
        }

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
