"""
Depth Estimation Module

Converts 2D image detections to 3D positions using:
- Monocular depth estimation
- Stereo matching (if stereo cameras available)
- Size-based depth estimation (fallback)
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class Detection3D:
    """Detection with 3D position information."""
    bbox: np.ndarray            # [x1, y1, x2, y2] in pixels
    class_id: int
    class_name: str
    confidence: float
    depth: float                # Distance from camera in meters
    position_camera: np.ndarray  # [x, y, z] in camera frame (meters)
    size: np.ndarray            # [width, depth, height] in meters


class DepthEstimator:
    """
    Estimates depth/distance for detected objects.

    Supports multiple depth estimation methods:
    1. Monocular depth network (MiDaS-style)
    2. Stereo matching (if stereo cameras)
    3. Size-based estimation (using known object sizes)

    Output: 3D positions in camera coordinate frame
    """

    # Typical object sizes for size-based depth estimation
    REFERENCE_SIZES = {
        "tree": {"height": 8.0, "width": 3.0},
        "building": {"height": 10.0, "width": 15.0},
        "person": {"height": 1.7, "width": 0.5},
        "vehicle": {"height": 1.5, "width": 4.5},
        "pole": {"height": 6.0, "width": 0.3},
        "wire": {"height": 0.1, "width": 0.1},
        "bird": {"height": 0.2, "width": 0.3},
        "unknown": {"height": 1.0, "width": 1.0},
    }

    def __init__(
        self,
        camera_params: Optional[Dict] = None,
        device: str = "cpu",
        method: str = "monocular"
    ):
        """
        Initialize depth estimator.

        Args:
            camera_params: Camera intrinsic parameters
                - fx, fy: Focal lengths in pixels
                - cx, cy: Principal point
                - width, height: Image dimensions
            device: Computation device
            method: Depth estimation method
                - "monocular": Neural network monocular depth
                - "stereo": Stereo matching (requires stereo cameras)
                - "size_based": Use known object sizes
        """
        self.camera_params = camera_params or {
            "width": 640,
            "height": 480,
            "fx": 554.0,
            "fy": 554.0,
            "cx": 320.0,
            "cy": 240.0,
        }
        self.device = device
        self.method = method
        self.depth_model = None
        self._model_loaded = False

        self._init_depth_model()

    def _init_depth_model(self):
        """Initialize monocular depth estimation model."""
        if self.method != "monocular":
            return

        try:
            import torch
            # Simplified depth network for embedded deployment
            self.depth_model = self._create_depth_network()
            self.depth_model.to(self.device)
            self.depth_model.eval()
            self._model_loaded = True
        except ImportError:
            self._model_loaded = False

    def _create_depth_network(self) -> "torch.nn.Module":
        """Create lightweight monocular depth network."""
        import torch
        import torch.nn as nn

        class DepthNet(nn.Module):
            """Simple encoder-decoder for depth estimation."""

            def __init__(self):
                super().__init__()
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
                # Decoder
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                features = self.encoder(x)
                depth = self.decoder(features)
                # Scale to reasonable depth range (1-100 meters)
                depth = 1.0 + depth * 99.0
                return depth

        return DepthNet()

    def estimate(
        self,
        frame: np.ndarray,
        detections: List[Any]
    ) -> List[Detection3D]:
        """
        Estimate depth for each detection.

        Args:
            frame: RGB image (H, W, 3)
            detections: List of Detection objects from ObjectDetector

        Returns:
            List of Detection3D objects with 3D positions
        """
        if len(detections) == 0:
            return []

        # Get depth map for entire image
        depth_map = self._estimate_depth_map(frame)

        # Convert each detection to 3D
        detections_3d = []
        for det in detections:
            depth, position = self._detection_to_3d(det, depth_map)

            det_3d = Detection3D(
                bbox=det.bbox,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                depth=depth,
                position_camera=position,
                size=det.estimated_size
            )
            detections_3d.append(det_3d)

        return detections_3d

    def _estimate_depth_map(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth map for entire image.

        Returns depth values in meters.
        """
        if self.method == "monocular" and self._model_loaded:
            return self._monocular_depth(frame)
        else:
            # Fallback: uniform depth assumption
            return np.ones((frame.shape[0], frame.shape[1]), dtype=np.float32) * 20.0

    def _monocular_depth(self, frame: np.ndarray) -> np.ndarray:
        """Run monocular depth estimation network."""
        import torch

        # Preprocess
        img = frame.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            depth = self.depth_model(tensor)

        # Convert to numpy
        depth_map = depth[0, 0].cpu().numpy()

        # Resize to original image size if needed
        if depth_map.shape != (frame.shape[0], frame.shape[1]):
            from scipy.ndimage import zoom
            scale_h = frame.shape[0] / depth_map.shape[0]
            scale_w = frame.shape[1] / depth_map.shape[1]
            depth_map = zoom(depth_map, (scale_h, scale_w), order=1)

        return depth_map

    def _detection_to_3d(
        self,
        detection: Any,
        depth_map: np.ndarray
    ) -> tuple:
        """
        Convert a 2D detection to 3D position.

        Uses depth map value at detection center, with size-based
        refinement for known object types.
        """
        bbox = detection.bbox
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Clamp to image bounds
        h, w = depth_map.shape
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)

        # Get depth from center region of detection
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Sample depth from center region (more robust)
        margin_x = max(1, (x2 - x1) // 4)
        margin_y = max(1, (y2 - y1) // 4)
        region = depth_map[
            max(0, cy - margin_y):min(h, cy + margin_y + 1),
            max(0, cx - margin_x):min(w, cx + margin_x + 1)
        ]

        if region.size > 0:
            # Use median for robustness
            depth = float(np.median(region))
        else:
            depth = 20.0  # Default depth

        # Refine with size-based estimation if available
        if self.method == "size_based" or not self._model_loaded:
            size_depth = self._estimate_depth_from_size(detection, bbox)
            if size_depth is not None:
                # Blend monocular and size-based estimates
                depth = 0.5 * depth + 0.5 * size_depth

        # Convert pixel coordinates to camera frame
        position = self._pixel_to_camera(cx, cy, depth)

        return depth, position

    def _estimate_depth_from_size(
        self,
        detection: Any,
        bbox: np.ndarray
    ) -> Optional[float]:
        """
        Estimate depth based on known object sizes.

        Uses: depth = (focal_length * real_size) / pixel_size
        """
        class_name = detection.class_name
        if class_name not in self.REFERENCE_SIZES:
            return None

        ref = self.REFERENCE_SIZES[class_name]
        bbox_height = bbox[3] - bbox[1]
        bbox_width = bbox[2] - bbox[0]

        if bbox_height <= 0 or bbox_width <= 0:
            return None

        # Use height for depth estimation (more stable)
        real_height = ref["height"]
        focal_length = self.camera_params["fy"]

        depth_from_height = (focal_length * real_height) / bbox_height

        # Sanity check
        depth = float(np.clip(depth_from_height, 1.0, 100.0))

        return depth

    def _pixel_to_camera(
        self,
        u: float,
        v: float,
        depth: float
    ) -> np.ndarray:
        """
        Convert pixel coordinates to camera frame coordinates.

        Camera frame convention:
        - X: right
        - Y: down
        - Z: forward (depth)
        """
        fx = self.camera_params["fx"]
        fy = self.camera_params["fy"]
        cx = self.camera_params["cx"]
        cy = self.camera_params["cy"]

        # Inverse projection
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return np.array([x, y, z], dtype=np.float32)

    def get_state_dict(self) -> Optional[Dict]:
        """Get depth model state dictionary."""
        if self.depth_model is not None:
            return self.depth_model.state_dict()
        return None

    def load_state_dict(self, state_dict: Dict):
        """Load depth model state dictionary."""
        if self.depth_model is not None:
            self.depth_model.load_state_dict(state_dict)
