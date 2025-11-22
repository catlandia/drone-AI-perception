"""
Object Detector Module

CNN-based object detection for aerial/drone imagery.
Detects obstacles and outputs bounding boxes with class labels.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class Detection:
    """Single object detection result."""
    bbox: np.ndarray          # [x1, y1, x2, y2] in pixels
    class_id: int             # Class index
    class_name: str           # Human-readable class name
    confidence: float         # Detection confidence (0-1)
    estimated_size: np.ndarray  # Estimated 3D size [w, d, h] meters


class ObjectDetector:
    """
    Object detector for aerial obstacle detection.

    Uses a CNN architecture optimized for drone imagery to detect:
    - Trees
    - Buildings
    - People
    - Vehicles
    - Poles/wires
    - Birds
    - Unknown obstacles

    Performance target: 30+ FPS on modern GPU, 15+ FPS on embedded
    """

    CLASS_NAMES = ["tree", "building", "person", "vehicle", "pole", "wire", "bird", "unknown"]

    # Default object sizes in meters [width, depth, height]
    DEFAULT_SIZES = {
        "tree": [3.0, 3.0, 8.0],
        "building": [15.0, 15.0, 10.0],
        "person": [0.5, 0.5, 1.8],
        "vehicle": [2.0, 4.5, 1.5],
        "pole": [0.3, 0.3, 6.0],
        "wire": [0.1, 50.0, 0.1],
        "bird": [0.3, 0.3, 0.2],
        "unknown": [1.0, 1.0, 1.0],
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ):
        """
        Initialize object detector.

        Args:
            model_path: Path to trained model weights
            device: Computation device ("cuda" or "cpu")
            confidence_threshold: Minimum confidence for detections
            nms_threshold: IoU threshold for NMS
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model = None
        self._model_loaded = False

        if model_path:
            self.load_model(model_path)
        else:
            self._init_default_model()

    def _init_default_model(self):
        """Initialize with default untrained model."""
        try:
            import torch
            from perception.detector.models import PerceptionDetector

            self.model = PerceptionDetector(pretrained=False)
            self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True
        except ImportError:
            # PyTorch not available, use placeholder
            self.model = None
            self._model_loaded = False

    def load_model(self, path: str):
        """Load trained model weights."""
        try:
            import torch
            from perception.detector.models import PerceptionDetector

            self.model = PerceptionDetector()
            state_dict = torch.load(path, map_location=self.device)
            if "detector" in state_dict:
                self.model.load_state_dict(state_dict["detector"])
            else:
                self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load model from {path}: {e}")
            self._init_default_model()

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in an image frame.

        Args:
            frame: RGB image as numpy array (H, W, 3)

        Returns:
            List of Detection objects
        """
        if not self._model_loaded or self.model is None:
            # Return placeholder detections for testing without trained model
            return self._detect_placeholder(frame)

        try:
            import torch

            # Preprocess
            tensor = self._preprocess(frame)

            # Inference
            with torch.no_grad():
                output = self.model(tensor)

            # Postprocess
            detections = self._postprocess(output, frame.shape)

            return detections

        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def _preprocess(self, frame: np.ndarray) -> "torch.Tensor":
        """Preprocess image for model input."""
        import torch

        # Normalize to [0, 1]
        img = frame.astype(np.float32) / 255.0

        # Transpose HWC to CHW
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension
        tensor = torch.from_numpy(img).unsqueeze(0)

        return tensor.to(self.device)

    def _postprocess(
        self,
        output: Dict[str, "torch.Tensor"],
        image_shape: tuple
    ) -> List[Detection]:
        """
        Convert model output to Detection objects.

        Applies confidence filtering and NMS.
        """
        import torch

        boxes = output['boxes'][0]  # (N, 4)
        classes = output['classes'][0]  # (N, num_classes)
        objectness = output['objectness'][0]  # (N, 1)
        sizes = output['sizes'][0]  # (N, 3)

        # Compute final scores
        class_probs = torch.softmax(classes, dim=-1)
        max_probs, class_ids = class_probs.max(dim=-1)
        scores = torch.sigmoid(objectness.squeeze(-1)) * max_probs

        # Filter by confidence
        mask = scores > self.confidence_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        scores = scores[mask]
        sizes = sizes[mask]

        if len(boxes) == 0:
            return []

        # Apply NMS
        keep = self._nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep]
        class_ids = class_ids[keep]
        scores = scores[keep]
        sizes = sizes[keep]

        # Convert to Detection objects
        detections = []
        for i in range(len(boxes)):
            class_id = int(class_ids[i].item())
            class_name = self.CLASS_NAMES[class_id] if class_id < len(self.CLASS_NAMES) else "unknown"

            # Get estimated size (from model or default)
            size_pred = sizes[i].cpu().numpy()
            if np.allclose(size_pred, 0):
                size_pred = np.array(self.DEFAULT_SIZES.get(class_name, [1.0, 1.0, 1.0]))
            else:
                # Model predicts log-scale sizes
                size_pred = np.exp(size_pred)

            det = Detection(
                bbox=boxes[i].cpu().numpy(),
                class_id=class_id,
                class_name=class_name,
                confidence=float(scores[i].item()),
                estimated_size=size_pred
            )
            detections.append(det)

        return detections

    def _nms(
        self,
        boxes: "torch.Tensor",
        scores: "torch.Tensor",
        threshold: float
    ) -> List[int]:
        """Non-maximum suppression."""
        import torch

        # Use torchvision NMS if available
        try:
            from torchvision.ops import nms
            keep = nms(boxes, scores, threshold)
            return keep.cpu().tolist()
        except ImportError:
            pass

        # Fallback implementation
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(descending=True)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break

            i = order[0].item()
            keep.append(i)

            # Compute IoU with remaining boxes
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])

            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # Keep boxes with low IoU
            mask = iou <= threshold
            order = order[1:][mask]

        return keep

    def _detect_placeholder(self, frame: np.ndarray) -> List[Detection]:
        """
        Generate placeholder detections for testing.

        This is used when no trained model is available.
        Simulates detections based on image features.
        """
        detections = []
        h, w = frame.shape[:2]

        # Simple edge detection to find potential objects
        gray = np.mean(frame, axis=2)
        edges = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()

        # Generate 0-3 random detections based on image content
        num_detections = min(3, int(edges / 10))

        for i in range(num_detections):
            # Random box
            x1 = np.random.randint(0, w - 100)
            y1 = np.random.randint(0, h - 100)
            x2 = x1 + np.random.randint(50, min(200, w - x1))
            y2 = y1 + np.random.randint(50, min(200, h - y1))

            # Random class
            class_id = np.random.randint(0, len(self.CLASS_NAMES))
            class_name = self.CLASS_NAMES[class_id]

            det = Detection(
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                class_id=class_id,
                class_name=class_name,
                confidence=0.5 + np.random.rand() * 0.4,
                estimated_size=np.array(self.DEFAULT_SIZES[class_name])
            )
            detections.append(det)

        return detections

    def get_state_dict(self) -> Optional[Dict]:
        """Get model state dictionary."""
        if self.model is not None:
            return self.model.state_dict()
        return None

    def load_state_dict(self, state_dict: Dict):
        """Load model state dictionary."""
        if self.model is not None:
            self.model.load_state_dict(state_dict)
