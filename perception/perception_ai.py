"""
Main Perception AI Interface

This is the primary interface class that combines all perception components
to provide unified obstacle detection and tracking for drone navigation.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import time

from perception.detector.object_detector import ObjectDetector
from perception.depth.depth_estimator import DepthEstimator
from perception.transform.coordinate_transformer import CoordinateTransformer
from perception.tracker.object_tracker import ObjectTracker


@dataclass
class Obstacle:
    """Represents a detected obstacle in world coordinates."""
    id: int                          # Tracking ID (same object = same ID)
    type: str                        # tree/building/person/vehicle/unknown
    position: List[float]            # World coordinates [x, y, z] in meters
    size: List[float]                # [width, depth, height] in meters
    velocity: List[float]            # [vx, vy, vz] in m/s (for moving objects)
    confidence: float                # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


class PerceptionAI:
    """
    Main Perception AI interface for drone obstacle detection.

    This class integrates:
    - Object Detector: CNN-based detection from camera images
    - Depth Estimator: 2D to 3D conversion
    - Coordinate Transformer: Camera to world coordinates
    - Object Tracker: Multi-object tracking with persistent IDs

    Usage:
        perception = PerceptionAI(model_path="models/A+ 30-11-2025 perception v1.pt")
        result = perception.detect(frame, drone_position, drone_orientation)

    Output format:
        {
            "obstacles": [
                {
                    "id": 1,
                    "type": "tree",
                    "position": [10.5, 5.2, 0.0],
                    "size": [2.0, 2.0, 8.0],
                    "velocity": [0.0, 0.0, 0.0],
                    "confidence": 0.92
                },
                ...
            ],
            "nearest_obstacle_distance": 10.5,
            "ground_distance": 25.0
        }
    """

    # Supported object types
    OBJECT_TYPES = ["tree", "building", "person", "vehicle", "pole", "wire", "bird", "unknown"]

    # Detection range limits (meters)
    MIN_DETECTION_RANGE = 5.0
    MAX_DETECTION_RANGE = 100.0

    def __init__(
        self,
        model_path: Optional[str] = None,
        camera_params: Optional[Dict] = None,
        device: str = "auto"
    ):
        """
        Initialize Perception AI with trained models.

        Args:
            model_path: Path to trained perception model (.pt file)
                       Format: "{Grade} {DD-MM-YYYY} perception v{N}.pt"
                       Example: "A+ 30-11-2025 perception v1.pt"
            camera_params: Camera intrinsic parameters (optional)
                          Default assumes 640x480 with typical drone camera FOV
            device: Computation device ("auto", "cuda", "cpu")
        """
        self.model_path = model_path
        self.device = self._resolve_device(device)

        # Default camera parameters (640x480, ~60 degree FOV)
        self.camera_params = camera_params or {
            "width": 640,
            "height": 480,
            "fx": 554.0,  # Focal length x
            "fy": 554.0,  # Focal length y
            "cx": 320.0,  # Principal point x
            "cy": 240.0,  # Principal point y
        }

        # Initialize components
        self.detector = ObjectDetector(model_path=model_path, device=self.device)
        self.depth_estimator = DepthEstimator(camera_params=self.camera_params, device=self.device)
        self.transformer = CoordinateTransformer(camera_params=self.camera_params)
        self.tracker = ObjectTracker(max_age=30, min_hits=3)

        # Performance tracking
        self._frame_count = 0
        self._total_time = 0.0
        self._last_fps = 0.0

    def _resolve_device(self, device: str) -> str:
        """Resolve computation device."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def detect(
        self,
        frame: np.ndarray,
        drone_position: np.ndarray,
        drone_orientation: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process a camera frame and detect obstacles in world coordinates.

        Args:
            frame: RGB image as numpy array with shape (H, W, 3)
                   Expected size: 640x480 or similar
            drone_position: Drone position [x, y, z] in meters (world frame)
            drone_orientation: Drone orientation as quaternion [w, x, y, z]
                              or euler angles [roll, pitch, yaw] in radians

        Returns:
            Dictionary containing:
                - obstacles: List of detected obstacles with world positions
                - nearest_obstacle_distance: Distance to closest obstacle (meters)
                - ground_distance: Height above ground (meters)

        Performance:
            Target: 15+ FPS, <100ms latency
        """
        start_time = time.time()

        # Validate inputs
        frame = self._validate_frame(frame)
        drone_position = np.asarray(drone_position, dtype=np.float32)
        drone_orientation = self._normalize_orientation(drone_orientation)

        # Step 1: Detect objects in image
        detections = self.detector.detect(frame)

        # Step 2: Estimate depth for each detection
        detections_with_depth = self.depth_estimator.estimate(frame, detections)

        # Step 3: Transform to world coordinates
        world_detections = self.transformer.camera_to_world(
            detections_with_depth,
            drone_position,
            drone_orientation
        )

        # Step 4: Track objects across frames
        tracked_obstacles = self.tracker.update(world_detections)

        # Build output
        obstacles = []
        for track in tracked_obstacles:
            obstacle = Obstacle(
                id=track["id"],
                type=track["type"],
                position=track["position"].tolist(),
                size=track["size"].tolist(),
                velocity=track["velocity"].tolist(),
                confidence=track["confidence"]
            )
            obstacles.append(obstacle.to_dict())

        # Calculate summary statistics
        nearest_distance = self._calculate_nearest_distance(obstacles, drone_position)
        ground_distance = self._estimate_ground_distance(frame, drone_position, drone_orientation)

        # Update performance metrics
        elapsed = time.time() - start_time
        self._update_performance(elapsed)

        return {
            "obstacles": obstacles,
            "nearest_obstacle_distance": nearest_distance,
            "ground_distance": ground_distance,
            "frame_time_ms": elapsed * 1000,
            "fps": self._last_fps
        }

    def _validate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Validate and normalize input frame."""
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {frame.shape}")

        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        return frame

    def _normalize_orientation(self, orientation: np.ndarray) -> np.ndarray:
        """Normalize orientation to quaternion format [w, x, y, z]."""
        orientation = np.asarray(orientation, dtype=np.float32)

        if len(orientation) == 3:
            # Convert euler angles [roll, pitch, yaw] to quaternion
            orientation = self._euler_to_quaternion(orientation)
        elif len(orientation) == 4:
            # Normalize quaternion
            norm = np.linalg.norm(orientation)
            if norm > 0:
                orientation = orientation / norm
        else:
            raise ValueError(f"Orientation must have 3 (euler) or 4 (quaternion) elements, got {len(orientation)}")

        return orientation

    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """Convert euler angles [roll, pitch, yaw] to quaternion [w, x, y, z]."""
        roll, pitch, yaw = euler

        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z], dtype=np.float32)

    def _calculate_nearest_distance(
        self,
        obstacles: List[Dict],
        drone_position: np.ndarray
    ) -> float:
        """Calculate distance to nearest obstacle."""
        if not obstacles:
            return float('inf')

        min_distance = float('inf')
        for obs in obstacles:
            pos = np.array(obs["position"])
            distance = np.linalg.norm(pos - drone_position)
            min_distance = min(min_distance, distance)

        return float(min_distance)

    def _estimate_ground_distance(
        self,
        frame: np.ndarray,
        drone_position: np.ndarray,
        drone_orientation: np.ndarray
    ) -> float:
        """Estimate height above ground."""
        # Use depth estimation for ground plane detection
        # For now, use drone z-coordinate as approximation
        # TODO: Implement proper ground plane detection from depth map
        return float(drone_position[2])

    def _update_performance(self, elapsed: float):
        """Update FPS and performance metrics."""
        self._frame_count += 1
        self._total_time += elapsed

        # Calculate running average FPS
        if self._total_time > 0:
            self._last_fps = self._frame_count / self._total_time

        # Reset counters periodically to avoid overflow
        if self._frame_count >= 1000:
            self._frame_count = 100
            self._total_time = 100 / self._last_fps if self._last_fps > 0 else 1.0

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        return {
            "average_fps": self._last_fps,
            "total_frames": self._frame_count,
            "total_time_s": self._total_time
        }

    def reset_tracker(self):
        """Reset the object tracker (e.g., for new scene)."""
        self.tracker.reset()

    def save_model(self, path: str):
        """Save the current model state."""
        import torch

        state = {
            "detector": self.detector.get_state_dict(),
            "depth_estimator": self.depth_estimator.get_state_dict(),
            "camera_params": self.camera_params,
            "version": "0.1.0"
        }
        torch.save(state, path)

    def load_model(self, path: str):
        """Load model state from file."""
        import torch

        state = torch.load(path, map_location=self.device)
        self.detector.load_state_dict(state["detector"])
        self.depth_estimator.load_state_dict(state["depth_estimator"])
        self.camera_params = state.get("camera_params", self.camera_params)
