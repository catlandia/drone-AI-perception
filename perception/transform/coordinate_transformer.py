"""
Coordinate Transformation Module

Transforms detections from camera frame to world coordinates
using drone position and orientation.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class WorldDetection:
    """Detection in world coordinates."""
    id: int                      # Placeholder ID (assigned by tracker)
    type: str                    # Object class name
    position: np.ndarray         # [x, y, z] world coordinates in meters
    size: np.ndarray             # [width, depth, height] in meters
    velocity: np.ndarray         # [vx, vy, vz] placeholder (tracker fills in)
    confidence: float            # Detection confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "position": self.position,
            "size": self.size,
            "velocity": self.velocity,
            "confidence": self.confidence
        }


class CoordinateTransformer:
    """
    Transforms coordinates between camera and world frames.

    Coordinate Conventions:
        Camera Frame:
            - X: right
            - Y: down
            - Z: forward

        World Frame (NED - North-East-Down):
            - X: North (forward)
            - Y: East (right)
            - Z: Down (into ground)

        Alternative World Frame (ENU - East-North-Up):
            - X: East
            - Y: North
            - Z: Up

    The transformation uses:
        P_world = R_drone * R_cam_to_body * P_camera + T_drone

    Where:
        - R_drone: Drone orientation in world frame
        - R_cam_to_body: Camera mounting rotation
        - P_camera: Point in camera frame
        - T_drone: Drone position in world frame
    """

    def __init__(
        self,
        camera_params: Optional[Dict] = None,
        camera_mount: str = "forward",
        world_frame: str = "NED"
    ):
        """
        Initialize coordinate transformer.

        Args:
            camera_params: Camera intrinsic parameters
            camera_mount: Camera mounting direction
                - "forward": Camera faces drone forward direction
                - "down": Camera faces down (nadir view)
                - "forward_down": 45 degree tilt
            world_frame: World coordinate system ("NED" or "ENU")
        """
        self.camera_params = camera_params or {
            "width": 640,
            "height": 480,
            "fx": 554.0,
            "fy": 554.0,
            "cx": 320.0,
            "cy": 240.0,
        }
        self.world_frame = world_frame

        # Camera to body frame rotation
        self.R_cam_to_body = self._get_camera_mount_rotation(camera_mount)

    def _get_camera_mount_rotation(self, mount: str) -> np.ndarray:
        """
        Get rotation matrix from camera to body frame.

        Camera frame: X-right, Y-down, Z-forward
        Body frame: X-forward, Y-right, Z-down
        """
        if mount == "forward":
            # Camera aligned with body forward
            # Rotate camera axes to match body axes
            return np.array([
                [0, 0, 1],   # Body X (forward) = Camera Z
                [1, 0, 0],   # Body Y (right) = Camera X
                [0, 1, 0],   # Body Z (down) = Camera Y
            ], dtype=np.float32)

        elif mount == "down":
            # Camera looking straight down (nadir)
            return np.array([
                [0, -1, 0],  # Body X (forward) = -Camera Y
                [1, 0, 0],   # Body Y (right) = Camera X
                [0, 0, 1],   # Body Z (down) = Camera Z
            ], dtype=np.float32)

        elif mount == "forward_down":
            # Camera tilted 45 degrees down
            angle = np.radians(-45)
            c, s = np.cos(angle), np.sin(angle)
            R_tilt = np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c],
            ], dtype=np.float32)
            R_forward = np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ], dtype=np.float32)
            return R_forward @ R_tilt

        else:
            # Default: identity (camera = body)
            return np.eye(3, dtype=np.float32)

    def camera_to_world(
        self,
        detections_3d: List[Any],
        drone_position: np.ndarray,
        drone_orientation: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Transform detections from camera frame to world frame.

        Args:
            detections_3d: List of Detection3D objects
            drone_position: [x, y, z] drone position in world frame (meters)
            drone_orientation: Quaternion [w, x, y, z] or euler [roll, pitch, yaw]

        Returns:
            List of detection dicts with world positions
        """
        # Get rotation matrix from drone orientation
        R_drone = self._quaternion_to_rotation(drone_orientation)

        # Combined rotation: world <- body <- camera
        R_total = R_drone @ self.R_cam_to_body

        world_detections = []
        for i, det in enumerate(detections_3d):
            # Transform position
            p_camera = det.position_camera
            p_world = R_total @ p_camera + drone_position

            # Transform size (approximate - just apply rotation to size vector)
            # For box sizes, we keep them axis-aligned in world frame
            size_world = np.abs(R_total @ det.size)

            world_det = {
                "id": -1,  # Placeholder, tracker assigns real ID
                "type": det.class_name,
                "position": p_world,
                "size": size_world,
                "velocity": np.zeros(3, dtype=np.float32),  # Tracker estimates
                "confidence": det.confidence,
                "bbox": det.bbox,  # Keep for tracking
                "depth": det.depth
            }
            world_detections.append(world_det)

        return world_detections

    def world_to_camera(
        self,
        world_point: np.ndarray,
        drone_position: np.ndarray,
        drone_orientation: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Project a world point to camera pixel coordinates.

        Args:
            world_point: [x, y, z] in world frame
            drone_position: Drone position in world frame
            drone_orientation: Drone orientation (quaternion)

        Returns:
            [u, v] pixel coordinates, or None if behind camera
        """
        # Get rotation matrix
        R_drone = self._quaternion_to_rotation(drone_orientation)
        R_total = R_drone @ self.R_cam_to_body

        # Transform to camera frame
        p_relative = world_point - drone_position
        p_camera = R_total.T @ p_relative

        # Check if point is behind camera
        if p_camera[2] <= 0:
            return None

        # Project to image
        fx = self.camera_params["fx"]
        fy = self.camera_params["fy"]
        cx = self.camera_params["cx"]
        cy = self.camera_params["cy"]

        u = fx * p_camera[0] / p_camera[2] + cx
        v = fy * p_camera[1] / p_camera[2] + cy

        # Check if in image bounds
        w = self.camera_params["width"]
        h = self.camera_params["height"]

        if 0 <= u < w and 0 <= v < h:
            return np.array([u, v], dtype=np.float32)
        return None

    def _quaternion_to_rotation(self, q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.

        Args:
            q: Quaternion [w, x, y, z] or euler [roll, pitch, yaw]

        Returns:
            3x3 rotation matrix
        """
        q = np.asarray(q, dtype=np.float64)

        if len(q) == 3:
            # Convert euler to quaternion first
            q = self._euler_to_quaternion(q)

        # Normalize quaternion
        q = q / np.linalg.norm(q)
        w, x, y, z = q

        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ], dtype=np.float32)

        return R

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

        return np.array([w, x, y, z], dtype=np.float64)

    def get_transform_matrix(
        self,
        drone_position: np.ndarray,
        drone_orientation: np.ndarray
    ) -> np.ndarray:
        """
        Get 4x4 transformation matrix from camera to world.

        Returns:
            4x4 homogeneous transformation matrix
        """
        R_drone = self._quaternion_to_rotation(drone_orientation)
        R_total = R_drone @ self.R_cam_to_body

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R_total
        T[:3, 3] = drone_position

        return T

    def compute_frustum_corners(
        self,
        drone_position: np.ndarray,
        drone_orientation: np.ndarray,
        near: float = 1.0,
        far: float = 100.0
    ) -> np.ndarray:
        """
        Compute camera frustum corners in world coordinates.

        Useful for determining visible area.

        Returns:
            8x3 array of corner positions (4 near + 4 far)
        """
        w = self.camera_params["width"]
        h = self.camera_params["height"]
        fx = self.camera_params["fx"]
        fy = self.camera_params["fy"]
        cx = self.camera_params["cx"]
        cy = self.camera_params["cy"]

        # Image corners
        corners_2d = [
            [0, 0], [w, 0], [w, h], [0, h]
        ]

        corners_world = []

        for depth in [near, far]:
            for u, v in corners_2d:
                # Pixel to camera coordinates
                x = (u - cx) * depth / fx
                y = (v - cy) * depth / fy
                z = depth

                p_camera = np.array([x, y, z], dtype=np.float32)

                # Transform to world
                R_drone = self._quaternion_to_rotation(drone_orientation)
                R_total = R_drone @ self.R_cam_to_body
                p_world = R_total @ p_camera + drone_position

                corners_world.append(p_world)

        return np.array(corners_world)
