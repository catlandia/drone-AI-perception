"""
Perception AI - Layer 2 of the Drone AI System

This module provides obstacle detection and tracking for drone navigation.
It processes camera frames along with drone pose to output world-coordinate
obstacle information for path planning.

Components:
    - ObjectDetector: CNN-based object detection from aerial imagery
    - DepthEstimator: Converts 2D detections to 3D positions
    - CoordinateTransformer: Camera to world coordinate transformation
    - ObjectTracker: Multi-object tracking with Kalman filtering

Main Interface:
    PerceptionAI: Unified interface combining all components
"""

from perception.perception_ai import PerceptionAI
from perception.detector.object_detector import ObjectDetector
from perception.depth.depth_estimator import DepthEstimator
from perception.transform.coordinate_transformer import CoordinateTransformer
from perception.tracker.object_tracker import ObjectTracker

__version__ = "0.1.0"
__all__ = [
    "PerceptionAI",
    "ObjectDetector",
    "DepthEstimator",
    "CoordinateTransformer",
    "ObjectTracker",
]
