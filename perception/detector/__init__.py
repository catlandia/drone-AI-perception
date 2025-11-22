"""Object Detection Module"""
from perception.detector.object_detector import ObjectDetector
from perception.detector.models import DetectorBackbone, DetectionHead

__all__ = ["ObjectDetector", "DetectorBackbone", "DetectionHead"]
