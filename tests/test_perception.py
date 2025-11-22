"""
Unit Tests for Perception AI

Run with: pytest tests/test_perception.py -v
"""

import pytest
import numpy as np
from datetime import datetime

# Import perception modules
import sys
sys.path.insert(0, '.')


class TestCoordinateTransformer:
    """Tests for coordinate transformation."""

    def test_quaternion_to_rotation_identity(self):
        """Test identity quaternion produces identity rotation."""
        from perception.transform.coordinate_transformer import CoordinateTransformer

        transformer = CoordinateTransformer()
        quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        R = transformer._quaternion_to_rotation(quat)

        expected = np.eye(3)
        np.testing.assert_array_almost_equal(R, expected, decimal=5)

    def test_euler_to_quaternion(self):
        """Test euler to quaternion conversion."""
        from perception.transform.coordinate_transformer import CoordinateTransformer

        transformer = CoordinateTransformer()

        # Zero rotation
        euler = np.array([0.0, 0.0, 0.0])
        quat = transformer._euler_to_quaternion(euler)
        np.testing.assert_array_almost_equal(quat, [1, 0, 0, 0], decimal=5)

    def test_camera_to_world_basic(self):
        """Test basic camera to world transformation."""
        from perception.transform.coordinate_transformer import CoordinateTransformer
        from perception.depth.depth_estimator import Detection3D

        transformer = CoordinateTransformer()

        # Create mock detection
        det = Detection3D(
            bbox=np.array([100, 100, 200, 200]),
            class_id=0,
            class_name="tree",
            confidence=0.9,
            depth=10.0,
            position_camera=np.array([0, 0, 10]),  # 10m forward
            size=np.array([2, 2, 8])
        )

        drone_pos = np.array([0, 0, 50])  # 50m altitude
        drone_orient = np.array([1, 0, 0, 0])  # Identity

        result = transformer.camera_to_world([det], drone_pos, drone_orient)

        assert len(result) == 1
        assert result[0]['type'] == "tree"
        assert result[0]['confidence'] == 0.9


class TestObjectTracker:
    """Tests for object tracking."""

    def test_tracker_creates_tracks(self):
        """Test that tracker creates new tracks from detections."""
        from perception.tracker.object_tracker import ObjectTracker

        tracker = ObjectTracker(min_hits=1)

        detections = [
            {
                'position': np.array([10, 5, 0]),
                'type': 'tree',
                'size': np.array([2, 2, 8]),
                'confidence': 0.9
            }
        ]

        tracks = tracker.update(detections)

        assert len(tracks) == 1
        assert tracks[0]['id'] == 1
        assert tracks[0]['type'] == 'tree'

    def test_tracker_maintains_ids(self):
        """Test that tracker maintains consistent IDs across frames."""
        from perception.tracker.object_tracker import ObjectTracker

        tracker = ObjectTracker(min_hits=1, distance_threshold=2.0)

        # Frame 1
        det1 = [{'position': np.array([10, 5, 0]), 'type': 'tree',
                 'size': np.array([2, 2, 8]), 'confidence': 0.9}]
        tracks1 = tracker.update(det1)
        id1 = tracks1[0]['id']

        # Frame 2 - same object moved slightly
        det2 = [{'position': np.array([10.1, 5.1, 0]), 'type': 'tree',
                 'size': np.array([2, 2, 8]), 'confidence': 0.9}]
        tracks2 = tracker.update(det2)
        id2 = tracks2[0]['id']

        assert id1 == id2, "Track ID should be consistent"

    def test_tracker_estimates_velocity(self):
        """Test that tracker estimates object velocity."""
        from perception.tracker.object_tracker import ObjectTracker

        tracker = ObjectTracker(min_hits=1)

        # Moving object
        for i in range(5):
            det = [{'position': np.array([10 + i*0.5, 5, 0]), 'type': 'vehicle',
                    'size': np.array([4, 2, 1.5]), 'confidence': 0.9}]
            tracks = tracker.update(det)

        assert len(tracks) == 1
        # Velocity should be positive in x direction
        assert tracks[0]['velocity'][0] > 0

    def test_tracker_reset(self):
        """Test tracker reset clears all tracks."""
        from perception.tracker.object_tracker import ObjectTracker

        tracker = ObjectTracker(min_hits=1)

        det = [{'position': np.array([10, 5, 0]), 'type': 'tree',
                'size': np.array([2, 2, 8]), 'confidence': 0.9}]
        tracker.update(det)

        assert tracker.get_track_count() > 0

        tracker.reset()

        assert tracker.get_track_count() == 0


class TestKalmanFilter:
    """Tests for Kalman filter."""

    def test_kalman_predict(self):
        """Test Kalman filter prediction step."""
        from perception.tracker.object_tracker import KalmanFilter3D

        kf = KalmanFilter3D(dt=0.033)

        # Initial state: position [10, 5, 0], velocity [1, 0, 0]
        state = np.array([10, 5, 0, 1, 0, 0], dtype=np.float32)
        cov = np.eye(6, dtype=np.float32)

        state_pred, cov_pred = kf.predict(state, cov)

        # Position should move in velocity direction
        assert state_pred[0] > 10
        assert abs(state_pred[1] - 5) < 0.01
        assert abs(state_pred[2] - 0) < 0.01

    def test_kalman_update(self):
        """Test Kalman filter update step."""
        from perception.tracker.object_tracker import KalmanFilter3D

        kf = KalmanFilter3D()

        state = np.array([10, 5, 0, 0, 0, 0], dtype=np.float32)
        cov = np.eye(6, dtype=np.float32) * 10  # High uncertainty

        measurement = np.array([10.5, 5.2, 0.1], dtype=np.float32)

        state_new, cov_new = kf.update(state, cov, measurement)

        # State should move toward measurement
        assert state_new[0] > state[0]
        assert state_new[1] > state[1]


class TestGrading:
    """Tests for model grading system."""

    def test_perfect_grade(self):
        """Test that perfect metrics get P grade."""
        from perception.utils.grading import ModelGrader

        grader = ModelGrader()
        metrics = {
            'detection_accuracy': 98.0,
            'false_positive_rate': 0.5,
            'position_error': 0.3,
            'fps': 35.0
        }

        grade = grader.compute_grade(metrics)
        assert grade == 'P'

    def test_failing_grade(self):
        """Test that poor metrics get F grade."""
        from perception.utils.grading import ModelGrader

        grader = ModelGrader()
        metrics = {
            'detection_accuracy': 38.0,
            'false_positive_rate': 35.0,
            'position_error': 10.0,
            'fps': 3.0
        }

        grade = grader.compute_grade(metrics)
        assert grade == 'W'

    def test_grade_comparison(self):
        """Test grade comparison."""
        from perception.utils.grading import ModelGrader

        grader = ModelGrader()

        assert grader.compare_grades('A+', 'B') == 1   # A+ is better
        assert grader.compare_grades('C', 'A') == -1   # C is worse
        assert grader.compare_grades('B', 'B') == 0    # Equal

    def test_model_name_generation(self):
        """Test model name generation."""
        from perception.utils.grading import generate_model_name
        from datetime import datetime

        name = generate_model_name('A+', 1, datetime(2025, 11, 22))
        assert name == "A+ 22-11-2025 perception v1.pt"

    def test_model_name_parsing(self):
        """Test model name parsing."""
        from perception.utils.grading import parse_model_name

        result = parse_model_name("A+ 22-11-2025 perception v1.pt")

        assert result is not None
        assert result['grade'] == 'A+'
        assert result['version'] == 1
        assert result['date'].year == 2025
        assert result['date'].month == 11
        assert result['date'].day == 22

    def test_invalid_model_name(self):
        """Test parsing invalid model name."""
        from perception.utils.grading import parse_model_name

        result = parse_model_name("invalid_name.pt")
        assert result is None


class TestObjectDetector:
    """Tests for object detector."""

    def test_detector_initialization(self):
        """Test detector initializes without errors."""
        from perception.detector.object_detector import ObjectDetector

        detector = ObjectDetector(device='cpu')
        assert detector is not None

    def test_detector_placeholder(self):
        """Test detector placeholder detection."""
        from perception.detector.object_detector import ObjectDetector

        detector = ObjectDetector(device='cpu')

        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame)

        # Should return list (possibly empty)
        assert isinstance(detections, list)


class TestDepthEstimator:
    """Tests for depth estimation."""

    def test_depth_estimator_initialization(self):
        """Test depth estimator initializes."""
        from perception.depth.depth_estimator import DepthEstimator

        estimator = DepthEstimator(device='cpu', method='size_based')
        assert estimator is not None

    def test_pixel_to_camera(self):
        """Test pixel to camera coordinate conversion."""
        from perception.depth.depth_estimator import DepthEstimator

        estimator = DepthEstimator()

        # Image center at 10m depth
        pos = estimator._pixel_to_camera(320, 240, 10.0)

        # Should be approximately at center (near 0,0) with z=10
        assert abs(pos[0]) < 0.1
        assert abs(pos[1]) < 0.1
        assert abs(pos[2] - 10.0) < 0.01


class TestPerceptionAI:
    """Integration tests for main PerceptionAI class."""

    def test_perception_initialization(self):
        """Test PerceptionAI initializes correctly."""
        from perception.perception_ai import PerceptionAI

        perception = PerceptionAI(device='cpu')
        assert perception is not None

    def test_perception_detect(self):
        """Test full detection pipeline."""
        from perception.perception_ai import PerceptionAI

        perception = PerceptionAI(device='cpu')

        # Create test inputs
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        drone_pos = np.array([0, 0, 50])
        drone_orient = np.array([1, 0, 0, 0])

        result = perception.detect(frame, drone_pos, drone_orient)

        # Check output format
        assert 'obstacles' in result
        assert 'nearest_obstacle_distance' in result
        assert 'ground_distance' in result
        assert isinstance(result['obstacles'], list)

    def test_perception_output_format(self):
        """Test that output matches specification."""
        from perception.perception_ai import PerceptionAI

        perception = PerceptionAI(device='cpu')

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = perception.detect(
            frame,
            np.array([0, 0, 50]),
            np.array([1, 0, 0, 0])
        )

        # Verify all required keys present
        required_keys = ['obstacles', 'nearest_obstacle_distance', 'ground_distance']
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # If there are obstacles, check their format
        for obs in result['obstacles']:
            assert 'id' in obs
            assert 'type' in obs
            assert 'position' in obs
            assert 'size' in obs
            assert 'velocity' in obs
            assert 'confidence' in obs

            # Type should be valid
            valid_types = ['tree', 'building', 'person', 'vehicle',
                          'pole', 'wire', 'bird', 'unknown']
            assert obs['type'] in valid_types

            # Position and size should be lists of 3
            assert len(obs['position']) == 3
            assert len(obs['size']) == 3
            assert len(obs['velocity']) == 3

            # Confidence between 0 and 1
            assert 0 <= obs['confidence'] <= 1

    def test_euler_orientation_input(self):
        """Test detection with euler angle orientation."""
        from perception.perception_ai import PerceptionAI

        perception = PerceptionAI(device='cpu')

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        drone_pos = np.array([0, 0, 50])
        # Euler angles: roll=0, pitch=0, yaw=45deg
        drone_euler = np.array([0, 0, np.pi/4])

        result = perception.detect(frame, drone_pos, drone_euler)

        assert 'obstacles' in result

    def test_tracker_reset(self):
        """Test tracker reset functionality."""
        from perception.perception_ai import PerceptionAI

        perception = PerceptionAI(device='cpu')

        # Run some detections
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            perception.detect(frame, np.array([0, 0, 50]), np.array([1, 0, 0, 0]))

        # Reset
        perception.reset_tracker()

        # Should work without error after reset
        result = perception.detect(frame, np.array([0, 0, 50]), np.array([1, 0, 0, 0]))
        assert 'obstacles' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
