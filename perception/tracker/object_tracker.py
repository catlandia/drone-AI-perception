"""
Object Tracking Module

Multi-object tracking with persistent IDs using Kalman filtering.
Maintains object identity across frames and estimates velocity.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Track:
    """Represents a tracked object."""
    id: int                          # Unique track ID
    type: str                        # Object class
    position: np.ndarray             # Current position [x, y, z]
    velocity: np.ndarray             # Estimated velocity [vx, vy, vz]
    size: np.ndarray                 # Object size [w, d, h]
    confidence: float                # Current confidence
    age: int = 0                     # Frames since creation
    hits: int = 0                    # Successful detection matches
    time_since_update: int = 0       # Frames since last detection match
    state: np.ndarray = field(default_factory=lambda: np.zeros(6))  # Kalman state
    covariance: np.ndarray = field(default_factory=lambda: np.eye(6))  # Kalman covariance


class KalmanFilter3D:
    """
    Kalman filter for 3D object tracking.

    State: [x, y, z, vx, vy, vz]
    Measurement: [x, y, z]

    Constant velocity motion model.
    """

    def __init__(self, dt: float = 1/30.0):
        """
        Initialize Kalman filter.

        Args:
            dt: Time step between frames (default 30 FPS)
        """
        self.dt = dt

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)

        # Process noise covariance
        q = 0.5  # Process noise magnitude
        self.Q = np.array([
            [dt**4/4, 0, 0, dt**3/2, 0, 0],
            [0, dt**4/4, 0, 0, dt**3/2, 0],
            [0, 0, dt**4/4, 0, 0, dt**3/2],
            [dt**3/2, 0, 0, dt**2, 0, 0],
            [0, dt**3/2, 0, 0, dt**2, 0],
            [0, 0, dt**3/2, 0, 0, dt**2],
        ], dtype=np.float32) * q**2

        # Measurement noise covariance
        r = 1.0  # Measurement noise (meters)
        self.R = np.eye(3, dtype=np.float32) * r**2

    def predict(self, state: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state.

        Args:
            state: Current state [x, y, z, vx, vy, vz]
            covariance: Current covariance matrix

        Returns:
            Predicted state and covariance
        """
        state_pred = self.F @ state
        cov_pred = self.F @ covariance @ self.F.T + self.Q
        return state_pred, cov_pred

    def update(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update state with measurement.

        Args:
            state: Predicted state
            covariance: Predicted covariance
            measurement: Observed position [x, y, z]

        Returns:
            Updated state and covariance
        """
        # Innovation
        y = measurement - self.H @ state

        # Innovation covariance
        S = self.H @ covariance @ self.H.T + self.R

        # Kalman gain
        K = covariance @ self.H.T @ np.linalg.inv(S)

        # Update state
        state_new = state + K @ y

        # Update covariance
        I = np.eye(6, dtype=np.float32)
        cov_new = (I - K @ self.H) @ covariance

        return state_new, cov_new

    def init_state(self, position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize state from first observation.

        Args:
            position: Initial position [x, y, z]

        Returns:
            Initial state and covariance
        """
        state = np.zeros(6, dtype=np.float32)
        state[:3] = position

        # High initial velocity uncertainty
        covariance = np.diag([1, 1, 1, 10, 10, 10]).astype(np.float32)

        return state, covariance


class ObjectTracker:
    """
    Multi-object tracker with persistent IDs.

    Uses Kalman filtering for motion prediction and
    Hungarian algorithm for detection-to-track association.

    Features:
    - Assigns unique IDs to objects
    - Handles occlusions (tracks survive brief disappearances)
    - Estimates object velocities
    - Filters out spurious detections (requires min hits)
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        distance_threshold: float = 5.0
    ):
        """
        Initialize tracker.

        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections before track is confirmed
            iou_threshold: IOU threshold for matching (if boxes available)
            distance_threshold: Distance threshold for matching (meters)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold

        self.tracks: List[Track] = []
        self.next_id = 1
        self.kalman = KalmanFilter3D()

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracks with new detections.

        Args:
            detections: List of detection dicts with 'position', 'type', 'size', 'confidence'

        Returns:
            List of tracked obstacle dicts with assigned IDs
        """
        # Predict all existing tracks
        for track in self.tracks:
            track.state, track.covariance = self.kalman.predict(
                track.state, track.covariance
            )
            track.position = track.state[:3].copy()
            track.velocity = track.state[3:6].copy()
            track.age += 1
            track.time_since_update += 1

        # Match detections to tracks
        if len(detections) > 0 and len(self.tracks) > 0:
            matched, unmatched_dets, unmatched_tracks = self._associate(detections)
        else:
            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = list(range(len(self.tracks)))

        # Update matched tracks
        for track_idx, det_idx in matched:
            track = self.tracks[track_idx]
            det = detections[det_idx]

            # Kalman update
            track.state, track.covariance = self.kalman.update(
                track.state, track.covariance, det['position']
            )
            track.position = track.state[:3].copy()
            track.velocity = track.state[3:6].copy()

            # Update other properties
            track.type = det['type']
            track.size = det['size']
            track.confidence = det['confidence']
            track.hits += 1
            track.time_since_update = 0

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self._create_track(det)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return confirmed tracks
        results = []
        for track in self.tracks:
            if track.hits >= self.min_hits and track.time_since_update == 0:
                results.append({
                    'id': track.id,
                    'type': track.type,
                    'position': track.position,
                    'size': track.size,
                    'velocity': track.velocity,
                    'confidence': track.confidence
                })

        return results

    def _associate(
        self,
        detections: List[Dict[str, Any]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to existing tracks.

        Uses distance-based matching with Hungarian algorithm.

        Returns:
            matched: List of (track_idx, det_idx) pairs
            unmatched_dets: List of unmatched detection indices
            unmatched_tracks: List of unmatched track indices
        """
        num_tracks = len(self.tracks)
        num_dets = len(detections)

        # Compute cost matrix (distance between predicted positions and detections)
        cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                # Euclidean distance
                dist = np.linalg.norm(track.position - det['position'])

                # Penalize type mismatch
                if track.type != det['type']:
                    dist += 2.0

                cost_matrix[i, j] = dist

        # Apply threshold
        cost_matrix[cost_matrix > self.distance_threshold] = 1e6

        # Hungarian algorithm (simple greedy implementation)
        matched, unmatched_dets, unmatched_tracks = self._hungarian_matching(
            cost_matrix, self.distance_threshold
        )

        return matched, unmatched_dets, unmatched_tracks

    def _hungarian_matching(
        self,
        cost_matrix: np.ndarray,
        threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Simple greedy matching (approximate Hungarian).

        For optimal matching, use scipy.optimize.linear_sum_assignment.
        """
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched = []
            unmatched_tracks = set(range(cost_matrix.shape[0]))
            unmatched_dets = set(range(cost_matrix.shape[1]))

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < threshold:
                    matched.append((r, c))
                    unmatched_tracks.discard(r)
                    unmatched_dets.discard(c)

            return matched, list(unmatched_dets), list(unmatched_tracks)

        except ImportError:
            # Fallback: greedy matching
            return self._greedy_matching(cost_matrix, threshold)

    def _greedy_matching(
        self,
        cost_matrix: np.ndarray,
        threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Greedy matching as fallback when scipy not available.
        """
        matched = []
        matched_tracks = set()
        matched_dets = set()

        # Find minimum cost matches greedily
        while True:
            # Find minimum cost
            min_cost = float('inf')
            min_idx = (-1, -1)

            for i in range(cost_matrix.shape[0]):
                if i in matched_tracks:
                    continue
                for j in range(cost_matrix.shape[1]):
                    if j in matched_dets:
                        continue
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_idx = (i, j)

            if min_cost >= threshold or min_idx[0] < 0:
                break

            matched.append(min_idx)
            matched_tracks.add(min_idx[0])
            matched_dets.add(min_idx[1])

        unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in matched_tracks]
        unmatched_dets = [j for j in range(cost_matrix.shape[1]) if j not in matched_dets]

        return matched, unmatched_dets, unmatched_tracks

    def _create_track(self, detection: Dict[str, Any]):
        """Create new track from detection."""
        state, covariance = self.kalman.init_state(detection['position'])

        track = Track(
            id=self.next_id,
            type=detection['type'],
            position=detection['position'].copy(),
            velocity=np.zeros(3, dtype=np.float32),
            size=detection['size'].copy(),
            confidence=detection['confidence'],
            age=1,
            hits=1,
            time_since_update=0,
            state=state,
            covariance=covariance
        )

        self.tracks.append(track)
        self.next_id += 1

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_id = 1

    def get_track_count(self) -> int:
        """Get number of active tracks."""
        return len(self.tracks)

    def get_confirmed_track_count(self) -> int:
        """Get number of confirmed tracks."""
        return sum(1 for t in self.tracks if t.hits >= self.min_hits)
