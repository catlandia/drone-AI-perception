"""
Model Grading and Naming Utilities

Provides standardized grading system and model naming format
compatible with the Flight Controller (drone-AI project).

Model Name Format:
    {Grade} {DD-MM-YYYY} perception v{N}.pt

Examples:
    C 25-11-2025 perception v1.pt
    A+ 30-11-2025 perception v2.pt
    S+ 22-11-2025 perception v3.pt

Grading Criteria:
    - Detection accuracy (% obstacles found)
    - False positive rate
    - Position accuracy (error in meters)
    - Speed (FPS)
"""

import re
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GradeThresholds:
    """Thresholds for each performance metric."""
    detection_accuracy: float     # Percentage of obstacles correctly detected
    false_positive_rate: float    # Percentage of false detections
    position_error: float         # Position error in meters
    fps: float                    # Frames per second


# Grade definitions with thresholds
GRADE_THRESHOLDS = {
    "P": GradeThresholds(
        detection_accuracy=95.0,
        false_positive_rate=1.0,
        position_error=0.5,
        fps=30.0
    ),
    "S+": GradeThresholds(
        detection_accuracy=93.0,
        false_positive_rate=2.0,
        position_error=0.6,
        fps=28.0
    ),
    "S": GradeThresholds(
        detection_accuracy=90.0,
        false_positive_rate=3.0,
        position_error=0.7,
        fps=25.0
    ),
    "S-": GradeThresholds(
        detection_accuracy=88.0,
        false_positive_rate=4.0,
        position_error=0.8,
        fps=23.0
    ),
    "A+": GradeThresholds(
        detection_accuracy=85.0,
        false_positive_rate=5.0,
        position_error=1.0,
        fps=20.0
    ),
    "A": GradeThresholds(
        detection_accuracy=82.0,
        false_positive_rate=6.0,
        position_error=1.2,
        fps=18.0
    ),
    "A-": GradeThresholds(
        detection_accuracy=80.0,
        false_positive_rate=7.0,
        position_error=1.4,
        fps=16.0
    ),
    "B+": GradeThresholds(
        detection_accuracy=77.0,
        false_positive_rate=8.0,
        position_error=1.6,
        fps=15.0
    ),
    "B": GradeThresholds(
        detection_accuracy=74.0,
        false_positive_rate=9.0,
        position_error=1.8,
        fps=14.0
    ),
    "B-": GradeThresholds(
        detection_accuracy=71.0,
        false_positive_rate=10.0,
        position_error=2.0,
        fps=13.0
    ),
    "C+": GradeThresholds(
        detection_accuracy=68.0,
        false_positive_rate=12.0,
        position_error=2.5,
        fps=12.0
    ),
    "C": GradeThresholds(
        detection_accuracy=65.0,
        false_positive_rate=14.0,
        position_error=3.0,
        fps=11.0
    ),
    "C-": GradeThresholds(
        detection_accuracy=62.0,
        false_positive_rate=16.0,
        position_error=3.5,
        fps=10.0
    ),
    "D+": GradeThresholds(
        detection_accuracy=58.0,
        false_positive_rate=18.0,
        position_error=4.0,
        fps=9.0
    ),
    "D": GradeThresholds(
        detection_accuracy=54.0,
        false_positive_rate=20.0,
        position_error=4.5,
        fps=8.0
    ),
    "D-": GradeThresholds(
        detection_accuracy=50.0,
        false_positive_rate=22.0,
        position_error=5.0,
        fps=7.0
    ),
    "F+": GradeThresholds(
        detection_accuracy=45.0,
        false_positive_rate=25.0,
        position_error=6.0,
        fps=6.0
    ),
    "F": GradeThresholds(
        detection_accuracy=40.0,
        false_positive_rate=28.0,
        position_error=7.0,
        fps=5.0
    ),
    "F-": GradeThresholds(
        detection_accuracy=35.0,
        false_positive_rate=30.0,
        position_error=8.0,
        fps=4.0
    ),
    "W": GradeThresholds(
        detection_accuracy=0.0,
        false_positive_rate=100.0,
        position_error=100.0,
        fps=0.0
    ),
}

# Grade descriptions
GRADE_DESCRIPTIONS = {
    "P": "PERFECT - Exceptional performance, production ready",
    "S+": "SUPREME+ - Outstanding performance",
    "S": "SUPREME - Excellent performance",
    "S-": "SUPREME- - Very good performance",
    "A+": "ALPHA+ - Great performance",
    "A": "ALPHA - Good performance",
    "A-": "ALPHA- - Above average",
    "B+": "BETTER+ - Solid performance",
    "B": "BETTER - Acceptable performance",
    "B-": "BETTER- - Needs improvement",
    "C+": "COOL+ - Basic functionality",
    "C": "COOL - Minimal viable",
    "C-": "COOL- - Below expectations",
    "D+": "DELUSIONAL+ - Poor performance",
    "D": "DELUSIONAL - Very poor",
    "D-": "DELUSIONAL- - Barely functional",
    "F+": "FAILURE+ - Critically poor",
    "F": "FAILURE - Not functional",
    "F-": "FAILURE- - Complete failure",
    "W": "WORST - Does not work",
}

# Grade ordering for comparison
GRADE_ORDER = [
    "P", "S+", "S", "S-", "A+", "A", "A-",
    "B+", "B", "B-", "C+", "C", "C-",
    "D+", "D", "D-", "F+", "F", "F-", "W"
]


class ModelGrader:
    """
    Evaluates and grades perception models based on performance metrics.

    Usage:
        grader = ModelGrader()
        metrics = {
            "detection_accuracy": 85.5,
            "false_positive_rate": 4.2,
            "position_error": 0.95,
            "fps": 22.0
        }
        grade = grader.compute_grade(metrics)
        print(f"Model grade: {grade}")  # "A+"
    """

    def __init__(self):
        self.thresholds = GRADE_THRESHOLDS
        self.descriptions = GRADE_DESCRIPTIONS

    def compute_grade(self, metrics: Dict[str, float]) -> str:
        """
        Compute overall grade from performance metrics.

        Args:
            metrics: Dictionary with keys:
                - detection_accuracy: % of obstacles detected (0-100)
                - false_positive_rate: % of false detections (0-100)
                - position_error: Average position error in meters
                - fps: Frames per second

        Returns:
            Grade string (e.g., "A+", "B", "C-")
        """
        # Validate metrics
        required = ["detection_accuracy", "false_positive_rate", "position_error", "fps"]
        for key in required:
            if key not in metrics:
                raise ValueError(f"Missing required metric: {key}")

        # Find best matching grade
        for grade in GRADE_ORDER:
            if self._meets_grade(metrics, grade):
                return grade

        return "W"

    def _meets_grade(self, metrics: Dict[str, float], grade: str) -> bool:
        """Check if metrics meet requirements for a grade."""
        thresholds = self.thresholds[grade]

        return (
            metrics["detection_accuracy"] >= thresholds.detection_accuracy and
            metrics["false_positive_rate"] <= thresholds.false_positive_rate and
            metrics["position_error"] <= thresholds.position_error and
            metrics["fps"] >= thresholds.fps
        )

    def get_grade_description(self, grade: str) -> str:
        """Get description for a grade."""
        return self.descriptions.get(grade, "Unknown grade")

    def get_detailed_report(self, metrics: Dict[str, float]) -> str:
        """
        Generate detailed grading report.

        Returns formatted string with grade breakdown.
        """
        grade = self.compute_grade(metrics)
        thresholds = self.thresholds.get(grade, self.thresholds["W"])

        report = f"""
=== PERCEPTION MODEL GRADE REPORT ===

Overall Grade: {grade} - {self.get_grade_description(grade)}

Performance Metrics:
  Detection Accuracy: {metrics['detection_accuracy']:.1f}% (threshold: >={thresholds.detection_accuracy}%)
  False Positive Rate: {metrics['false_positive_rate']:.1f}% (threshold: <={thresholds.false_positive_rate}%)
  Position Error: {metrics['position_error']:.2f}m (threshold: <={thresholds.position_error}m)
  Speed: {metrics['fps']:.1f} FPS (threshold: >={thresholds.fps} FPS)

Grade Thresholds for {grade}:
  - Detection: >={thresholds.detection_accuracy}%
  - False Positives: <={thresholds.false_positive_rate}%
  - Position Error: <={thresholds.position_error}m
  - Speed: >={thresholds.fps} FPS
"""
        return report

    def compare_grades(self, grade1: str, grade2: str) -> int:
        """
        Compare two grades.

        Returns:
            -1 if grade1 < grade2 (worse)
             0 if equal
             1 if grade1 > grade2 (better)
        """
        try:
            idx1 = GRADE_ORDER.index(grade1)
            idx2 = GRADE_ORDER.index(grade2)
            # Lower index = better grade
            if idx1 < idx2:
                return 1
            elif idx1 > idx2:
                return -1
            return 0
        except ValueError:
            return 0


def generate_model_name(
    grade: str,
    version: int,
    date: Optional[datetime] = None
) -> str:
    """
    Generate standardized model file name.

    Format: {Grade} {DD-MM-YYYY} perception v{N}.pt

    Args:
        grade: Model grade (e.g., "A+", "B", "C-")
        version: Model version number
        date: Date (default: today)

    Returns:
        Formatted model name

    Example:
        >>> generate_model_name("A+", 1)
        "A+ 22-11-2025 perception v1.pt"
    """
    if date is None:
        date = datetime.now()

    date_str = date.strftime("%d-%m-%Y")
    return f"{grade} {date_str} perception v{version}.pt"


def parse_model_name(filename: str) -> Optional[Dict[str, any]]:
    """
    Parse model name to extract grade, date, and version.

    Args:
        filename: Model filename

    Returns:
        Dictionary with 'grade', 'date', 'version', or None if invalid

    Example:
        >>> parse_model_name("A+ 22-11-2025 perception v1.pt")
        {'grade': 'A+', 'date': datetime(2025, 11, 22), 'version': 1}
    """
    # Pattern: {Grade} {DD-MM-YYYY} perception v{N}.pt
    pattern = r"^([PSABCDFW][+-]?) (\d{2}-\d{2}-\d{4}) perception v(\d+)\.pt$"
    match = re.match(pattern, filename)

    if not match:
        return None

    grade = match.group(1)
    date_str = match.group(2)
    version = int(match.group(3))

    try:
        date = datetime.strptime(date_str, "%d-%m-%Y")
    except ValueError:
        return None

    return {
        "grade": grade,
        "date": date,
        "version": version
    }


def get_next_version(model_dir: str, grade: str) -> int:
    """
    Get next available version number for a grade.

    Args:
        model_dir: Directory containing models
        grade: Model grade

    Returns:
        Next version number
    """
    import os

    max_version = 0
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            parsed = parse_model_name(filename)
            if parsed and parsed["grade"] == grade:
                max_version = max(max_version, parsed["version"])

    return max_version + 1
