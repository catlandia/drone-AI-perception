# Perception AI

**Layer 2 of the Drone AI System**

Perception AI provides real-time obstacle detection and tracking for autonomous drone navigation. It processes camera frames and drone pose information to output world-coordinate obstacle data.

## System Architecture

```
Layer 4: Mission Planner    (NOT BUILT)
Layer 3: Path Planner       (NOT BUILT)
Layer 2: Perception AI      <-- THIS PROJECT
Layer 1: Flight Controller  (DONE - separate repo "drone-AI")
```

## Quick Start

```python
from perception import PerceptionAI
import numpy as np

# Initialize
perception = PerceptionAI(model_path="models/A+ 25-11-2025 perception v1.pt")

# Detect obstacles
frame = camera.capture()  # RGB image (H, W, 3)
drone_position = np.array([x, y, z])  # meters
drone_orientation = np.array([w, x, y, z])  # quaternion

result = perception.detect(frame, drone_position, drone_orientation)

# Use results
for obstacle in result['obstacles']:
    print(f"ID: {obstacle['id']}, Type: {obstacle['type']}")
    print(f"Position: {obstacle['position']} meters")
```

## Output Format

```python
{
    "obstacles": [
        {
            "id": 1,                           # Tracking ID (same object = same ID)
            "type": "tree",                    # tree/building/person/vehicle/unknown
            "position": [10.5, 5.2, 0.0],     # World coordinates (meters)
            "size": [2.0, 2.0, 8.0],          # Width, depth, height (meters)
            "velocity": [0.0, 0.0, 0.0],      # For moving objects
            "confidence": 0.92                 # 0.0 to 1.0
        },
        # ... more obstacles
    ],
    "nearest_obstacle_distance": 10.5,        # Meters to closest obstacle
    "ground_distance": 25.0                   # Height above ground
}
```

## Components

| Component | Description |
|-----------|-------------|
| **Object Detector** | CNN-based detection from aerial imagery |
| **Depth Estimator** | 2D to 3D position conversion |
| **Coordinate Transformer** | Camera to world coordinate transformation |
| **Object Tracker** | Multi-object tracking with Kalman filter |

## Installation

```bash
# Clone repository
git clone <repo-url>
cd drone-AI-perception

# Install dependencies
pip install -e .

# With PyTorch support (recommended)
pip install -e ".[torch]"

# Run demo
python examples/demo.py
```

## Performance Requirements

| Metric | Target |
|--------|--------|
| Speed | 15+ FPS (30+ preferred) |
| Latency | <100ms frame-to-output |
| Range | 5m to 100m detection |
| Position Accuracy | <1m error at 20m distance |

## Model Naming Convention

Models are saved with standardized names:

```
{Grade} {DD-MM-YYYY} perception v{N}.pt
```

Examples:
- `C 25-11-2025 perception v1.pt`
- `A+ 30-11-2025 perception v2.pt`

This matches the Flight Controller format:
- `S+ 22-11-2025 flycontrol v1.pt`

## Grading System

| Grade | Description |
|-------|-------------|
| P | PERFECT (>95% detection, <0.5m error) |
| S/S+/S- | SUPREME |
| A/A+/A- | ALPHA |
| B/B+/B- | BETTER |
| C/C+/C- | COOL |
| D/D+/D- | DELUSIONAL |
| F/F+/F- | FAILURE |
| W | WORST |

Grading is based on:
- Detection accuracy (% obstacles found)
- False positive rate
- Position accuracy (meters)
- Speed (FPS)

## Integration with Flight Controller

```python
# Perception AI outputs obstacles
perception_output = perception_ai.detect(camera_frame, drone_pos, drone_orientation)

# Path Planner uses obstacles to plan route (future)
waypoints = path_planner.plan(current_pos, goal_pos, perception_output["obstacles"])

# Flight Controller flies to waypoints
for waypoint in waypoints:
    action = flight_controller.get_action(observation)
    drone.send_motors(action)
```

## Training Data

For object detector training:
- Aerial/drone perspective images
- Labeled bounding boxes for: trees, buildings, people, vehicles, poles, wires, birds

Recommended datasets:
- VisDrone
- UAV123
- Custom data collection

## Testing

```bash
# Run tests
pytest tests/test_perception.py -v
```

## Project Structure

```
drone-AI-perception/
├── perception/
│   ├── __init__.py
│   ├── perception_ai.py         # Main interface
│   ├── detector/
│   │   ├── object_detector.py   # CNN detection
│   │   └── models.py            # Neural network architectures
│   ├── depth/
│   │   └── depth_estimator.py   # Depth estimation
│   ├── transform/
│   │   └── coordinate_transformer.py  # Coordinate transforms
│   ├── tracker/
│   │   └── object_tracker.py    # Kalman filter tracking
│   └── utils/
│       └── grading.py           # Model grading system
├── models/                       # Saved models
├── data/                         # Training data
├── tests/
│   └── test_perception.py
├── examples/
│   └── demo.py
├── requirements.txt
├── setup.py
└── README.md
```

## API Reference

### PerceptionAI

```python
class PerceptionAI:
    def __init__(self, model_path: str = None, device: str = "auto"):
        """Initialize with trained model."""

    def detect(self,
               frame: np.ndarray,           # RGB image (H, W, 3)
               drone_position: np.ndarray,  # [x, y, z] meters
               drone_orientation: np.ndarray  # quaternion [w, x, y, z]
              ) -> dict:
        """Detect obstacles and return world-coordinate positions."""

    def reset_tracker(self):
        """Reset object tracker for new scene."""
```

## What NOT to Change in Flight Controller

If modifying drone-AI for integration:
- Do NOT change `agent.py` (PPO agent works)
- Do NOT change `simulation.py` physics
- Do NOT change reward weights
- You CAN extend observation space in `environment.py` if needed
