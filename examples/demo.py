#!/usr/bin/env python3
"""
Perception AI Demo

Demonstrates basic usage of the Perception AI system.

Usage:
    python examples/demo.py
"""

import sys
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, '.')

from perception import PerceptionAI
from perception.utils.grading import ModelGrader, generate_model_name


def create_synthetic_scene():
    """Create a synthetic camera frame for testing."""
    # Create a simple scene with some objects
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Sky (top half)
    frame[:240, :] = [135, 206, 235]  # Sky blue

    # Ground (bottom half)
    frame[240:, :] = [34, 139, 34]  # Forest green

    # Add some "trees" (dark green rectangles)
    for i in range(3):
        x = 100 + i * 200
        frame[200:350, x:x+50] = [0, 100, 0]

    # Add a "building" (gray rectangle)
    frame[180:320, 400:550] = [128, 128, 128]

    return frame


def demo_basic_usage():
    """Demonstrate basic perception usage."""
    print("=" * 60)
    print("PERCEPTION AI - BASIC USAGE DEMO")
    print("=" * 60)

    # Initialize Perception AI
    print("\n1. Initializing Perception AI...")
    perception = PerceptionAI(device='cpu')
    print("   Done!")

    # Create synthetic test frame
    print("\n2. Creating synthetic scene...")
    frame = create_synthetic_scene()
    print(f"   Frame shape: {frame.shape}")

    # Simulate drone position and orientation
    drone_position = np.array([0.0, 0.0, 50.0])  # 50m altitude
    drone_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

    print(f"   Drone position: {drone_position} meters")
    print(f"   Drone orientation (quaternion): {drone_orientation}")

    # Run detection
    print("\n3. Running perception...")
    start_time = time.time()
    result = perception.detect(frame, drone_position, drone_orientation)
    elapsed = time.time() - start_time

    print(f"   Detection completed in {elapsed*1000:.1f}ms")

    # Display results
    print("\n4. Results:")
    print(f"   Obstacles detected: {len(result['obstacles'])}")
    print(f"   Nearest obstacle: {result['nearest_obstacle_distance']:.2f}m")
    print(f"   Ground distance: {result['ground_distance']:.2f}m")

    if result['obstacles']:
        print("\n   Obstacle details:")
        for i, obs in enumerate(result['obstacles']):
            print(f"\n   Obstacle {i+1}:")
            print(f"     ID: {obs['id']}")
            print(f"     Type: {obs['type']}")
            print(f"     Position: [{obs['position'][0]:.1f}, {obs['position'][1]:.1f}, {obs['position'][2]:.1f}]m")
            print(f"     Size: [{obs['size'][0]:.1f}, {obs['size'][1]:.1f}, {obs['size'][2]:.1f}]m")
            print(f"     Confidence: {obs['confidence']:.2f}")


def demo_tracking():
    """Demonstrate object tracking across frames."""
    print("\n" + "=" * 60)
    print("PERCEPTION AI - TRACKING DEMO")
    print("=" * 60)

    perception = PerceptionAI(device='cpu')

    # Simulate drone flying forward
    print("\nSimulating drone flight with moving obstacles...")

    for frame_num in range(5):
        # Create frame
        frame = create_synthetic_scene()

        # Drone moving forward
        drone_position = np.array([0.0 + frame_num * 2.0, 0.0, 50.0])
        drone_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        result = perception.detect(frame, drone_position, drone_orientation)

        print(f"\nFrame {frame_num + 1}:")
        print(f"  Drone position: [{drone_position[0]:.1f}, {drone_position[1]:.1f}, {drone_position[2]:.1f}]")
        print(f"  Obstacles: {len(result['obstacles'])}")

        for obs in result['obstacles']:
            print(f"    ID {obs['id']}: {obs['type']} at pos={[f'{p:.1f}' for p in obs['position']]}, vel={[f'{v:.2f}' for v in obs['velocity']]}")


def demo_grading():
    """Demonstrate model grading system."""
    print("\n" + "=" * 60)
    print("PERCEPTION AI - GRADING DEMO")
    print("=" * 60)

    grader = ModelGrader()

    # Simulate different performance levels
    test_cases = [
        {
            "name": "Excellent Model",
            "metrics": {
                "detection_accuracy": 92.5,
                "false_positive_rate": 2.5,
                "position_error": 0.7,
                "fps": 28.0
            }
        },
        {
            "name": "Average Model",
            "metrics": {
                "detection_accuracy": 75.0,
                "false_positive_rate": 8.5,
                "position_error": 1.7,
                "fps": 15.0
            }
        },
        {
            "name": "Poor Model",
            "metrics": {
                "detection_accuracy": 55.0,
                "false_positive_rate": 20.0,
                "position_error": 4.2,
                "fps": 8.0
            }
        }
    ]

    for case in test_cases:
        print(f"\n{case['name']}:")
        print(f"  Detection Accuracy: {case['metrics']['detection_accuracy']}%")
        print(f"  False Positive Rate: {case['metrics']['false_positive_rate']}%")
        print(f"  Position Error: {case['metrics']['position_error']}m")
        print(f"  FPS: {case['metrics']['fps']}")

        grade = grader.compute_grade(case['metrics'])
        desc = grader.get_grade_description(grade)

        print(f"  -> Grade: {grade} ({desc})")

        # Generate model name
        model_name = generate_model_name(grade, 1)
        print(f"  -> Model name: {model_name}")


def demo_integration_with_flight_controller():
    """Show how Perception AI integrates with Flight Controller."""
    print("\n" + "=" * 60)
    print("PERCEPTION AI - FLIGHT CONTROLLER INTEGRATION")
    print("=" * 60)

    print("""
This demonstrates the interface between Perception AI and the
Flight Controller (drone-AI project).

Integration Flow:
1. Camera captures frame
2. Perception AI processes frame + drone pose
3. Output goes to Path Planner (Layer 3)
4. Path Planner generates waypoints
5. Flight Controller executes waypoints

Example code:

```python
from perception import PerceptionAI

# Initialize
perception = PerceptionAI(model_path="models/A+ 25-11-2025 perception v1.pt")

# Main loop
while flying:
    # Get inputs
    frame = camera.capture()
    position = drone.get_position()      # [x, y, z] meters
    orientation = drone.get_quaternion() # [w, x, y, z]

    # Run perception
    result = perception.detect(frame, position, orientation)

    # Output to path planner
    obstacles = result['obstacles']
    nearest = result['nearest_obstacle_distance']

    # Path planner uses obstacles to compute safe path
    waypoints = path_planner.plan(
        current_pos=position,
        goal_pos=goal,
        obstacles=obstacles
    )

    # Flight controller follows waypoints
    for waypoint in waypoints:
        action = flight_controller.get_action(observation)
        drone.send_motors(action)
```
""")


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#" + " " * 20 + "PERCEPTION AI" + " " * 21 + "#")
    print("#" + " " * 14 + "Layer 2 of Drone AI System" + " " * 14 + "#")
    print("#" * 60)

    demo_basic_usage()
    demo_tracking()
    demo_grading()
    demo_integration_with_flight_controller()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Collect training data (aerial images with labels)")
    print("2. Train detector on drone imagery dataset (VisDrone, UAV123)")
    print("3. Train depth estimator or integrate depth camera")
    print("4. Evaluate and grade model performance")
    print("5. Integrate with Flight Controller (drone-AI)")


if __name__ == '__main__':
    main()
