# 6-DoF Racket Lock Implementation

Mathematical solution to keep the racket rigidly locked to the player's hand throughout the stroke.

## The Problem
During fast strokes (backhand, serve), the racket can appear to "slip" from the hand due to:
- Tracking jitter
- Brief occlusions
- Inconsistent keypoint detection

## The Solution: 6-DoF Lock

The racket is **mathematically locked** to the wrist with 6 degrees of freedom:

**Position (3 DoF):**
```
racket_position = wrist_position + (forearm_direction × FIXED_OFFSET)
```

**Orientation (3 DoF):**
```
racket_quaternion = quaternion_from_forearm_direction(wrist - elbow)
```

Since `FIXED_OFFSET` is constant (0.55m default), the racket **cannot slip**.

## Files

| File | Purpose |
|------|---------|
| `racket_6dof_lock.py` | Core implementation |
| `test_6dof_lock.py` | Mathematical verification tests |
| `motion_occlusion_solver.py` | Generic occlusion handling (optional) |
| `test_solver.py` | Occlusion solver tests |

## Quick Start

### Python Usage
```python
from racket_6dof_lock import Racket6DoFLock

lock = Racket6DoFLock()

# Each frame:
pose = lock.compute(
    wrist=[0.5, 1.0, 0.3],    # Wrist position from pose detection
    elbow=[0.4, 1.2, 0.3],    # Elbow position
    timestamp=0.033            # Frame time
)

# Output:
print(pose.position)      # Racket face center [x, y, z]
print(pose.quaternion)    # Orientation [w, x, y, z]
print(pose.direction)     # Face normal vector
```

### JSON Batch Processing
```bash
python racket_6dof_lock.py input.json output.json
```

**Input JSON format:**
```json
{
  "frames": [
    {
      "timestamp": 0.0,
      "keypoints": [
        {"name": "right_wrist", "x": 0.5, "y": 1.0, "z": 0.3},
        {"name": "right_elbow", "x": 0.4, "y": 1.2, "z": 0.3}
      ]
    }
  ]
}
```

**Output JSON format:**
```json
{
  "frames": [
    {
      "frame_index": 0,
      "timestamp": 0.0,
      "racket_position": [1.05, 0.508, 0.3],
      "racket_quaternion": [0.707, 0.0, 0.707, 0.0],
      "racket_direction": [0.894, -0.447, 0.0],
      "wrist_position": [0.5, 1.0, 0.3]
    }
  ]
}
```

## Configuration

```python
from racket_6dof_lock import Racket6DoFLock, GripConfig

config = GripConfig(
    wrist_to_grip=np.array([0.1, 0.0, 0.03]),  # Kalpesh's offset
    wrist_to_face=0.55                          # Wrist to racket face distance
)

lock = Racket6DoFLock(
    grip_config=config,
    smoothing=0.7  # 0-1, higher = more responsive
)
```

## Run Tests

```bash
cd delivery
python test_6dof_lock.py
```

Expected output:
```
ALL TESTS PASSED
The racket is mathematically guaranteed to stay locked to the hand.
```

## Integration Notes

### For Unity (C++ Port)
The math is straightforward to port:

```cpp
// Compute forearm direction
vec3 forearm = wrist - elbow;
vec3 direction = normalize(forearm);

// Position: wrist + offset along forearm
vec3 racket_pos = wrist + direction * WRIST_TO_FACE;

// Orientation: quaternion from direction
quat racket_rot = quaternion_look_rotation(direction, vec3(0,1,0));
```

### Coordinate System
- Input: Right-handed (Apple Vision)
- For Unity (left-handed): flip X axis
  ```python
  position[0] *= -1
  quaternion[1] *= -1  # x component
  ```

## Mathematical Guarantee

The 6-DoF lock guarantees:
1. **No slip**: Racket stays at fixed distance from wrist
2. **Smooth motion**: Minimum jerk filtering reduces tracking jitter
3. **Valid rotations**: All quaternions are unit length
4. **Continuous**: Even during brief tracking gaps

This was verified with the test suite in `test_6dof_lock.py`.
