# Occlusion Handling Module

## What This Is

Processes pose data frame-by-frame and outputs smooth racket position even during occlusion (when tracking is lost).

---

## Files

```
motion_occlusion_solver.py   <- Main solver
test_solver.py               <- Validation tests
README.md                    <- This file
```

---

## Batch Processing (JSON to JSON)

**Command line:**
```bash
python motion_occlusion_solver.py input.json output.json 0.55
```

**Input JSON format:**
```json
{
    "frames": [
        {
            "timestamp": 0.0,
            "keypoints": [
                {"name": "right_wrist", "x": 0.5, "y": 1.0, "z": 0.3, "score": 0.95},
                {"name": "right_elbow", "x": 0.4, "y": 1.2, "z": 0.3, "score": 0.95}
            ]
        },
        {
            "timestamp": 0.033,
            "keypoints": [...]
        }
    ]
}
```

**Output JSON format:**
```json
{
    "source_file": "input.json",
    "object_length": 0.55,
    "total_frames": 100,
    "frames": [
        {
            "frame_index": 0,
            "timestamp": 0.0,
            "input_confidence": 0.95,
            "position": [0.75, 0.51, 0.30],
            "quaternion": [0.99, 0.01, 0.02, 0.03],
            "direction": [0.45, -0.89, 0.0],
            "velocity": [0.01, -0.02, 0.01],
            "output_confidence": 0.95,
            "is_predicted": false,
            "state": "tracking"
        }
    ]
}
```

---

## Python API

```python
from motion_occlusion_solver import OcclusionSolver, process_json_file

# Option 1: Batch process JSON file
process_json_file(
    input_path='pose_data.json',
    output_path='output.json',
    object_length=0.55
)

# Option 2: Frame-by-frame processing
solver = OcclusionSolver(object_length=0.55)

for frame in frames:
    result = solver.process(
        anchor=frame['wrist'],           # [x, y, z]
        reference=frame['elbow'],        # [x, y, z]
        confidence=frame['score'],       # 0.0 - 1.0
        timestamp=frame['timestamp']     # seconds (important for prediction!)
    )
    
    # Use result
    position = result.position
    quaternion = result.quaternion
    is_predicted = result.is_predicted
```

---

## Timestamp Parameter

**Yes, timestamp is supported and important.**

```python
result = solver.process(
    anchor=[x, y, z],
    reference=[x, y, z],
    confidence=0.95,
    timestamp=0.033  # <-- Time in seconds
)
```

- If not provided, assumes 30fps and auto-increments
- For accurate prediction during occlusion, provide actual timestamps
- Prediction uses time elapsed to calculate trajectory

---

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `position` | `[x,y,z]` | Predicted object position |
| `quaternion` | `[w,x,y,z]` | Rotation (None for 2D) |
| `direction` | `[x,y,z]` | Direction unit vector |
| `velocity` | `[x,y,z]` | Velocity estimate |
| `output_confidence` | `float` | How reliable (0-1) |
| `is_predicted` | `bool` | True = prediction, False = observed |
| `state` | `string` | "tracking", "predicted", "blending", etc |

---

## Validation

Run tests before using:
```bash
python test_solver.py
```

All 10 tests should pass.

---

## Configuration

```python
solver = OcclusionSolver(
    object_length=0.55,           # wrist to object center (meters)
    high_conf=0.7,                # above = good tracking
    low_conf=0.3,                 # below = switch to prediction
    smoothing=0.7,                # position filter
    expected_occlusion_time=0.3   # typical gap duration (seconds)
)
```
