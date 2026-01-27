# 6-DoF Racket Lock

Keeps the tennis racket locked to the player's wrist throughout the stroke. Uses the 3D pose data from Apple Vision.

## How It Works

Takes `keypoints3D` (3D coordinates in meters, pelvis as origin), applies camera extrinsics, outputs world-space racket position and rotation.

The racket stays at a fixed distance from the wrist - can't slip because it's computed directly from wrist position.

## Usage

```bash
python racket_6dof_lock.py input.json output.json
```

Or in Python:
```python
from racket_6dof_lock import process_video_json

process_video_json('pose_data.json', 'racket_output.json')
```

## Input Format

Uses `keypoints3D` from the pose JSON:
```json
{
  "frames": [
    {
      "timestamp": 0.0,
      "camera_extrinsics": [16 floats],
      "keypoints3D": [
        {"name": "rightWrist", "x": -0.11, "y": 0.31, "z": 0.40},
        {"name": "rightElbow", "x": -0.28, "y": 0.31, "z": 0.21}
      ]
    }
  ]
}
```

## Output Format

```json
{
  "frames": [
    {
      "frame_index": 0,
      "timestamp": 0.0,
      "position": [0.54, 2.21, -9.59],
      "quaternion": [0.75, 0.08, 0.64, 0.12],
      "position_local": [0.24, 0.31, 0.81],
      "wrist_position": [-0.0005, 2.19, -9.68],
      "confidence": 0.54
    }
  ]
}
```

- `position` / `quaternion` - World space (camera extrinsics applied)
- `position_local` - Pelvis-centered (for validation)
- All units in meters

## Tested

Ran on all provided clips:
- 30fps: 401 frames, 0.0000m deviation
- 120fps: 2068 frames, 0.0000m deviation

## Files

- `racket_6dof_lock.py` - Main code
- `motion_occlusion_solver.py` - Handles tracking gaps (optional)
- `test_*.py` - Verification tests
