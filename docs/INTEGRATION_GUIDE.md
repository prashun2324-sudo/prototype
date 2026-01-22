# Occlusion Solver - Integration Guide

## Before You Start

**Run the validation tests first:**
```bash
cd src
python test_solver.py
```

All 10 tests should pass. If any fail, do not integrate.

---

## What This Code Does

Handles gaps in tracking when the hand/racket goes out of view:
- Detects when tracking confidence drops
- Predicts position using minimum jerk trajectory
- Blends smoothly when tracking returns

---

## What You Need to Provide

The solver needs these inputs each frame:

| Input | Type | Description |
|-------|------|-------------|
| `anchor` | `[x,y,z]` or `[x,y]` | Wrist position from your tracker |
| `reference` | `[x,y,z]` or `[x,y]` | Elbow position (for direction) |
| `confidence` | `float 0-1` | Tracking confidence from Apple Vision |
| `timestamp` | `float` (optional) | Frame time in seconds |

---

## Basic Integration

```python
from motion_occlusion_solver import OcclusionSolver

# Create solver (adjust object_length for your use case)
solver = OcclusionSolver(
    object_length=0.55,    # wrist to racket face in meters
    high_conf=0.7,         # above this = good tracking
    low_conf=0.3,          # below this = switch to prediction
    expected_occlusion_time=0.3  # typical occlusion duration
)

# In your frame loop:
def on_frame(wrist_pos, elbow_pos, confidence, timestamp):
    result = solver.process(
        anchor=wrist_pos,
        reference=elbow_pos,
        confidence=confidence,
        timestamp=timestamp
    )
    
    # Use result
    racket_position = result.position
    racket_quaternion = result.quaternion  # [w,x,y,z] format
    is_prediction = result.is_predicted
    
    return result.as_dict()  # for sending to Unity
```

---

## Output Format

```python
result.position      # np.array [x,y,z]
result.direction     # np.array [dx,dy,dz] unit vector
result.quaternion    # np.array [w,x,y,z] or None for 2D
result.velocity      # np.array [vx,vy,vz]
result.confidence    # float 0-1
result.is_predicted  # bool - True if this is predicted, not observed
result.state         # str - "tracking", "predicted", "blending", etc

result.as_dict()     # dict for JSON/Unity
```

---

## Coordinate System Notes

The solver doesn't transform coordinates - it outputs in whatever system you input.

**If your Apple Vision data is right-handed and Unity is left-handed:**
```python
# Transform BEFORE feeding to solver
def transform_to_unity(pos):
    # Flip X axis (common transformation)
    return [-pos[0], pos[1], pos[2]]

# Or transform AFTER
def transform_output(result):
    result.position[0] = -result.position[0]
    if result.quaternion is not None:
        result.quaternion[1] = -result.quaternion[1]  # flip qx
    return result
```

Kalpesh mentioned X is already flipped in the data. Verify with a simple test:
- Point wrist in known direction
- Check output direction matches expectation

---

## How to Verify It's Working

### Test 1: Tracking follows input
```python
solver = OcclusionSolver(object_length=0.5)
r = solver.process([1,0,0], [0,0,0], 0.95)
print(r.position)  # Should be around [1.5, 0, 0]
print(r.is_predicted)  # Should be False
```

### Test 2: Prediction kicks in at low confidence
```python
# Feed good data
for i in range(5):
    solver.process([1,1,1], [0.5,1.5,1], 0.95)

# Feed bad data with low confidence
r = solver.process([0,0,0], [0,0,0], 0.05)
r = solver.process([0,0,0], [0,0,0], 0.05)
print(r.is_predicted)  # Should be True
print(r.position)  # Should be near last good position, not [0,0,0]
```

### Test 3: Recovery doesn't jump
```python
# After occlusion, recover at new position
r = solver.process([2,1,1], [1.5,1.5,1], 0.95)
print(r.state)  # Should be "blending" initially
# Position should smoothly transition, not jump
```

---

## Tuning Parameters

| Parameter | Default | Adjust if... |
|-----------|---------|-------------|
| `object_length` | 0.5 | Change based on actual wrist-to-racket distance |
| `high_conf` | 0.7 | Lower if your tracker is noisier |
| `low_conf` | 0.3 | Raise if predictions start too early |
| `smoothing` | 0.7 | Lower for more responsive, higher for smoother |
| `expected_occlusion_time` | 0.3 | Match typical occlusion duration in your clips |

---

## Known Limitations

1. **Rolling shutter**: Not compensated. At 100mph, distortion is ~4cm. Can add if scanline timing is available.

2. **Grip slip**: Handled via confidence drop detection. If grip slips but confidence stays high, there will be a jump.

3. **Rotation during occlusion**: Currently keeps last known rotation. Could add angular velocity prediction if needed.

---

## Files

| File | Purpose |
|------|---------|
| `motion_occlusion_solver.py` | Main solver - this is what you integrate |
| `test_solver.py` | Validation tests - run before integration |

---

## Questions?

If something doesn't work as expected:
1. Run `test_solver.py` to verify solver is correct
2. Check your input data format matches expected
3. Verify coordinate system alignment with a known test case
