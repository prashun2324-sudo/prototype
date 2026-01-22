# Occlusion Handling Module

## What This Is

A standalone module that handles tracking gaps when the racket/hand goes out of view (behind body, back-scratch position, etc). 

Feed it your Apple Vision tracking data, it outputs smooth racket position even during occlusion.

---

## Files Included

```
delivery/
├── motion_occlusion_solver.py   <- Main solver (integrate this)
├── test_solver.py               <- Run this to validate
└── README.md                    <- You're reading this
```

---

## Quick Start

### Step 1: Validate
```bash
python test_solver.py
```
All 10 tests should pass. If any fail, don't integrate.

### Step 2: Integrate
```python
from motion_occlusion_solver import OcclusionSolver

solver = OcclusionSolver(object_length=0.55)

# Each frame:
result = solver.process(
    anchor=wrist_xyz,        # from Apple Vision
    reference=elbow_xyz,     # from Apple Vision
    confidence=conf_score    # from Apple Vision
)

# Output for Unity:
position = result.position       # [x, y, z]
quaternion = result.quaternion   # [w, x, y, z]
is_predicted = result.is_predicted
```

---

## How It Works

1. **Tracks confidence** - when Apple Vision confidence drops, we know tracking is unreliable

2. **Predicts during occlusion** - uses minimum jerk trajectory (biomechanically accurate human motion model)

3. **Blends on recovery** - when tracking returns, smoothly transitions back (no jumping)

---

## Inputs Required

| Input | Type | Source |
|-------|------|--------|
| `anchor` | `[x,y,z]` | Wrist position from Apple Vision |
| `reference` | `[x,y,z]` | Elbow position from Apple Vision |
| `confidence` | `0.0-1.0` | Tracking confidence from Apple Vision |

---

## Output

| Field | Type | Description |
|-------|------|-------------|
| `position` | `[x,y,z]` | Racket face center position |
| `quaternion` | `[w,x,y,z]` | Racket rotation for Unity |
| `is_predicted` | `bool` | True = predicted, False = observed |
| `confidence` | `float` | Reliability score |
| `state` | `string` | "tracking", "predicted", "blending" |

---

## Configuration

```python
solver = OcclusionSolver(
    object_length=0.55,           # wrist to racket face (meters)
    high_conf=0.7,                # good tracking threshold
    low_conf=0.3,                 # occlusion threshold
    expected_occlusion_time=0.3   # typical gap duration (seconds)
)
```

Adjust `object_length` based on your racket model.

---

## Coordinate System

The solver outputs in the same coordinate system as your input. 

If you need Apple Vision (right-handed) to Unity (left-handed):
- Negate X position
- Negate X component of quaternion

---

## Hybrid Integration

This module handles the **occlusion prediction** part. Your existing Apple Vision tracking handles the **observation** part.

```
Apple Vision Data
       │
       ▼
┌─────────────────┐
│ Occlusion       │
│ Solver          │──▶ Smooth output even during gaps
└─────────────────┘
       │
       ▼
   Unity Rig
```

When confidence is high → uses Apple Vision data directly
When confidence drops → switches to prediction
When confidence returns → blends back smoothly

---

## Questions

If tests pass but integration doesn't work:
1. Check coordinate system alignment
2. Verify confidence score range (should be 0-1)
3. Check wrist/elbow positions are in meters (not pixels)

