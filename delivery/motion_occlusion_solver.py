"""
Motion Occlusion Solver
-----------------------

Handles tracking gaps when the hand/object goes behind the body
or out of camera view. Uses prediction to fill in missing frames.

Basic idea:
- When confidence drops, we know tracking is getting unreliable
- When it drops below threshold, switch to prediction mode
- Predict using velocity + minimum jerk (smooth motion)
- When tracking comes back, blend smoothly to avoid jumps

Usage:
    solver = OcclusionSolver(object_length=0.55)
    
    for frame in frames:
        result = solver.process(
            wrist_pos, elbow_pos, confidence, time
        )
        # result has position, direction, whether its predicted, etc

"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class State(Enum):
    TRACKING = "tracking"
    LOW_CONF = "low_confidence" 
    PREDICTED = "predicted"
    BLENDING = "blending"


@dataclass
class Result:
    """Output from the solver."""
    position: np.ndarray
    direction: np.ndarray
    quaternion: Optional[np.ndarray]
    velocity: np.ndarray
    confidence: float
    is_predicted: bool
    state: str
    
    def as_dict(self):
        return {
            'position': self.position.tolist(),
            'direction': self.direction.tolist(),
            'quaternion': self.quaternion.tolist() if self.quaternion is not None else None,
            'velocity': self.velocity.tolist(),
            'confidence': self.confidence,
            'occluded': self.is_predicted,
            'state': self.state
        }


def quat_from_direction(dir_vec, up=None):
    """
    Make a quaternion that points in the given direction.
    Returns [w, x, y, z] format.
    """
    if len(dir_vec) != 3:
        return None
    
    if up is None:
        up = np.array([0, 1, 0])
    
    # normalize direction
    fwd = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)
    
    # make orthogonal basis
    right = np.cross(up, fwd)
    rlen = np.linalg.norm(right)
    if rlen < 1e-6:
        # direction is parallel to up, pick different up
        up = np.array([0, 0, 1])
        right = np.cross(up, fwd)
        rlen = np.linalg.norm(right)
    
    right = right / rlen
    up = np.cross(fwd, right)
    
    # rotation matrix
    R = np.column_stack([right, up, fwd])
    
    # convert to quaternion (standard algorithm)
    tr = R[0,0] + R[1,1] + R[2,2]
    
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def slerp(q1, q2, t):
    """Interpolate between two quaternions."""
    if q1 is None or q2 is None:
        return q2 if q2 is not None else q1
    
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    dot = np.dot(q1, q2)
    
    # take shorter path
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    # if very close, just lerp
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    theta0 = np.arccos(dot)
    theta = theta0 * t
    
    q2_perp = q2 - q1 * dot
    q2_perp = q2_perp / np.linalg.norm(q2_perp)
    
    return q1 * np.cos(theta) + q2_perp * np.sin(theta)


def min_jerk_predict(start_pos, start_vel, duration, t):
    """
    Predict position using minimum jerk trajectory.
    
    This is based on the observation that human movements
    tend to minimize jerk (derivative of acceleration).
    Makes predictions look natural.
    """
    # clamp t to [0, duration]
    tau = min(1.0, t / duration) if duration > 0 else 1.0
    
    # estimate where we'll end up
    end_pos = start_pos + start_vel * duration * 0.7  # slow down a bit
    end_vel = start_vel * 0.3
    
    # polynomial coefficients for min jerk
    t2 = tau * tau
    t3 = t2 * tau
    t4 = t3 * tau
    t5 = t4 * tau
    
    # position basis functions
    h1 = 1 - 10*t3 + 15*t4 - 6*t5
    h2 = tau - 6*t3 + 8*t4 - 3*t5
    h3 = 10*t3 - 15*t4 + 6*t5
    h4 = -4*t3 + 7*t4 - 3*t5
    
    pos = h1*start_pos + h2*duration*start_vel + h3*end_pos + h4*duration*end_vel
    
    # velocity (derivative)
    dh1 = -30*t2 + 60*t3 - 30*t4
    dh2 = 1 - 18*t2 + 32*t3 - 15*t4
    dh3 = 30*t2 - 60*t3 + 30*t4
    dh4 = -12*t2 + 28*t3 - 15*t4
    
    if duration > 0:
        vel = (dh1*start_pos + dh2*duration*start_vel + dh3*end_pos + dh4*duration*end_vel) / duration
    else:
        vel = start_vel
    
    return pos, vel


class OcclusionSolver:
    """
    Main solver class.
    
    Tracks an object attached to a body point (like a racket on a wrist).
    When tracking confidence drops, switches to prediction mode.
    """
    
    def __init__(
        self,
        object_length=0.5,
        high_conf=0.7,
        low_conf=0.3,
        smoothing=0.7,
        expected_occlusion_time=0.3
    ):
        """
        Args:
            object_length: how far from anchor to object center
            high_conf: above this = good tracking
            low_conf: below this = switch to prediction
            smoothing: position filter (0-1)
            expected_occlusion_time: roughly how long occlusions last
        """
        self.obj_len = object_length
        self.high_conf = high_conf
        self.low_conf = low_conf
        self.smooth = smoothing
        self.occ_time = expected_occlusion_time
        
        # current state
        self.state = State.TRACKING
        self.is_3d = None
        
        # tracking history
        self.conf_history = []
        
        # position/velocity estimates
        self.pos = None
        self.vel = None
        self.dir = None
        self.quat = None
        
        # for occlusion handling
        self.occ_start_t = 0
        self.occ_start_pos = None
        self.occ_start_vel = None
        self.blend_start_t = 0
        
        # timing
        self.last_t = 0
        self.frame_num = 0
    
    def process(
        self,
        anchor,
        reference,
        confidence,
        timestamp=None,
        second_anchor=None
    ):
        """
        Process one frame of tracking data.
        
        Args:
            anchor: position of attachment point (wrist) - list or array
            reference: reference point for direction (elbow)
            confidence: tracking confidence 0-1
            timestamp: time in seconds (optional, will auto-increment)
            second_anchor: optional second hand position
            
        Returns:
            Result object with position, direction, etc
        """
        # convert inputs
        anchor = np.array(anchor, dtype=float)
        reference = np.array(reference, dtype=float)
        if second_anchor is not None:
            second_anchor = np.array(second_anchor, dtype=float)
        
        # figure out if 2D or 3D
        if self.is_3d is None:
            self.is_3d = (len(anchor) == 3)
        
        # handle time
        self.frame_num += 1
        if timestamp is None:
            timestamp = self.frame_num / 30.0
        dt = timestamp - self.last_t if self.last_t > 0 else 1/30
        self.last_t = timestamp
        
        # track confidence
        self.conf_history.append(confidence)
        if len(self.conf_history) > 10:
            self.conf_history.pop(0)
        
        # average recent confidence
        if len(self.conf_history) >= 3:
            avg_conf = sum(self.conf_history[-3:]) / 3
        else:
            avg_conf = confidence
        
        # figure out what state we should be in
        new_state = self._decide_state(avg_conf)
        
        # handle transitions
        if new_state != self.state:
            self._on_transition(new_state, timestamp)
        self.state = new_state
        
        # compute output based on state
        if self.state == State.TRACKING:
            return self._do_tracking(anchor, reference, second_anchor, confidence)
        elif self.state == State.PREDICTED:
            return self._do_prediction(timestamp)
        elif self.state == State.LOW_CONF:
            return self._do_low_conf(anchor, reference, second_anchor, confidence)
        else:  # BLENDING
            return self._do_blending(anchor, reference, second_anchor, timestamp, confidence)
    
    def _decide_state(self, conf):
        """Figure out which state we should be in."""
        if conf >= self.high_conf:
            if self.state == State.PREDICTED:
                return State.BLENDING
            return State.TRACKING
        elif conf >= self.low_conf:
            if self.state == State.TRACKING:
                return State.LOW_CONF
            elif self.state == State.PREDICTED:
                return State.BLENDING
            return self.state
        else:
            return State.PREDICTED
    
    def _on_transition(self, new_state, t):
        """Handle state changes."""
        if new_state == State.PREDICTED:
            # entering occlusion - save current state for prediction
            self.occ_start_t = t
            if self.pos is not None:
                self.occ_start_pos = self.pos.copy()
                self.occ_start_vel = self.vel.copy() if self.vel is not None else np.zeros_like(self.pos)
        elif new_state == State.BLENDING:
            self.blend_start_t = t
    
    def _do_tracking(self, anchor, ref, second, conf):
        """Normal tracking - just follow the data."""
        # direction from reference to anchor
        d = anchor - ref
        dlen = np.linalg.norm(d)
        
        if dlen > 1e-6:
            direction = d / dlen
        elif self.dir is not None:
            direction = self.dir
        else:
            direction = np.zeros_like(anchor)
            direction[0] = 1
        
        # if two-handed, adjust direction
        if second is not None:
            dist = np.linalg.norm(second - anchor)
            if dist < dlen * 2:  # hands are close
                grip = second - anchor
                glen = np.linalg.norm(grip)
                if glen > 1e-6:
                    grip_dir = grip / glen
                    direction = 0.7 * direction + 0.3 * grip_dir
                    direction = direction / np.linalg.norm(direction)
        
        # object position
        pos = anchor + direction * self.obj_len
        
        # compute velocity
        if self.pos is not None:
            new_vel = pos - self.pos
            if self.vel is not None:
                vel = 0.3 * new_vel + 0.7 * self.vel
            else:
                vel = new_vel
        else:
            vel = np.zeros_like(pos)
        
        # smooth position
        if self.pos is not None:
            pos = self.smooth * pos + (1 - self.smooth) * self.pos
        
        # save state
        self.pos = pos.copy()
        self.vel = vel.copy()
        self.dir = direction.copy()
        
        # quaternion for 3D
        quat = None
        if self.is_3d:
            quat = quat_from_direction(direction)
            if self.quat is not None:
                quat = slerp(self.quat, quat, 0.6)
            self.quat = quat
        
        return Result(
            position=pos,
            direction=direction,
            quaternion=quat,
            velocity=vel,
            confidence=conf,
            is_predicted=False,
            state="tracking"
        )
    
    def _do_prediction(self, t):
        """Predict position during occlusion."""
        if self.occ_start_pos is None:
            # no history, just return last known
            dims = 3 if self.is_3d else 2
            return Result(
                position=self.pos if self.pos is not None else np.zeros(dims),
                direction=self.dir if self.dir is not None else np.array([1,0,0] if self.is_3d else [1,0]),
                quaternion=self.quat,
                velocity=self.vel if self.vel is not None else np.zeros(dims),
                confidence=0.2,
                is_predicted=True,
                state="predicted"
            )
        
        # time since occlusion started
        dt = t - self.occ_start_t
        
        # predict using min jerk
        pos, vel = min_jerk_predict(
            self.occ_start_pos,
            self.occ_start_vel,
            self.occ_time,
            dt
        )
        
        self.pos = pos.copy()
        self.vel = vel.copy()
        
        # confidence decays over time
        conf = max(0.1, 0.5 - dt)
        
        return Result(
            position=pos,
            direction=self.dir if self.dir is not None else np.array([1,0,0] if self.is_3d else [1,0]),
            quaternion=self.quat,
            velocity=vel,
            confidence=conf,
            is_predicted=True,
            state="predicted"
        )
    
    def _do_low_conf(self, anchor, ref, second, conf):
        """Low confidence - use last known position, don't trust bad data."""
        # If confidence is very low, don't update position from bad data
        if conf < 0.2 and self.pos is not None:
            # Return last known good position
            return Result(
                position=self.pos.copy(),
                direction=self.dir if self.dir is not None else np.array([1,0,0] if self.is_3d else [1,0]),
                quaternion=self.quat,
                velocity=self.vel if self.vel is not None else np.zeros_like(self.pos),
                confidence=conf,
                is_predicted=True,
                state="low_confidence"
            )
        
        result = self._do_tracking(anchor, ref, second, conf)
        result.confidence = conf * 0.8
        result.state = "low_confidence"
        return result
    
    def _do_blending(self, anchor, ref, second, t, conf):
        """Blend from prediction back to tracking."""
        # get fresh tracking result
        tracked = self._do_tracking(anchor, ref, second, conf)
        
        # blend factor
        dt = t - self.blend_start_t
        blend = min(1.0, dt / 0.15)  # 150ms blend
        
        if blend < 1.0 and self.occ_start_pos is not None:
            # mix predicted and tracked
            tracked.position = blend * tracked.position + (1-blend) * self.pos
            
            if tracked.quaternion is not None and self.quat is not None:
                tracked.quaternion = slerp(self.quat, tracked.quaternion, blend)
        
        tracked.confidence = conf * blend
        tracked.state = "blending" if blend < 1.0 else "tracking"
        tracked.is_predicted = blend < 1.0
        
        return tracked
    
    def reset(self):
        """Clear all state."""
        self.state = State.TRACKING
        self.conf_history = []
        self.pos = None
        self.vel = None
        self.dir = None
        self.quat = None
        self.occ_start_pos = None
        self.occ_start_vel = None
        self.last_t = 0
        self.frame_num = 0


# quick test
if __name__ == '__main__':
    print("Testing occlusion solver...")
    print("-" * 40)
    
    solver = OcclusionSolver(object_length=0.55)
    
    # some test data
    test_frames = [
        {'w': [0.5, 1.0, 0.3], 'e': [0.4, 1.2, 0.3], 'c': 0.95},
        {'w': [0.55, 0.95, 0.35], 'e': [0.45, 1.15, 0.35], 'c': 0.90},
        {'w': [0.6, 0.9, 0.4], 'e': [0.5, 1.1, 0.4], 'c': 0.50},
        {'w': [0, 0, 0], 'e': [0, 0, 0], 'c': 0.05},  # lost tracking
        {'w': [0, 0, 0], 'e': [0, 0, 0], 'c': 0.05},
        {'w': [0.7, 0.8, 0.5], 'e': [0.6, 1.0, 0.5], 'c': 0.85},  # recovered
        {'w': [0.75, 0.75, 0.55], 'e': [0.65, 0.95, 0.55], 'c': 0.95},
    ]
    
    for i, f in enumerate(test_frames):
        r = solver.process(f['w'], f['e'], f['c'])
        p = r.position
        print(f"frame {i}: conf={f['c']:.2f}, state={r.state:15s}, "
              f"pos=[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}], predicted={r.is_predicted}")
    
    print("-" * 40)
    print("done")
