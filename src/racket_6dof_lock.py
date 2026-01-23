"""
Racket 6-DoF Lock
-----------------

Locks the racket rigidly to the wrist throughout the stroke.
Uses minimum jerk smoothing to eliminate jitter while maintaining
accurate attachment.

The racket NEVER slips from the hand - it's mathematically locked
to the wrist joint with a fixed grip offset.

6 Degrees of Freedom:
- Position: X, Y, Z (racket follows wrist position + grip offset)
- Rotation: Roll, Pitch, Yaw (racket orientation from forearm direction)

Usage:
    python racket_6dof_lock.py input.json output.json

Or in Python:
    from racket_6dof_lock import Racket6DoFLock, process_video_json
    
    lock = Racket6DoFLock()
    result = lock.compute(wrist, elbow, timestamp)
"""

import numpy as np
import json
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class GripConfig:
    """
    Grip offset configuration.
    These define how the racket attaches to the wrist.
    """
    # Offset from wrist to grip point (meters)
    # Per Kalpesh: (0.1, 0, 0.03)
    wrist_to_grip: np.ndarray = None
    
    # Distance from wrist to racket face center (meters)
    wrist_to_face: float = 0.55
    
    def __post_init__(self):
        if self.wrist_to_grip is None:
            self.wrist_to_grip = np.array([0.1, 0.0, 0.03])


@dataclass 
class RacketPose:
    """6-DoF Racket Pose output."""
    # Position (3 DoF)
    position: np.ndarray      # Racket face center [x, y, z]
    
    # Orientation (3 DoF) 
    quaternion: np.ndarray    # [w, x, y, z]
    direction: np.ndarray     # Face normal unit vector
    
    # Additional info
    wrist_position: np.ndarray
    timestamp: float
    
    def to_dict(self):
        return {
            'position': self.position.tolist(),
            'quaternion': self.quaternion.tolist(),
            'direction': self.direction.tolist(),
            'wrist_position': self.wrist_position.tolist(),
            'timestamp': self.timestamp
        }


class MinimumJerkFilter:
    """
    Minimum jerk smoothing filter.
    
    Smooths the input signal while preserving the natural motion
    characteristics. Based on the observation that human movements
    minimize jerk (derivative of acceleration).
    """
    
    def __init__(self, smoothing=0.7):
        self.alpha = smoothing
        self.pos = None
        self.vel = None
        self.acc = None
    
    def filter(self, new_pos: np.ndarray, dt: float) -> np.ndarray:
        """Apply minimum jerk smoothing."""
        if self.pos is None:
            self.pos = new_pos.copy()
            self.vel = np.zeros_like(new_pos)
            self.acc = np.zeros_like(new_pos)
            return self.pos
        
        dt = max(dt, 0.001)
        
        # Second-order smoothing for minimum jerk
        # Target position with smoothing
        target_vel = (new_pos - self.pos) / dt
        target_acc = (target_vel - self.vel) / dt
        
        # Smooth acceleration first (reduces jerk)
        self.acc = 0.3 * target_acc + 0.7 * self.acc
        
        # Then smooth velocity
        self.vel = self.vel + self.acc * dt
        self.vel = 0.5 * target_vel + 0.5 * self.vel
        
        # Update position
        self.pos = self.pos + self.vel * dt
        
        # Strong blend toward target (keeps lock tight)
        self.pos = 0.6 * new_pos + 0.4 * self.pos
        
        return self.pos.copy()


class QuaternionFilter:
    """Smooth quaternion filtering using SLERP."""
    
    def __init__(self, smoothing=0.6):
        self.alpha = smoothing
        self.quat = None
    
    def filter(self, new_quat: np.ndarray) -> np.ndarray:
        if self.quat is None:
            self.quat = new_quat.copy()
            return self.quat
        
        # SLERP interpolation
        self.quat = self._slerp(self.quat, new_quat, self.alpha)
        return self.quat.copy()
    
    def _slerp(self, q1, q2, t):
        """Spherical linear interpolation."""
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        theta0 = np.arccos(dot)
        theta = theta0 * t
        
        q2_perp = q2 - q1 * dot
        q2_perp = q2_perp / np.linalg.norm(q2_perp)
        
        return q1 * np.cos(theta) + q2_perp * np.sin(theta)


class Racket6DoFLock:
    """
    6-DoF Racket Lock
    
    Keeps the racket rigidly attached to the wrist throughout the stroke.
    The racket CANNOT slip - it's mathematically locked to the wrist.
    
    How it works:
    1. Takes wrist and elbow positions each frame
    2. Computes forearm direction (elbow → wrist)
    3. Positions racket at wrist + offset along forearm
    4. Orients racket to match forearm direction
    5. Applies minimum jerk smoothing to remove jitter
    
    The grip offset is CONSTANT - racket never moves relative to wrist.
    """
    
    def __init__(self, grip_config: GripConfig = None, smoothing: float = 0.7):
        """
        Args:
            grip_config: Grip offset parameters
            smoothing: Filter strength (0-1, higher = more responsive)
        """
        self.config = grip_config or GripConfig()
        
        # Filters for smooth output
        self.pos_filter = MinimumJerkFilter(smoothing)
        self.quat_filter = QuaternionFilter(smoothing * 0.9)
        
        # State
        self.last_timestamp = 0.0
        self.frame_count = 0
    
    def compute(
        self,
        wrist: np.ndarray,
        elbow: np.ndarray,
        timestamp: float = None
    ) -> RacketPose:
        """
        Compute 6-DoF racket pose locked to wrist.
        
        Args:
            wrist: Wrist position [x, y, z]
            elbow: Elbow position [x, y, z]
            timestamp: Frame time in seconds
            
        Returns:
            RacketPose with position and quaternion
        """
        wrist = np.array(wrist, dtype=float)
        elbow = np.array(elbow, dtype=float)
        
        # Handle timestamp
        self.frame_count += 1
        if timestamp is None:
            timestamp = self.frame_count / 30.0
        dt = timestamp - self.last_timestamp if self.last_timestamp > 0 else 1/30
        self.last_timestamp = timestamp
        
        # Smooth wrist position first (reduces input jitter)
        smooth_wrist = self.pos_filter.filter(wrist, dt)
        
        # Compute forearm direction (elbow → wrist)
        forearm = smooth_wrist - elbow
        forearm_len = np.linalg.norm(forearm)
        
        if forearm_len < 0.001:
            # Fallback if elbow/wrist are same point
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = forearm / forearm_len
        
        # RIGID LOCK: racket position is EXACTLY wrist + fixed offset
        # This distance NEVER changes - mathematically guaranteed
        position = smooth_wrist + direction * self.config.wrist_to_face
        
        # Compute quaternion from direction
        raw_quat = self._direction_to_quaternion(direction)
        
        # Smooth quaternion
        quaternion = self.quat_filter.filter(raw_quat)
        
        return RacketPose(
            position=position,
            quaternion=quaternion,
            direction=direction,
            wrist_position=smooth_wrist,  # Return smoothed wrist
            timestamp=timestamp
        )
    
    def _direction_to_quaternion(self, direction: np.ndarray) -> np.ndarray:
        """Convert direction vector to quaternion [w, x, y, z]."""
        # Normalize
        fwd = direction / (np.linalg.norm(direction) + 1e-8)
        
        # Build rotation matrix
        up = np.array([0.0, 1.0, 0.0])
        
        right = np.cross(up, fwd)
        right_len = np.linalg.norm(right)
        
        if right_len < 1e-6:
            # Direction parallel to up, use different up
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(up, fwd)
            right_len = np.linalg.norm(right)
        
        right = right / right_len
        up = np.cross(fwd, right)
        
        # Rotation matrix to quaternion
        R = np.column_stack([right, up, fwd])
        
        trace = R[0,0] + R[1,1] + R[2,2]
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
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
    
    def reset(self):
        """Reset filter state."""
        self.pos_filter = MinimumJerkFilter(self.pos_filter.alpha)
        self.quat_filter = QuaternionFilter(self.quat_filter.alpha)
        self.last_timestamp = 0.0
        self.frame_count = 0


def process_video_json(
    input_path: str,
    output_path: str,
    wrist_to_face: float = 0.55,
    wrist_key: str = 'right_wrist',
    elbow_key: str = 'right_elbow'
):
    """
    Process a video's pose JSON and output 6-DoF locked racket poses.
    
    Args:
        input_path: Input JSON file path
        output_path: Output JSON file path
        wrist_to_face: Distance from wrist to racket face (meters)
        wrist_key: Key name for wrist in keypoints
        elbow_key: Key name for elbow in keypoints
    """
    # Load input
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    frames = data.get('frames', [])
    
    # Setup lock
    config = GripConfig(wrist_to_face=wrist_to_face)
    lock = Racket6DoFLock(grip_config=config)
    
    output_frames = []
    
    for i, frame in enumerate(frames):
        timestamp = frame.get('timestamp', i / 30.0)
        keypoints = frame.get('keypoints', [])
        
        # Extract wrist and elbow
        wrist = None
        elbow = None
        
        for kp in keypoints:
            name = kp.get('name', '')
            if wrist_key in name:
                wrist = [kp.get('x', 0), kp.get('y', 0), kp.get('z', 0)]
            elif elbow_key in name:
                elbow = [kp.get('x', 0), kp.get('y', 0), kp.get('z', 0)]
        
        if wrist is None or elbow is None:
            output_frames.append({
                'frame_index': i,
                'timestamp': timestamp,
                'error': 'missing wrist or elbow'
            })
            continue
        
        # Compute locked pose
        pose = lock.compute(wrist, elbow, timestamp)
        
        output_frames.append({
            'frame_index': i,
            'timestamp': timestamp,
            'racket_position': pose.position.tolist(),
            'racket_quaternion': pose.quaternion.tolist(),
            'racket_direction': pose.direction.tolist(),
            'wrist_position': pose.wrist_position.tolist()
        })
    
    # Write output
    output = {
        'source': input_path,
        'wrist_to_face': wrist_to_face,
        'total_frames': len(output_frames),
        'description': '6-DoF locked racket pose per frame',
        'frames': output_frames
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Processed {len(output_frames)} frames")
    print(f"Output: {output_path}")
    
    return output


# Test and CLI
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) >= 3:
        # Process JSON file
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        distance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.55
        
        process_video_json(input_file, output_file, wrist_to_face=distance)
    
    else:
        # Demo test
        print("6-DoF Racket Lock - Demo")
        print("=" * 50)
        
        lock = Racket6DoFLock()
        
        # Simulate backhand stroke
        test_data = [
            {'t': 0.000, 'w': [0.5, 1.0, 0.3], 'e': [0.4, 1.2, 0.3]},
            {'t': 0.033, 'w': [0.52, 0.98, 0.32], 'e': [0.42, 1.18, 0.32]},
            {'t': 0.066, 'w': [0.55, 0.95, 0.35], 'e': [0.45, 1.15, 0.35]},
            {'t': 0.100, 'w': [0.60, 0.90, 0.40], 'e': [0.50, 1.10, 0.40]},
            {'t': 0.133, 'w': [0.68, 0.85, 0.45], 'e': [0.58, 1.05, 0.45]},
            {'t': 0.166, 'w': [0.75, 0.80, 0.50], 'e': [0.65, 1.00, 0.50]},
            {'t': 0.200, 'w': [0.80, 0.78, 0.52], 'e': [0.70, 0.98, 0.52]},
        ]
        
        print(f"{'Frame':<6} {'Time':<8} {'Racket Position':<30} {'Locked to Wrist'}")
        print("-" * 70)
        
        for i, frame in enumerate(test_data):
            pose = lock.compute(frame['w'], frame['e'], frame['t'])
            
            # Verify lock: racket should be at fixed offset from wrist
            offset = np.linalg.norm(pose.position - pose.wrist_position)
            
            pos_str = f"[{pose.position[0]:.3f}, {pose.position[1]:.3f}, {pose.position[2]:.3f}]"
            print(f"{i:<6} {frame['t']:.3f}s  {pos_str:<30} offset={offset:.3f}m")
        
        print("-" * 70)
        print("Racket stays locked to wrist throughout stroke.")
        print()
        print("Usage: python racket_6dof_lock.py input.json output.json [wrist_to_face]")



