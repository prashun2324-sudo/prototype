"""
Racket 6-DoF Lock with World-Space Transform
=============================================

Senior Kinematics Implementation for Tennis Racket Tracking

This module provides production-ready 6-DoF racket tracking with proper
coordinate system handling:

Coordinate Systems:
    - LOCAL SPACE: Pelvis-centered coordinates from Apple Vision (keypoints3D)
                   Origin at subject's pelvis, Y-up, units in meters
    
    - WORLD SPACE: Global coordinates after applying camera extrinsics
                   Suitable for Unity integration

Pipeline:
    1. Extract 3D keypoints (rightWrist, rightElbow) from keypoints3D
    2. Compute racket pose in local space (locked to wrist)
    3. Apply camera extrinsics to transform to world space
    4. Output 6-DoF: position (x,y,z) + quaternion (w,x,y,z)

Camera Extrinsics:
    The 4x4 transformation matrix provided per-frame transforms from
    pelvis-local space to world space. Stored as 16 floats in column-major
    order (OpenGL/Unity convention).

Usage:
    from racket_6dof_lock import Racket6DoFLock, process_video_json
    
    # Process entire video
    process_video_json('input.json', 'output.json')
    
    # Or frame-by-frame
    lock = Racket6DoFLock()
    result = lock.compute(wrist_3d, elbow_3d, timestamp, extrinsics)

Author: Kinematics Engineering Team
"""

import numpy as np
import json
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GripConfig:
    """
    Racket grip configuration parameters.
    
    These define the geometric relationship between the wrist joint
    and the racket face center. Based on standard tennis grip mechanics.
    
    Attributes:
        wrist_to_grip: Offset from wrist joint to grip point (meters)
        wrist_to_face: Distance from wrist to racket face center (meters)
                       Standard tennis racket ~0.55m from grip to face
    """
    wrist_to_grip: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.0, 0.03]))
    wrist_to_face: float = 0.55


@dataclass
class RacketPose:
    """
    6-DoF Racket Pose output structure.
    
    Contains complete pose information in both local (pelvis-centered)
    and world coordinate spaces.
    
    Attributes:
        position_local: Racket face center in pelvis-local space [x,y,z] meters
        position_world: Racket face center in world space [x,y,z] meters
        quaternion_local: Orientation in local space [w,x,y,z]
        quaternion_world: Orientation in world space [w,x,y,z]
        direction: Face normal unit vector (local space)
        wrist_local: Wrist position in local space
        wrist_world: Wrist position in world space
        timestamp: Frame timestamp in seconds
        frame_index: Frame number
    """
    position_local: np.ndarray
    position_world: np.ndarray
    quaternion_local: np.ndarray
    quaternion_world: np.ndarray
    direction: np.ndarray
    wrist_local: np.ndarray
    wrist_world: np.ndarray
    timestamp: float
    frame_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'frame_index': self.frame_index,
            'timestamp': self.timestamp,
            # World space (primary output for Unity)
            'position': self.position_world.tolist(),
            'quaternion': self.quaternion_world.tolist(),
            # Local space (for debugging/validation)
            'position_local': self.position_local.tolist(),
            'quaternion_local': self.quaternion_local.tolist(),
            # Additional data
            'direction': self.direction.tolist(),
            'wrist_position': self.wrist_world.tolist(),
            'wrist_position_local': self.wrist_local.tolist()
        }


# =============================================================================
# COORDINATE TRANSFORMATIONS
# =============================================================================

class CoordinateTransform:
    """
    Handles coordinate system transformations using camera extrinsics.
    
    The camera extrinsics matrix transforms from pelvis-local space
    to world space. It's a 4x4 homogeneous transformation matrix
    stored in column-major order.
    
    Matrix structure:
        [R00 R01 R02 Tx]     Columns 0-2: Rotation (3x3)
        [R10 R11 R12 Ty]     Column 3: Translation (3x1)
        [R20 R21 R22 Tz]
        [0   0   0   1 ]
    """
    
    @staticmethod
    def parse_extrinsics(extrinsics_array: List[float]) -> np.ndarray:
        """
        Parse camera extrinsics from JSON array to 4x4 matrix.
        
        Args:
            extrinsics_array: 16 floats in column-major order
            
        Returns:
            4x4 transformation matrix (row-major for numpy operations)
        """
        if len(extrinsics_array) != 16:
            raise ValueError(f"Expected 16 values for 4x4 matrix, got {len(extrinsics_array)}")
        
        # Column-major to row-major conversion
        matrix = np.array(extrinsics_array).reshape(4, 4, order='F')
        return matrix
    
    @staticmethod
    def transform_point(point: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Transform a 3D point using the extrinsics matrix.
        
        Args:
            point: 3D point [x, y, z]
            matrix: 4x4 transformation matrix
            
        Returns:
            Transformed 3D point in world space
        """
        # Convert to homogeneous coordinates
        point_h = np.array([point[0], point[1], point[2], 1.0])
        
        # Apply transformation
        world_h = matrix @ point_h
        
        # Return 3D point (drop homogeneous coordinate)
        return world_h[:3]
    
    @staticmethod
    def transform_direction(direction: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Transform a direction vector (rotation only, no translation).
        
        Args:
            direction: Unit direction vector [x, y, z]
            matrix: 4x4 transformation matrix
            
        Returns:
            Transformed direction vector (normalized)
        """
        # Extract rotation part (3x3)
        rotation = matrix[:3, :3]
        
        # Apply rotation
        world_dir = rotation @ direction
        
        # Normalize
        norm = np.linalg.norm(world_dir)
        if norm > 1e-8:
            world_dir = world_dir / norm
        
        return world_dir
    
    @staticmethod
    def transform_quaternion(quat: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Transform a quaternion to world space.
        
        Args:
            quat: Quaternion [w, x, y, z]
            matrix: 4x4 transformation matrix
            
        Returns:
            Transformed quaternion in world space
        """
        # Extract rotation matrix from extrinsics
        rot_extrinsics = matrix[:3, :3]
        
        # Convert input quaternion to rotation matrix
        rot_local = CoordinateTransform._quat_to_matrix(quat)
        
        # Combine rotations: world = extrinsics * local
        rot_world = rot_extrinsics @ rot_local
        
        # Convert back to quaternion
        return CoordinateTransform._matrix_to_quat(rot_world)
    
    @staticmethod
    def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
    
    @staticmethod
    def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [w,x,y,z]."""
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


# =============================================================================
# SMOOTHING FILTERS
# =============================================================================

class MinimumJerkFilter:
    """
    Minimum jerk smoothing filter for natural motion.
    
    Based on the biomechanical principle that human movements
    minimize jerk (derivative of acceleration). This produces
    smooth, natural-looking motion while maintaining responsiveness.
    """
    
    def __init__(self, smoothing: float = 0.7):
        """
        Args:
            smoothing: Filter strength 0-1 (higher = more responsive)
        """
        self.alpha = smoothing
        self.pos = None
        self.vel = None
        self.acc = None
    
    def filter(self, new_pos: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply minimum jerk smoothing to position.
        
        Args:
            new_pos: New position measurement
            dt: Time delta since last frame
            
        Returns:
            Smoothed position
        """
        if self.pos is None:
            self.pos = new_pos.copy()
            self.vel = np.zeros_like(new_pos)
            self.acc = np.zeros_like(new_pos)
            return self.pos.copy()
        
        dt = max(dt, 0.001)  # Prevent division by zero
        
        # Compute target velocity and acceleration
        target_vel = (new_pos - self.pos) / dt
        target_acc = (target_vel - self.vel) / dt
        
        # Smooth acceleration first (reduces jerk)
        self.acc = 0.3 * target_acc + 0.7 * self.acc
        
        # Update velocity with smoothed acceleration
        self.vel = self.vel + self.acc * dt
        self.vel = 0.5 * target_vel + 0.5 * self.vel
        
        # Update position
        self.pos = self.pos + self.vel * dt
        
        # Strong blend toward target (keeps lock tight)
        self.pos = 0.6 * new_pos + 0.4 * self.pos
        
        return self.pos.copy()
    
    def reset(self):
        """Reset filter state."""
        self.pos = None
        self.vel = None
        self.acc = None


class QuaternionFilter:
    """
    Smooth quaternion filtering using SLERP interpolation.
    
    Maintains smooth rotation without gimbal lock issues.
    """
    
    def __init__(self, smoothing: float = 0.6):
        self.alpha = smoothing
        self.quat = None
    
    def filter(self, new_quat: np.ndarray) -> np.ndarray:
        """Apply SLERP smoothing to quaternion."""
        if self.quat is None:
            self.quat = new_quat.copy()
            return self.quat.copy()
        
        self.quat = self._slerp(self.quat, new_quat, self.alpha)
        return self.quat.copy()
    
    def _slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between quaternions."""
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        dot = np.dot(q1, q2)
        
        # Take shorter path
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        # Linear interpolation for very close quaternions
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        theta0 = np.arccos(dot)
        theta = theta0 * t
        
        q2_perp = q2 - q1 * dot
        q2_perp = q2_perp / np.linalg.norm(q2_perp)
        
        return q1 * np.cos(theta) + q2_perp * np.sin(theta)
    
    def reset(self):
        """Reset filter state."""
        self.quat = None


# =============================================================================
# MAIN RACKET LOCK CLASS
# =============================================================================

class Racket6DoFLock:
    """
    6-DoF Racket Lock with World-Space Transform
    
    Keeps the racket rigidly attached to the wrist throughout the stroke.
    Outputs both local (pelvis-centered) and world space coordinates.
    
    The lock is mathematically guaranteed - the racket CANNOT slip
    from the wrist because position is computed directly from wrist position.
    
    Pipeline per frame:
        1. Smooth wrist position (minimum jerk filter)
        2. Compute forearm direction (elbow -> wrist)
        3. Position racket at wrist + offset along forearm direction
        4. Compute orientation quaternion from direction
        5. Transform to world space using camera extrinsics
    """
    
    def __init__(
        self,
        grip_config: GripConfig = None,
        smoothing: float = 0.7
    ):
        """
        Initialize the racket lock.
        
        Args:
            grip_config: Grip offset parameters (uses defaults if None)
            smoothing: Filter responsiveness 0-1 (higher = more responsive)
        """
        self.config = grip_config or GripConfig()
        
        # Smoothing filters
        self.pos_filter = MinimumJerkFilter(smoothing)
        self.quat_filter = QuaternionFilter(smoothing * 0.9)
        
        # State tracking
        self.last_timestamp = 0.0
        self.frame_count = 0
    
    def compute(
        self,
        wrist: np.ndarray,
        elbow: np.ndarray,
        timestamp: float = None,
        extrinsics: np.ndarray = None,
        frame_index: int = 0
    ) -> RacketPose:
        """
        Compute 6-DoF racket pose locked to wrist.
        
        Args:
            wrist: Wrist position in local space [x, y, z] meters
            elbow: Elbow position in local space [x, y, z] meters
            timestamp: Frame time in seconds (auto-increments if None)
            extrinsics: 4x4 camera extrinsics matrix (identity if None)
            frame_index: Frame number for output
            
        Returns:
            RacketPose with local and world space coordinates
        """
        wrist = np.array(wrist, dtype=np.float64)
        elbow = np.array(elbow, dtype=np.float64)
        
        # Handle timestamp
        self.frame_count += 1
        if timestamp is None:
            timestamp = self.frame_count / 30.0
        dt = timestamp - self.last_timestamp if self.last_timestamp > 0 else 1/30
        self.last_timestamp = timestamp
        
        # Default extrinsics (identity matrix)
        if extrinsics is None:
            extrinsics = np.eye(4)
        
        # =====================================================================
        # LOCAL SPACE COMPUTATION
        # =====================================================================
        
        # Smooth wrist position
        smooth_wrist = self.pos_filter.filter(wrist, dt)
        
        # Compute forearm direction (elbow -> wrist)
        forearm = smooth_wrist - elbow
        forearm_len = np.linalg.norm(forearm)
        
        if forearm_len < 0.001:
            # Fallback if elbow/wrist coincide
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = forearm / forearm_len
        
        # RIGID LOCK: racket position = wrist + fixed offset along forearm
        # This is mathematically guaranteed to keep constant distance
        position_local = smooth_wrist + direction * self.config.wrist_to_face
        
        # Compute quaternion from direction
        raw_quat = self._direction_to_quaternion(direction)
        quaternion_local = self.quat_filter.filter(raw_quat)
        
        # =====================================================================
        # WORLD SPACE TRANSFORMATION
        # =====================================================================
        
        # Transform positions to world space
        position_world = CoordinateTransform.transform_point(position_local, extrinsics)
        wrist_world = CoordinateTransform.transform_point(smooth_wrist, extrinsics)
        
        # Transform quaternion to world space
        quaternion_world = CoordinateTransform.transform_quaternion(quaternion_local, extrinsics)
        
        return RacketPose(
            position_local=position_local,
            position_world=position_world,
            quaternion_local=quaternion_local,
            quaternion_world=quaternion_world,
            direction=direction,
            wrist_local=smooth_wrist,
            wrist_world=wrist_world,
            timestamp=timestamp,
            frame_index=frame_index
        )
    
    def _direction_to_quaternion(self, direction: np.ndarray) -> np.ndarray:
        """
        Convert direction vector to quaternion [w, x, y, z].
        
        Creates a rotation that aligns the forward axis with the direction.
        """
        fwd = direction / (np.linalg.norm(direction) + 1e-8)
        
        # Build orthonormal basis
        up = np.array([0.0, 1.0, 0.0])
        
        right = np.cross(up, fwd)
        right_len = np.linalg.norm(right)
        
        if right_len < 1e-6:
            # Direction is parallel to up, use different up
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(up, fwd)
            right_len = np.linalg.norm(right)
        
        right = right / right_len
        up = np.cross(fwd, right)
        
        # Rotation matrix
        R = np.column_stack([right, up, fwd])
        
        return CoordinateTransform._matrix_to_quat(R)
    
    def reset(self):
        """Reset all filter states."""
        self.pos_filter.reset()
        self.quat_filter.reset()
        self.last_timestamp = 0.0
        self.frame_count = 0


# =============================================================================
# JSON PROCESSING
# =============================================================================

def extract_keypoint_3d(keypoints3D: List[Dict], name: str) -> Optional[np.ndarray]:
    """
    Extract a named keypoint from keypoints3D array.
    
    Args:
        keypoints3D: List of keypoint dictionaries
        name: Keypoint name (e.g., 'rightWrist', 'rightElbow')
        
    Returns:
        3D position array [x, y, z] or None if not found
    """
    for kp in keypoints3D:
        if kp.get('name') == name:
            return np.array([
                kp.get('x', 0.0),
                kp.get('y', 0.0),
                kp.get('z', 0.0)
            ])
    return None


def process_video_json(
    input_path: str,
    output_path: str,
    wrist_to_face: float = 0.55,
    wrist_key: str = 'rightWrist',
    elbow_key: str = 'rightElbow',
    left_wrist_key: str = 'leftWrist'
) -> Dict[str, Any]:
    """
    Process a video's pose JSON and output world-space 6-DoF racket poses.
    
    This function:
        1. Reads keypoints3D (3D coordinates in pelvis-local space, meters)
        2. Computes locked racket pose per frame
        3. Applies camera extrinsics for world-space output
        4. Outputs JSON with both local and world coordinates
    
    Args:
        input_path: Input JSON file path (Apple Vision pose data)
        output_path: Output JSON file path
        wrist_to_face: Distance from wrist to racket face (meters)
        wrist_key: Key name for wrist in keypoints3D
        elbow_key: Key name for elbow in keypoints3D
        left_wrist_key: Key name for left wrist (for two-handed detection)
    
    Returns:
        Output data dictionary
        
    Input Format (Apple Vision):
        {
            "frames": [
                {
                    "timestamp": 0.0,
                    "camera_extrinsics": [16 floats, column-major 4x4],
                    "keypoints3D": [
                        {"name": "rightWrist", "x": 0.1, "y": 0.3, "z": 0.4},
                        {"name": "rightElbow", "x": 0.2, "y": 0.5, "z": 0.3},
                        ...
                    ],
                    "handTrackingQuality": 0.95,
                    ...
                }
            ]
        }
    
    Output Format:
        {
            "metadata": {...},
            "frames": [
                {
                    "frame_index": 0,
                    "timestamp": 0.0,
                    "position": [x, y, z],        // World space (meters)
                    "quaternion": [w, x, y, z],   // World space
                    "position_local": [x, y, z],  // Pelvis-local (meters)
                    "quaternion_local": [w, x, y, z],
                    "wrist_position": [x, y, z],  // World space
                    "direction": [dx, dy, dz],    // Local space
                    "confidence": 0.95
                }
            ]
        }
    """
    # Load input
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    frames = data.get('frames', [])
    
    # Setup lock
    config = GripConfig(wrist_to_face=wrist_to_face)
    lock = Racket6DoFLock(grip_config=config)
    
    output_frames = []
    processed_count = 0
    skipped_count = 0
    
    for i, frame in enumerate(frames):
        # Extract frame data
        timestamp = frame.get('timestamp', i / 30.0)
        keypoints3D = frame.get('keypoints3D', [])
        extrinsics_array = frame.get('camera_extrinsics', None)
        confidence = frame.get('handTrackingQuality', 0.5)
        
        # Extract 3D keypoints
        wrist = extract_keypoint_3d(keypoints3D, wrist_key)
        elbow = extract_keypoint_3d(keypoints3D, elbow_key)
        left_wrist = extract_keypoint_3d(keypoints3D, left_wrist_key)
        
        # Skip if missing required data
        if wrist is None or elbow is None:
            output_frames.append({
                'frame_index': i,
                'timestamp': timestamp,
                'error': f'Missing keypoints3D: wrist={wrist is not None}, elbow={elbow is not None}'
            })
            skipped_count += 1
            continue
        
        # Parse extrinsics
        extrinsics = None
        if extrinsics_array and len(extrinsics_array) == 16:
            try:
                extrinsics = CoordinateTransform.parse_extrinsics(extrinsics_array)
            except ValueError as e:
                print(f"Warning: Frame {i} - {e}")
        
        # Compute locked pose
        pose = lock.compute(
            wrist=wrist,
            elbow=elbow,
            timestamp=timestamp,
            extrinsics=extrinsics,
            frame_index=i
        )
        
        # Build output frame
        output_frame = pose.to_dict()
        output_frame['confidence'] = confidence
        output_frame['has_extrinsics'] = extrinsics is not None
        
        output_frames.append(output_frame)
        processed_count += 1
    
    # Build output
    output = {
        'metadata': {
            'source_file': input_path,
            'wrist_to_face_distance': wrist_to_face,
            'coordinate_system': {
                'local': 'Pelvis-centered, Y-up, units in meters',
                'world': 'Camera extrinsics applied, units in meters'
            },
            'quaternion_format': '[w, x, y, z]',
            'total_frames': len(output_frames),
            'processed_frames': processed_count,
            'skipped_frames': skipped_count
        },
        'frames': output_frames
    }
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"=" * 60)
    print(f"Racket 6-DoF Lock Processing Complete")
    print(f"=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Frames: {processed_count} processed, {skipped_count} skipped")
    print(f"Coordinate output: World space (meters)")
    print(f"=" * 60)
    
    return output


# =============================================================================
# VALIDATION & TESTING
# =============================================================================

def validate_lock_distance(output_data: Dict) -> Tuple[float, float, float]:
    """
    Validate that racket stays locked to wrist.
    
    Returns:
        Tuple of (min_distance, max_distance, mean_distance)
    """
    expected_distance = output_data['metadata']['wrist_to_face_distance']
    distances = []
    
    for frame in output_data['frames']:
        if 'error' in frame:
            continue
        
        pos = np.array(frame['position_local'])
        wrist = np.array(frame['wrist_position_local'])
        dist = np.linalg.norm(pos - wrist)
        distances.append(dist)
    
    if not distances:
        return 0, 0, 0
    
    return min(distances), max(distances), np.mean(distances)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        distance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.55
        
        result = process_video_json(input_file, output_file, wrist_to_face=distance)
        
        # Validate
        min_d, max_d, mean_d = validate_lock_distance(result)
        print(f"\nLock Validation:")
        print(f"  Expected distance: {distance:.4f}m")
        print(f"  Actual range: {min_d:.4f}m - {max_d:.4f}m")
        print(f"  Mean distance: {mean_d:.4f}m")
        print(f"  Max deviation: {abs(max_d - distance):.6f}m")
    
    else:
        print("Racket 6-DoF Lock with World-Space Transform")
        print("=" * 50)
        print()
        print("Usage:")
        print("  python racket_6dof_lock.py <input.json> <output.json> [wrist_to_face]")
        print()
        print("Example:")
        print("  python racket_6dof_lock.py pose_data.json racket_output.json 0.55")
        print()
        print("Features:")
        print("  - Uses keypoints3D (pelvis-centered, meters)")
        print("  - Applies camera extrinsics for world-space output")
        print("  - Mathematically guaranteed racket lock")
        print("  - Minimum jerk smoothing for natural motion")
