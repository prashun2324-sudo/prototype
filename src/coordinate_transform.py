"""
Coordinate Transformation Module
================================
Handles Apple Vision (Right-Handed) to Unity (Left-Handed) coordinate conversion.
Includes rolling shutter compensation and camera extrinsics handling.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TransformConfig:
    """Configuration for coordinate transformation."""
    # Rolling shutter parameters
    readout_time_30fps: float = 0.020  # ~20ms readout for iPhone at 30fps
    readout_time_120fps: float = 0.006  # ~6ms at 120fps
    
    # Frame dimensions (will be updated from data)
    frame_height: int = 1280
    frame_width: int = 720


class CoordinateTransformer:
    """
    Transforms coordinates between Apple Vision Pro (right-handed) 
    and Unity (left-handed) coordinate systems.
    """
    
    def __init__(self, config: Optional[TransformConfig] = None):
        self.config = config or TransformConfig()
        
    def apple_to_unity_position(self, pos: np.ndarray) -> np.ndarray:
        """
        Convert position from Apple Vision to Unity coordinate system.
        
        Apple Vision (Right-Handed):
            X: Right, Y: Up, Z: Backward (toward viewer)
        
        Unity (Left-Handed):
            X: Right, Y: Up, Z: Forward (into screen)
        
        Transform: Negate Z-axis
        """
        result = pos.copy()
        if len(result.shape) == 1:
            result[2] = -result[2]
        else:
            result[:, 2] = -result[:, 2]
        return result
    
    def apple_to_unity_quaternion(self, quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion from Apple Vision to Unity.
        
        For handedness flip, negate X and Y components of quaternion.
        Quaternion format: [w, x, y, z]
        """
        w, x, y, z = quat
        return np.array([w, -x, -y, z])
    
    def apple_to_unity_rotation_matrix(self, R: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 rotation matrix from Apple to Unity.
        
        Flip: Negate third column and third row.
        """
        R_unity = R.copy()
        R_unity[:, 2] = -R_unity[:, 2]
        R_unity[2, :] = -R_unity[2, :]
        return R_unity
    
    def parse_camera_extrinsics(self, extrinsics_flat: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse the 16-element camera extrinsics array into rotation and translation.
        
        Apple provides column-major 4x4 matrix:
        [R00 R10 R20 0]
        [R01 R11 R21 0]
        [R02 R12 R22 0]
        [Tx  Ty  Tz  1]
        """
        # Reshape to 4x4 column-major
        M = np.array(extrinsics_flat).reshape(4, 4).T  # Transpose for row-major
        
        # Extract rotation (3x3) and translation (3x1)
        R = M[:3, :3]
        T = M[:3, 3]
        
        return R, T
    
    def transform_extrinsics_to_unity(self, extrinsics_flat: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full pipeline: Parse extrinsics and convert to Unity coordinate system.
        """
        R_apple, T_apple = self.parse_camera_extrinsics(extrinsics_flat)
        
        R_unity = self.apple_to_unity_rotation_matrix(R_apple)
        T_unity = self.apple_to_unity_position(T_apple)
        
        return R_unity, T_unity


class RollingShutterCompensator:
    """
    Compensates for rolling shutter distortion in iPhone video.
    
    Rolling shutter captures image line-by-line from top to bottom.
    Fast-moving objects appear distorted based on their vertical position.
    """
    
    def __init__(self, frame_height: int, fps: float):
        self.frame_height = frame_height
        self.fps = fps
        
        # Estimate readout time based on fps
        if fps >= 100:
            self.readout_time = 0.006  # ~6ms for 120fps
        else:
            self.readout_time = 0.020  # ~20ms for 30fps
    
    def get_scanline_time_offset(self, y_pixel: float) -> float:
        """
        Calculate time offset for a specific scanline.
        
        Returns offset in seconds from frame timestamp.
        """
        normalized_y = y_pixel / self.frame_height
        return normalized_y * self.readout_time
    
    def compensate_position(self, 
                           position_2d: np.ndarray, 
                           velocity_2d: np.ndarray) -> np.ndarray:
        """
        Compensate 2D position for rolling shutter effect.
        
        Args:
            position_2d: [x, y] detected position in pixels
            velocity_2d: [vx, vy] estimated velocity in pixels/second
        
        Returns:
            Corrected [x, y] position
        """
        time_offset = self.get_scanline_time_offset(position_2d[1])
        
        # Back-project to actual position at frame timestamp
        corrected = position_2d - velocity_2d * time_offset
        
        return corrected
    
    def compensate_keypoints(self,
                            keypoints: dict,
                            velocities: dict) -> dict:
        """
        Compensate all keypoints for rolling shutter.
        
        Args:
            keypoints: Dict of joint_name -> [x, y] position
            velocities: Dict of joint_name -> [vx, vy] velocity
        
        Returns:
            Corrected keypoints dict
        """
        corrected = {}
        
        for name, pos in keypoints.items():
            if name in velocities:
                vel = velocities[name]
                corrected[name] = self.compensate_position(
                    np.array(pos), np.array(vel)
                ).tolist()
            else:
                corrected[name] = pos
        
        return corrected
    
    def estimate_velocity_from_frames(self,
                                      prev_keypoints: dict,
                                      curr_keypoints: dict,
                                      dt: float) -> dict:
        """
        Estimate keypoint velocities from consecutive frames.
        """
        velocities = {}
        
        for name in curr_keypoints:
            if name in prev_keypoints:
                prev_pos = np.array(prev_keypoints[name])
                curr_pos = np.array(curr_keypoints[name])
                
                if prev_pos[0] != -1 and curr_pos[0] != -1:
                    velocities[name] = ((curr_pos - prev_pos) / dt).tolist()
        
        return velocities


class GripSlipDetector:
    """
    Detects grip slip events during racket strokes.
    
    A grip slip causes instantaneous change in wrist-to-racket offset
    that violates the minimum jerk assumption.
    """
    
    def __init__(self, 
                 angle_threshold: float = 8.0,  # degrees
                 rate_threshold: float = 500.0,  # degrees/second
                 smoothing_alpha: float = 0.3):
        self.angle_threshold = angle_threshold
        self.rate_threshold = rate_threshold
        self.smoothing_alpha = smoothing_alpha
        
        # State
        self.offset_estimate = np.zeros(3)  # Roll, pitch, yaw offset
        self.prev_residual = np.zeros(3)
        self.slip_detected = False
        self.frames_since_slip = 0
    
    def compute_residual(self,
                        predicted_orientation: np.ndarray,
                        measured_orientation: np.ndarray) -> np.ndarray:
        """
        Compute orientation residual in degrees.
        
        Args:
            predicted_orientation: Kinematic prediction [roll, pitch, yaw]
            measured_orientation: Observed orientation [roll, pitch, yaw]
        
        Returns:
            Residual in degrees
        """
        residual = measured_orientation - predicted_orientation - self.offset_estimate
        
        # Normalize to [-180, 180]
        residual = np.mod(residual + 180, 360) - 180
        
        return residual
    
    def detect_slip(self,
                   residual: np.ndarray,
                   dt: float) -> bool:
        """
        Detect if a grip slip has occurred.
        
        Uses both magnitude and rate of change of residual.
        """
        # Residual magnitude
        magnitude = np.linalg.norm(residual)
        
        # Rate of change
        residual_rate = np.linalg.norm(residual - self.prev_residual) / dt
        
        # Slip detection
        is_slip = (magnitude > self.angle_threshold and 
                   residual_rate > self.rate_threshold)
        
        self.prev_residual = residual.copy()
        
        return is_slip
    
    def update(self,
              predicted_orientation: np.ndarray,
              measured_orientation: np.ndarray,
              dt: float,
              motion_phase: str = 'unknown') -> dict:
        """
        Update slip detection state and return correction.
        
        Args:
            predicted_orientation: Kinematic prediction
            measured_orientation: Observed orientation
            dt: Time since last update
            motion_phase: Current phase of motion
        
        Returns:
            Dict with slip status and corrected offset
        """
        residual = self.compute_residual(predicted_orientation, measured_orientation)
        
        # Adjust threshold based on phase
        phase_multiplier = {
            'impact': 2.0,      # Higher tolerance during impact
            'acceleration': 1.5,
            'backscratch': 1.2,
            'preparation': 1.0,
            'trophy': 1.0,
            'follow_through': 1.3,
        }.get(motion_phase.lower(), 1.0)
        
        effective_threshold = self.angle_threshold * phase_multiplier
        
        # Check for slip
        is_slip = self.detect_slip(residual, dt)
        
        if is_slip:
            self.slip_detected = True
            self.frames_since_slip = 0
            
            # Immediate offset correction
            self.offset_estimate = self.offset_estimate + residual
        else:
            self.frames_since_slip += 1
            
            # Gradual offset update (exponential smoothing)
            if np.linalg.norm(residual) < effective_threshold:
                self.offset_estimate = (self.smoothing_alpha * residual + 
                                       (1 - self.smoothing_alpha) * self.offset_estimate)
            
            # Clear slip flag after stabilization
            if self.frames_since_slip > 10:
                self.slip_detected = False
        
        return {
            'slip_detected': self.slip_detected,
            'residual': residual.tolist(),
            'offset_estimate': self.offset_estimate.tolist(),
            'corrected_orientation': (measured_orientation - self.offset_estimate).tolist()
        }
    
    def reset(self):
        """Reset slip detector state (e.g., for new stroke)."""
        self.offset_estimate = np.zeros(3)
        self.prev_residual = np.zeros(3)
        self.slip_detected = False
        self.frames_since_slip = 0


class HybridFusion:
    """
    Fuses Apple Vision ML predictions with kinematic physics model.
    
    This is the core of our patent-worthy hybrid approach.
    """
    
    def __init__(self,
                 base_kinematic_weight: float = 0.3,
                 high_speed_kinematic_weight: float = 0.7,
                 occlusion_kinematic_weight: float = 1.0,
                 speed_threshold: float = 15.0):  # m/s
        self.base_weight = base_kinematic_weight
        self.high_speed_weight = high_speed_kinematic_weight
        self.occlusion_weight = occlusion_kinematic_weight
        self.speed_threshold = speed_threshold
    
    def compute_adaptive_weight(self,
                               motion_speed: float,
                               ml_confidence: float,
                               is_occluded: bool) -> float:
        """
        Compute adaptive weight for kinematic model.
        
        Higher weight = more trust in kinematic prediction
        """
        if is_occluded:
            return self.occlusion_weight
        
        # Speed factor: Higher speed = trust kinematics more
        speed_factor = min(1.0, motion_speed / self.speed_threshold)
        
        # Confidence factor: Lower confidence = trust kinematics more
        confidence_factor = 1.0 - ml_confidence
        
        # Combine factors
        weight = (self.base_weight + 
                 (self.high_speed_weight - self.base_weight) * 
                 (0.5 * speed_factor + 0.5 * confidence_factor))
        
        return np.clip(weight, 0.0, 1.0)
    
    def fuse_position(self,
                     ml_position: np.ndarray,
                     kinematic_position: np.ndarray,
                     weight: float) -> np.ndarray:
        """
        Fuse ML and kinematic position estimates.
        """
        return weight * kinematic_position + (1 - weight) * ml_position
    
    def fuse_orientation(self,
                        ml_quaternion: np.ndarray,
                        kinematic_quaternion: np.ndarray,
                        weight: float) -> np.ndarray:
        """
        Fuse ML and kinematic orientation using SLERP.
        
        Spherical linear interpolation for smooth quaternion blending.
        """
        # Ensure quaternions are unit length
        ml_q = ml_quaternion / np.linalg.norm(ml_quaternion)
        kin_q = kinematic_quaternion / np.linalg.norm(kinematic_quaternion)
        
        # Compute dot product
        dot = np.dot(ml_q, kin_q)
        
        # If negative dot, negate one quaternion (shorter path)
        if dot < 0:
            kin_q = -kin_q
            dot = -dot
        
        # Clamp dot to valid range
        dot = np.clip(dot, -1.0, 1.0)
        
        # SLERP
        theta = np.arccos(dot)
        
        if theta < 0.001:
            # Very close, use linear interpolation
            return (1 - weight) * ml_q + weight * kin_q
        
        sin_theta = np.sin(theta)
        factor_ml = np.sin((1 - weight) * theta) / sin_theta
        factor_kin = np.sin(weight * theta) / sin_theta
        
        result = factor_ml * ml_q + factor_kin * kin_q
        return result / np.linalg.norm(result)
    
    def fuse(self,
            ml_pose: dict,
            kinematic_pose: dict,
            motion_speed: float,
            ml_confidence: float,
            is_occluded: bool) -> dict:
        """
        Full fusion of ML and kinematic pose estimates.
        
        Args:
            ml_pose: {'position': [x,y,z], 'orientation': [w,x,y,z]}
            kinematic_pose: Same format
            motion_speed: Current motion speed
            ml_confidence: ML detection confidence
            is_occluded: Whether joint is occluded
        
        Returns:
            Fused pose dict
        """
        weight = self.compute_adaptive_weight(motion_speed, ml_confidence, is_occluded)
        
        fused_position = self.fuse_position(
            np.array(ml_pose['position']),
            np.array(kinematic_pose['position']),
            weight
        )
        
        fused_orientation = self.fuse_orientation(
            np.array(ml_pose['orientation']),
            np.array(kinematic_pose['orientation']),
            weight
        )
        
        return {
            'position': fused_position.tolist(),
            'orientation': fused_orientation.tolist(),
            'kinematic_weight': weight,
            'is_occluded': is_occluded
        }

