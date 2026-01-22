"""
Universal 6DoF Motion Tracking System
=====================================

A mathematically rigorous framework for tracking human body and held objects
(rackets, bats, clubs) with full 6 Degrees of Freedom.

Mathematical Foundations:
- Forward Kinematics using Denavit-Hartenberg convention
- Quaternion-based orientation representation (avoids gimbal lock)
- Extended Kalman Filter for state estimation
- Rigid body dynamics for held object tracking

References:
- Craig, J.J. (2005). Introduction to Robotics: Mechanics and Control
- Diebel, J. (2006). Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors
- Zatsiorsky, V.M. (2002). Kinetics of Human Motion
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from scipy.spatial.transform import Rotation as SciRotation
from scipy.interpolate import CubicSpline
import copy


# =============================================================================
# MATHEMATICAL PRIMITIVES
# =============================================================================

class Quaternion:
    """
    Unit quaternion for 3D rotation representation.
    
    Quaternion q = w + xi + yj + zk where ||q|| = 1
    
    Advantages over Euler angles:
    - No gimbal lock
    - Smooth interpolation (SLERP)
    - Efficient composition
    """
    
    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Create quaternion from axis-angle representation.
        
        q = [cos(θ/2), sin(θ/2)*axis]
        """
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = np.sin(half_angle) * axis
        return np.array([w, xyz[0], xyz[1], xyz[2]])
    
    @staticmethod
    def from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Create quaternion from Euler angles (ZYX convention).
        
        roll: rotation around X-axis
        pitch: rotation around Y-axis  
        yaw: rotation around Z-axis
        """
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    @staticmethod
    def to_euler(q: np.ndarray) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles (ZYX)."""
        w, x, y, z = q
        
        # Roll (X)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (Y) - handle gimbal lock
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (Z)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Quaternion multiplication (Hamilton product).
        
        q1 * q2 = [w1*w2 - v1·v2, w1*v2 + w2*v1 + v1×v2]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def conjugate(q: np.ndarray) -> np.ndarray:
        """Quaternion conjugate: q* = [w, -x, -y, -z]"""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    @staticmethod
    def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate vector v by quaternion q.
        
        v' = q * [0, v] * q*
        """
        v_quat = np.array([0, v[0], v[1], v[2]])
        q_conj = Quaternion.conjugate(q)
        result = Quaternion.multiply(Quaternion.multiply(q, v_quat), q_conj)
        return result[1:4]
    
    @staticmethod
    def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to 3x3 rotation matrix.
        
        R = [1-2(y²+z²)  2(xy-wz)   2(xz+wy)]
            [2(xy+wz)    1-2(x²+z²) 2(yz-wx)]
            [2(xz-wy)    2(yz+wx)   1-2(x²+y²)]
        """
        w, x, y, z = q
        
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
    
    @staticmethod
    def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical Linear Interpolation between quaternions.
        
        SLERP(q1, q2, t) = sin((1-t)θ)/sin(θ) * q1 + sin(tθ)/sin(θ) * q2
        """
        # Normalize
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Compute dot product
        dot = np.dot(q1, q2)
        
        # If negative, negate one quaternion (shorter path)
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        # Clamp to valid range
        dot = np.clip(dot, -1, 1)
        
        # If very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        
        s1 = np.sin((1 - t) * theta) / sin_theta
        s2 = np.sin(t * theta) / sin_theta
        
        return s1 * q1 + s2 * q2


class RotationMatrix:
    """3x3 rotation matrix operations."""
    
    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Rodrigues' rotation formula.
        
        R = I + sin(θ)K + (1-cos(θ))K²
        
        where K is the skew-symmetric matrix of axis.
        """
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        return (np.eye(3) + 
                np.sin(angle) * K + 
                (1 - np.cos(angle)) * (K @ K))
    
    @staticmethod
    def rotation_x(angle: float) -> np.ndarray:
        """Rotation matrix around X-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    @staticmethod
    def rotation_y(angle: float) -> np.ndarray:
        """Rotation matrix around Y-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    @staticmethod
    def rotation_z(angle: float) -> np.ndarray:
        """Rotation matrix around Z-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])


# =============================================================================
# 6DOF STATE REPRESENTATION
# =============================================================================

@dataclass
class Pose6DoF:
    """
    Complete 6 Degrees of Freedom pose representation.
    
    Position: [x, y, z] in meters
    Orientation: [w, x, y, z] unit quaternion
    
    Optional derivatives for dynamics:
    - velocity: [vx, vy, vz] m/s
    - angular_velocity: [wx, wy, wz] rad/s
    """
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [w, x, y, z] quaternion
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    timestamp: float = 0.0
    confidence: float = 1.0
    
    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float64)
        self.orientation = np.array(self.orientation, dtype=np.float64)
        self.velocity = np.array(self.velocity, dtype=np.float64)
        self.angular_velocity = np.array(self.angular_velocity, dtype=np.float64)
        
        # Normalize quaternion
        self.orientation = self.orientation / np.linalg.norm(self.orientation)
    
    def get_rotation_matrix(self) -> np.ndarray:
        """Get 3x3 rotation matrix from orientation quaternion."""
        return Quaternion.to_rotation_matrix(self.orientation)
    
    def get_transform_matrix(self) -> np.ndarray:
        """
        Get 4x4 homogeneous transformation matrix.
        
        T = [R  t]
            [0  1]
        """
        T = np.eye(4)
        T[:3, :3] = self.get_rotation_matrix()
        T[:3, 3] = self.position
        return T
    
    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Transform a point from local to global coordinates."""
        return self.get_rotation_matrix() @ point + self.position
    
    def get_forward_vector(self) -> np.ndarray:
        """Get the forward direction vector (local Z-axis in world frame)."""
        return Quaternion.rotate_vector(self.orientation, np.array([0, 0, 1]))
    
    def get_up_vector(self) -> np.ndarray:
        """Get the up direction vector (local Y-axis in world frame)."""
        return Quaternion.rotate_vector(self.orientation, np.array([0, 1, 0]))
    
    def get_right_vector(self) -> np.ndarray:
        """Get the right direction vector (local X-axis in world frame)."""
        return Quaternion.rotate_vector(self.orientation, np.array([1, 0, 0]))


# =============================================================================
# KINEMATIC CHAIN (DENAVIT-HARTENBERG)
# =============================================================================

@dataclass
class DHParameters:
    """
    Denavit-Hartenberg parameters for a joint.
    
    θ (theta): Joint angle (rotation around z-axis)
    d: Link offset (translation along z-axis)
    a: Link length (translation along x-axis)
    α (alpha): Link twist (rotation around x-axis)
    
    Convention: Modified DH (Craig's convention)
    """
    theta: float = 0.0  # Joint angle (variable for revolute joints)
    d: float = 0.0      # Link offset
    a: float = 0.0      # Link length
    alpha: float = 0.0  # Link twist
    is_revolute: bool = True  # True for rotation, False for prismatic
    
    def get_transform(self, joint_value: float = None) -> np.ndarray:
        """
        Compute 4x4 transformation matrix for this joint.
        
        T = Rz(θ) * Tz(d) * Tx(a) * Rx(α)
        
        For revolute joints, joint_value is added to theta.
        For prismatic joints, joint_value is added to d.
        """
        theta = self.theta + (joint_value if self.is_revolute else 0)
        d = self.d + (0 if self.is_revolute else joint_value)
        
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)
        
        return np.array([
            [ct, -st*ca, st*sa, self.a*ct],
            [st, ct*ca, -ct*sa, self.a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])


class HumanArmKinematicChain:
    """
    7-DOF human arm kinematic chain from shoulder to wrist.
    
    Joint definitions (following ISB recommendations):
    - J1: Shoulder flexion/extension (sagittal plane)
    - J2: Shoulder abduction/adduction (frontal plane)
    - J3: Shoulder internal/external rotation (transverse plane)
    - J4: Elbow flexion/extension
    - J5: Forearm pronation/supination
    - J6: Wrist flexion/extension
    - J7: Wrist radial/ulnar deviation
    
    Segment lengths are scaled by body height.
    """
    
    # Standard segment proportions relative to body height
    UPPER_ARM_RATIO = 0.186  # Upper arm / height
    FOREARM_RATIO = 0.146    # Forearm / height
    HAND_RATIO = 0.108       # Hand / height
    
    def __init__(self, body_height: float = 1.75, side: str = 'right'):
        """
        Initialize kinematic chain.
        
        Args:
            body_height: Height in meters for scaling
            side: 'left' or 'right' arm
        """
        self.body_height = body_height
        self.side = side
        self.mirror = -1 if side == 'left' else 1
        
        # Compute segment lengths
        self.upper_arm_length = body_height * self.UPPER_ARM_RATIO
        self.forearm_length = body_height * self.FOREARM_RATIO
        self.hand_length = body_height * self.HAND_RATIO
        
        # Joint limits (radians)
        self.joint_limits = {
            'shoulder_flexion': (-np.pi/3, np.pi),        # -60° to 180°
            'shoulder_abduction': (-np.pi/4, np.pi),      # -45° to 180°
            'shoulder_rotation': (-np.pi/2, np.pi/2),     # -90° to 90°
            'elbow_flexion': (0, 2.5),                    # 0° to 143°
            'forearm_pronation': (-np.pi/2, np.pi/2),     # -90° to 90°
            'wrist_flexion': (-np.pi/3, np.pi/3),         # -60° to 60°
            'wrist_deviation': (-np.pi/6, np.pi/4),       # -30° to 45°
        }
        
        # Define DH parameters
        self._setup_dh_chain()
    
    def _setup_dh_chain(self):
        """Setup Denavit-Hartenberg parameters for the arm chain."""
        m = self.mirror
        
        self.dh_params = [
            # Shoulder complex (3 DOF)
            DHParameters(theta=0, d=0, a=0, alpha=-np.pi/2),           # Flexion
            DHParameters(theta=-np.pi/2, d=0, a=0, alpha=np.pi/2),    # Abduction
            DHParameters(theta=0, d=self.upper_arm_length, a=0, alpha=-np.pi/2 * m),  # Rotation
            
            # Elbow (1 DOF)
            DHParameters(theta=0, d=0, a=0, alpha=np.pi/2 * m),        # Flexion
            
            # Forearm (1 DOF)
            DHParameters(theta=0, d=self.forearm_length, a=0, alpha=-np.pi/2 * m),  # Pronation
            
            # Wrist (2 DOF)
            DHParameters(theta=0, d=0, a=0, alpha=np.pi/2 * m),        # Flexion
            DHParameters(theta=0, d=self.hand_length * 0.5, a=0, alpha=0),  # Deviation
        ]
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> Pose6DoF:
        """
        Compute end-effector (wrist) pose from joint angles.
        
        Args:
            joint_angles: Array of 7 joint angles in radians
            
        Returns:
            Pose6DoF of the wrist in shoulder frame
        """
        assert len(joint_angles) == 7, "Need 7 joint angles"
        
        # Start with identity transform
        T = np.eye(4)
        
        # Chain transforms
        for i, (dh, angle) in enumerate(zip(self.dh_params, joint_angles)):
            T = T @ dh.get_transform(angle)
        
        # Extract position and orientation
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        
        # Convert rotation matrix to quaternion
        r = SciRotation.from_matrix(rotation_matrix)
        orientation = r.as_quat()  # [x, y, z, w]
        orientation = np.array([orientation[3], orientation[0], orientation[1], orientation[2]])  # [w, x, y, z]
        
        return Pose6DoF(position=position, orientation=orientation)
    
    def get_all_joint_positions(self, joint_angles: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get positions of all joints in the chain.
        
        Returns dict with: shoulder, elbow, wrist positions
        """
        positions = {'shoulder': np.zeros(3)}
        
        T = np.eye(4)
        
        # After shoulder complex (3 joints)
        for i in range(3):
            T = T @ self.dh_params[i].get_transform(joint_angles[i])
        
        # Elbow position
        T_elbow = T @ self.dh_params[3].get_transform(joint_angles[3])
        positions['elbow'] = T_elbow[:3, 3].copy()
        
        # Continue to wrist
        T = T_elbow
        for i in range(4, 7):
            T = T @ self.dh_params[i].get_transform(joint_angles[i])
        
        positions['wrist'] = T[:3, 3].copy()
        
        return positions


# =============================================================================
# RACKET AS RIGID BODY EXTENSION
# =============================================================================

@dataclass
class RacketGeometry:
    """
    Tennis racket geometric parameters.
    
    All measurements in meters.
    """
    handle_length: float = 0.19      # Handle length
    shaft_length: float = 0.12       # Shaft (throat) length
    head_length: float = 0.35        # Head length (tip to tip)
    head_width: float = 0.26         # Head width
    total_length: float = 0.69       # Total racket length
    
    # Grip offset from wrist center
    grip_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, -0.05, 0.02]))
    
    # Local frame orientation offset (depends on grip style)
    grip_rotation: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))  # Identity


class RacketTracker:
    """
    Track racket as rigid body attached to wrist.
    
    The racket is treated as a kinematic extension of the arm chain.
    Its pose is computed from wrist pose plus grip offset and rotation.
    """
    
    def __init__(self, geometry: RacketGeometry = None, grip_style: str = 'continental'):
        self.geometry = geometry or RacketGeometry()
        self.grip_style = grip_style
        
        # Grip-specific rotation offsets
        self.grip_rotations = {
            'continental': Quaternion.from_euler(0, 0, 0),
            'eastern_forehand': Quaternion.from_euler(np.pi/8, 0, 0),
            'semi_western': Quaternion.from_euler(np.pi/4, 0, 0),
            'western': Quaternion.from_euler(np.pi/3, 0, 0),
            'eastern_backhand': Quaternion.from_euler(-np.pi/8, 0, 0),
        }
        
        self.grip_rotation = self.grip_rotations.get(grip_style, 
                                                      self.grip_rotations['continental'])
    
    def compute_racket_pose(self, wrist_pose: Pose6DoF) -> Pose6DoF:
        """
        Compute racket pose from wrist pose.
        
        Racket frame:
        - Origin: Center of strings
        - Z-axis: Normal to string bed (face direction)
        - Y-axis: Along racket length (toward tip)
        - X-axis: Across racket width
        """
        # Apply grip rotation to wrist orientation
        racket_orientation = Quaternion.multiply(wrist_pose.orientation, self.grip_rotation)
        
        # Transform grip offset to world frame
        grip_offset_world = Quaternion.rotate_vector(wrist_pose.orientation, 
                                                     self.geometry.grip_offset)
        
        # Compute racket head center position
        # Head center is handle_length + shaft_length + head_length/2 from wrist
        head_offset_local = np.array([0, self.geometry.handle_length + 
                                        self.geometry.shaft_length + 
                                        self.geometry.head_length / 2, 0])
        head_offset_world = Quaternion.rotate_vector(racket_orientation, head_offset_local)
        
        racket_position = wrist_pose.position + grip_offset_world + head_offset_world
        
        # Compute racket velocity from wrist velocity and angular velocity
        # v_racket = v_wrist + ω × r
        r = racket_position - wrist_pose.position
        racket_velocity = wrist_pose.velocity + np.cross(wrist_pose.angular_velocity, r)
        
        return Pose6DoF(
            position=racket_position,
            orientation=racket_orientation,
            velocity=racket_velocity,
            angular_velocity=wrist_pose.angular_velocity,
            timestamp=wrist_pose.timestamp,
            confidence=wrist_pose.confidence
        )
    
    def get_face_normal(self, racket_pose: Pose6DoF) -> np.ndarray:
        """Get racket face normal vector (perpendicular to string bed)."""
        return racket_pose.get_forward_vector()
    
    def get_string_bed_corners(self, racket_pose: Pose6DoF) -> List[np.ndarray]:
        """Get 4 corner points of the string bed in world coordinates."""
        hw = self.geometry.head_width / 2
        hl = self.geometry.head_length / 2
        
        corners_local = [
            np.array([-hw, -hl, 0]),
            np.array([hw, -hl, 0]),
            np.array([hw, hl, 0]),
            np.array([-hw, hl, 0]),
        ]
        
        return [racket_pose.transform_point(c) for c in corners_local]


# =============================================================================
# EXTENDED KALMAN FILTER FOR 6DOF STATE ESTIMATION
# =============================================================================

class EKF6DoF:
    """
    Extended Kalman Filter for 6DoF pose estimation.
    
    State vector [13 elements]:
    - Position: [x, y, z]
    - Velocity: [vx, vy, vz]
    - Orientation: [qw, qx, qy, qz]
    - Angular velocity: [wx, wy, wz]
    
    This provides:
    - Sensor fusion (combine multiple measurements)
    - Noise filtering
    - Prediction during occlusion
    - Smooth trajectory estimation
    """
    
    def __init__(self, 
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.05,
                 dt: float = 1/30):
        """
        Initialize EKF.
        
        Args:
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            dt: Time step (default 30fps)
        """
        self.dt = dt
        
        # State dimension
        self.n = 13
        
        # Initialize state
        self.x = np.zeros(self.n)
        self.x[6] = 1.0  # qw = 1 (identity quaternion)
        
        # State covariance
        self.P = np.eye(self.n) * 0.1
        
        # Process noise covariance
        self.Q = np.eye(self.n) * process_noise
        self.Q[0:3, 0:3] *= 0.001  # Position noise
        self.Q[3:6, 3:6] *= 0.01   # Velocity noise
        self.Q[6:10, 6:10] *= 0.001  # Orientation noise
        self.Q[10:13, 10:13] *= 0.01  # Angular velocity noise
        
        # Measurement noise covariance (position + orientation)
        self.R = np.eye(7) * measurement_noise
        
        self.initialized = False
    
    def _normalize_quaternion(self):
        """Normalize quaternion part of state."""
        q = self.x[6:10]
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            self.x[6:10] = q / norm
    
    def _state_transition(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        State transition function f(x, dt).
        
        Position update: p' = p + v*dt
        Velocity: constant velocity model
        Orientation: q' = q ⊗ exp(ω*dt/2)
        Angular velocity: constant angular velocity model
        """
        x_new = x.copy()
        
        # Position update
        x_new[0:3] = x[0:3] + x[3:6] * dt
        
        # Orientation update using angular velocity
        omega = x[10:13]
        omega_norm = np.linalg.norm(omega)
        
        if omega_norm > 1e-10:
            # Quaternion exponential: exp(ω*dt/2)
            angle = omega_norm * dt
            axis = omega / omega_norm
            dq = Quaternion.from_axis_angle(axis, angle)
            
            # Update orientation: q' = q ⊗ dq
            x_new[6:10] = Quaternion.multiply(x[6:10], dq)
        
        return x_new
    
    def _compute_jacobian_F(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute Jacobian of state transition function.
        
        F = ∂f/∂x
        """
        F = np.eye(self.n)
        
        # Position depends on velocity
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Orientation Jacobian (simplified)
        # Full derivation involves quaternion calculus
        omega = x[10:13]
        omega_norm = np.linalg.norm(omega)
        
        if omega_norm > 1e-10:
            # Approximate: orientation depends on angular velocity
            F[6:10, 10:13] = np.zeros((4, 3))
            # First-order approximation
            q = x[6:10]
            F[6, 10:13] = -0.5 * dt * q[1:4]
            F[7:10, 10:13] = 0.5 * dt * (q[0] * np.eye(3) + self._skew(q[1:4]))
        
        return F
    
    def _skew(self, v: np.ndarray) -> np.ndarray:
        """Skew-symmetric matrix for cross product."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def _measurement_function(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement function h(x).
        
        We measure position and orientation directly.
        z = [x, y, z, qw, qx, qy, qz]
        """
        return np.concatenate([x[0:3], x[6:10]])
    
    def _compute_jacobian_H(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of measurement function.
        
        H = ∂h/∂x
        """
        H = np.zeros((7, self.n))
        H[0:3, 0:3] = np.eye(3)  # Position
        H[3:7, 6:10] = np.eye(4)  # Orientation
        return H
    
    def predict(self, dt: float = None):
        """
        EKF prediction step.
        
        x' = f(x, dt)
        P' = F * P * F^T + Q
        """
        if dt is None:
            dt = self.dt
        
        # State prediction
        self.x = self._state_transition(self.x, dt)
        self._normalize_quaternion()
        
        # Covariance prediction
        F = self._compute_jacobian_F(self.x, dt)
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, measurement: np.ndarray, R: np.ndarray = None):
        """
        EKF update step.
        
        Args:
            measurement: [x, y, z, qw, qx, qy, qz]
            R: Optional custom measurement noise covariance
        """
        if R is None:
            R = self.R
        
        # Innovation
        z_pred = self._measurement_function(self.x)
        y = measurement - z_pred
        
        # Handle quaternion sign ambiguity
        if np.dot(measurement[3:7], z_pred[3:7]) < 0:
            y[3:7] = -measurement[3:7] - z_pred[3:7]
        
        # Innovation covariance
        H = self._compute_jacobian_H(self.x)
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        self._normalize_quaternion()
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        self.initialized = True
    
    def initialize(self, pose: Pose6DoF):
        """Initialize filter with a pose."""
        self.x[0:3] = pose.position
        self.x[3:6] = pose.velocity
        self.x[6:10] = pose.orientation
        self.x[10:13] = pose.angular_velocity
        self.initialized = True
    
    def get_pose(self) -> Pose6DoF:
        """Get current estimated pose."""
        return Pose6DoF(
            position=self.x[0:3].copy(),
            orientation=self.x[6:10].copy(),
            velocity=self.x[3:6].copy(),
            angular_velocity=self.x[10:13].copy()
        )


# =============================================================================
# INVERSE KINEMATICS SOLVER
# =============================================================================

class InverseKinematicsSolver:
    """
    Solve for joint angles given end-effector pose.
    
    Uses iterative Jacobian-based method (Damped Least Squares).
    """
    
    def __init__(self, kinematic_chain: HumanArmKinematicChain):
        self.chain = kinematic_chain
        self.damping = 0.01
        self.max_iterations = 50
        self.tolerance = 1e-4
    
    def _compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute geometric Jacobian numerically.
        
        J = [∂p/∂θ]  (6x7 for position only, or 6x7 for full)
        """
        J = np.zeros((3, 7))
        epsilon = 1e-6
        
        # Current end-effector position
        pose_current = self.chain.forward_kinematics(joint_angles)
        p_current = pose_current.position
        
        for i in range(7):
            # Perturb joint i
            angles_plus = joint_angles.copy()
            angles_plus[i] += epsilon
            
            pose_plus = self.chain.forward_kinematics(angles_plus)
            p_plus = pose_plus.position
            
            J[:, i] = (p_plus - p_current) / epsilon
        
        return J
    
    def solve(self, target_position: np.ndarray, 
              initial_angles: np.ndarray = None) -> np.ndarray:
        """
        Solve IK for target position.
        
        Args:
            target_position: Desired end-effector position
            initial_angles: Starting joint configuration
            
        Returns:
            Joint angles that achieve the target (approximately)
        """
        if initial_angles is None:
            initial_angles = np.zeros(7)
        
        angles = initial_angles.copy()
        
        for _ in range(self.max_iterations):
            # Current position
            pose = self.chain.forward_kinematics(angles)
            error = target_position - pose.position
            
            if np.linalg.norm(error) < self.tolerance:
                break
            
            # Compute Jacobian
            J = self._compute_jacobian(angles)
            
            # Damped Least Squares
            # Δθ = J^T (J J^T + λ²I)^{-1} e
            JJT = J @ J.T
            damped = JJT + self.damping**2 * np.eye(3)
            delta = J.T @ np.linalg.solve(damped, error)
            
            # Update angles
            angles += delta
            
            # Apply joint limits
            for i, (name, (lo, hi)) in enumerate(self.chain.joint_limits.items()):
                angles[i] = np.clip(angles[i], lo, hi)
        
        return angles


# =============================================================================
# UNIVERSAL TRACKER
# =============================================================================

class Universal6DoFTracker:
    """
    Universal 6DoF tracking system for body and held objects.
    
    Combines:
    - Forward kinematics for body pose
    - Rigid body tracking for held objects
    - EKF for temporal filtering
    - IK for pose estimation from observations
    
    Works with any pose estimation input (MediaPipe, Apple Vision, OpenPose, etc.)
    """
    
    def __init__(self, 
                 body_height: float = 1.75,
                 grip_style: str = 'continental',
                 fps: float = 30.0):
        """
        Initialize universal tracker.
        
        Args:
            body_height: Height in meters for scaling kinematic chain
            grip_style: Tennis grip style for racket offset
            fps: Frame rate for temporal filtering
        """
        self.body_height = body_height
        
        # Kinematic chains for both arms
        self.arm_chains = {
            'right': HumanArmKinematicChain(body_height, 'right'),
            'left': HumanArmKinematicChain(body_height, 'left'),
        }
        
        # IK solvers
        self.ik_solvers = {
            'right': InverseKinematicsSolver(self.arm_chains['right']),
            'left': InverseKinematicsSolver(self.arm_chains['left']),
        }
        
        # Racket tracker
        self.racket_tracker = RacketTracker(grip_style=grip_style)
        
        # EKF for each tracked object
        self.ekf_wrist = {
            'right': EKF6DoF(dt=1/fps),
            'left': EKF6DoF(dt=1/fps),
        }
        self.ekf_racket = EKF6DoF(dt=1/fps)
        
        # State history for trajectory analysis
        self.history: List[Dict] = []
        self.max_history = 300
        
        # Occlusion handling
        self.occlusion_threshold = 0.3
        self.frames_since_observation = {'right': 0, 'left': 0, 'racket': 0}
    
    def process_keypoints(self, 
                         keypoints: Dict[str, np.ndarray],
                         confidences: Dict[str, float],
                         timestamp: float = 0.0,
                         side: str = 'right') -> Dict[str, Pose6DoF]:
        """
        Process detected keypoints to compute 6DoF poses.
        
        Args:
            keypoints: Dict mapping joint names to 3D positions
                Required: 'shoulder', 'elbow', 'wrist' (in shoulder frame)
            confidences: Dict mapping joint names to detection confidence
            timestamp: Current timestamp
            side: 'left' or 'right' arm
            
        Returns:
            Dict with 'wrist' and 'racket' Pose6DoF
        """
        results = {}
        
        # Check for occlusion
        wrist_conf = confidences.get('wrist', 0)
        is_occluded = wrist_conf < self.occlusion_threshold
        
        if is_occluded:
            self.frames_since_observation[side] += 1
            self.frames_since_observation['racket'] += 1
            
            # Use EKF prediction only
            self.ekf_wrist[side].predict()
            self.ekf_racket.predict()
            
            results['wrist'] = self.ekf_wrist[side].get_pose()
            results['racket'] = self.ekf_racket.get_pose()
            results['is_occluded'] = True
            
        else:
            self.frames_since_observation[side] = 0
            self.frames_since_observation['racket'] = 0
            
            # Compute wrist pose from keypoints
            wrist_pose = self._compute_wrist_pose(keypoints, confidences, side)
            
            # EKF update
            measurement = np.concatenate([
                wrist_pose.position,
                wrist_pose.orientation
            ])
            
            if not self.ekf_wrist[side].initialized:
                self.ekf_wrist[side].initialize(wrist_pose)
            else:
                self.ekf_wrist[side].predict()
                self.ekf_wrist[side].update(measurement)
            
            results['wrist'] = self.ekf_wrist[side].get_pose()
            
            # Compute racket pose
            racket_pose = self.racket_tracker.compute_racket_pose(results['wrist'])
            
            if not self.ekf_racket.initialized:
                self.ekf_racket.initialize(racket_pose)
            else:
                racket_measurement = np.concatenate([
                    racket_pose.position,
                    racket_pose.orientation
                ])
                self.ekf_racket.predict()
                self.ekf_racket.update(racket_measurement)
            
            results['racket'] = self.ekf_racket.get_pose()
            results['is_occluded'] = False
        
        results['timestamp'] = timestamp
        results['confidence'] = wrist_conf
        
        # Store in history
        self._add_to_history(results)
        
        return results
    
    def _compute_wrist_pose(self, 
                           keypoints: Dict[str, np.ndarray],
                           confidences: Dict[str, float],
                           side: str) -> Pose6DoF:
        """
        Compute wrist 6DoF pose from keypoint positions.
        
        Uses geometric constraints and IK to determine orientation.
        """
        shoulder = keypoints.get('shoulder', np.zeros(3))
        elbow = keypoints.get('elbow', np.zeros(3))
        wrist = keypoints.get('wrist', np.zeros(3))
        
        # Compute arm vectors
        upper_arm = elbow - shoulder
        forearm = wrist - elbow
        
        # Normalize
        upper_arm_norm = upper_arm / (np.linalg.norm(upper_arm) + 1e-10)
        forearm_norm = forearm / (np.linalg.norm(forearm) + 1e-10)
        
        # Compute wrist orientation from arm geometry
        # Z-axis: Along forearm (toward hand)
        z_axis = forearm_norm
        
        # Y-axis: Perpendicular to arm plane
        arm_normal = np.cross(upper_arm_norm, forearm_norm)
        arm_normal_len = np.linalg.norm(arm_normal)
        if arm_normal_len > 0.1:
            y_axis = arm_normal / arm_normal_len
        else:
            # Arms nearly straight - use world up
            y_axis = np.array([0, 1, 0])
            y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
            y_axis = y_axis / np.linalg.norm(y_axis)
        
        # X-axis: Cross product
        x_axis = np.cross(y_axis, z_axis)
        
        # Rotation matrix
        rot_mat = np.column_stack([x_axis, y_axis, z_axis])
        
        # Convert to quaternion
        r = SciRotation.from_matrix(rot_mat)
        q_scipy = r.as_quat()  # [x, y, z, w]
        orientation = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
        
        # Estimate velocity from history
        velocity = np.zeros(3)
        angular_velocity = np.zeros(3)
        
        if len(self.history) > 0:
            prev = self.history[-1]
            if side in prev and 'wrist' in prev[side]:
                dt = 1/30  # Assume 30fps
                prev_pos = prev[side]['wrist'].position
                velocity = (wrist - prev_pos) / dt
        
        return Pose6DoF(
            position=wrist,
            orientation=orientation,
            velocity=velocity,
            angular_velocity=angular_velocity,
            confidence=confidences.get('wrist', 0.5)
        )
    
    def _add_to_history(self, results: Dict):
        """Add tracking results to history."""
        self.history.append(copy.deepcopy(results))
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_racket_face_normal(self) -> np.ndarray:
        """Get current racket face normal vector."""
        pose = self.ekf_racket.get_pose()
        return pose.get_forward_vector()
    
    def get_trajectory(self, duration: float = 1.0) -> List[Pose6DoF]:
        """
        Get recent trajectory for specified duration.
        
        Args:
            duration: How far back to look (seconds)
            
        Returns:
            List of Pose6DoF for racket
        """
        n_frames = int(duration * 30)  # Assume 30fps
        trajectory = []
        
        for entry in self.history[-n_frames:]:
            if 'racket' in entry:
                trajectory.append(entry['racket'])
        
        return trajectory
    
    def reset(self):
        """Reset tracker state."""
        for side in ['right', 'left']:
            self.ekf_wrist[side] = EKF6DoF(dt=1/30)
        self.ekf_racket = EKF6DoF(dt=1/30)
        self.history.clear()
        self.frames_since_observation = {'right': 0, 'left': 0, 'racket': 0}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_tracker(body_height: float = 1.75,
                  grip_style: str = 'continental',
                  fps: float = 30.0) -> Universal6DoFTracker:
    """Create and return a configured Universal6DoFTracker."""
    return Universal6DoFTracker(body_height, grip_style, fps)


def pose_to_dict(pose: Pose6DoF) -> Dict:
    """Convert Pose6DoF to dictionary for JSON serialization."""
    return {
        'position': pose.position.tolist(),
        'orientation': pose.orientation.tolist(),
        'velocity': pose.velocity.tolist(),
        'angular_velocity': pose.angular_velocity.tolist(),
        'timestamp': pose.timestamp,
        'confidence': pose.confidence
    }


def dict_to_pose(d: Dict) -> Pose6DoF:
    """Create Pose6DoF from dictionary."""
    return Pose6DoF(
        position=np.array(d['position']),
        orientation=np.array(d['orientation']),
        velocity=np.array(d.get('velocity', [0, 0, 0])),
        angular_velocity=np.array(d.get('angular_velocity', [0, 0, 0])),
        timestamp=d.get('timestamp', 0),
        confidence=d.get('confidence', 1.0)
    )

