"""
Extended Kalman Filter for 6DoF Racket Pose Estimation
Handles occlusion recovery in the "Parallax Corridor"
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

from six_dof_tracker import MotionPhase, quaternion_from_rotation_matrix


@dataclass
class EKFState:
    """Full EKF state for visualization/debugging"""
    position: np.ndarray
    orientation: np.ndarray  # Quaternion
    velocity: np.ndarray
    angular_velocity: np.ndarray
    covariance_trace: float  # Uncertainty measure
    in_prediction_mode: bool


class BiomechanicalPriors:
    """
    Biomechanical priors for serve motion.
    Used to constrain EKF predictions during occlusion.
    """
    
    # Phase-specific velocity directions (unit vectors)
    VELOCITY_PRIORS = {
        MotionPhase.TROPHY: {
            'direction': np.array([0, 1, -0.3]),  # Up and back
            'magnitude_range': (1, 5),  # m/s
        },
        MotionPhase.BACKSCRATCH: {
            'direction': np.array([0, -0.5, -0.8]),  # Down and back
            'magnitude_range': (3, 15),
        },
        MotionPhase.ACCELERATION: {
            'direction': np.array([0, 0.7, 0.7]),  # Up and forward
            'magnitude_range': (15, 50),
        },
        MotionPhase.IMPACT: {
            'direction': np.array([0, 0.3, 0.95]),  # Forward
            'magnitude_range': (30, 50),
        },
        MotionPhase.FOLLOW_THROUGH: {
            'direction': np.array([-0.5, -0.3, 0.8]),  # Across body
            'magnitude_range': (10, 30),
        }
    }
    
    # Angular velocity ranges (deg/s)
    ANGULAR_VELOCITY_PRIORS = {
        MotionPhase.TROPHY: 100,
        MotionPhase.BACKSCRATCH: 500,
        MotionPhase.ACCELERATION: 2500,  # Peak pronation
        MotionPhase.IMPACT: 1500,
        MotionPhase.FOLLOW_THROUGH: 800
    }
    
    @classmethod
    def apply_velocity_prior(cls, 
                              velocity: np.ndarray,
                              phase: MotionPhase,
                              blend_factor: float = 0.3) -> np.ndarray:
        """
        Blend velocity toward phase-appropriate direction.
        
        Args:
            velocity: Current velocity estimate
            phase: Motion phase
            blend_factor: How much to pull toward prior (0-1)
        
        Returns:
            Adjusted velocity
        """
        if phase not in cls.VELOCITY_PRIORS:
            return velocity
        
        prior = cls.VELOCITY_PRIORS[phase]
        target_dir = prior['direction'] / np.linalg.norm(prior['direction'])
        
        v_mag = np.linalg.norm(velocity)
        if v_mag < 1e-6:
            return velocity
        
        v_dir = velocity / v_mag
        
        # Blend direction
        blended_dir = v_dir * (1 - blend_factor) + target_dir * blend_factor
        blended_dir = blended_dir / np.linalg.norm(blended_dir)
        
        # Constrain magnitude
        v_min, v_max = prior['magnitude_range']
        v_mag = np.clip(v_mag, v_min, v_max)
        
        return blended_dir * v_mag


class RacketEKF:
    """
    Extended Kalman Filter for 6DoF racket pose estimation.
    
    State vector (13 dimensions):
    x = [px, py, pz,           # Position (3)
         qw, qx, qy, qz,       # Orientation quaternion (4)
         vx, vy, vz,           # Linear velocity (3)
         wx, wy, wz]           # Angular velocity (3)
    """
    
    def __init__(self, dt: float = 0.033):
        """
        Args:
            dt: Time step (default ~30fps)
        """
        self.dt = dt
        self.n = 13  # State dimension
        
        # State vector
        self.x = np.zeros(self.n)
        self.x[3] = 1.0  # Quaternion w component (identity rotation)
        
        # State covariance
        self.P = np.eye(self.n) * 0.1
        
        # Process noise covariance
        self.Q = self._build_process_noise()
        
        # Measurement noise covariance
        self.R = np.eye(7) * 0.01  # Position (3) + Quaternion (4)
        
        # Physical constraints
        self.max_velocity = 50.0  # m/s (extreme serve)
        self.max_angular_velocity = np.radians(3000)  # rad/s
        
        # Prediction counter
        self.prediction_count = 0
        self.max_predictions = 15  # ~500ms at 30fps
    
    def _build_process_noise(self) -> np.ndarray:
        """Build process noise covariance matrix"""
        Q = np.eye(self.n)
        
        # Position noise (low - physics is predictable)
        Q[0:3, 0:3] *= 0.001
        
        # Orientation noise (medium)
        Q[3:7, 3:7] *= 0.01
        
        # Velocity noise (higher - we're estimating)
        Q[7:10, 7:10] *= 0.1
        
        # Angular velocity noise (highest)
        Q[10:13, 10:13] *= 0.5
        
        return Q
    
    def initialize(self, position: np.ndarray, orientation: np.ndarray,
                   velocity: np.ndarray = None, angular_velocity: np.ndarray = None):
        """Initialize filter state"""
        self.x[0:3] = position
        self.x[3:7] = orientation / np.linalg.norm(orientation)
        self.x[7:10] = velocity if velocity is not None else np.zeros(3)
        self.x[10:13] = angular_velocity if angular_velocity is not None else np.zeros(3)
        
        # Reset covariance
        self.P = np.eye(self.n) * 0.1
        self.prediction_count = 0
    
    def predict(self, phase: MotionPhase = MotionPhase.UNKNOWN):
        """
        Predict next state using physics model.
        
        Args:
            phase: Current motion phase for biomechanical priors
        """
        dt = self.dt
        
        # Extract current state
        p = self.x[0:3].copy()
        q = self.x[3:7].copy()
        v = self.x[7:10].copy()
        w = self.x[10:13].copy()
        
        # Apply biomechanical priors during extended prediction
        if self.prediction_count > 3:
            v = BiomechanicalPriors.apply_velocity_prior(v, phase, blend_factor=0.2)
        
        # Position update: p' = p + v*dt
        p_new = p + v * dt
        
        # Quaternion update: q' = q ⊗ Δq
        w_mag = np.linalg.norm(w)
        if w_mag > 1e-6:
            axis = w / w_mag
            angle = w_mag * dt
            half_angle = angle / 2
            
            dq = np.array([
                np.cos(half_angle),
                axis[0] * np.sin(half_angle),
                axis[1] * np.sin(half_angle),
                axis[2] * np.sin(half_angle)
            ])
            
            q_new = self._quaternion_multiply(q, dq)
            q_new = q_new / np.linalg.norm(q_new)
        else:
            q_new = q
        
        # Velocity decay (energy dissipation)
        if phase != MotionPhase.ACCELERATION:
            decay = 0.98
        else:
            decay = 1.0  # No decay during explosive phase
        
        v_new = v * decay
        w_new = w * decay
        
        # Update state
        self.x[0:3] = p_new
        self.x[3:7] = q_new
        self.x[7:10] = v_new
        self.x[10:13] = w_new
        
        # Update covariance: P' = F*P*F' + Q
        F = self._compute_jacobian()
        
        # Increase process noise during extended prediction
        Q_scaled = self.Q * (1 + 0.2 * self.prediction_count)
        
        self.P = F @ self.P @ F.T + Q_scaled
        
        # Enforce constraints
        self._enforce_constraints()
        
        self.prediction_count += 1
    
    def update(self, measurement: np.ndarray, confidence: float = 1.0):
        """
        Update state with measurement.
        
        Args:
            measurement: [px, py, pz, qw, qx, qy, qz]
            confidence: Measurement confidence (0-1)
        """
        # Reset prediction counter
        self.prediction_count = 0
        
        # Measurement matrix
        H = np.zeros((7, self.n))
        H[0:3, 0:3] = np.eye(3)  # Position
        H[3:7, 3:7] = np.eye(4)  # Quaternion
        
        # Adjust measurement noise by confidence
        R = self.R / (confidence + 0.01)
        
        # Predicted measurement
        z_pred = H @ self.x
        
        # Handle quaternion sign ambiguity
        if np.dot(measurement[3:7], z_pred[3:7]) < 0:
            measurement = measurement.copy()
            measurement[3:7] = -measurement[3:7]
        
        # Innovation
        y = measurement - z_pred
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Normalize quaternion
        self.x[3:7] = self.x[3:7] / np.linalg.norm(self.x[3:7])
        
        # Covariance update (Joseph form for numerical stability)
        I = np.eye(self.n)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        
        # Update velocity from position change
        if hasattr(self, '_last_position'):
            dt = self.dt
            v_observed = (self.x[0:3] - self._last_position) / dt
            # Blend observed velocity with predicted
            self.x[7:10] = 0.6 * self.x[7:10] + 0.4 * v_observed
        
        self._last_position = self.x[0:3].copy()
    
    def _compute_jacobian(self) -> np.ndarray:
        """Compute state transition Jacobian"""
        F = np.eye(self.n)
        dt = self.dt
        
        # Position depends on velocity
        F[0:3, 7:10] = np.eye(3) * dt
        
        # Quaternion depends on angular velocity (linearized)
        q = self.x[3:7]
        F[3:7, 10:13] = self._quaternion_omega_jacobian(q) * dt
        
        return F
    
    def _quaternion_omega_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Jacobian of quaternion kinematics w.r.t. angular velocity"""
        w, x, y, z = q
        return 0.5 * np.array([
            [-x, -y, -z],
            [w, -z, y],
            [z, w, -x],
            [-y, x, w]
        ])
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion multiplication q1 ⊗ q2"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _enforce_constraints(self):
        """Enforce physical constraints"""
        # Velocity magnitude
        v_mag = np.linalg.norm(self.x[7:10])
        if v_mag > self.max_velocity:
            self.x[7:10] = self.x[7:10] / v_mag * self.max_velocity
        
        # Angular velocity magnitude
        w_mag = np.linalg.norm(self.x[10:13])
        if w_mag > self.max_angular_velocity:
            self.x[10:13] = self.x[10:13] / w_mag * self.max_angular_velocity
        
        # Quaternion normalization
        self.x[3:7] = self.x[3:7] / np.linalg.norm(self.x[3:7])
    
    def get_state(self) -> EKFState:
        """Get current state as structured object"""
        return EKFState(
            position=self.x[0:3].copy(),
            orientation=self.x[3:7].copy(),
            velocity=self.x[7:10].copy(),
            angular_velocity=self.x[10:13].copy(),
            covariance_trace=np.trace(self.P),
            in_prediction_mode=self.prediction_count > 0
        )
    
    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get position and orientation"""
        return self.x[0:3].copy(), self.x[3:7].copy()
    
    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get linear and angular velocity"""
        return self.x[7:10].copy(), self.x[10:13].copy()
    
    @property
    def is_prediction_valid(self) -> bool:
        """Check if prediction is still reliable"""
        return self.prediction_count < self.max_predictions


class OcclusionRecoverySystem:
    """
    Manages occlusion detection and recovery using EKF.
    Provides smooth transitions when entering/exiting occlusion.
    """
    
    def __init__(self, dt: float = 0.033):
        self.ekf = RacketEKF(dt)
        self.dt = dt
        
        # State
        self.in_occlusion = False
        self.occlusion_start_time = 0.0
        self.max_occlusion_duration = 0.5  # 500ms
        
        # Recovery blending
        self.recovery_duration = 0.15  # 150ms
        self.recovery_start_time = 0.0
        self.recovery_start_pose = None
        
        # Statistics
        self.total_occlusion_time = 0.0
        self.occlusion_count = 0
    
    def reset(self):
        """Reset system"""
        self.in_occlusion = False
        self.recovery_start_pose = None
        self.total_occlusion_time = 0.0
        self.occlusion_count = 0
    
    def process(self,
                timestamp: float,
                tracked_pose: Optional[Tuple[np.ndarray, np.ndarray]],
                tracking_confidence: float,
                motion_phase: MotionPhase) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Process frame and return best pose estimate.
        
        Args:
            timestamp: Current time
            tracked_pose: (position, quaternion) or None if occluded
            tracking_confidence: 0-1
            motion_phase: Current motion phase
        
        Returns:
            (position, quaternion, confidence)
        """
        # Determine if occluded
        is_occluded = tracked_pose is None or tracking_confidence < 0.3
        
        if not is_occluded:
            # Good tracking available
            position, quaternion = tracked_pose
            
            if self.in_occlusion:
                # Exiting occlusion - start recovery blend
                self.in_occlusion = False
                self.total_occlusion_time += timestamp - self.occlusion_start_time
                
                self.recovery_start_time = timestamp
                self.recovery_start_pose = self.ekf.get_pose()
            
            # Update EKF with measurement
            measurement = np.concatenate([position, quaternion])
            self.ekf.update(measurement, tracking_confidence)
            
            # Check if in recovery blend
            if self.recovery_start_pose is not None:
                elapsed = timestamp - self.recovery_start_time
                
                if elapsed < self.recovery_duration:
                    # Blend from prediction to tracked
                    t = self._smoothstep(elapsed / self.recovery_duration)
                    
                    blended_pos = self._lerp(
                        self.recovery_start_pose[0], position, t
                    )
                    blended_quat = self._slerp(
                        self.recovery_start_pose[1], quaternion, t
                    )
                    
                    # Confidence ramps up during recovery
                    confidence = 0.5 + 0.5 * t
                    
                    return blended_pos, blended_quat, confidence
                else:
                    # Recovery complete
                    self.recovery_start_pose = None
            
            return position, quaternion, tracking_confidence
        
        else:
            # Occlusion - use EKF prediction
            
            if not self.in_occlusion:
                # Entering occlusion
                self.in_occlusion = True
                self.occlusion_start_time = timestamp
                self.occlusion_count += 1
            
            # Predict
            self.ekf.predict(phase=motion_phase)
            
            # Get predicted pose
            position, quaternion = self.ekf.get_pose()
            
            # Confidence decays during occlusion
            occlusion_duration = timestamp - self.occlusion_start_time
            confidence = max(0.1, 0.8 - occlusion_duration * 1.5)
            
            # Check if prediction is still valid
            if not self.ekf.is_prediction_valid:
                confidence *= 0.5  # Further reduce confidence
            
            return position, quaternion, confidence
    
    def _lerp(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        """Linear interpolation"""
        return a + t * (b - a)
    
    def _slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation"""
        # Handle quaternion sign
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        # If very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        # SLERP
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        
        s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0
        
        return s1 * q1 + s2 * q2
    
    def _smoothstep(self, t: float) -> float:
        """Smooth step function"""
        t = np.clip(t, 0, 1)
        return t * t * (3 - 2 * t)
    
    def get_statistics(self) -> Dict:
        """Get occlusion statistics"""
        return {
            'total_occlusion_time': self.total_occlusion_time,
            'occlusion_count': self.occlusion_count,
            'currently_occluded': self.in_occlusion,
            'ekf_prediction_count': self.ekf.prediction_count,
            'ekf_covariance_trace': np.trace(self.ekf.P)
        }


if __name__ == "__main__":
    import sys
    
    print("EKF Tracker Test")
    print("=" * 50)
    
    # Simple test
    ekf = RacketEKF(dt=0.033)
    
    # Initialize at origin with identity rotation
    ekf.initialize(
        position=np.array([0, 1.5, 0]),
        orientation=np.array([1, 0, 0, 0]),
        velocity=np.array([0, 5, 10])  # Moving up and forward
    )
    
    print("Initial state:")
    state = ekf.get_state()
    print(f"  Position: {state.position}")
    print(f"  Velocity: {state.velocity}")
    
    # Predict forward (simulate occlusion)
    print("\nPredicting through 10 frames of occlusion...")
    for i in range(10):
        ekf.predict(phase=MotionPhase.ACCELERATION)
    
    state = ekf.get_state()
    print(f"After prediction:")
    print(f"  Position: {state.position}")
    print(f"  Velocity: {state.velocity}")
    print(f"  Covariance trace: {state.covariance_trace:.2f}")
    print(f"  Prediction valid: {ekf.is_prediction_valid}")
    
    # Test full system with data
    if len(sys.argv) > 1:
        from six_dof_tracker import SixDoFTracker, load_body_tracking_json
        
        body_path = sys.argv[1]
        print(f"\nTesting with data: {body_path}")
        
        frames = load_body_tracking_json(body_path)
        six_dof = SixDoFTracker()
        recovery = OcclusionRecoverySystem()
        
        results = []
        for frame in frames:
            # Get 6DoF pose
            pose = six_dof.compute_pose(frame, 'right')
            
            if pose:
                tracked_pose = (pose.position, pose.orientation)
                confidence = pose.confidence
            else:
                tracked_pose = None
                confidence = 0.0
            
            # Process through recovery system
            pos, quat, conf = recovery.process(
                frame.timestamp,
                tracked_pose,
                confidence,
                pose.phase if pose else MotionPhase.UNKNOWN
            )
            
            results.append({
                'timestamp': frame.timestamp,
                'position': pos,
                'confidence': conf,
                'occluded': tracked_pose is None
            })
        
        # Statistics
        stats = recovery.get_statistics()
        print(f"\nResults:")
        print(f"  Total frames: {len(results)}")
        print(f"  Occlusion events: {stats['occlusion_count']}")
        print(f"  Total occlusion time: {stats['total_occlusion_time']*1000:.0f}ms")
        
        occluded_frames = sum(1 for r in results if r['occluded'])
        print(f"  Occluded frames: {occluded_frames} ({occluded_frames/len(results)*100:.1f}%)")

