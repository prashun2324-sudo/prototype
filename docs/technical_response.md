# Technical Response: 4-Point Implementation Specification

## Executive Summary

This document provides exact implementation logic for patent-grade racket tracking accuracy, addressing the 3 missing Degrees of Freedom, occlusion recovery, ball-racket impact physics, and Unity integration.

---

## 1. Solving for the Missing 3 DoFs (Supination, Pronation, Flexion)

### Problem Statement
The wrist anchor provides only 3 DoF (position). We need additional 3 DoF for orientation:
- **Pronation/Supination**: Forearm rotation along its longitudinal axis (±90°)
- **Wrist Flexion/Extension**: Bending toward palm/back of hand (±70°)  
- **Radial/Ulnar Deviation**: Side-to-side wrist movement (±30°)

### Solution: Quaternion-Based IK with Forearm Twist Extraction

#### Mathematical Model

**Step 1: Extract Forearm Twist (Pronation/Supination)**

The forearm twist cannot be directly observed from elbow-wrist positions alone. We derive it from the kinematic chain:

```
Given:
  P_shoulder, P_elbow, P_wrist from Apple Vision
  
Compute forearm coordinate frame:
  Z_forearm = normalize(P_wrist - P_elbow)           // Forearm axis
  Y_temp = normalize(P_elbow - P_shoulder)           // Upper arm direction
  X_forearm = normalize(cross(Y_temp, Z_forearm))    // Perpendicular to arm plane
  Y_forearm = cross(Z_forearm, X_forearm)            // Complete orthonormal basis
  
Forearm rotation matrix:
  R_forearm = [X_forearm | Y_forearm | Z_forearm]    // 3x3 matrix
```

**Step 2: Pronation Angle from Elbow Flexion Coupling**

Biomechanical constraint: Pronation couples with elbow angle during tennis strokes.

```python
def estimate_pronation_angle(elbow_angle: float, 
                              motion_phase: str,
                              angular_velocity: float) -> float:
    """
    Estimate forearm pronation from biomechanical coupling.
    
    Tennis serve pronation pattern:
    - Trophy position: ~0° (neutral)
    - Back-scratch: -45° to -60° (supinated)
    - Acceleration: Rapid pronation
    - Impact: +30° to +45° (pronated)
    - Follow-through: +60° to +90° (full pronation)
    """
    
    # Base pronation from phase
    phase_pronation = {
        'preparation': 0.0,
        'trophy': 0.0,
        'backscratch': -50.0,      # Supinated
        'acceleration': 15.0,      # Transitioning
        'impact': 40.0,            # Pronated for power
        'follow_through': 70.0     # Full pronation
    }
    
    base = phase_pronation.get(motion_phase, 0.0)
    
    # Coupling with elbow extension rate
    # During acceleration, pronation velocity correlates with elbow extension
    if motion_phase == 'acceleration':
        # Peak pronation velocity: 1500-2500 deg/s
        pronation_velocity = angular_velocity * 0.6  # Coupling factor
        base += pronation_velocity * 0.033  # Per-frame contribution
    
    return np.clip(base, -90, 90)
```

**Step 3: Wrist Flexion from Angle Data**

Your JSON already contains `L Wrist` / `R Wrist` angles. Convert to flexion:

```python
def wrist_angle_to_flexion(wrist_angle: float) -> float:
    """
    Convert Apple Vision wrist angle to flexion/extension.
    
    Apple Vision reports angle between forearm and hand vectors.
    Neutral (straight) = 180°
    Flexion (bent toward palm) = < 180°
    Extension (bent backward) = > 180° (reported as < 180° from other side)
    """
    # Normalize to flexion angle
    # 180° = 0° flexion (neutral)
    # 120° = 60° flexion
    # 200° = -20° extension
    
    flexion = 180.0 - wrist_angle
    return np.clip(flexion, -70, 80)  # Anatomical limits
```

**Step 4: Construct Full 6DoF Racket Pose**

```python
def compute_racket_6dof(P_shoulder: np.ndarray,
                        P_elbow: np.ndarray,
                        P_wrist: np.ndarray,
                        wrist_angle: float,
                        motion_phase: str,
                        angular_velocity: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute full 6DoF racket pose.
    
    Returns:
        position: [x, y, z] grip point in world space
        orientation: Quaternion [w, x, y, z]
    """
    
    # 1. Forearm basis
    Z_forearm = normalize(P_wrist - P_elbow)
    Y_temp = normalize(P_elbow - P_shoulder)
    X_forearm = normalize(np.cross(Y_temp, Z_forearm))
    Y_forearm = np.cross(Z_forearm, X_forearm)
    R_forearm = np.column_stack([X_forearm, Y_forearm, Z_forearm])
    
    # 2. Pronation rotation (about forearm axis Z)
    pronation_deg = estimate_pronation_angle(
        compute_elbow_angle(P_shoulder, P_elbow, P_wrist),
        motion_phase,
        angular_velocity
    )
    R_pronation = rotation_matrix_z(np.radians(pronation_deg))
    
    # 3. Wrist flexion rotation (about X axis after pronation)
    flexion_deg = wrist_angle_to_flexion(wrist_angle)
    R_flexion = rotation_matrix_x(np.radians(flexion_deg))
    
    # 4. Grip offset rotation (for specific grip type)
    # Continental grip: racket face ~15° closed from forearm plane
    R_grip = rotation_matrix_x(np.radians(-15))  # Adjust per grip style
    
    # 5. Combine rotations: Forearm → Pronation → Flexion → Grip
    R_racket = R_forearm @ R_pronation @ R_flexion @ R_grip
    
    # 6. Position: Wrist + grip offset along racket shaft
    grip_offset_local = np.array([0, 0, 0.08])  # 8cm along shaft
    grip_offset_world = R_racket @ grip_offset_local
    P_racket = P_wrist + grip_offset_world
    
    # 7. Convert to quaternion
    Q_racket = quaternion_from_rotation_matrix(R_racket)
    
    return P_racket, Q_racket


def rotation_matrix_x(angle: float) -> np.ndarray:
    """Rotation matrix about X axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rotation_matrix_z(angle: float) -> np.ndarray:
    """Rotation matrix about Z axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
```

### Quaternion Representation

For smooth interpolation and gimbal-lock avoidance, all rotations stored as quaternions:

```python
def quaternion_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]"""
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z]) / np.linalg.norm([w, x, y, z])
```

---

## 2. Occlusion Recovery: Extended Kalman Filter for 6DoF Pose

### Problem Statement
When racket enters the "parallax corridor" (45° behind player), tracking is lost. We need state estimation to predict 6DoF pose and prevent "teleportation" when tracking resumes.

### Solution: Extended Kalman Filter (EKF) with Biomechanical Constraints

#### State Vector (13 dimensions)

```
x = [
    px, py, pz,           // Position (3)
    qw, qx, qy, qz,       // Orientation quaternion (4)
    vx, vy, vz,           // Linear velocity (3)
    wx, wy, wz            // Angular velocity (3)
]
```

#### Process Model (State Transition)

```python
class RacketEKF:
    """Extended Kalman Filter for 6DoF racket pose estimation"""
    
    def __init__(self, dt: float = 0.033):
        self.dt = dt
        
        # State dimension
        self.n = 13
        
        # State vector
        self.x = np.zeros(self.n)
        self.x[3] = 1.0  # Quaternion w component (identity)
        
        # State covariance
        self.P = np.eye(self.n) * 0.1
        
        # Process noise covariance
        self.Q = self._build_process_noise()
        
        # Measurement noise covariance
        self.R = np.eye(7) * 0.01  # Position (3) + Quaternion (4)
        
        # Biomechanical constraints
        self.max_velocity = 50.0  # m/s (serve peak)
        self.max_angular_velocity = 3000.0  # deg/s (pronation peak)
    
    def _build_process_noise(self) -> np.ndarray:
        """Build process noise matrix with physics-based uncertainty"""
        Q = np.eye(self.n)
        
        # Position uncertainty grows with velocity
        Q[0:3, 0:3] *= 0.001  # Low position noise
        
        # Orientation uncertainty
        Q[3:7, 3:7] *= 0.01
        
        # Velocity uncertainty (higher - we're predicting)
        Q[7:10, 7:10] *= 0.1
        
        # Angular velocity uncertainty
        Q[10:13, 10:13] *= 0.5
        
        return Q
    
    def predict(self, phase: str = 'unknown'):
        """
        Predict next state using physics model.
        
        Args:
            phase: Current motion phase for biomechanical priors
        """
        dt = self.dt
        
        # Extract current state
        p = self.x[0:3]    # Position
        q = self.x[3:7]    # Quaternion
        v = self.x[7:10]   # Velocity
        w = self.x[10:13]  # Angular velocity
        
        # Apply biomechanical priors based on phase
        v, w = self._apply_biomechanical_priors(v, w, phase)
        
        # Position update: p' = p + v*dt + 0.5*a*dt^2
        # Assume constant velocity (no acceleration in predict)
        p_new = p + v * dt
        
        # Quaternion update: q' = q ⊗ Δq
        # where Δq represents rotation by w*dt
        w_magnitude = np.linalg.norm(w)
        if w_magnitude > 1e-6:
            axis = w / w_magnitude
            angle = w_magnitude * dt
            half_angle = angle / 2
            dq = np.array([
                np.cos(half_angle),
                axis[0] * np.sin(half_angle),
                axis[1] * np.sin(half_angle),
                axis[2] * np.sin(half_angle)
            ])
            q_new = self._quaternion_multiply(q, dq)
            q_new = q_new / np.linalg.norm(q_new)  # Normalize
        else:
            q_new = q
        
        # Velocity decay (air resistance, muscle deceleration)
        decay = 0.98 if phase != 'acceleration' else 1.0
        v_new = v * decay
        w_new = w * decay
        
        # Update state
        self.x[0:3] = p_new
        self.x[3:7] = q_new
        self.x[7:10] = v_new
        self.x[10:13] = w_new
        
        # Update covariance: P' = F*P*F' + Q
        F = self._compute_jacobian()
        self.P = F @ self.P @ F.T + self.Q
        
        # Enforce constraints
        self._enforce_constraints()
    
    def _apply_biomechanical_priors(self, v: np.ndarray, w: np.ndarray, 
                                      phase: str) -> Tuple[np.ndarray, np.ndarray]:
        """Apply biomechanical priors based on serve phase"""
        
        # Phase-specific velocity priors
        phase_priors = {
            'trophy': {
                'v_direction': np.array([0, 1, -0.3]),  # Up and back
                'v_magnitude_range': (1, 5),
                'w_pronation_range': (-100, 100)  # deg/s
            },
            'backscratch': {
                'v_direction': np.array([0, -0.5, -0.8]),  # Down and back
                'v_magnitude_range': (3, 15),
                'w_pronation_range': (-500, 0)  # Supinating
            },
            'acceleration': {
                'v_direction': np.array([0, 0.7, 0.7]),  # Up and forward
                'v_magnitude_range': (15, 50),
                'w_pronation_range': (1000, 2500)  # Rapid pronation
            },
            'impact': {
                'v_direction': np.array([0, 0.3, 0.95]),  # Forward
                'v_magnitude_range': (30, 50),
                'w_pronation_range': (500, 1500)
            },
            'follow_through': {
                'v_direction': np.array([-0.5, -0.3, 0.8]),  # Across body
                'v_magnitude_range': (10, 30),
                'w_pronation_range': (200, 800)
            }
        }
        
        if phase in phase_priors:
            prior = phase_priors[phase]
            
            # Soft constraint on velocity direction
            v_mag = np.linalg.norm(v)
            if v_mag > 0:
                v_dir = v / v_mag
                target_dir = prior['v_direction'] / np.linalg.norm(prior['v_direction'])
                
                # Blend toward expected direction
                blend = 0.3  # 30% pull toward prior
                v_dir_new = v_dir * (1 - blend) + target_dir * blend
                v_dir_new = v_dir_new / np.linalg.norm(v_dir_new)
                
                # Constrain magnitude
                v_min, v_max = prior['v_magnitude_range']
                v_mag = np.clip(v_mag, v_min, v_max)
                
                v = v_dir_new * v_mag
        
        return v, w
    
    def update(self, measurement: np.ndarray, confidence: float):
        """
        Update state with measurement.
        
        Args:
            measurement: [px, py, pz, qw, qx, qy, qz]
            confidence: Measurement confidence (0-1)
        """
        # Measurement matrix (observe position and orientation)
        H = np.zeros((7, self.n))
        H[0:3, 0:3] = np.eye(3)  # Position
        H[3:7, 3:7] = np.eye(4)  # Quaternion
        
        # Adjust measurement noise by confidence
        R = self.R / (confidence + 0.01)  # Higher confidence = lower noise
        
        # Innovation (measurement residual)
        z_pred = H @ self.x
        y = measurement - z_pred
        
        # Handle quaternion discontinuity (flip if dot product < 0)
        if np.dot(measurement[3:7], z_pred[3:7]) < 0:
            y[3:7] = measurement[3:7] + z_pred[3:7]  # Opposite hemisphere
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Normalize quaternion
        self.x[3:7] = self.x[3:7] / np.linalg.norm(self.x[3:7])
        
        # Covariance update
        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P
        
        # Update velocity estimate from position change
        if hasattr(self, '_last_position'):
            v_observed = (self.x[0:3] - self._last_position) / self.dt
            self.x[7:10] = 0.7 * self.x[7:10] + 0.3 * v_observed
        self._last_position = self.x[0:3].copy()
    
    def _compute_jacobian(self) -> np.ndarray:
        """Compute state transition Jacobian"""
        F = np.eye(self.n)
        dt = self.dt
        
        # Position depends on velocity
        F[0:3, 7:10] = np.eye(3) * dt
        
        # Quaternion depends on angular velocity (simplified)
        # Full derivation requires quaternion calculus
        F[3:7, 10:13] = np.eye(4, 3) * dt * 0.5
        
        return F
    
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
        """Enforce physical constraints on state"""
        # Velocity magnitude constraint
        v_mag = np.linalg.norm(self.x[7:10])
        if v_mag > self.max_velocity:
            self.x[7:10] = self.x[7:10] / v_mag * self.max_velocity
        
        # Angular velocity constraint
        w_mag = np.linalg.norm(self.x[10:13])
        max_w_rad = np.radians(self.max_angular_velocity)
        if w_mag > max_w_rad:
            self.x[10:13] = self.x[10:13] / w_mag * max_w_rad
        
        # Quaternion normalization
        self.x[3:7] = self.x[3:7] / np.linalg.norm(self.x[3:7])
    
    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current position and orientation"""
        return self.x[0:3].copy(), self.x[3:7].copy()
    
    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current linear and angular velocity"""
        return self.x[7:10].copy(), self.x[10:13].copy()
```

### Occlusion Detection and Recovery

```python
class OcclusionRecoverySystem:
    """Manages transition in/out of occlusion using EKF"""
    
    def __init__(self):
        self.ekf = RacketEKF()
        self.in_occlusion = False
        self.occlusion_start_time = 0.0
        self.max_occlusion_duration = 0.5  # 500ms max prediction
        
        # Recovery blending
        self.recovery_duration = 0.15  # 150ms blend
        self.recovery_start_time = 0.0
        self.recovery_start_pose = None
    
    def process(self, 
                timestamp: float,
                tracked_pose: Optional[Tuple[np.ndarray, np.ndarray]],
                tracking_confidence: float,
                motion_phase: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process frame and return best pose estimate.
        
        Args:
            timestamp: Current time
            tracked_pose: (position, quaternion) if tracking available, None if occluded
            tracking_confidence: 0-1 confidence in tracking
            motion_phase: Current serve phase
        
        Returns:
            (position, quaternion) - best estimate
        """
        
        occlusion_detected = tracked_pose is None or tracking_confidence < 0.3
        
        if not occlusion_detected:
            # Good tracking available
            
            if self.in_occlusion:
                # Just exited occlusion - start recovery blend
                self.in_occlusion = False
                self.recovery_start_time = timestamp
                self.recovery_start_pose = self.ekf.get_pose()
            
            # Update EKF with measurement
            position, quaternion = tracked_pose
            measurement = np.concatenate([position, quaternion])
            self.ekf.update(measurement, tracking_confidence)
            
            # Check if still in recovery blend
            if self.recovery_start_pose is not None:
                recovery_elapsed = timestamp - self.recovery_start_time
                
                if recovery_elapsed < self.recovery_duration:
                    # Blend from EKF prediction to tracked pose
                    t = recovery_elapsed / self.recovery_duration
                    t = self._smoothstep(t)
                    
                    blended_pos = self._lerp(
                        self.recovery_start_pose[0], position, t
                    )
                    blended_quat = self._slerp(
                        self.recovery_start_pose[1], quaternion, t
                    )
                    
                    return blended_pos, blended_quat
                else:
                    # Recovery complete
                    self.recovery_start_pose = None
            
            return position, quaternion
        
        else:
            # Occlusion - use EKF prediction
            
            if not self.in_occlusion:
                # Just entered occlusion
                self.in_occlusion = True
                self.occlusion_start_time = timestamp
            
            # Check occlusion duration limit
            occlusion_duration = timestamp - self.occlusion_start_time
            if occlusion_duration > self.max_occlusion_duration:
                # Too long - reduce confidence in prediction
                pass
            
            # Predict forward
            self.ekf.predict(phase=motion_phase)
            
            return self.ekf.get_pose()
    
    def _lerp(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        return a + t * (b - a)
    
    def _slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation"""
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        
        q2_perp = q2 - q1 * dot
        q2_perp = q2_perp / np.linalg.norm(q2_perp)
        
        return q1 * np.cos(theta) + q2_perp * np.sin(theta)
    
    def _smoothstep(self, t: float) -> float:
        t = np.clip(t, 0, 1)
        return t * t * (3 - 2 * t)
```

---

## 3. Ball-Racket Impact Physics: Normal Vector and Spin Calculation

### Problem Statement
At the millisecond of impact, we need:
- Racket face normal vector
- Ball trajectory calculation
- Spin (RPM) calculation

### Solution: Contact Mechanics Model

#### Racket Face Normal Vector

```python
def compute_racket_face_normal(quaternion: np.ndarray) -> np.ndarray:
    """
    Compute racket face normal vector from orientation quaternion.
    
    Racket coordinate system:
    - Local Z = face normal (pointing away from strings)
    - Local Y = shaft direction (handle to head)
    - Local X = lateral (left to right across face)
    
    Args:
        quaternion: [w, x, y, z] racket orientation
    
    Returns:
        Unit normal vector in world coordinates
    """
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(quaternion)
    
    # Face normal is the Z-axis of the racket frame
    face_normal = R[:, 2]
    
    return face_normal / np.linalg.norm(face_normal)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix"""
    w, x, y, z = q
    
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
```

#### Ball Trajectory at Impact

```python
@dataclass
class ImpactResult:
    """Result of ball-racket impact calculation"""
    exit_velocity: np.ndarray      # m/s
    exit_direction: np.ndarray     # Unit vector
    spin_axis: np.ndarray          # Unit vector
    spin_rpm: float                # Revolutions per minute
    contact_point: np.ndarray      # World position
    contact_time: float            # Timestamp


class ImpactPhysics:
    """
    Ball-racket impact physics model.
    Based on ITF technical specifications and research.
    """
    
    # Physical constants
    BALL_MASS = 0.057  # kg (tennis ball: 56-59.4g)
    BALL_RADIUS = 0.0335  # m (6.54-6.86 cm diameter)
    BALL_COR = 0.75  # Coefficient of restitution (0.73-0.76 for tennis)
    
    # Racket properties
    RACKET_MASS = 0.320  # kg (typical strung racket)
    STRING_BED_DWELL_TIME = 0.004  # seconds (3-5ms typical)
    
    # Friction coefficient (string-ball)
    MU_FRICTION = 0.4  # Tangential friction
    
    def compute_impact(self,
                       ball_velocity: np.ndarray,
                       ball_spin: np.ndarray,  # Angular velocity rad/s
                       racket_position: np.ndarray,
                       racket_quaternion: np.ndarray,
                       racket_velocity: np.ndarray,
                       racket_angular_velocity: np.ndarray,
                       contact_point_local: np.ndarray = None
                       ) -> ImpactResult:
        """
        Compute ball trajectory and spin after impact.
        
        Uses oblique impact mechanics with friction.
        
        Args:
            ball_velocity: Incoming ball velocity (m/s)
            ball_spin: Incoming ball angular velocity (rad/s)
            racket_position: Racket position (grip point)
            racket_quaternion: Racket orientation
            racket_velocity: Racket velocity at impact (m/s)
            racket_angular_velocity: Racket angular velocity (rad/s)
            contact_point_local: Contact point in racket local coords (default: center)
        
        Returns:
            ImpactResult with exit trajectory and spin
        """
        
        # 1. Get racket face normal
        R_racket = quaternion_to_rotation_matrix(racket_quaternion)
        face_normal = R_racket[:, 2]
        
        # 2. Compute contact point in world space
        if contact_point_local is None:
            contact_point_local = np.array([0, 0.35, 0])  # Center of strings
        contact_point = racket_position + R_racket @ contact_point_local
        
        # 3. Racket velocity at contact point (including rotation)
        r_contact = contact_point - racket_position
        v_racket_at_contact = racket_velocity + np.cross(racket_angular_velocity, r_contact)
        
        # 4. Relative velocity (ball relative to racket)
        v_rel = ball_velocity - v_racket_at_contact
        
        # 5. Decompose into normal and tangential components
        v_normal = np.dot(v_rel, face_normal) * face_normal
        v_tangent = v_rel - v_normal
        
        # 6. Normal impulse (coefficient of restitution)
        # J_n = -(1 + e) * m_eff * v_n
        m_eff = (self.BALL_MASS * self.RACKET_MASS) / (self.BALL_MASS + self.RACKET_MASS)
        J_normal = -(1 + self.BALL_COR) * m_eff * np.dot(v_rel, face_normal)
        
        # 7. Tangential impulse (friction limited)
        # Account for ball spin at contact
        ball_surface_velocity = np.cross(ball_spin, self.BALL_RADIUS * face_normal)
        v_slip = v_tangent + ball_surface_velocity
        
        J_tangent_max = self.MU_FRICTION * abs(J_normal)
        J_tangent_required = m_eff * np.linalg.norm(v_slip)
        
        if J_tangent_required <= J_tangent_max:
            # No slip - full grip
            J_tangent = -m_eff * v_slip
        else:
            # Slip - friction limited
            slip_direction = v_slip / np.linalg.norm(v_slip)
            J_tangent = -J_tangent_max * slip_direction
        
        # 8. Total impulse
        J_total = J_normal * face_normal + J_tangent
        
        # 9. Ball exit velocity
        v_ball_exit = ball_velocity + J_total / self.BALL_MASS
        
        # 10. Ball exit spin
        # Δω = (r × J) / I_ball
        I_ball = (2/3) * self.BALL_MASS * self.BALL_RADIUS**2  # Hollow sphere
        torque_impulse = np.cross(-self.BALL_RADIUS * face_normal, J_tangent)
        delta_spin = torque_impulse / I_ball
        spin_exit = ball_spin + delta_spin
        
        # 11. Convert to RPM
        spin_magnitude = np.linalg.norm(spin_exit)
        spin_rpm = spin_magnitude * 60 / (2 * np.pi)
        spin_axis = spin_exit / spin_magnitude if spin_magnitude > 0 else np.array([0, 1, 0])
        
        # 12. Exit direction
        exit_speed = np.linalg.norm(v_ball_exit)
        exit_direction = v_ball_exit / exit_speed if exit_speed > 0 else face_normal
        
        return ImpactResult(
            exit_velocity=v_ball_exit,
            exit_direction=exit_direction,
            spin_axis=spin_axis,
            spin_rpm=spin_rpm,
            contact_point=contact_point,
            contact_time=0.0  # Set externally
        )
    
    def estimate_spin_type(self, spin_axis: np.ndarray, 
                           ball_direction: np.ndarray) -> str:
        """
        Classify spin type based on axis orientation.
        
        Returns:
            'topspin', 'backspin', 'sidespin_left', 'sidespin_right', 'mixed'
        """
        # Spin axis relative to ball direction
        # Topspin: axis horizontal, perpendicular to direction
        # Sidespin: axis vertical
        
        up = np.array([0, 1, 0])
        forward = ball_direction / np.linalg.norm(ball_direction)
        right = np.cross(up, forward)
        
        # Project spin axis
        vertical_component = abs(np.dot(spin_axis, up))
        horizontal_component = np.sqrt(1 - vertical_component**2)
        
        # Determine which side (for sidespin)
        side_dot = np.dot(spin_axis, right)
        
        if vertical_component > 0.7:
            return 'sidespin_right' if side_dot > 0 else 'sidespin_left'
        elif horizontal_component > 0.7:
            # Check topspin vs backspin
            topspin_axis = np.cross(forward, up)
            if np.dot(spin_axis, topspin_axis) > 0:
                return 'topspin'
            else:
                return 'backspin'
        else:
            return 'mixed'
```

#### Usage Example

```python
# At impact detection
impact_physics = ImpactPhysics()

# Ball state (from ball tracking)
ball_velocity = np.array([0, 5, -15])  # Coming toward player
ball_spin = np.array([0, 0, 0])  # No incoming spin (serve toss)

# Racket state from our tracking system
racket_pos, racket_quat = tracking_system.get_pose()
racket_vel, racket_ang_vel = tracking_system.get_velocity()

# Compute impact
result = impact_physics.compute_impact(
    ball_velocity=ball_velocity,
    ball_spin=ball_spin,
    racket_position=racket_pos,
    racket_quaternion=racket_quat,
    racket_velocity=racket_vel,
    racket_angular_velocity=racket_ang_vel
)

print(f"Exit speed: {np.linalg.norm(result.exit_velocity):.1f} m/s")
print(f"Spin: {result.spin_rpm:.0f} RPM")
print(f"Spin type: {impact_physics.estimate_spin_type(result.spin_axis, result.exit_direction)}")
```

---

## 4. Logic Integration: Unity C# Interface

### Problem Statement
Pass tracking data to Unity's skeletal rig "Grip Constraint" without physics engine conflicts.

### Solution: Message-Based Bridge with Constraint Priority

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Python Tracking System                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │   EKF    │  │ Template │  │  Blend   │  │ Impact Physics   │ │
│  │  Filter  │  │ Matcher  │  │  System  │  │                  │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
│                          │                                       │
│                          ▼                                       │
│                  ┌──────────────┐                                │
│                  │  Data Export │                                │
│                  │   (JSON/ZMQ) │                                │
│                  └──────────────┘                                │
└─────────────────────────│────────────────────────────────────────┘
                          │ TCP/ZeroMQ
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Unity Runtime                            │
│                  ┌──────────────┐                                │
│                  │ TrackingBridge│  (C# - receives data)         │
│                  └──────────────┘                                │
│                          │                                       │
│           ┌──────────────┼──────────────┐                       │
│           ▼              ▼              ▼                        │
│  ┌─────────────┐  ┌────────────┐  ┌───────────────┐             │
│  │  Animator   │  │GripConstrain│ │ PhysicsRacket │             │
│  │ (Skeleton)  │  │  (Custom)  │  │  (Rigidbody)  │             │
│  └─────────────┘  └────────────┘  └───────────────┘             │
│           │              │              │                        │
│           └──────────────┴──────────────┘                        │
│                          │                                       │
│                          ▼                                       │
│               [Final Racket Transform]                           │
└─────────────────────────────────────────────────────────────────┘
```

#### Python Data Export Class

```python
import json
import zmq
import struct
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrackingPacket:
    """Data packet sent to Unity"""
    timestamp: float
    frame_id: int
    
    # Racket 6DoF pose
    position: np.ndarray       # [x, y, z]
    rotation: np.ndarray       # [w, x, y, z] quaternion
    
    # Velocities for physics
    velocity: np.ndarray       # [vx, vy, vz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    
    # Confidence and mode
    confidence: float
    tracking_mode: str         # 'vision', 'template', 'blend', 'predicted'
    motion_phase: str          # Current serve phase
    
    # Impact data (if impact detected)
    impact_detected: bool
    impact_normal: Optional[np.ndarray]
    impact_velocity: Optional[np.ndarray]
    
    def to_bytes(self) -> bytes:
        """Serialize for network transmission"""
        # Header: timestamp (double), frame_id (int), flags (int)
        flags = (
            (1 if self.impact_detected else 0) |
            (self._mode_to_int(self.tracking_mode) << 1) |
            (self._phase_to_int(self.motion_phase) << 4)
        )
        
        header = struct.pack('<diI', self.timestamp, self.frame_id, flags)
        
        # Pose: position (3 floats), rotation (4 floats)
        pose = struct.pack('<7f', 
            *self.position, 
            *self.rotation
        )
        
        # Velocity: linear (3), angular (3)
        vel = struct.pack('<6f',
            *self.velocity,
            *self.angular_velocity
        )
        
        # Confidence
        conf = struct.pack('<f', self.confidence)
        
        # Impact data (if present)
        if self.impact_detected:
            impact = struct.pack('<6f',
                *self.impact_normal,
                *self.impact_velocity
            )
        else:
            impact = b''
        
        return header + pose + vel + conf + impact
    
    def to_json(self) -> str:
        """Serialize as JSON (for debugging/logging)"""
        return json.dumps({
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'position': self.position.tolist(),
            'rotation': self.rotation.tolist(),
            'velocity': self.velocity.tolist(),
            'angular_velocity': self.angular_velocity.tolist(),
            'confidence': self.confidence,
            'tracking_mode': self.tracking_mode,
            'motion_phase': self.motion_phase,
            'impact_detected': self.impact_detected,
            'impact_normal': self.impact_normal.tolist() if self.impact_normal is not None else None,
            'impact_velocity': self.impact_velocity.tolist() if self.impact_velocity is not None else None
        })
    
    def _mode_to_int(self, mode: str) -> int:
        return {'vision': 0, 'template': 1, 'blend': 2, 'predicted': 3}.get(mode, 0)
    
    def _phase_to_int(self, phase: str) -> int:
        phases = ['preparation', 'trophy', 'backscratch', 'acceleration', 'impact', 'follow_through']
        return phases.index(phase) if phase in phases else 0


class UnityBridge:
    """Sends tracking data to Unity via ZeroMQ"""
    
    def __init__(self, port: int = 5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        self.frame_id = 0
    
    def send(self, packet: TrackingPacket):
        """Send tracking packet to Unity"""
        self.socket.send(packet.to_bytes())
        self.frame_id += 1
    
    def close(self):
        self.socket.close()
        self.context.term()
```

#### Unity C# Receiver

```csharp
// TrackingBridge.cs
using System;
using System.Runtime.InteropServices;
using UnityEngine;
using NetMQ;
using NetMQ.Sockets;

[Serializable]
public struct TrackingData
{
    public double timestamp;
    public int frameId;
    public Vector3 position;
    public Quaternion rotation;
    public Vector3 velocity;
    public Vector3 angularVelocity;
    public float confidence;
    public TrackingMode mode;
    public MotionPhase phase;
    public bool impactDetected;
    public Vector3 impactNormal;
    public Vector3 impactVelocity;
}

public enum TrackingMode { Vision, Template, Blend, Predicted }
public enum MotionPhase { Preparation, Trophy, Backscratch, Acceleration, Impact, FollowThrough }

public class TrackingBridge : MonoBehaviour
{
    [SerializeField] private string serverAddress = "tcp://localhost:5555";
    [SerializeField] private GripConstraint gripConstraint;
    [SerializeField] private Animator avatarAnimator;
    
    private SubscriberSocket subscriber;
    private TrackingData latestData;
    private bool hasNewData = false;
    
    // For interpolation
    private TrackingData previousData;
    private float interpolationTime = 0f;
    
    void Start()
    {
        AsyncIO.ForceDotNet.Force();
        subscriber = new SubscriberSocket();
        subscriber.Connect(serverAddress);
        subscriber.Subscribe("");
        
        Debug.Log($"TrackingBridge connected to {serverAddress}");
    }
    
    void Update()
    {
        // Receive all pending messages
        while (subscriber.TryReceiveFrameBytes(out byte[] data))
        {
            previousData = latestData;
            latestData = ParseTrackingData(data);
            hasNewData = true;
            interpolationTime = 0f;
        }
        
        if (hasNewData)
        {
            // Interpolate between frames for smooth motion
            interpolationTime += Time.deltaTime;
            float t = Mathf.Clamp01(interpolationTime / 0.033f); // Assuming 30fps source
            
            TrackingData interpolated = InterpolateData(previousData, latestData, t);
            
            // Apply to grip constraint
            ApplyToGripConstraint(interpolated);
        }
    }
    
    private TrackingData ParseTrackingData(byte[] data)
    {
        TrackingData result = new TrackingData();
        int offset = 0;
        
        // Header
        result.timestamp = BitConverter.ToDouble(data, offset); offset += 8;
        result.frameId = BitConverter.ToInt32(data, offset); offset += 4;
        int flags = BitConverter.ToInt32(data, offset); offset += 4;
        
        result.impactDetected = (flags & 1) != 0;
        result.mode = (TrackingMode)((flags >> 1) & 0x7);
        result.phase = (MotionPhase)((flags >> 4) & 0xF);
        
        // Position (convert from right-hand to Unity left-hand)
        float px = BitConverter.ToSingle(data, offset); offset += 4;
        float py = BitConverter.ToSingle(data, offset); offset += 4;
        float pz = BitConverter.ToSingle(data, offset); offset += 4;
        result.position = new Vector3(-px, py, pz);  // Flip X for Unity
        
        // Rotation
        float qw = BitConverter.ToSingle(data, offset); offset += 4;
        float qx = BitConverter.ToSingle(data, offset); offset += 4;
        float qy = BitConverter.ToSingle(data, offset); offset += 4;
        float qz = BitConverter.ToSingle(data, offset); offset += 4;
        result.rotation = new Quaternion(qx, -qy, -qz, qw);  // Convert handedness
        
        // Velocity
        float vx = BitConverter.ToSingle(data, offset); offset += 4;
        float vy = BitConverter.ToSingle(data, offset); offset += 4;
        float vz = BitConverter.ToSingle(data, offset); offset += 4;
        result.velocity = new Vector3(-vx, vy, vz);
        
        // Angular velocity
        float wx = BitConverter.ToSingle(data, offset); offset += 4;
        float wy = BitConverter.ToSingle(data, offset); offset += 4;
        float wz = BitConverter.ToSingle(data, offset); offset += 4;
        result.angularVelocity = new Vector3(wx, -wy, -wz);
        
        // Confidence
        result.confidence = BitConverter.ToSingle(data, offset); offset += 4;
        
        // Impact data
        if (result.impactDetected && data.Length > offset + 24)
        {
            float nx = BitConverter.ToSingle(data, offset); offset += 4;
            float ny = BitConverter.ToSingle(data, offset); offset += 4;
            float nz = BitConverter.ToSingle(data, offset); offset += 4;
            result.impactNormal = new Vector3(-nx, ny, nz);
            
            float ivx = BitConverter.ToSingle(data, offset); offset += 4;
            float ivy = BitConverter.ToSingle(data, offset); offset += 4;
            float ivz = BitConverter.ToSingle(data, offset); offset += 4;
            result.impactVelocity = new Vector3(-ivx, ivy, ivz);
        }
        
        return result;
    }
    
    private TrackingData InterpolateData(TrackingData a, TrackingData b, float t)
    {
        TrackingData result = b;
        result.position = Vector3.Lerp(a.position, b.position, t);
        result.rotation = Quaternion.Slerp(a.rotation, b.rotation, t);
        result.velocity = Vector3.Lerp(a.velocity, b.velocity, t);
        return result;
    }
    
    private void ApplyToGripConstraint(TrackingData data)
    {
        if (gripConstraint != null)
        {
            gripConstraint.SetRacketPose(
                data.position,
                data.rotation,
                data.confidence,
                data.mode
            );
        }
    }
    
    void OnDestroy()
    {
        subscriber?.Close();
        subscriber?.Dispose();
        NetMQConfig.Cleanup();
    }
}
```

#### Grip Constraint Component

```csharp
// GripConstraint.cs
using UnityEngine;
using UnityEngine.Animations.Rigging;

public class GripConstraint : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private Transform racketTransform;
    [SerializeField] private Transform wristBone;
    [SerializeField] private Transform elbowBone;
    [SerializeField] private Rigidbody racketRigidbody;
    
    [Header("Constraint Settings")]
    [SerializeField] private Vector3 gripOffset = new Vector3(0, 0, 0.08f);
    [SerializeField] private float positionBlendSpeed = 15f;
    [SerializeField] private float rotationBlendSpeed = 10f;
    
    [Header("Physics Settings")]
    [SerializeField] private bool useKinematicDuringTracking = true;
    [SerializeField] private float physicsBlendThreshold = 0.3f;
    
    // Internal state
    private Vector3 targetPosition;
    private Quaternion targetRotation;
    private float currentConfidence;
    private TrackingMode currentMode;
    private bool isKinematic = true;
    
    // For smooth transitions
    private Vector3 smoothedPosition;
    private Quaternion smoothedRotation;

    void Start()
    {
        smoothedPosition = racketTransform.position;
        smoothedRotation = racketTransform.rotation;
        
        if (racketRigidbody != null)
        {
            racketRigidbody.isKinematic = true;
        }
    }

    public void SetRacketPose(Vector3 position, Quaternion rotation, 
                              float confidence, TrackingMode mode)
    {
        targetPosition = position;
        targetRotation = rotation;
        currentConfidence = confidence;
        currentMode = mode;
        
        // Decide kinematic vs physics
        UpdatePhysicsMode();
    }

    void LateUpdate()
    {
        // This runs after the Animator, so we can override
        
        if (isKinematic)
        {
            // Smooth tracking
            smoothedPosition = Vector3.Lerp(
                smoothedPosition, 
                targetPosition, 
                Time.deltaTime * positionBlendSpeed
            );
            
            smoothedRotation = Quaternion.Slerp(
                smoothedRotation,
                targetRotation,
                Time.deltaTime * rotationBlendSpeed
            );
            
            // Apply to transform
            racketTransform.position = smoothedPosition;
            racketTransform.rotation = smoothedRotation;
        }
        // If not kinematic, physics engine handles it
    }

    private void UpdatePhysicsMode()
    {
        if (racketRigidbody == null) return;
        
        bool shouldBeKinematic = useKinematicDuringTracking && 
                                  currentConfidence > physicsBlendThreshold;
        
        if (shouldBeKinematic != isKinematic)
        {
            isKinematic = shouldBeKinematic;
            racketRigidbody.isKinematic = isKinematic;
            
            if (!isKinematic)
            {
                // Transitioning to physics - set initial velocity
                racketRigidbody.velocity = Vector3.zero; // Could set from tracking velocity
            }
        }
    }

    // Called by Unity's Animation Rigging if using that system
    public void OnRigUpdate()
    {
        // Additional IK adjustments if needed
        // This integrates with Unity's Animation Rigging package
    }
    
    #if UNITY_EDITOR
    void OnDrawGizmos()
    {
        if (racketTransform != null)
        {
            // Draw grip point
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(racketTransform.position, 0.02f);
            
            // Draw racket face normal
            Gizmos.color = Color.blue;
            Vector3 normal = racketTransform.forward;
            Gizmos.DrawRay(racketTransform.position, normal * 0.3f);
            
            // Draw shaft direction
            Gizmos.color = Color.green;
            Vector3 shaft = racketTransform.up;
            Gizmos.DrawRay(racketTransform.position, shaft * 0.5f);
        }
    }
    #endif
}
```

#### Integration Manager

```csharp
// RacketTrackingManager.cs
using UnityEngine;
using UnityEngine.Events;

public class RacketTrackingManager : MonoBehaviour
{
    [Header("Components")]
    [SerializeField] private TrackingBridge bridge;
    [SerializeField] private GripConstraint gripConstraint;
    [SerializeField] private BallPhysics ballPhysics;
    
    [Header("Events")]
    public UnityEvent<ImpactData> OnBallImpact;
    public UnityEvent<MotionPhase> OnPhaseChange;
    
    private MotionPhase lastPhase;
    
    void Update()
    {
        // Phase change detection
        if (bridge.LatestData.phase != lastPhase)
        {
            lastPhase = bridge.LatestData.phase;
            OnPhaseChange?.Invoke(lastPhase);
        }
        
        // Impact detection
        if (bridge.LatestData.impactDetected)
        {
            HandleImpact(bridge.LatestData);
        }
    }
    
    private void HandleImpact(TrackingData data)
    {
        // Calculate ball physics
        if (ballPhysics != null)
        {
            ImpactData impact = new ImpactData
            {
                contactPoint = data.position,
                racketNormal = data.impactNormal,
                racketVelocity = data.impactVelocity,
                timestamp = (float)data.timestamp
            };
            
            ballPhysics.ApplyImpact(impact);
            OnBallImpact?.Invoke(impact);
        }
    }
}

[System.Serializable]
public struct ImpactData
{
    public Vector3 contactPoint;
    public Vector3 racketNormal;
    public Vector3 racketVelocity;
    public float timestamp;
}
```

---

## Summary

| Question | Solution | Key Components |
|----------|----------|----------------|
| **3 DoF (Supination, Pronation, Flexion)** | Quaternion-based IK with forearm twist extraction | `compute_racket_6dof()`, biomechanical coupling model |
| **Occlusion Recovery (Parallax Corridor)** | Extended Kalman Filter with biomechanical priors | `RacketEKF`, `OcclusionRecoverySystem` |
| **Ball-Racket Impact Physics** | Oblique impact mechanics with friction | `ImpactPhysics.compute_impact()`, normal vector calculation |
| **Unity Integration** | ZeroMQ bridge with kinematic/physics switching | `TrackingBridge.cs`, `GripConstraint.cs` |

### Next Steps

1. **Implement EKF** in Python tracking pipeline (2-3 days)
2. **Add 3DoF extraction** to existing code (1-2 days)
3. **Create Unity bridge** with ZeroMQ (2-3 days)
4. **Integrate impact physics** (2-3 days)
5. **Testing and tuning** (1 week)

Total estimated time: **2-3 weeks** for core implementation, plus testing.

