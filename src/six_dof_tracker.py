"""
6DoF Racket Tracker with Full 3 Missing DoFs
Implements:
- Supination/Pronation estimation
- Wrist Flexion/Extension
- Quaternion-based IK
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class JointAngles:
    """Joint angles from body tracking"""
    l_shoulder: float = -1
    r_shoulder: float = -1
    l_elbow: float = -1
    r_elbow: float = -1
    l_wrist: float = -1
    r_wrist: float = -1


@dataclass
class BodyFrame:
    """Single frame of body tracking data"""
    timestamp: float
    keypoints3D: Dict[str, List[float]] = field(default_factory=dict)
    angles: JointAngles = field(default_factory=JointAngles)
    motion_speed: float = 0.0
    hand_tracking_quality: float = 0.5
    
    def get_shoulder_position(self, side: str = 'right') -> Optional[np.ndarray]:
        key = f'{side}Shoulder'
        if key in self.keypoints3D and self.keypoints3D[key][0] != -1:
            return np.array(self.keypoints3D[key])
        return None
    
    def get_elbow_position(self, side: str = 'right') -> Optional[np.ndarray]:
        key = f'{side}Elbow'
        if key in self.keypoints3D and self.keypoints3D[key][0] != -1:
            return np.array(self.keypoints3D[key])
        return None
    
    def get_wrist_position(self, side: str = 'right') -> Optional[np.ndarray]:
        key = f'{side}Wrist'
        if key in self.keypoints3D and self.keypoints3D[key][0] != -1:
            return np.array(self.keypoints3D[key])
        return None


class MotionPhase(Enum):
    PREPARATION = "preparation"
    TROPHY = "trophy"
    BACKSCRATCH = "backscratch"
    ACCELERATION = "acceleration"
    IMPACT = "impact"
    FOLLOW_THROUGH = "follow_through"
    UNKNOWN = "unknown"


@dataclass
class RacketPose6DoF:
    """Full 6DoF racket pose"""
    position: np.ndarray  # [x, y, z] grip point
    orientation: np.ndarray  # Quaternion [w, x, y, z]
    
    # Individual DoF values (for debugging/analysis)
    pronation_deg: float  # Forearm rotation
    flexion_deg: float  # Wrist bend
    deviation_deg: float  # Radial/ulnar deviation
    
    # Derived
    face_normal: np.ndarray  # Racket face normal vector
    shaft_direction: np.ndarray  # Handle to head direction
    
    confidence: float
    phase: MotionPhase


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length"""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


def rotation_matrix_x(angle: float) -> np.ndarray:
    """Rotation matrix about X axis (radians)"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_matrix_y(angle: float) -> np.ndarray:
    """Rotation matrix about Y axis (radians)"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_z(angle: float) -> np.ndarray:
    """Rotation matrix about Z axis (radians)"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def quaternion_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]"""
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


class PronationEstimator:
    """
    Estimates forearm pronation/supination from biomechanical coupling.
    
    Tennis serve pronation pattern:
    - Trophy position: ~0° (neutral)
    - Back-scratch: -45° to -60° (supinated)
    - Acceleration: Rapid pronation (1500-2500 deg/s)
    - Impact: +30° to +45° (pronated)
    - Follow-through: +60° to +90° (full pronation)
    """
    
    # Phase-based pronation angles (degrees)
    PHASE_PRONATION = {
        MotionPhase.PREPARATION: 0.0,
        MotionPhase.TROPHY: 0.0,
        MotionPhase.BACKSCRATCH: -50.0,  # Supinated
        MotionPhase.ACCELERATION: 15.0,   # Transitioning
        MotionPhase.IMPACT: 40.0,         # Pronated
        MotionPhase.FOLLOW_THROUGH: 70.0, # Full pronation
        MotionPhase.UNKNOWN: 0.0
    }
    
    # Peak angular velocities by phase (deg/s)
    PHASE_ANGULAR_VELOCITY = {
        MotionPhase.PREPARATION: 200,
        MotionPhase.TROPHY: 100,
        MotionPhase.BACKSCRATCH: 500,
        MotionPhase.ACCELERATION: 2000,  # Peak pronation speed
        MotionPhase.IMPACT: 1500,
        MotionPhase.FOLLOW_THROUGH: 800,
        MotionPhase.UNKNOWN: 0
    }
    
    def __init__(self):
        self.last_pronation = 0.0
        self.last_timestamp = 0.0
    
    def estimate(self, 
                 elbow_angle: float,
                 motion_speed: float,
                 phase: MotionPhase,
                 timestamp: float) -> float:
        """
        Estimate pronation angle.
        
        Args:
            elbow_angle: Current elbow flexion angle (degrees)
            motion_speed: Overall motion speed from tracking
            phase: Current motion phase
            timestamp: Current time
        
        Returns:
            Estimated pronation angle in degrees (-90 to +90)
            Negative = supinated, Positive = pronated
        """
        # Base angle from phase
        target_pronation = self.PHASE_PRONATION.get(phase, 0.0)
        
        # Coupling with elbow extension
        # During acceleration, elbow extends rapidly which couples with pronation
        if phase == MotionPhase.ACCELERATION:
            # Elbow angle decreases (extends) from ~70° to ~170°
            # Map this to pronation progression
            elbow_extension_progress = (elbow_angle - 70) / 100.0
            elbow_extension_progress = np.clip(elbow_extension_progress, 0, 1)
            
            # Pronation accelerates as elbow extends
            target_pronation = -50 + elbow_extension_progress * 90  # -50° to +40°
        
        # Smooth transition
        dt = timestamp - self.last_timestamp if self.last_timestamp > 0 else 0.033
        max_change = self.PHASE_ANGULAR_VELOCITY.get(phase, 500) * dt
        
        # Limit rate of change
        delta = target_pronation - self.last_pronation
        delta = np.clip(delta, -max_change, max_change)
        
        pronation = self.last_pronation + delta
        pronation = np.clip(pronation, -90, 90)
        
        self.last_pronation = pronation
        self.last_timestamp = timestamp
        
        return pronation


class FlexionEstimator:
    """
    Estimates wrist flexion/extension from tracking angle data.
    
    Apple Vision wrist angle interpretation:
    - 180° = neutral (straight wrist)
    - < 180° = flexion (bent toward palm)
    - > 180° = extension (bent backward)
    """
    
    # Anatomical limits
    MAX_FLEXION = 80.0  # Degrees
    MAX_EXTENSION = 70.0  # Degrees
    
    def __init__(self):
        self.last_flexion = 0.0
    
    def estimate(self, wrist_angle: float) -> float:
        """
        Convert Apple Vision wrist angle to flexion/extension.
        
        Args:
            wrist_angle: Wrist angle from tracking (degrees)
                        -1 if not tracked
        
        Returns:
            Flexion angle (positive = flexion, negative = extension)
        """
        if wrist_angle < 0:
            # Not tracked - return last known
            return self.last_flexion
        
        # Convert: 180° = neutral
        flexion = 180.0 - wrist_angle
        
        # Clamp to anatomical limits
        flexion = np.clip(flexion, -self.MAX_EXTENSION, self.MAX_FLEXION)
        
        self.last_flexion = flexion
        return flexion


class DeviationEstimator:
    """
    Estimates radial/ulnar deviation.
    This is harder to estimate from typical tracking data.
    Uses biomechanical priors based on motion phase.
    """
    
    # Phase-based deviation priors
    PHASE_DEVIATION = {
        MotionPhase.PREPARATION: 0.0,
        MotionPhase.TROPHY: -10.0,  # Slight ulnar
        MotionPhase.BACKSCRATCH: -15.0,  # Ulnar deviation
        MotionPhase.ACCELERATION: 5.0,  # Transitioning to radial
        MotionPhase.IMPACT: 15.0,  # Radial deviation at impact
        MotionPhase.FOLLOW_THROUGH: 10.0,
        MotionPhase.UNKNOWN: 0.0
    }
    
    # Anatomical limits
    MAX_RADIAL = 30.0
    MAX_ULNAR = 20.0
    
    def estimate(self, phase: MotionPhase) -> float:
        """
        Estimate radial/ulnar deviation.
        
        Returns:
            Deviation angle (positive = radial, negative = ulnar)
        """
        deviation = self.PHASE_DEVIATION.get(phase, 0.0)
        return np.clip(deviation, -self.MAX_ULNAR, self.MAX_RADIAL)


class PhaseDetector:
    """Detects current motion phase from tracking data"""
    
    def __init__(self):
        self.last_phase = MotionPhase.UNKNOWN
        self.phase_start_time = 0.0
        self.peak_height_seen = 0.0
    
    def detect(self, 
               frame: BodyFrame,
               side: str = 'right') -> MotionPhase:
        """
        Detect motion phase from body frame.
        
        Uses:
        - Wrist height relative to shoulder
        - Motion speed
        - Temporal continuity
        """
        wrist = frame.get_wrist_position(side)
        shoulder = frame.get_shoulder_position(side)
        
        if wrist is None or shoulder is None:
            return self.last_phase  # Can't detect, keep last
        
        # Relative height
        height_diff = wrist[1] - shoulder[1]
        
        # Update peak height
        if height_diff > self.peak_height_seen:
            self.peak_height_seen = height_diff
        
        # Speed
        speed = frame.motion_speed
        
        # Phase detection logic
        new_phase = self._classify_phase(height_diff, speed, frame.timestamp)
        
        # Enforce phase progression (can't go backwards in serve)
        if self._is_valid_transition(self.last_phase, new_phase):
            self.last_phase = new_phase
            self.phase_start_time = frame.timestamp
        
        return self.last_phase
    
    def _classify_phase(self, height_diff: float, speed: float, timestamp: float) -> MotionPhase:
        """Classify phase based on metrics - tuned for real tennis serve data"""
        
        # PREPARATION: Low speed, arm down
        if speed < 5.0 and height_diff < 0.0:
            return MotionPhase.PREPARATION
        
        # TROPHY: Arm raised high (wrist above shoulder by >30cm)
        if height_diff > 0.3 and speed < 20.0:
            return MotionPhase.TROPHY
        
        # BACKSCRATCH: Arm going back/down with moderate speed
        if height_diff < 0.2 and height_diff > -0.2 and speed > 10.0 and speed < 25.0:
            return MotionPhase.BACKSCRATCH
        
        # ACCELERATION: High speed with arm moving up or forward
        if speed > 20.0:
            if height_diff > 0.0:
                return MotionPhase.ACCELERATION
            else:
                # FOLLOW_THROUGH: High speed, arm coming down
                return MotionPhase.FOLLOW_THROUGH
        
        # IMPACT: Very high speed at extended position
        if speed > 30.0 and height_diff > 0.1:
            return MotionPhase.IMPACT
        
        return self.last_phase
    
    def _is_valid_transition(self, old: MotionPhase, new: MotionPhase) -> bool:
        """Check if phase transition is valid"""
        # Allow any detected phase - the classification logic handles validity
        # This allows the detector to respond to actual motion patterns
        # rather than enforcing a strict sequence
        if new == MotionPhase.UNKNOWN:
            return False  # Don't transition TO unknown
        return True  # Allow any valid phase detection
    
    def reset(self):
        """Reset for new serve"""
        self.last_phase = MotionPhase.UNKNOWN
        self.peak_height_seen = 0.0


class SixDoFTracker:
    """
    Full 6DoF racket tracker using quaternion-based IK.
    
    Combines:
    - 3 DoF position from wrist tracking
    - 3 DoF orientation from:
        - Forearm direction (from elbow-wrist)
        - Pronation estimation
        - Wrist flexion
        - Grip offset
    """
    
    # Racket dimensions
    GRIP_OFFSET = 0.08  # 8cm from wrist to grip center
    RACKET_LENGTH = 0.685  # 68.5cm total length
    
    def __init__(self, grip_style: str = 'continental'):
        """
        Args:
            grip_style: 'eastern', 'continental', 'western'
        """
        self.grip_style = grip_style
        
        # Grip offset angles
        self.grip_offsets = {
            'eastern': 0.0,
            'continental': -15.0,  # Degrees
            'western': 30.0
        }
        
        # Sub-estimators
        self.pronation_estimator = PronationEstimator()
        self.flexion_estimator = FlexionEstimator()
        self.deviation_estimator = DeviationEstimator()
        self.phase_detector = PhaseDetector()
    
    def compute_pose(self,
                     frame: BodyFrame,
                     side: str = 'right') -> Optional[RacketPose6DoF]:
        """
        Compute full 6DoF racket pose.
        
        Args:
            frame: Body tracking frame
            side: 'right' or 'left'
        
        Returns:
            RacketPose6DoF or None if insufficient tracking
        """
        # Get joint positions
        shoulder = frame.get_shoulder_position(side)
        elbow = frame.get_elbow_position(side)
        wrist = frame.get_wrist_position(side)
        
        if shoulder is None or elbow is None or wrist is None:
            return None
        
        # Detect phase
        phase = self.phase_detector.detect(frame, side)
        
        # Get wrist angle
        wrist_angle_attr = f'{side[0]}_wrist'
        wrist_angle = getattr(frame.angles, wrist_angle_attr, -1)
        
        elbow_angle_attr = f'{side[0]}_elbow'
        elbow_angle = getattr(frame.angles, elbow_angle_attr, 120)
        if elbow_angle < 0:
            elbow_angle = 120  # Default
        
        # Estimate DoFs
        pronation_deg = self.pronation_estimator.estimate(
            elbow_angle, frame.motion_speed, phase, frame.timestamp
        )
        flexion_deg = self.flexion_estimator.estimate(wrist_angle)
        deviation_deg = self.deviation_estimator.estimate(phase)
        
        # Build forearm coordinate frame
        Z_forearm = normalize(wrist - elbow)  # Forearm axis
        Y_temp = normalize(elbow - shoulder)  # Upper arm for reference
        X_forearm = normalize(np.cross(Y_temp, Z_forearm))
        Y_forearm = np.cross(Z_forearm, X_forearm)
        
        R_forearm = np.column_stack([X_forearm, Y_forearm, Z_forearm])
        
        # Apply pronation (rotation about forearm axis Z)
        R_pronation = rotation_matrix_z(np.radians(pronation_deg))
        
        # Apply wrist flexion (rotation about X after pronation)
        R_flexion = rotation_matrix_x(np.radians(flexion_deg))
        
        # Apply deviation (rotation about Y)
        R_deviation = rotation_matrix_y(np.radians(deviation_deg))
        
        # Apply grip offset
        grip_offset_deg = self.grip_offsets.get(self.grip_style, 0)
        R_grip = rotation_matrix_x(np.radians(grip_offset_deg))
        
        # Combine: Forearm → Pronation → Flexion → Deviation → Grip
        R_racket = R_forearm @ R_pronation @ R_flexion @ R_deviation @ R_grip
        
        # Compute position (grip point)
        grip_offset_local = np.array([0, 0, self.GRIP_OFFSET])
        grip_offset_world = R_racket @ grip_offset_local
        position = wrist + grip_offset_world
        
        # Convert to quaternion
        orientation = quaternion_from_rotation_matrix(R_racket)
        
        # Extract face normal and shaft direction
        face_normal = R_racket[:, 2]  # Z-axis
        shaft_direction = R_racket[:, 1]  # Y-axis
        
        # Confidence based on tracking quality
        confidence = frame.hand_tracking_quality
        if wrist_angle < 0:
            confidence *= 0.7  # Reduce if wrist not tracked
        
        return RacketPose6DoF(
            position=position,
            orientation=orientation,
            pronation_deg=pronation_deg,
            flexion_deg=flexion_deg,
            deviation_deg=deviation_deg,
            face_normal=face_normal,
            shaft_direction=shaft_direction,
            confidence=confidence,
            phase=phase
        )
    
    def reset(self):
        """Reset tracker state"""
        self.phase_detector.reset()
        self.pronation_estimator.last_pronation = 0.0
        self.flexion_estimator.last_flexion = 0.0


def load_body_tracking_json(json_path: str) -> List[BodyFrame]:
    """Load body tracking data from JSON file"""
    import json
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames = []
    for frame_data in data.get('frames', []):
        angles = JointAngles(
            l_shoulder=frame_data.get('angles', {}).get('L Shoulder', -1),
            r_shoulder=frame_data.get('angles', {}).get('R Shoulder', -1),
            l_elbow=frame_data.get('angles', {}).get('L Elbow', -1),
            r_elbow=frame_data.get('angles', {}).get('R Elbow', -1),
            l_wrist=frame_data.get('angles', {}).get('L Wrist', -1),
            r_wrist=frame_data.get('angles', {}).get('R Wrist', -1)
        )
        
        frame = BodyFrame(
            timestamp=frame_data.get('timestamp', 0.0),
            keypoints3D=frame_data.get('keypoints3D', {}),
            angles=angles,
            motion_speed=frame_data.get('motionSpeed', 0.0),
            hand_tracking_quality=frame_data.get('handTrackingQuality', 0.5)
        )
        frames.append(frame)
    
    return frames


if __name__ == "__main__":
    import sys
    
    print("6DoF Racket Tracker Test")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        body_path = sys.argv[1]
        print(f"Loading: {body_path}")
        
        frames = load_body_tracking_json(body_path)
        tracker = SixDoFTracker(grip_style='continental')
        
        print(f"\nProcessing {len(frames)} frames...")
        
        results = []
        for frame in frames:
            pose = tracker.compute_pose(frame, side='right')
            if pose:
                results.append(pose)
        
        print(f"Successfully computed {len(results)} poses")
        
        if results:
            # Analyze phases
            phase_counts = {}
            for pose in results:
                phase_counts[pose.phase.value] = phase_counts.get(pose.phase.value, 0) + 1
            
            print("\nPhase distribution:")
            for phase, count in sorted(phase_counts.items()):
                print(f"  {phase}: {count} frames ({count/len(results)*100:.1f}%)")
            
            # Pronation range
            pronations = [p.pronation_deg for p in results]
            print(f"\nPronation range: {min(pronations):.1f}° to {max(pronations):.1f}°")
            
            # Flexion range
            flexions = [p.flexion_deg for p in results]
            print(f"Flexion range: {min(flexions):.1f}° to {max(flexions):.1f}°")
        else:
            print("No poses computed - check tracking data quality")
        
    else:
        print("Usage: python six_dof_tracker.py <body_json>")

