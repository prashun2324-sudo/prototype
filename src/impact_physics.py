"""
Ball-Racket Impact Physics
Calculates:
- Racket face normal vector
- Ball exit trajectory
- Spin (RPM) at impact
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from six_dof_tracker import quaternion_to_rotation_matrix


@dataclass
class ImpactResult:
    """Result of ball-racket impact calculation"""
    # Exit trajectory
    exit_velocity: np.ndarray  # m/s
    exit_speed: float  # m/s
    exit_direction: np.ndarray  # Unit vector
    
    # Spin
    spin_axis: np.ndarray  # Unit vector
    spin_rpm: float  # Revolutions per minute
    spin_type: str  # 'topspin', 'backspin', 'sidespin', 'flat'
    
    # Contact info
    contact_point: np.ndarray  # World position
    contact_normal: np.ndarray  # Racket face normal at contact
    
    # Energy transfer
    coefficient_of_restitution: float
    energy_transfer_percent: float


class ImpactPhysics:
    """
    Ball-racket impact physics using oblique impact mechanics.
    
    Based on:
    - ITF technical specifications
    - Cross & Lindsey "Technical Tennis" research
    - Brody "Tennis Science for Tennis Players"
    """
    
    # Ball constants (ITF specifications)
    BALL_MASS = 0.057  # kg (56-59.4g)
    BALL_RADIUS = 0.0335  # m (diameter 6.54-6.86cm)
    BALL_MOMENT_OF_INERTIA = (2/3) * BALL_MASS * BALL_RADIUS**2  # Hollow sphere
    
    # Racket constants
    RACKET_MASS = 0.320  # kg (typical strung racket)
    RACKET_SWEET_SPOT_OFFSET = 0.35  # m from grip
    
    # Impact constants
    DEFAULT_COR = 0.75  # Coefficient of restitution (0.73-0.76)
    STRING_FRICTION = 0.4  # Ball-string friction coefficient
    DWELL_TIME = 0.004  # seconds (3-5ms typical)
    
    def __init__(self):
        self.cor = self.DEFAULT_COR
    
    def compute_face_normal(self, racket_quaternion: np.ndarray) -> np.ndarray:
        """
        Compute racket face normal vector from orientation.
        
        Racket coordinate frame:
        - X: Lateral (across face)
        - Y: Shaft direction (handle to head)
        - Z: Face normal (perpendicular to strings)
        
        Args:
            racket_quaternion: [w, x, y, z]
        
        Returns:
            Unit normal vector in world coordinates
        """
        R = quaternion_to_rotation_matrix(racket_quaternion)
        
        # Face normal is Z-axis of racket frame
        face_normal = R[:, 2]
        
        return face_normal / np.linalg.norm(face_normal)
    
    def compute_shaft_direction(self, racket_quaternion: np.ndarray) -> np.ndarray:
        """Get shaft direction (handle to head)"""
        R = quaternion_to_rotation_matrix(racket_quaternion)
        return R[:, 1] / np.linalg.norm(R[:, 1])
    
    def compute_impact(self,
                       ball_position: np.ndarray,
                       ball_velocity: np.ndarray,
                       ball_spin: np.ndarray,  # Angular velocity rad/s
                       racket_position: np.ndarray,
                       racket_quaternion: np.ndarray,
                       racket_velocity: np.ndarray,
                       racket_angular_velocity: np.ndarray,
                       contact_offset: np.ndarray = None) -> ImpactResult:
        """
        Compute ball trajectory and spin after impact.
        
        Uses oblique impact mechanics with friction.
        
        Args:
            ball_position: Ball center position
            ball_velocity: Incoming ball velocity (m/s)
            ball_spin: Incoming ball angular velocity (rad/s)
            racket_position: Racket grip position
            racket_quaternion: Racket orientation [w, x, y, z]
            racket_velocity: Racket velocity at grip (m/s)
            racket_angular_velocity: Racket angular velocity (rad/s)
            contact_offset: Contact point offset from grip in local coords
        
        Returns:
            ImpactResult with exit trajectory and spin
        """
        # Get racket frame
        R_racket = quaternion_to_rotation_matrix(racket_quaternion)
        face_normal = R_racket[:, 2]
        
        # Compute contact point
        if contact_offset is None:
            # Default to sweet spot
            contact_offset = np.array([0, self.RACKET_SWEET_SPOT_OFFSET, 0])
        
        contact_point = racket_position + R_racket @ contact_offset
        
        # Racket velocity at contact point (including rotation)
        r_contact = contact_point - racket_position
        v_racket_contact = racket_velocity + np.cross(racket_angular_velocity, r_contact)
        
        # Relative velocity (ball relative to racket surface)
        v_rel = ball_velocity - v_racket_contact
        
        # Decompose into normal and tangential
        v_normal_mag = np.dot(v_rel, face_normal)
        v_normal = v_normal_mag * face_normal
        v_tangent = v_rel - v_normal
        
        # Effective mass for collision
        m_ball = self.BALL_MASS
        m_racket = self.RACKET_MASS
        m_eff = (m_ball * m_racket) / (m_ball + m_racket)
        
        # Normal impulse (coefficient of restitution)
        J_normal = -(1 + self.cor) * m_eff * v_normal_mag
        
        # Tangential impulse (friction + spin effect)
        # Ball surface velocity due to spin
        ball_surface_vel = np.cross(ball_spin, self.BALL_RADIUS * face_normal)
        
        # Total slip velocity
        v_slip = v_tangent + ball_surface_vel
        v_slip_mag = np.linalg.norm(v_slip)
        
        # Friction impulse (limited by Coulomb friction)
        J_friction_max = self.STRING_FRICTION * abs(J_normal)
        J_friction_required = m_eff * v_slip_mag
        
        if v_slip_mag < 1e-6:
            J_tangent = np.zeros(3)
        elif J_friction_required <= J_friction_max:
            # Full grip (no slip)
            J_tangent = -m_eff * v_slip
        else:
            # Slip (friction limited)
            slip_dir = v_slip / v_slip_mag
            J_tangent = -J_friction_max * slip_dir
        
        # Total impulse
        J_total = J_normal * face_normal + J_tangent
        
        # Ball exit velocity
        v_exit = ball_velocity + J_total / m_ball
        
        # Ball exit spin
        # Torque impulse: τ = r × F
        torque_impulse = np.cross(-self.BALL_RADIUS * face_normal, J_tangent)
        delta_spin = torque_impulse / self.BALL_MOMENT_OF_INERTIA
        spin_exit = ball_spin + delta_spin
        
        # Convert to useful quantities
        exit_speed = np.linalg.norm(v_exit)
        exit_direction = v_exit / exit_speed if exit_speed > 0 else face_normal
        
        spin_magnitude = np.linalg.norm(spin_exit)
        spin_rpm = spin_magnitude * 60 / (2 * np.pi)
        spin_axis = spin_exit / spin_magnitude if spin_magnitude > 0 else np.array([0, 1, 0])
        
        # Classify spin type
        spin_type = self._classify_spin(spin_axis, exit_direction)
        
        # Energy transfer
        ke_in = 0.5 * m_ball * np.dot(ball_velocity, ball_velocity)
        ke_out = 0.5 * m_ball * np.dot(v_exit, v_exit)
        energy_transfer = (ke_out - ke_in) / ke_in if ke_in > 0 else 0
        
        return ImpactResult(
            exit_velocity=v_exit,
            exit_speed=exit_speed,
            exit_direction=exit_direction,
            spin_axis=spin_axis,
            spin_rpm=spin_rpm,
            spin_type=spin_type,
            contact_point=contact_point,
            contact_normal=face_normal,
            coefficient_of_restitution=self.cor,
            energy_transfer_percent=energy_transfer * 100
        )
    
    def _classify_spin(self, spin_axis: np.ndarray, ball_direction: np.ndarray) -> str:
        """
        Classify spin type based on axis orientation.
        
        Spin types:
        - Topspin: Axis horizontal, perpendicular to direction
        - Backspin: Opposite rotation to topspin
        - Sidespin: Axis vertical
        - Flat: Minimal spin
        """
        # Reference vectors
        up = np.array([0, 1, 0])
        forward = ball_direction / np.linalg.norm(ball_direction)
        right = np.cross(up, forward)
        if np.linalg.norm(right) > 1e-6:
            right = right / np.linalg.norm(right)
        else:
            right = np.array([1, 0, 0])
        
        # Project spin axis
        vertical_component = abs(np.dot(spin_axis, up))
        
        if vertical_component > 0.7:
            # Primarily vertical axis = sidespin
            side_dot = np.dot(spin_axis, up)
            return 'sidespin_right' if side_dot > 0 else 'sidespin_left'
        
        # Horizontal spin axis
        topspin_axis = np.cross(forward, up)
        if np.linalg.norm(topspin_axis) > 1e-6:
            topspin_axis = topspin_axis / np.linalg.norm(topspin_axis)
            
            spin_dot = np.dot(spin_axis, topspin_axis)
            if abs(spin_dot) < 0.3:
                return 'flat'
            elif spin_dot > 0:
                return 'topspin'
            else:
                return 'backspin'
        
        return 'flat'
    
    def estimate_serve_spin(self, 
                            serve_type: str,
                            racket_speed: float) -> Tuple[float, np.ndarray]:
        """
        Estimate typical spin for serve types.
        
        Args:
            serve_type: 'flat', 'slice', 'kick', 'topspin'
            racket_speed: Racket head speed at impact (m/s)
        
        Returns:
            (spin_rpm, spin_axis)
        """
        # Typical serve spin values (from research)
        serve_params = {
            'flat': {
                'spin_factor': 0.1,  # Minimal spin
                'axis': np.array([0, 1, 0])  # Minimal
            },
            'slice': {
                'spin_factor': 0.4,  # ~1500-2000 RPM
                'axis': np.array([0, 0.3, 0.95])  # Mostly sidespin
            },
            'kick': {
                'spin_factor': 0.7,  # ~2500-3500 RPM
                'axis': np.array([0.7, 0.3, -0.65])  # Topspin + sidespin
            },
            'topspin': {
                'spin_factor': 0.6,  # ~2000-3000 RPM
                'axis': np.array([1, 0, 0])  # Pure topspin
            }
        }
        
        params = serve_params.get(serve_type, serve_params['flat'])
        
        # Spin increases with racket speed
        # Base: ~40 RPM per m/s of racket speed
        base_rpm = racket_speed * 40 * params['spin_factor']
        
        return base_rpm, params['axis'] / np.linalg.norm(params['axis'])


class ImpactDetector:
    """
    Detects ball-racket impact events from tracking data.
    """
    
    def __init__(self, 
                 impact_distance_threshold: float = 0.1,
                 velocity_threshold: float = 10.0):
        self.impact_distance_threshold = impact_distance_threshold
        self.velocity_threshold = velocity_threshold
        
        self.last_ball_position = None
        self.last_racket_position = None
    
    def check_impact(self,
                     ball_position: np.ndarray,
                     ball_velocity: np.ndarray,
                     racket_position: np.ndarray,
                     racket_face_normal: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if impact occurred this frame.
        
        Returns:
            (is_impact, contact_point)
        """
        # Distance check
        distance = np.linalg.norm(ball_position - racket_position)
        
        if distance > self.impact_distance_threshold:
            self.last_ball_position = ball_position
            self.last_racket_position = racket_position
            return False, None
        
        # Velocity direction check (ball moving toward racket)
        to_racket = racket_position - ball_position
        if np.dot(ball_velocity, to_racket) < 0:
            return False, None
        
        # Speed check
        if np.linalg.norm(ball_velocity) < self.velocity_threshold:
            return False, None
        
        # Impact detected
        # Estimate contact point
        contact_point = ball_position + self.BALL_RADIUS * racket_face_normal
        
        return True, contact_point
    
    BALL_RADIUS = 0.0335


def simulate_serve_impact(serve_type: str = 'flat',
                          racket_speed: float = 50.0,  # m/s
                          ball_toss_height: float = 2.5) -> ImpactResult:
    """
    Simulate a serve impact with typical parameters.
    
    Args:
        serve_type: 'flat', 'slice', 'kick', 'topspin'
        racket_speed: Racket head speed at impact
        ball_toss_height: Ball position height at contact
    
    Returns:
        ImpactResult
    """
    physics = ImpactPhysics()
    
    # Ball state (just after toss apex)
    ball_position = np.array([0, ball_toss_height, 0])
    ball_velocity = np.array([0, -2, 0])  # Falling slowly
    ball_spin = np.array([0, 0, 0])  # No spin on toss
    
    # Racket state at contact
    # Position at contact point
    racket_position = np.array([0, ball_toss_height - 0.35, -0.1])
    
    # Racket moving up and forward
    racket_velocity = np.array([0, racket_speed * 0.3, racket_speed * 0.95])
    
    # Racket rotation (pronation adds spin)
    # Angular velocity depends on serve type
    pronation_rates = {
        'flat': 15,
        'slice': 25,
        'kick': 35,
        'topspin': 30
    }
    pronation_rate = pronation_rates.get(serve_type, 15)
    racket_angular_velocity = np.array([0, 0, pronation_rate])
    
    # Racket orientation (slightly tilted for serve types)
    tilt_angles = {
        'flat': 0,
        'slice': -15,  # Face slightly open
        'kick': 10,    # Face slightly closed
        'topspin': 5
    }
    tilt = np.radians(tilt_angles.get(serve_type, 0))
    
    # Simple rotation matrix for tilt
    c, s = np.cos(tilt), np.sin(tilt)
    R = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    
    # Convert to quaternion
    from six_dof_tracker import quaternion_from_rotation_matrix
    racket_quaternion = quaternion_from_rotation_matrix(R)
    
    # Compute impact
    result = physics.compute_impact(
        ball_position=ball_position,
        ball_velocity=ball_velocity,
        ball_spin=ball_spin,
        racket_position=racket_position,
        racket_quaternion=racket_quaternion,
        racket_velocity=racket_velocity,
        racket_angular_velocity=racket_angular_velocity
    )
    
    return result


if __name__ == "__main__":
    print("Impact Physics Test")
    print("=" * 50)
    
    # Test different serve types
    serve_types = ['flat', 'slice', 'kick', 'topspin']
    
    for serve_type in serve_types:
        print(f"\n{serve_type.upper()} SERVE:")
        print("-" * 30)
        
        result = simulate_serve_impact(serve_type=serve_type)
        
        print(f"  Exit speed: {result.exit_speed:.1f} m/s ({result.exit_speed * 2.237:.0f} mph)")
        print(f"  Exit direction: [{result.exit_direction[0]:.2f}, {result.exit_direction[1]:.2f}, {result.exit_direction[2]:.2f}]")
        print(f"  Spin: {result.spin_rpm:.0f} RPM ({result.spin_type})")
        print(f"  Face normal: [{result.contact_normal[0]:.2f}, {result.contact_normal[1]:.2f}, {result.contact_normal[2]:.2f}]")
    
    # Test face normal calculation
    print("\n\nFACE NORMAL TEST:")
    print("-" * 30)
    
    physics = ImpactPhysics()
    
    # Identity quaternion (racket facing +Z)
    q_identity = np.array([1, 0, 0, 0])
    normal = physics.compute_face_normal(q_identity)
    print(f"Identity rotation -> Normal: {normal}")
    
    # 90° rotation about Y (racket facing +X)
    q_90y = np.array([0.707, 0, 0.707, 0])
    normal = physics.compute_face_normal(q_90y)
    print(f"90° Y rotation -> Normal: {normal}")

