"""
Racket Tracking Solution

Provides accurate 6DoF racket tracking for tennis motion analysis.

Modules:
- six_dof_tracker: Full 6DoF pose computation from body tracking
- ekf_tracker: Extended Kalman Filter for occlusion recovery
- impact_physics: Ball-racket impact physics calculations
- unity_bridge: Real-time data streaming to Unity

Usage:
    from src.six_dof_tracker import SixDoFTracker, BodyFrame
    from src.ekf_tracker import OcclusionRecoverySystem
    from src.impact_physics import ImpactPhysics
    from src.unity_bridge import UnityBridge
"""

from .six_dof_tracker import (
    SixDoFTracker,
    BodyFrame,
    JointAngles,
    RacketPose6DoF,
    MotionPhase,
    load_body_tracking_json
)

from .ekf_tracker import (
    RacketEKF,
    OcclusionRecoverySystem,
    EKFState,
    BiomechanicalPriors
)

from .impact_physics import (
    ImpactPhysics,
    ImpactResult,
    ImpactDetector,
    simulate_serve_impact
)

from .unity_bridge import (
    UnityBridge,
    UnityBridgeJSON,
    TrackingPacket,
    generate_unity_code
)

from .coordinate_transform import (
    CoordinateTransformer,
    RollingShutterCompensator,
    GripSlipDetector,
    HybridFusion
)

from .universal_6dof import (
    Universal6DoFTracker,
    Pose6DoF,
    Quaternion,
    RotationMatrix,
    HumanArmKinematicChain,
    RacketTracker,
    EKF6DoF,
    create_tracker,
    pose_to_dict,
    dict_to_pose
)

__version__ = "1.0.0"
__all__ = [
    # Core trackers
    "SixDoFTracker",
    "RacketEKF", 
    "OcclusionRecoverySystem",
    
    # Data structures
    "BodyFrame",
    "JointAngles",
    "RacketPose6DoF",
    "EKFState",
    "ImpactResult",
    "TrackingPacket",
    
    # Enums
    "MotionPhase",
    
    # Physics
    "ImpactPhysics",
    "ImpactDetector",
    "BiomechanicalPriors",
    
    # Unity integration
    "UnityBridge",
    "UnityBridgeJSON",
    
    # Utilities
    "load_body_tracking_json",
    "simulate_serve_impact",
    "generate_unity_code",
    
    # Coordinate Transform
    "CoordinateTransformer",
    "RollingShutterCompensator",
    "GripSlipDetector",
    "HybridFusion"
]

