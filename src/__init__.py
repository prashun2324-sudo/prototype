"""
6-DoF Racket Lock Solution

Keeps the tennis racket mathematically locked to the player's hand
throughout the stroke. No slip possible.

Core Modules:
- racket_6dof_lock: 6-DoF lock implementation
- motion_occlusion_solver: Generic occlusion handling

Usage:
    from src.racket_6dof_lock import Racket6DoFLock, GripConfig
    from src.motion_occlusion_solver import OcclusionSolver
"""

from .racket_6dof_lock import (
    Racket6DoFLock,
    GripConfig,
    RacketPose,
    MinimumJerkFilter,
    QuaternionFilter,
    process_video_json
)

from .motion_occlusion_solver import (
    OcclusionSolver,
    Result,
    State
)

__version__ = "1.0.0"
__all__ = [
    # Core
    "Racket6DoFLock",
    "OcclusionSolver",
    
    # Config
    "GripConfig",
    
    # Data structures
    "RacketPose",
    "Result",
    "State",
    
    # Filters
    "MinimumJerkFilter",
    "QuaternionFilter",
    
    # Utilities
    "process_video_json"
]
