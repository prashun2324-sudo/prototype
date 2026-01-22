"""
Unity Integration Configuration
================================

Based on Kalpesh's specifications:
- X-axis is FLIPPED (video mirror)
- Z → Forward (to camera)
- Y → Up

Primary Anchor: Right Hand (Continental Grip)
Secondary Anchor: Not mapped (noise creates contradictions)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GripConfig:
    """
    Parameterized grip configuration for racket attachment.
    
    These values define how the racket attaches to the wrist joint.
    Can be adjusted for different grip styles.
    """
    # Offset from wrist to racket handle (meters)
    # Kalpesh's current values: (0.1, 0, 0.03)
    wrist_to_handle: Tuple[float, float, float] = (0.1, 0.0, 0.03)
    
    # Distance from wrist to racket face center (meters)
    # Approximately handle_length + shaft_length + head_length/2
    wrist_to_face_center: float = 0.55
    
    # Grip style rotation offsets (Euler angles in degrees)
    # These rotate the racket relative to wrist orientation
    grip_rotation_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass 
class CoordinateConfig:
    """
    Coordinate system configuration for Apple Vision → Unity.
    
    Apple Vision data has X-axis flipped due to video mirroring.
    """
    # Flip X-axis (video mirror correction)
    flip_x: bool = True
    
    # Unity coordinate system
    # Z → Forward (to camera)
    # Y → Up
    # X → Right (after flip correction)
    forward_axis: str = 'Z'
    up_axis: str = 'Y'


# Predefined grip styles
GRIP_STYLES = {
    'continental': GripConfig(
        wrist_to_handle=(0.1, 0.0, 0.03),
        wrist_to_face_center=0.55,
        grip_rotation_euler=(0, 0, 0)
    ),
    'eastern_forehand': GripConfig(
        wrist_to_handle=(0.1, 0.0, 0.03),
        wrist_to_face_center=0.55,
        grip_rotation_euler=(22.5, 0, 0)  # ~1/8 turn
    ),
    'semi_western': GripConfig(
        wrist_to_handle=(0.1, 0.0, 0.03),
        wrist_to_face_center=0.55,
        grip_rotation_euler=(45, 0, 0)  # ~1/4 turn
    ),
    'western': GripConfig(
        wrist_to_handle=(0.1, 0.0, 0.03),
        wrist_to_face_center=0.55,
        grip_rotation_euler=(67.5, 0, 0)  # ~3/8 turn
    ),
    'eastern_backhand': GripConfig(
        wrist_to_handle=(0.1, 0.0, 0.03),
        wrist_to_face_center=0.55,
        grip_rotation_euler=(-22.5, 0, 0)  # -1/8 turn
    ),
    'two_handed_backhand_right': GripConfig(
        # Right hand (bottom) - Continental
        wrist_to_handle=(0.1, 0.0, 0.03),
        wrist_to_face_center=0.55,
        grip_rotation_euler=(0, 0, 0)
    ),
}


def apply_coordinate_transform(position: np.ndarray, config: CoordinateConfig = None) -> np.ndarray:
    """
    Transform position from Apple Vision to Unity coordinates.
    
    Args:
        position: [x, y, z] in Apple Vision coordinates
        config: Coordinate configuration
        
    Returns:
        Transformed position for Unity
    """
    if config is None:
        config = CoordinateConfig()
    
    result = position.copy()
    
    # Flip X-axis if needed (video mirror correction)
    if config.flip_x:
        result[0] = -result[0]
    
    return result


def apply_coordinate_transform_2d(x: float, y: float, frame_width: float, 
                                  config: CoordinateConfig = None) -> Tuple[float, float]:
    """
    Transform 2D position from Apple Vision to Unity-compatible coordinates.
    
    For 2D keypoints, we flip the X coordinate around the frame center.
    """
    if config is None:
        config = CoordinateConfig()
    
    if config.flip_x:
        x = frame_width - x
    
    return x, y

