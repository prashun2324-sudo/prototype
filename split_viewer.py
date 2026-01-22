"""
Split Screen Viewer: Video + 3D Mannequin
==========================================

Two adjacent panels:
- Left: Original video with skeleton overlay
- Right: 3D rendered mannequin following the movements
"""

import numpy as np
import json
import cv2
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

sys.path.insert(0, 'src')
from universal_6dof import Universal6DoFTracker, Pose6DoF, Quaternion, create_tracker


class UnifiedDataLoader:
    """Loads tracking data from multiple JSON formats."""
    
    NAME_MAP = {
        'left_shoulder': 'leftShoulder', 'right_shoulder': 'rightShoulder',
        'left_elbow': 'leftElbow', 'right_elbow': 'rightElbow',
        'left_wrist': 'leftWrist', 'right_wrist': 'rightWrist',
        'left_hip': 'leftHip', 'right_hip': 'rightHip',
        'left_knee': 'leftKnee', 'right_knee': 'rightKnee',
        'left_ankle': 'leftAnkle', 'right_ankle': 'rightAnkle',
        'nose': 'nose', 'centerHead': 'centerHead', 'topHead': 'topHead',
        'spine': 'spine', 'root': 'root',
    }
    
    @staticmethod
    def detect_format(data: dict) -> str:
        if 'original_video_info' in data:
            return 'apple_vision'
        return 'legacy'
    
    @staticmethod
    def normalize_name(name: str) -> str:
        return UnifiedDataLoader.NAME_MAP.get(name, name)
    
    @staticmethod
    def extract_keypoints(frame: dict, format_type: str) -> tuple:
        kp2d, kp3d, conf = {}, {}, {}
        
        kp_key = 'keypoints' if format_type == 'apple_vision' else 'keypoints2D'
        for kp in frame.get(kp_key, []):
            name = UnifiedDataLoader.normalize_name(kp['name'])
            score = kp.get('score', 0.5)
            if score > 0.2 and kp['x'] != -1:
                kp2d[name] = (int(kp['x']), int(kp['y']))
                conf[name] = score
        
        for kp in frame.get('keypoints3D', []):
            name = UnifiedDataLoader.normalize_name(kp['name'])
            if kp['x'] != -1:
                kp3d[name] = np.array([kp['x'], kp['y'], kp['z']])
        
        return kp2d, kp3d, conf


class Mannequin3DRenderer:
    """
    Renders a 3D mannequin based on skeleton keypoints.
    Uses OpenCV for rendering (no external 3D libraries needed).
    """
    
    # Body segment definitions
    SEGMENTS = [
        # Torso
        ('leftShoulder', 'rightShoulder', (100, 180, 220), 12),
        ('leftHip', 'rightHip', (100, 180, 220), 12),
        ('leftShoulder', 'leftHip', (100, 180, 220), 10),
        ('rightShoulder', 'rightHip', (100, 180, 220), 10),
        # Left arm
        ('leftShoulder', 'leftElbow', (120, 200, 240), 8),
        ('leftElbow', 'leftWrist', (120, 200, 240), 7),
        # Right arm
        ('rightShoulder', 'rightElbow', (140, 220, 255), 9),
        ('rightElbow', 'rightWrist', (140, 220, 255), 8),
        # Left leg
        ('leftHip', 'leftKnee', (100, 160, 200), 9),
        ('leftKnee', 'leftAnkle', (100, 160, 200), 8),
        # Right leg
        ('rightHip', 'rightKnee', (110, 170, 210), 9),
        ('rightKnee', 'rightAnkle', (110, 170, 210), 8),
    ]
    
    def __init__(self, width: int = 640, height: int = 640):
        self.width = width
        self.height = height
        self.prev_joints = {}
        self.smooth_alpha = 0.7
        
        # Camera parameters for 3D projection
        self.focal_length = 800
        self.camera_distance = 3.0
        
        # View rotation (radians)
        self.view_angle_y = 0.3  # Slight rotation for 3D effect
        
    def project_3d_to_2d(self, point_3d: np.ndarray) -> Tuple[int, int]:
        """
        Project 3D point to 2D screen coordinates.
        Uses perspective projection.
        """
        # Apply view rotation around Y-axis
        cos_a, sin_a = np.cos(self.view_angle_y), np.sin(self.view_angle_y)
        x = point_3d[0] * cos_a - point_3d[2] * sin_a
        z = point_3d[0] * sin_a + point_3d[2] * cos_a
        y = point_3d[1]
        
        # Perspective projection
        z_offset = z + self.camera_distance
        if z_offset < 0.1:
            z_offset = 0.1
        
        scale = self.focal_length / z_offset
        
        # Map to screen (centered)
        screen_x = int(self.width / 2 + x * scale)
        screen_y = int(self.height / 2 - y * scale)  # Y is inverted
        
        return screen_x, screen_y
    
    def smooth_joints(self, joints_3d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply temporal smoothing to reduce jitter."""
        smoothed = {}
        for name, pos in joints_3d.items():
            if name in self.prev_joints:
                smoothed[name] = self.smooth_alpha * pos + (1 - self.smooth_alpha) * self.prev_joints[name]
            else:
                smoothed[name] = pos
        self.prev_joints = smoothed.copy()
        return smoothed
    
    def draw_capsule(self, img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int],
                    radius: int, color: Tuple[int, int, int], depth: float = 0):
        """Draw a capsule-like limb segment."""
        # Adjust color based on depth (darker = further away)
        depth_factor = max(0.5, min(1.0, 1.0 - depth * 0.2))
        adjusted_color = tuple(int(c * depth_factor) for c in color)
        
        # Draw thick line
        cv2.line(img, p1, p2, adjusted_color, radius * 2, cv2.LINE_AA)
        
        # Draw end caps
        cv2.circle(img, p1, radius, adjusted_color, -1, cv2.LINE_AA)
        cv2.circle(img, p2, radius, adjusted_color, -1, cv2.LINE_AA)
        
        # Add highlight
        highlight_color = tuple(min(255, int(c * 1.3)) for c in adjusted_color)
        highlight_offset = max(1, radius // 3)
        cv2.line(img, 
                (p1[0] - highlight_offset, p1[1] - highlight_offset),
                (p2[0] - highlight_offset, p2[1] - highlight_offset),
                highlight_color, max(1, radius // 2), cv2.LINE_AA)
    
    def draw_joint_sphere(self, img: np.ndarray, center: Tuple[int, int], 
                         radius: int, color: Tuple[int, int, int]):
        """Draw a sphere-like joint."""
        # Main sphere
        cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)
        
        # Highlight
        highlight = tuple(min(255, int(c * 1.4)) for c in color)
        highlight_pos = (center[0] - radius // 3, center[1] - radius // 3)
        cv2.circle(img, highlight_pos, radius // 3, highlight, -1, cv2.LINE_AA)
        
        # Shadow edge
        shadow = tuple(int(c * 0.6) for c in color)
        cv2.circle(img, center, radius, shadow, 1, cv2.LINE_AA)
    
    def draw_head(self, img: np.ndarray, neck_pos: Tuple[int, int], 
                 up_direction: np.ndarray, scale: float = 1.0):
        """Draw a head at neck position."""
        head_radius = int(25 * scale)
        
        # Head center is above neck
        head_center = (neck_pos[0], neck_pos[1] - int(head_radius * 1.2))
        
        # Head base color
        head_color = (180, 200, 220)
        
        # Draw head sphere
        cv2.circle(img, head_center, head_radius, head_color, -1, cv2.LINE_AA)
        
        # Highlight
        highlight = (220, 235, 250)
        highlight_pos = (head_center[0] - head_radius // 3, head_center[1] - head_radius // 3)
        cv2.circle(img, highlight_pos, head_radius // 3, highlight, -1, cv2.LINE_AA)
        
        # Shadow edge
        shadow = (120, 140, 160)
        cv2.circle(img, head_center, head_radius, shadow, 2, cv2.LINE_AA)
        
        # Neck
        neck_bottom = neck_pos
        neck_top = (head_center[0], head_center[1] + head_radius - 5)
        cv2.line(img, neck_bottom, neck_top, head_color, 10, cv2.LINE_AA)
    
    def draw_racket(self, img: np.ndarray, wrist_pos: Tuple[int, int],
                   racket_pose: Pose6DoF, scale: float = 1.0):
        """Draw tennis racket at wrist."""
        if racket_pose is None:
            return
        
        # Get orientation vectors
        forward = racket_pose.get_forward_vector()
        up = racket_pose.get_up_vector()
        
        # Handle direction (projected to 2D)
        handle_len = int(60 * scale)
        handle_end = (
            wrist_pos[0] + int(up[0] * handle_len),
            wrist_pos[1] - int(up[1] * handle_len)
        )
        
        # Draw handle
        cv2.line(img, wrist_pos, handle_end, (80, 60, 40), 8, cv2.LINE_AA)
        cv2.line(img, wrist_pos, handle_end, (60, 45, 30), 5, cv2.LINE_AA)
        
        # Racket head center
        head_offset = int(100 * scale)
        head_center = (
            wrist_pos[0] + int(up[0] * head_offset),
            wrist_pos[1] - int(up[1] * head_offset)
        )
        
        # Head dimensions
        major = int(45 * scale)
        minor = int(32 * scale)
        angle = np.degrees(np.arctan2(-up[1], up[0])) - 90
        
        # Frame
        cv2.ellipse(img, head_center, (major, minor), angle, 0, 360, (200, 50, 50), 5, cv2.LINE_AA)
        
        # Strings (filled ellipse with transparency effect)
        cv2.ellipse(img, head_center, (major - 4, minor - 4), angle, 0, 360, (230, 230, 200), -1, cv2.LINE_AA)
        
        # String pattern (simple lines)
        for i in range(-3, 4):
            offset = i * 8
            p1 = (head_center[0] + offset - 10, head_center[1] - minor + 10)
            p2 = (head_center[0] + offset + 10, head_center[1] + minor - 10)
            cv2.line(img, p1, p2, (200, 200, 170), 1, cv2.LINE_AA)
        
        # Face normal indicator
        normal_len = int(40 * scale)
        normal_end = (
            head_center[0] + int(forward[0] * normal_len),
            head_center[1] - int(forward[1] * normal_len)
        )
        cv2.arrowedLine(img, head_center, normal_end, (255, 100, 100), 3, cv2.LINE_AA, tipLength=0.3)
    
    def render(self, joints_3d: Dict[str, np.ndarray], 
              racket_pose: Pose6DoF = None) -> np.ndarray:
        """
        Render the 3D mannequin.
        
        Args:
            joints_3d: Dictionary of joint names to 3D positions
            racket_pose: Optional racket pose for rendering
            
        Returns:
            Rendered image as numpy array
        """
        # Create dark gradient background
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Gradient background
        for y in range(self.height):
            intensity = int(30 + 20 * (y / self.height))
            img[y, :] = (intensity, intensity + 5, intensity + 10)
        
        # Draw grid floor
        floor_y = self.height - 80
        for x in range(0, self.width, 40):
            cv2.line(img, (x, floor_y), (x, self.height), (50, 55, 60), 1)
        for y in range(floor_y, self.height, 20):
            cv2.line(img, (0, y), (self.width, y), (50, 55, 60), 1)
        
        if not joints_3d:
            return img
        
        # Smooth joints
        joints_3d = self.smooth_joints(joints_3d)
        
        # Find body center and scale
        if 'root' in joints_3d:
            center = joints_3d['root']
        elif 'leftHip' in joints_3d and 'rightHip' in joints_3d:
            center = (joints_3d['leftHip'] + joints_3d['rightHip']) / 2
        else:
            center = np.zeros(3)
        
        # Center the body
        centered_joints = {k: v - center for k, v in joints_3d.items()}
        
        # Project to 2D
        joints_2d = {}
        joints_depth = {}
        for name, pos in centered_joints.items():
            joints_2d[name] = self.project_3d_to_2d(pos)
            joints_depth[name] = pos[2]
        
        # Sort segments by depth (draw far ones first)
        sorted_segments = sorted(self.SEGMENTS, 
                                key=lambda s: -(joints_depth.get(s[0], 0) + joints_depth.get(s[1], 0)) / 2)
        
        # Draw body segments
        for start, end, color, thickness in sorted_segments:
            if start in joints_2d and end in joints_2d:
                p1, p2 = joints_2d[start], joints_2d[end]
                depth = (joints_depth.get(start, 0) + joints_depth.get(end, 0)) / 2
                self.draw_capsule(img, p1, p2, thickness, color, depth)
        
        # Draw joints
        joint_colors = {
            'leftShoulder': (150, 200, 250), 'rightShoulder': (150, 200, 250),
            'leftElbow': (130, 180, 230), 'rightElbow': (130, 180, 230),
            'leftWrist': (120, 170, 220), 'rightWrist': (120, 170, 220),
            'leftHip': (130, 180, 220), 'rightHip': (130, 180, 220),
            'leftKnee': (120, 170, 210), 'rightKnee': (120, 170, 210),
            'leftAnkle': (110, 160, 200), 'rightAnkle': (110, 160, 200),
        }
        
        for name, pos in joints_2d.items():
            if name in joint_colors:
                radius = 8 if 'Shoulder' in name or 'Hip' in name else 6
                self.draw_joint_sphere(img, pos, radius, joint_colors[name])
        
        # Draw head
        if 'leftShoulder' in joints_2d and 'rightShoulder' in joints_2d:
            neck_x = (joints_2d['leftShoulder'][0] + joints_2d['rightShoulder'][0]) // 2
            neck_y = (joints_2d['leftShoulder'][1] + joints_2d['rightShoulder'][1]) // 2 - 20
            up_dir = np.array([0, 1, 0])
            self.draw_head(img, (neck_x, neck_y), up_dir)
        
        # Draw racket
        if racket_pose and 'rightWrist' in joints_2d:
            self.draw_racket(img, joints_2d['rightWrist'], racket_pose)
        
        return img


class SplitScreenViewer:
    """
    Main viewer class that combines video and 3D mannequin.
    """
    
    def __init__(self, output_width: int = 1280, output_height: int = 640):
        self.output_width = output_width
        self.output_height = output_height
        self.panel_width = output_width // 2
        
        self.mannequin = Mannequin3DRenderer(self.panel_width, output_height)
        self.tracker = create_tracker(body_height=1.75, fps=30.0)
        
        self.smoothed_kp2d = {}
    
    def smooth_keypoints(self, kp2d: dict, alpha: float = 0.6) -> dict:
        smoothed = {}
        for name, pt in kp2d.items():
            if name in self.smoothed_kp2d:
                prev = self.smoothed_kp2d[name]
                smoothed[name] = (
                    int(alpha * pt[0] + (1-alpha) * prev[0]),
                    int(alpha * pt[1] + (1-alpha) * prev[1])
                )
            else:
                smoothed[name] = pt
        self.smoothed_kp2d = smoothed.copy()
        return smoothed
    
    def draw_video_overlay(self, frame: np.ndarray, kp2d: dict) -> np.ndarray:
        """Draw skeleton overlay on video frame."""
        bones = [
            ('leftShoulder', 'rightShoulder', (255, 200, 100)),
            ('leftHip', 'rightHip', (255, 200, 100)),
            ('leftShoulder', 'leftHip', (255, 200, 100)),
            ('rightShoulder', 'rightHip', (255, 200, 100)),
            ('leftShoulder', 'leftElbow', (255, 255, 100)),
            ('leftElbow', 'leftWrist', (255, 255, 100)),
            ('rightShoulder', 'rightElbow', (100, 200, 255)),
            ('rightElbow', 'rightWrist', (100, 200, 255)),
            ('leftHip', 'leftKnee', (255, 150, 200)),
            ('leftKnee', 'leftAnkle', (255, 150, 200)),
            ('rightHip', 'rightKnee', (200, 150, 255)),
            ('rightKnee', 'rightAnkle', (200, 150, 255)),
        ]
        
        # Draw bones
        for start, end, color in bones:
            if start in kp2d and end in kp2d:
                cv2.line(frame, kp2d[start], kp2d[end], color, 3, cv2.LINE_AA)
        
        # Draw joints
        for name, pt in kp2d.items():
            if any(name in b[:2] for b in bones):
                color = (100, 200, 255) if 'right' in name.lower() else (255, 255, 100)
                cv2.circle(frame, pt, 6, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame
    
    def process_frame(self, video_frame: np.ndarray, kp2d: dict, kp3d: dict,
                     confidences: dict, frame_num: int, total_frames: int) -> np.ndarray:
        """Process one frame and return split-screen output."""
        
        # Smooth 2D keypoints
        kp2d = self.smooth_keypoints(kp2d)
        
        # LEFT PANEL: Video with skeleton overlay
        video_panel = video_frame.copy()
        video_panel = self.draw_video_overlay(video_panel, kp2d)
        
        # Resize video to panel size
        video_panel = cv2.resize(video_panel, (self.panel_width, self.output_height))
        
        # Add label
        cv2.putText(video_panel, "ORIGINAL VIDEO", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Compute racket pose for 3D rendering
        racket_pose = None
        if 'rightShoulder' in kp3d and 'rightWrist' in kp3d:
            shoulder = kp3d['rightShoulder']
            tracker_kp = {
                'shoulder': np.zeros(3),
                'wrist': kp3d.get('rightWrist', np.zeros(3)) - shoulder,
                'elbow': kp3d.get('rightElbow', np.zeros(3)) - shoulder if 'rightElbow' in kp3d else None,
            }
            tracker_conf = {k: confidences.get(f'right{k.capitalize()}', 0.5) for k in tracker_kp}
            
            results = self.tracker.process_keypoints(tracker_kp, tracker_conf, 
                                                     frame_num / 30.0, side='right')
            racket_pose = results.get('racket')
        
        # RIGHT PANEL: 3D Mannequin
        mannequin_panel = self.mannequin.render(kp3d, racket_pose)
        
        # Add label
        cv2.putText(mannequin_panel, "3D MANNEQUIN", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 2)
        
        # Add frame counter
        cv2.putText(mannequin_panel, f"Frame: {frame_num}/{total_frames}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 170, 200), 1)
        
        # Combine panels
        combined = np.hstack([video_panel, mannequin_panel])
        
        # Draw divider line
        cv2.line(combined, (self.panel_width, 0), (self.panel_width, self.output_height),
                (100, 100, 100), 2)
        
        return combined


def find_video_for_json(json_path: str) -> str:
    """Find original video file for JSON."""
    folder = os.path.dirname(json_path)
    for f in os.listdir(folder):
        if '_tracked' in f or '_6dof' in f or 'comparison' in f:
            continue
        if f.startswith('original_') and f.endswith('.mp4'):
            return os.path.join(folder, f)
    for f in os.listdir(folder):
        if '_tracked' in f or '_6dof' in f or 'comparison' in f:
            continue
        if f.endswith('.mp4') or f.endswith('.mov'):
            return os.path.join(folder, f)
    return None


def run_split_viewer(json_path: str, video_path: str = None,
                    start_frame: int = 0, end_frame: int = None):
    """Run split-screen viewer."""
    
    print("=" * 60)
    print("  SPLIT SCREEN: VIDEO + 3D MANNEQUIN")
    print("=" * 60)
    
    if video_path is None:
        video_path = find_video_for_json(json_path)
        if video_path is None:
            print("ERROR: No video found")
            return
    
    print(f"JSON: {json_path}")
    print(f"Video: {video_path}")
    
    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    format_type = UnifiedDataLoader.detect_format(data)
    frames = data.get('frames', [])
    print(f"Format: {format_type}, Frames: {len(frames)}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Frame range
    if end_frame is None:
        end_frame = min(len(frames), total_video_frames) - 1
    start_frame = max(0, start_frame)
    end_frame = min(end_frame, len(frames) - 1, total_video_frames - 1)
    
    # Create viewer
    viewer = SplitScreenViewer(output_width=1280, output_height=640)
    
    # Window
    cv2.namedWindow('Split View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Split View', 1280, 640)
    
    # Output
    output_path = json_path.replace('.json', '_split.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (1280, 640))
    
    current = start_frame
    paused = False
    
    print(f"\nProcessing frames {start_frame} to {end_frame}")
    print("Controls: SPACE=Pause  A/D=Step  Q=Quit\n")
    
    while current <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract keypoints
        kp2d, kp3d, conf = UnifiedDataLoader.extract_keypoints(frames[current], format_type)
        
        # Process
        combined = viewer.process_frame(frame, kp2d, kp3d, conf, current, end_frame)
        
        # Display
        cv2.imshow('Split View', combined)
        out.write(combined)
        
        # Progress
        if current % 30 == 0:
            pct = (current - start_frame) / max(1, end_frame - start_frame) * 100
            print(f"Progress: {pct:.0f}%", end='\r')
        
        # Input
        wait = 0 if paused else max(1, int(1000/fps))
        key = cv2.waitKey(wait) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"\n{'PAUSED' if paused else 'PLAYING'}")
        elif key == ord('a'):
            current = max(start_frame, current - 1)
            paused = True
        elif key == ord('d'):
            current = min(end_frame, current + 1)
            paused = True
        elif not paused:
            current += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n\nSaved: {output_path}")


if __name__ == '__main__':
    data_sources = [
        'vids1/30Fps-1/angles_processed_1768283456.angles.json',
        'vids1/30Fps-2/angles_processed_1768283542.angles.json',
        'vids1/120Fps-1/angles_processed_1768285883.angles.json',
        '2/1/1.json',
    ]
    
    for json_path in data_sources:
        if os.path.exists(json_path):
            print(f"Found: {json_path}")
            if '120Fps' in json_path:
                run_split_viewer(json_path, start_frame=0, end_frame=400)
            else:
                run_split_viewer(json_path)
            break
    else:
        print("No data files found")

