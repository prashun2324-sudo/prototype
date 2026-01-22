"""
Universal 6DoF Motion Tracking Visualization
=============================================

Professional overlay visualization using the Universal6DoFTracker.
Works with any pose estimation input (Apple Vision, MediaPipe, OpenPose, etc.)
"""

import numpy as np
import json
import cv2
import sys
import os

sys.path.insert(0, 'src')

from universal_6dof import (
    Universal6DoFTracker, Pose6DoF, Quaternion,
    pose_to_dict, create_tracker
)


class UnifiedDataLoader:
    """Loads tracking data from multiple JSON formats."""
    
    # Joint name mapping
    NAME_MAP = {
        'left_shoulder': 'leftShoulder',
        'right_shoulder': 'rightShoulder',
        'left_elbow': 'leftElbow',
        'right_elbow': 'rightElbow',
        'left_wrist': 'leftWrist',
        'right_wrist': 'rightWrist',
        'left_hip': 'leftHip',
        'right_hip': 'rightHip',
        'left_knee': 'leftKnee',
        'right_knee': 'rightKnee',
        'left_ankle': 'leftAnkle',
        'right_ankle': 'rightAnkle',
    }
    
    @staticmethod
    def detect_format(data: dict) -> str:
        """Detect JSON format type."""
        if 'original_video_info' in data:
            return 'apple_vision'
        elif 'frames' in data:
            first_frame = data['frames'][0] if data['frames'] else {}
            if 'keypoints' in first_frame and 'keypoints2D' not in first_frame:
                return 'apple_vision'
            return 'legacy'
        return 'unknown'
    
    @staticmethod
    def normalize_joint_name(name: str) -> str:
        """Normalize joint name to camelCase format."""
        return UnifiedDataLoader.NAME_MAP.get(name, name)
    
    @staticmethod
    def extract_keypoints(frame: dict, format_type: str) -> tuple:
        """Extract 2D and 3D keypoints from frame data."""
        kp2d = {}
        kp3d = {}
        confidences = {}
        
        # Handle Apple Vision format
        if format_type == 'apple_vision':
            for kp in frame.get('keypoints', []):
                name = UnifiedDataLoader.normalize_joint_name(kp['name'])
                score = kp.get('score', 0)
                if score > 0.3 and kp['x'] != -1:
                    kp2d[name] = (int(kp['x']), int(kp['y']))
                    confidences[name] = score
            
            for kp in frame.get('keypoints3D', []):
                name = UnifiedDataLoader.normalize_joint_name(kp['name'])
                if kp['x'] != -1:
                    kp3d[name] = np.array([kp['x'], kp['y'], kp['z']])
        
        # Handle legacy format
        else:
            for kp in frame.get('keypoints2D', []):
                name = kp['name']
                score = kp.get('score', 0.5)
                if score > 0.3 and kp['x'] != -1:
                    kp2d[name] = (int(kp['x']), int(kp['y']))
                    confidences[name] = score
            
            for kp in frame.get('keypoints3D', []):
                name = kp['name']
                if kp['x'] != -1:
                    kp3d[name] = np.array([kp['x'], kp['y'], kp['z']])
        
        return kp2d, kp3d, confidences


class Professional6DoFOverlay:
    """
    Professional motion tracking overlay with full 6DoF visualization.
    """
    
    BONES = [
        # Torso
        ('leftShoulder', 'rightShoulder', (200, 200, 0), 3),
        ('leftHip', 'rightHip', (200, 200, 0), 3),
        ('leftShoulder', 'leftHip', (200, 200, 0), 2),
        ('rightShoulder', 'rightHip', (200, 200, 0), 2),
        # Left Arm
        ('leftShoulder', 'leftElbow', (255, 255, 0), 3),
        ('leftElbow', 'leftWrist', (255, 255, 0), 3),
        # Right Arm (highlight)
        ('rightShoulder', 'rightElbow', (0, 150, 255), 4),
        ('rightElbow', 'rightWrist', (0, 150, 255), 4),
        # Left Leg
        ('leftHip', 'leftKnee', (255, 100, 180), 2),
        ('leftKnee', 'leftAnkle', (255, 100, 180), 2),
        # Right Leg
        ('rightHip', 'rightKnee', (180, 100, 255), 2),
        ('rightKnee', 'rightAnkle', (180, 100, 255), 2),
    ]
    
    def __init__(self, fps: float = 30.0, body_height: float = 1.75):
        self.tracker = create_tracker(body_height=body_height, fps=fps)
        self.fps = fps
        self.prev_kp2d = {}
        self.smoothed_kp2d = {}
        
    def smooth_keypoints(self, kp2d: dict, alpha: float = 0.6) -> dict:
        """Temporal smoothing for stable visualization."""
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
    
    def draw_skeleton(self, frame: np.ndarray, kp2d: dict) -> np.ndarray:
        """Draw skeleton overlay."""
        overlay = frame.copy()
        
        # Draw glow
        for start, end, color, thickness in self.BONES:
            if start in kp2d and end in kp2d:
                cv2.line(overlay, kp2d[start], kp2d[end], color, thickness + 6, cv2.LINE_AA)
        
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Draw solid bones
        for start, end, color, thickness in self.BONES:
            if start in kp2d and end in kp2d:
                cv2.line(frame, kp2d[start], kp2d[end], color, thickness, cv2.LINE_AA)
        
        # Draw joints
        for name, pt in kp2d.items():
            if any(name in bone[:2] for bone in self.BONES):
                if 'right' in name.lower():
                    color = (0, 150, 255)
                    size = 8
                elif 'left' in name.lower():
                    color = (255, 255, 0)
                    size = 6
                else:
                    color = (200, 200, 0)
                    size = 6
                
                cv2.circle(frame, pt, size + 3, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, pt, size, color, -1, cv2.LINE_AA)
        
        return frame
    
    def draw_racket_6dof(self, frame: np.ndarray, racket_pose: Pose6DoF, 
                         wrist_2d: tuple, frame_size: tuple) -> np.ndarray:
        """
        Draw racket with full 6DoF orientation visualization.
        
        Shows:
        - Racket handle and head
        - Face normal direction
        - Orientation axes (RGB = XYZ)
        """
        if racket_pose is None or wrist_2d is None:
            return frame
        
        h, w = frame_size
        scale = min(h, w) / 5
        
        # Get orientation vectors
        forward = racket_pose.get_forward_vector()  # Face normal (Z)
        up = racket_pose.get_up_vector()            # Racket length (Y)
        right = racket_pose.get_right_vector()      # Racket width (X)
        
        # Project to 2D (simplified orthographic)
        def proj(vec_3d):
            return (int(vec_3d[0] * scale), int(-vec_3d[1] * scale))
        
        # Handle
        handle_end = (
            wrist_2d[0] + int(up[0] * scale * 0.25),
            wrist_2d[1] - int(up[1] * scale * 0.25)
        )
        
        # Handle with glow
        cv2.line(frame, wrist_2d, handle_end, (150, 220, 255), 14, cv2.LINE_AA)
        cv2.line(frame, wrist_2d, handle_end, (80, 120, 180), 8, cv2.LINE_AA)
        cv2.line(frame, wrist_2d, handle_end, (50, 80, 120), 5, cv2.LINE_AA)
        
        # Racket head center
        head_center = (
            wrist_2d[0] + int(up[0] * scale * 0.5),
            wrist_2d[1] - int(up[1] * scale * 0.5)
        )
        
        # Draw ellipse for head
        angle = np.degrees(np.arctan2(-up[1], up[0])) - 90
        major = int(scale * 0.2)
        minor = int(scale * 0.14)
        
        # Outer glow
        cv2.ellipse(frame, head_center, (major + 4, minor + 4), angle,
                   0, 360, (200, 230, 255), 4, cv2.LINE_AA)
        
        # Frame
        cv2.ellipse(frame, head_center, (major, minor), angle,
                   0, 360, (60, 100, 150), 5, cv2.LINE_AA)
        
        # Strings (transparent fill)
        overlay = frame.copy()
        cv2.ellipse(overlay, head_center, (major - 3, minor - 3), angle,
                   0, 360, (220, 240, 250), -1, cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Draw face normal (Z-axis, blue arrow)
        normal_end = (
            head_center[0] + int(forward[0] * scale * 0.3),
            head_center[1] - int(forward[1] * scale * 0.3)
        )
        cv2.arrowedLine(frame, head_center, normal_end, (255, 100, 100), 3, cv2.LINE_AA, tipLength=0.3)
        
        # Draw coordinate axes at head center (smaller)
        axis_scale = scale * 0.1
        
        # X-axis (red)
        x_end = (
            head_center[0] + int(right[0] * axis_scale),
            head_center[1] - int(right[1] * axis_scale)
        )
        cv2.line(frame, head_center, x_end, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Y-axis (green)
        y_end = (
            head_center[0] + int(up[0] * axis_scale),
            head_center[1] - int(up[1] * axis_scale)
        )
        cv2.line(frame, head_center, y_end, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Z-axis (blue) - already drawn as normal
        
        return frame
    
    def draw_info_panel(self, frame: np.ndarray, info: dict) -> np.ndarray:
        """Draw professional info panel with 6DoF data."""
        h, w = frame.shape[:2]
        
        # Background panel
        panel = frame.copy()
        cv2.rectangle(panel, (10, 10), (320, 220), (0, 0, 0), -1)
        frame = cv2.addWeighted(panel, 0.75, frame, 0.25, 0)
        cv2.rectangle(frame, (10, 10), (320, 220), (100, 200, 200), 2)
        
        # Title
        cv2.putText(frame, "UNIVERSAL 6DOF TRACKING", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 255), 2)
        
        y = 55
        
        # Frame info
        cv2.putText(frame, f"Frame: {info.get('frame', 0)}/{info.get('total', 0)}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 20
        
        # Position
        pos = info.get('position', [0, 0, 0])
        cv2.putText(frame, f"Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 18
        
        # Orientation (Euler angles)
        euler = info.get('euler', [0, 0, 0])
        cv2.putText(frame, f"Euler (deg): [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 18
        
        # Face normal
        normal = info.get('face_normal', [0, 0, 1])
        cv2.putText(frame, f"Face Normal: [{normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}]",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 255), 1)
        y += 22
        
        # Velocity
        vel = info.get('velocity', 0)
        vel_bar = min(int(vel * 5), 200)
        cv2.rectangle(frame, (20, y), (220, y + 12), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, y), (20 + vel_bar, y + 12), (0, 200, 255), -1)
        cv2.putText(frame, f"Velocity: {vel:.1f} m/s", (20, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 35
        
        # Confidence
        conf = info.get('confidence', 0)
        conf_bar = int(conf * 200)
        cv2.rectangle(frame, (20, y), (220, y + 12), (50, 50, 50), -1)
        color = (0, int(255 * conf), int(255 * (1-conf)))
        cv2.rectangle(frame, (20, y), (20 + conf_bar, y + 12), color, -1)
        cv2.putText(frame, f"Confidence: {conf:.0%}", (20, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 35
        
        # Occlusion status
        is_occluded = info.get('is_occluded', False)
        status = "OCCLUDED (Predicting)" if is_occluded else "TRACKING"
        status_color = (0, 100, 255) if is_occluded else (0, 255, 100)
        cv2.putText(frame, f"Status: {status}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
        
        return frame
    
    def process_frame(self, video_frame: np.ndarray, kp2d: dict, kp3d: dict,
                     confidences: dict, frame_num: int, total_frames: int,
                     timestamp: float) -> np.ndarray:
        """Process frame with full 6DoF tracking and visualization."""
        h, w = video_frame.shape[:2]
        
        # Smooth keypoints
        kp2d = self.smooth_keypoints(kp2d)
        
        # Draw skeleton
        video_frame = self.draw_skeleton(video_frame, kp2d)
        
        # Prepare keypoints for Universal6DoFTracker
        # Convert to tracker format (shoulder frame)
        tracker_keypoints = {}
        if 'rightShoulder' in kp3d:
            shoulder = kp3d['rightShoulder']
            tracker_keypoints['shoulder'] = np.zeros(3)  # Origin
            
            if 'rightElbow' in kp3d:
                tracker_keypoints['elbow'] = kp3d['rightElbow'] - shoulder
            if 'rightWrist' in kp3d:
                tracker_keypoints['wrist'] = kp3d['rightWrist'] - shoulder
        
        tracker_confidences = {
            'shoulder': confidences.get('rightShoulder', 0.5),
            'elbow': confidences.get('rightElbow', 0.5),
            'wrist': confidences.get('rightWrist', 0.5),
        }
        
        # Process through Universal6DoFTracker
        results = None
        if 'wrist' in tracker_keypoints:
            results = self.tracker.process_keypoints(
                tracker_keypoints, tracker_confidences, timestamp, side='right'
            )
        
        # Draw racket with 6DoF
        wrist_2d = kp2d.get('rightWrist')
        if results and wrist_2d:
            racket_pose = results.get('racket')
            if racket_pose:
                video_frame = self.draw_racket_6dof(
                    video_frame, racket_pose, wrist_2d, (h, w)
                )
        
        # Prepare info for panel
        info = {
            'frame': frame_num,
            'total': total_frames,
            'confidence': confidences.get('rightWrist', 0),
            'is_occluded': results.get('is_occluded', False) if results else True,
        }
        
        if results and 'racket' in results:
            rp = results['racket']
            info['position'] = rp.position.tolist()
            info['velocity'] = np.linalg.norm(rp.velocity)
            info['face_normal'] = rp.get_forward_vector().tolist()
            
            # Convert quaternion to Euler for display
            euler = Quaternion.to_euler(rp.orientation)
            info['euler'] = [np.degrees(e) for e in euler]
        else:
            info['position'] = [0, 0, 0]
            info['velocity'] = 0
            info['face_normal'] = [0, 0, 1]
            info['euler'] = [0, 0, 0]
        
        video_frame = self.draw_info_panel(video_frame, info)
        
        return video_frame


def find_video_for_json(json_path: str) -> str:
    """Find the video file corresponding to a JSON file."""
    folder = os.path.dirname(json_path)
    for f in os.listdir(folder):
        # Skip tracked output videos
        if '_tracked' in f or '_6dof' in f:
            continue
        if f.startswith('original_') and (f.endswith('.mp4') or f.endswith('.mov')):
            return os.path.join(folder, f)
    # Fallback to any video
    for f in os.listdir(folder):
        if '_tracked' in f or '_6dof' in f:
            continue
        if f.endswith('.mp4') or f.endswith('.mov'):
            return os.path.join(folder, f)
    return None


def run_universal_tracker(json_path: str, video_path: str = None,
                         start_frame: int = 0, end_frame: int = None):
    """Run universal 6DoF tracking visualization."""
    
    print("=" * 60)
    print("  UNIVERSAL 6DOF MOTION TRACKING")
    print("=" * 60)
    
    # Find video
    if video_path is None:
        video_path = find_video_for_json(json_path)
        if video_path is None:
            print("ERROR: No video found")
            return
    
    # Load JSON
    print(f"Loading: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    format_type = UnifiedDataLoader.detect_format(data)
    print(f"Format: {format_type}")
    
    frames = data.get('frames', [])
    print(f"Frames: {len(frames)}")
    
    # Open video
    print(f"Video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {orig_w}x{orig_h} @ {fps:.1f} fps")
    
    # Frame range
    if end_frame is None:
        end_frame = min(len(frames), total_video_frames) - 1
    start_frame = max(0, start_frame)
    end_frame = min(end_frame, len(frames) - 1, total_video_frames - 1)
    
    # Create overlay processor
    overlay = Professional6DoFOverlay(fps=fps)
    
    # Window
    cv2.namedWindow('Universal 6DoF Tracking', cv2.WINDOW_NORMAL)
    display_h = 720
    display_w = int(orig_w * display_h / orig_h)
    cv2.resizeWindow('Universal 6DoF Tracking', display_w, display_h)
    
    # Output video
    output_path = json_path.replace('.json', '_6dof.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (orig_w, orig_h))
    
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
        kp2d, kp3d, confidences = UnifiedDataLoader.extract_keypoints(
            frames[current], format_type
        )
        
        timestamp = frames[current].get('timestamp', current / fps)
        
        # Process frame
        processed = overlay.process_frame(
            frame.copy(), kp2d, kp3d, confidences,
            current, end_frame, timestamp
        )
        
        # Display
        display = cv2.resize(processed, (display_w, display_h))
        cv2.imshow('Universal 6DoF Tracking', display)
        
        # Save
        out.write(processed)
        
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
            print(f"\n{'PAUSED' if paused else 'PLAYING'} at frame {current}")
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
    # Priority: new Apple Vision data
    data_sources = [
        'vids1/30Fps-1/angles_processed_1768283456.angles.json',
        'vids1/30Fps-2/angles_processed_1768283542.angles.json',
        'vids1/30Fps-3/angles_processed_1768283629.angles.json',
        'vids1/120Fps-1/angles_processed_1768285883.angles.json',
        '2/1/1.json',
        '2/2/2.json',
    ]
    
    for json_path in data_sources:
        if os.path.exists(json_path):
            print(f"Found: {json_path}")
            
            # Set frame range based on data type
            if '120Fps' in json_path:
                # 120fps - process first 500 frames (~4 sec)
                run_universal_tracker(json_path, start_frame=0, end_frame=500)
            elif 'vids1' in json_path:
                # 30fps Apple Vision - full video
                run_universal_tracker(json_path)
            else:
                # Legacy format - specific range
                run_universal_tracker(json_path, start_frame=440, end_frame=490)
            
            break
    else:
        print("No data files found")
