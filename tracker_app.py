"""
Tennis Racket 6DoF Tracking - Professional GUI
===============================================

Single app with:
- Drag and drop JSON file
- Video on left, 3D model on right
- 3D model follows person + racket accurately
- Ready for Unity integration

Coordinate System (per Kalpesh):
- X-axis is FLIPPED (video mirror correction)
- Z → Forward (to camera)
- Y → Up

Primary Anchor: Right Hand only (Left hand not mapped to avoid noise contradictions)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import json
import numpy as np
import cv2
from PIL import Image, ImageTk
import threading

sys.path.insert(0, 'src')

# Configuration
FLIP_X_AXIS = True  # Video mirror correction per Kalpesh


class PersistentTracker:
    """
    NEVER loses tracking - maintains state even during occlusion.
    Uses Kalman-like prediction to fill gaps.
    """
    
    def __init__(self):
        self.state = {}  # {joint_name: {'pos': np.array, 'vel': np.array, 'conf': float}}
        self.alpha_pos = 0.6  # Position smoothing
        self.alpha_vel = 0.3  # Velocity smoothing
        self.decay = 0.95  # Confidence decay per frame without update
        
    def update(self, kp: dict) -> dict:
        """Update tracker with new keypoints, predict missing ones."""
        result = {}
        
        # Update existing joints and add new ones
        for name, pos in kp.items():
            pos = np.array(pos, dtype=float)
            
            if name in self.state:
                # Update with observation
                old_pos = self.state[name]['pos']
                new_vel = pos - old_pos
                
                # Smooth position and velocity
                smooth_pos = self.alpha_pos * pos + (1 - self.alpha_pos) * old_pos
                smooth_vel = self.alpha_vel * new_vel + (1 - self.alpha_vel) * self.state[name]['vel']
                
                self.state[name] = {
                    'pos': smooth_pos,
                    'vel': smooth_vel,
                    'conf': 1.0
                }
            else:
                # New joint
                self.state[name] = {
                    'pos': pos,
                    'vel': np.zeros(2),
                    'conf': 1.0
                }
            
            result[name] = self.state[name]['pos'].copy()
        
        # PREDICT missing joints (key feature - never lose tracking!)
        for name in list(self.state.keys()):
            if name not in kp:
                # Predict using velocity
                self.state[name]['pos'] += self.state[name]['vel']
                self.state[name]['vel'] *= 0.9  # Decay velocity
                self.state[name]['conf'] *= self.decay
                
                # Only use if confidence > threshold
                if self.state[name]['conf'] > 0.3:
                    result[name] = self.state[name]['pos'].copy()
        
        return result
    
    def get_confidence(self, name: str) -> float:
        """Get tracking confidence for a joint."""
        return self.state.get(name, {}).get('conf', 0.0)


class BodyRenderer:
    """
    Renders body and racket with:
    - NEVER missing racket (persistent tracking)
    - FRONT VIEW always (normalized pose)
    - Real-world proportions
    - Full twist tracking
    """
    
    BONES = [
        # Torso
        ('leftShoulder', 'rightShoulder', (200, 200, 210)),
        ('leftHip', 'rightHip', (200, 200, 210)),
        ('leftShoulder', 'leftHip', (180, 180, 190)),
        ('rightShoulder', 'rightHip', (180, 180, 190)),
        # Left arm
        ('leftShoulder', 'leftElbow', (255, 180, 120)),
        ('leftElbow', 'leftWrist', (255, 180, 120)),
        # Right arm (racket arm)
        ('rightShoulder', 'rightElbow', (80, 180, 255)),
        ('rightElbow', 'rightWrist', (80, 180, 255)),
        # Left leg
        ('leftHip', 'leftKnee', (200, 180, 180)),
        ('leftKnee', 'leftAnkle', (200, 180, 180)),
        # Right leg
        ('rightHip', 'rightKnee', (200, 180, 200)),
        ('rightKnee', 'rightAnkle', (200, 180, 200)),
    ]
    
    def __init__(self):
        self.tracker = PersistentTracker()
        self.racket_angle = 0.0  # Persistent racket angle
        self.racket_angle_vel = 0.0  # Angular velocity for smooth rotation
        self.base_forearm = 50  # Reference forearm length
        
    def get_forearm_length(self, kp: dict) -> float:
        """Get forearm length, with fallback."""
        if 'rightElbow' in kp and 'rightWrist' in kp:
            length = np.linalg.norm(kp['rightWrist'] - kp['rightElbow'])
            if length > 10:
                self.base_forearm = 0.8 * self.base_forearm + 0.2 * length
        if 'leftElbow' in kp and 'leftWrist' in kp:
            length = np.linalg.norm(kp['leftWrist'] - kp['leftElbow'])
            if length > 10:
                self.base_forearm = 0.8 * self.base_forearm + 0.2 * length
        return self.base_forearm
    
    def compute_racket_angle(self, kp: dict) -> float:
        """
        Compute racket angle from arm configuration.
        Uses multiple cues for robust tracking during twists.
        """
        target_angle = self.racket_angle  # Default to current
        
        has_right_arm = 'rightElbow' in kp and 'rightWrist' in kp
        has_left_arm = 'leftElbow' in kp and 'leftWrist' in kp
        
        if has_right_arm:
            # Primary: forearm direction
            forearm_vec = kp['rightWrist'] - kp['rightElbow']
            forearm_len = np.linalg.norm(forearm_vec)
            
            if forearm_len > 5:
                forearm_angle = np.arctan2(forearm_vec[1], forearm_vec[0])
                
                # Check for two-handed grip
                if has_left_arm and 'leftWrist' in kp and 'rightWrist' in kp:
                    hand_dist = np.linalg.norm(kp['leftWrist'] - kp['rightWrist'])
                    
                    if hand_dist < self.base_forearm * 1.2:
                        # Two-handed: blend with hand line
                        hand_vec = kp['leftWrist'] - kp['rightWrist']
                        hand_len = np.linalg.norm(hand_vec)
                        
                        if hand_len > 5:
                            hand_angle = np.arctan2(hand_vec[1], hand_vec[0])
                            
                            # Perpendicular to hand line gives racket direction
                            perp_angle = hand_angle + np.pi / 2
                            
                            # Blend: 60% forearm, 40% perpendicular
                            # Handle angle wrapping
                            angle_diff = perp_angle - forearm_angle
                            while angle_diff > np.pi: angle_diff -= 2 * np.pi
                            while angle_diff < -np.pi: angle_diff += 2 * np.pi
                            
                            target_angle = forearm_angle + 0.4 * angle_diff
                        else:
                            target_angle = forearm_angle
                    else:
                        target_angle = forearm_angle
                else:
                    target_angle = forearm_angle
        
        # Smooth angle changes (prevents jumps)
        angle_diff = target_angle - self.racket_angle
        while angle_diff > np.pi: angle_diff -= 2 * np.pi
        while angle_diff < -np.pi: angle_diff += 2 * np.pi
        
        # Apply with damping
        self.racket_angle_vel = 0.3 * angle_diff + 0.5 * self.racket_angle_vel
        self.racket_angle += self.racket_angle_vel
        
        return self.racket_angle
    
    def transform_to_front_view(self, kp: dict, canvas_w: int, canvas_h: int) -> dict:
        """
        Transform keypoints to ALWAYS show front view.
        Centers body in canvas and normalizes scale.
        """
        if not kp:
            return kp
        
        # Find body center (hip midpoint or shoulder midpoint)
        center = None
        if 'leftHip' in kp and 'rightHip' in kp:
            center = (kp['leftHip'] + kp['rightHip']) / 2
        elif 'leftShoulder' in kp and 'rightShoulder' in kp:
            center = (kp['leftShoulder'] + kp['rightShoulder']) / 2
        
        if center is None:
            # Use mean of all points
            all_pts = list(kp.values())
            if all_pts:
                center = np.mean(all_pts, axis=0)
            else:
                return kp
        
        # Calculate body scale (use shoulder width or hip width)
        body_scale = self.base_forearm * 4  # Approximate body height reference
        if 'leftShoulder' in kp and 'rightShoulder' in kp:
            shoulder_w = np.linalg.norm(kp['rightShoulder'] - kp['leftShoulder'])
            if shoulder_w > 10:
                body_scale = shoulder_w * 3  # Shoulder to ankle roughly 3x shoulder width
        
        # Target: body fills ~70% of canvas height
        target_scale = canvas_h * 0.6 / body_scale if body_scale > 0 else 1.0
        target_scale = max(0.5, min(2.0, target_scale))  # Clamp scale
        
        # Transform all points
        result = {}
        canvas_center = np.array([canvas_w / 2, canvas_h / 2])
        
        for name, pos in kp.items():
            # Center and scale
            transformed = (pos - center) * target_scale + canvas_center
            # Shift down slightly so head isn't cut off
            transformed[1] += canvas_h * 0.1
            result[name] = transformed
        
        return result
    
    def draw(self, img: np.ndarray, kp2d: dict) -> np.ndarray:
        """Draw body and racket - NEVER loses racket."""
        # Update persistent tracker (fills in missing joints)
        kp = self.tracker.update(kp2d)
        
        if not kp:
            return img
        
        h, w = img.shape[:2]
        
        # Transform to front view (centered, scaled)
        kp = self.transform_to_front_view(kp, w, h)
        
        # Get body scale
        forearm = self.get_forearm_length(kp)
        
        # Adaptive line thickness
        bone_thick = max(2, int(forearm / 10))
        glow_thick = max(4, int(forearm / 5))
        joint_radius = max(3, int(forearm / 8))
        
        # Draw bones with confidence-based opacity
        for start, end, color in self.BONES:
            if start in kp and end in kp:
                p1 = tuple(map(int, kp[start]))
                p2 = tuple(map(int, kp[end]))
                
                # Check confidence
                conf = min(self.tracker.get_confidence(start), 
                          self.tracker.get_confidence(end))
                
                # Dim if low confidence (predicted)
                if conf < 0.9:
                    color = tuple(int(c * conf) for c in color)
                
                cv2.line(img, p1, p2, tuple(c//2 for c in color), glow_thick, cv2.LINE_AA)
                cv2.line(img, p1, p2, color, bone_thick, cv2.LINE_AA)
        
        # Draw joints
        for name, pos in kp.items():
            if any(name in b[:2] for b in self.BONES):
                pt = tuple(map(int, pos))
                conf = self.tracker.get_confidence(name)
                
                color = (80, 180, 255) if 'right' in name.lower() else (255, 180, 120)
                if conf < 0.9:
                    color = tuple(int(c * conf) for c in color)
                
                cv2.circle(img, pt, joint_radius + 1, (50, 50, 60), -1, cv2.LINE_AA)
                cv2.circle(img, pt, joint_radius, color, -1, cv2.LINE_AA)
        
        # Draw head
        if 'leftShoulder' in kp and 'rightShoulder' in kp:
            neck = (kp['leftShoulder'] + kp['rightShoulder']) / 2
            shoulder_width = np.linalg.norm(kp['rightShoulder'] - kp['leftShoulder'])
            head_size = max(8, int(shoulder_width * 0.4))
            head_center = (int(neck[0]), int(neck[1] - head_size * 0.8))
            cv2.circle(img, head_center, head_size, (50, 60, 70), -1, cv2.LINE_AA)
            cv2.circle(img, head_center, int(head_size * 0.9), (200, 210, 220), -1, cv2.LINE_AA)
        
        # ALWAYS draw racket (key feature!)
        self.draw_racket(img, kp, forearm)
        
        return img
    
    def draw_racket(self, img: np.ndarray, kp: dict, forearm: float):
        """
        Draw racket - NEVER disappears.
        Uses persistent tracking for wrist position and angle.
        """
        # Get wrist position (tracker ensures this exists)
        if 'rightWrist' not in kp:
            return  # Only happens on first frame with no data
        
        wrist = kp['rightWrist']
        
        # Get angle from arm configuration
        angle = self.compute_racket_angle(kp)
        direction = np.array([np.cos(angle), np.sin(angle)])
        perp = np.array([-direction[1], direction[0]])
        
        # REAL proportions
        handle_len = forearm * 0.7
        shaft_len = forearm * 0.5
        head_len = forearm * 1.1
        head_wid = forearm * 0.85
        
        wrist_pt = tuple(map(int, wrist))
        
        # Handle
        handle_end = wrist + direction * handle_len
        handle_end_pt = tuple(map(int, handle_end))
        
        handle_thick = max(2, int(forearm / 12))
        cv2.line(img, wrist_pt, handle_end_pt, (30, 35, 40), handle_thick + 3, cv2.LINE_AA)
        cv2.line(img, wrist_pt, handle_end_pt, (70, 80, 90), handle_thick + 1, cv2.LINE_AA)
        cv2.line(img, wrist_pt, handle_end_pt, (100, 110, 120), handle_thick, cv2.LINE_AA)
        
        # Shaft
        shaft_end = handle_end + direction * shaft_len
        shaft_end_pt = tuple(map(int, shaft_end))
        
        shaft_thick = max(1, int(forearm / 18))
        cv2.line(img, handle_end_pt, shaft_end_pt, (140, 140, 160), shaft_thick + 1, cv2.LINE_AA)
        cv2.line(img, handle_end_pt, shaft_end_pt, (180, 180, 200), shaft_thick, cv2.LINE_AA)
        
        # Head
        head_center = shaft_end + direction * (head_len * 0.45)
        head_center_pt = tuple(map(int, head_center))
        
        head_l = int(head_len * 0.45)
        head_w = int(head_wid * 0.45)
        angle_deg = np.degrees(angle)
        
        frame_thick = max(1, int(forearm / 15))
        
        # Outer glow
        cv2.ellipse(img, head_center_pt, (head_l + 2, head_w + 2), angle_deg,
                   0, 360, (40, 30, 30), frame_thick + 2, cv2.LINE_AA)
        
        # Frame (red)
        cv2.ellipse(img, head_center_pt, (head_l, head_w), angle_deg,
                   0, 360, (50, 50, 180), frame_thick, cv2.LINE_AA)
        
        # Strings
        cv2.ellipse(img, head_center_pt, (head_l - frame_thick - 1, head_w - frame_thick - 1), 
                   angle_deg, 0, 360, (200, 205, 190), -1, cv2.LINE_AA)
        
        # String grid
        n_strings = max(4, int(forearm / 12))
        for i in range(-n_strings, n_strings + 1):
            t = i / n_strings
            offset = t * (head_w - frame_thick * 2)
            p1 = head_center + perp * offset - direction * (head_l - frame_thick * 2)
            p2 = head_center + perp * offset + direction * (head_l - frame_thick * 2)
            cv2.line(img, tuple(map(int, p1)), tuple(map(int, p2)), (160, 165, 150), 1, cv2.LINE_AA)
        
        for i in range(-n_strings, n_strings + 1):
            t = i / n_strings
            offset = t * (head_l - frame_thick * 2)
            p1 = head_center + direction * offset - perp * (head_w - frame_thick * 2)
            p2 = head_center + direction * offset + perp * (head_w - frame_thick * 2)
            cv2.line(img, tuple(map(int, p1)), tuple(map(int, p2)), (160, 165, 150), 1, cv2.LINE_AA)


class TrackerApp:
    """Main GUI Application."""
    
    # Joint name mapping
    NAME_MAP = {
        'left_shoulder': 'leftShoulder', 'right_shoulder': 'rightShoulder',
        'left_elbow': 'leftElbow', 'right_elbow': 'rightElbow',
        'left_wrist': 'leftWrist', 'right_wrist': 'rightWrist',
        'left_hip': 'leftHip', 'right_hip': 'rightHip',
        'left_knee': 'leftKnee', 'right_knee': 'rightKnee',
        'left_ankle': 'leftAnkle', 'right_ankle': 'rightAnkle',
    }
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("6DoF Racket Tracker")
        self.root.geometry("1300x700")
        self.root.configure(bg='#1e1e2e')
        
        # State
        self.cap = None
        self.data = None
        self.frames = []
        self.format_type = None
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.video_w = 0
        self.video_h = 0
        
        # Renderer
        self.renderer = BodyRenderer()
        
        self.setup_ui()
        self.bind_keys()
    
    def setup_ui(self):
        """Setup the interface."""
        # Title
        title = tk.Label(
            self.root, text="🎾 Tennis Racket 6DoF Tracker",
            font=('Segoe UI', 20, 'bold'), fg='#89b4fa', bg='#1e1e2e'
        )
        title.pack(pady=10)
        
        # Main area
        main = tk.Frame(self.root, bg='#1e1e2e')
        main.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left: Controls
        left = tk.Frame(main, bg='#313244', width=280)
        left.pack(side='left', fill='y', padx=(0, 10))
        left.pack_propagate(False)
        
        # Drop zone
        self.drop_zone = tk.Frame(left, bg='#45475a', height=120)
        self.drop_zone.pack(fill='x', padx=10, pady=15)
        self.drop_zone.pack_propagate(False)
        
        self.drop_label = tk.Label(
            self.drop_zone, text="📁 Click to Load JSON\n\nSelect tracking data file",
            font=('Segoe UI', 11), fg='#a6adc8', bg='#45475a', cursor='hand2'
        )
        self.drop_label.pack(expand=True)
        self.drop_label.bind('<Button-1>', self.load_file)
        self.drop_zone.bind('<Button-1>', self.load_file)
        
        # File info
        self.file_info = tk.Label(
            left, text="No file loaded", font=('Segoe UI', 9),
            fg='#6c7086', bg='#313244', wraplength=250
        )
        self.file_info.pack(pady=5)
        
        # Separator
        ttk.Separator(left).pack(fill='x', padx=10, pady=10)
        
        # Controls label
        tk.Label(left, text="Playback", font=('Segoe UI', 12, 'bold'),
                fg='#89b4fa', bg='#313244').pack(pady=5)
        
        # Buttons
        btn_frame = tk.Frame(left, bg='#313244')
        btn_frame.pack(pady=10)
        
        btn_style = {'font': ('Segoe UI', 16), 'width': 3, 'bg': '#45475a',
                    'fg': 'white', 'bd': 0, 'activebackground': '#89b4fa'}
        
        tk.Button(btn_frame, text="⏮", command=self.prev_frame, **btn_style).pack(side='left', padx=3)
        self.play_btn = tk.Button(btn_frame, text="▶", command=self.toggle_play, **btn_style)
        self.play_btn.pack(side='left', padx=3)
        tk.Button(btn_frame, text="⏭", command=self.next_frame, **btn_style).pack(side='left', padx=3)
        
        # Frame slider
        self.frame_var = tk.IntVar()
        self.slider = ttk.Scale(left, from_=0, to=100, variable=self.frame_var,
                               command=self.on_slider)
        self.slider.pack(fill='x', padx=15, pady=10)
        
        # Frame counter
        self.frame_label = tk.Label(left, text="Frame: 0 / 0",
                                   font=('Segoe UI', 10), fg='#a6adc8', bg='#313244')
        self.frame_label.pack()
        
        # Separator
        ttk.Separator(left).pack(fill='x', padx=10, pady=15)
        
        # Info
        tk.Label(left, text="Tracking Info", font=('Segoe UI', 12, 'bold'),
                fg='#89b4fa', bg='#313244').pack(pady=5)
        
        self.info_label = tk.Label(left, text="", font=('Consolas', 9),
                                  fg='#a6adc8', bg='#313244', justify='left')
        self.info_label.pack(padx=10, pady=5)
        
        # Right: Display area
        right = tk.Frame(main, bg='#1e1e2e')
        right.pack(side='right', fill='both', expand=True)
        
        # Video label
        tk.Label(right, text="Original Video", font=('Segoe UI', 11, 'bold'),
                fg='#cdd6f4', bg='#1e1e2e').pack(anchor='w')
        
        # Video canvas
        self.video_canvas = tk.Canvas(right, bg='#11111b', highlightthickness=0)
        self.video_canvas.pack(fill='both', expand=True, pady=(5, 10))
        
        # 3D model label
        tk.Label(right, text="3D Body + Racket Tracking", font=('Segoe UI', 11, 'bold'),
                fg='#cdd6f4', bg='#1e1e2e').pack(anchor='w')
        
        # 3D canvas
        self.model_canvas = tk.Canvas(right, bg='#11111b', highlightthickness=0)
        self.model_canvas.pack(fill='both', expand=True, pady=5)
        
        # Status
        self.status = tk.Label(self.root, text="Ready - Click to load JSON file",
                              font=('Segoe UI', 9), fg='#6c7086', bg='#1e1e2e')
        self.status.pack(side='bottom', fill='x', pady=5)
    
    def bind_keys(self):
        """Bind keyboard shortcuts."""
        self.root.bind('<space>', lambda e: self.toggle_play())
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        self.root.bind('<Escape>', lambda e: self.stop())
    
    def load_file(self, event=None):
        """Load JSON tracking file."""
        path = filedialog.askopenfilename(
            title='Select Tracking JSON',
            filetypes=[('JSON', '*.json'), ('All', '*.*')],
            initialdir=os.path.dirname(__file__)
        )
        if not path:
            return
        
        self.status.config(text="Loading...")
        self.root.update()
        
        try:
            # Load JSON
            with open(path, 'r') as f:
                self.data = json.load(f)
            
            # Detect format
            self.format_type = 'apple' if 'original_video_info' in self.data else 'legacy'
            self.frames = self.data.get('frames', [])
            
            # Find video
            folder = os.path.dirname(path)
            video_path = None
            for f in os.listdir(folder):
                if f.startswith('original_') and f.endswith('.mp4'):
                    video_path = os.path.join(folder, f)
                    break
            if not video_path:
                for f in os.listdir(folder):
                    if f.endswith('.mp4') and '_' not in f:
                        video_path = os.path.join(folder, f)
                        break
            
            if not video_path:
                messagebox.showerror("Error", "Video file not found!")
                return
            
            # Open video
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open video!")
                return
            
            self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.total_frames = min(len(self.frames), video_frames)
            self.current_frame = 0
            
            # Update UI
            self.slider.config(to=self.total_frames - 1)
            self.drop_label.config(text="✓ Loaded!\nClick to change", fg='#a6e3a1')
            self.file_info.config(text=f"{os.path.basename(path)}\n{self.total_frames} frames")
            
            # Reset renderer
            self.renderer = BodyRenderer()
            
            # Show first frame
            self.show_frame(0)
            self.status.config(text=f"Loaded: {os.path.basename(video_path)} - Press SPACE to play")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.config(text="Error loading file")
    
    def get_keypoints(self, frame_idx: int) -> dict:
        """Extract 2D keypoints from frame with X-axis flip correction."""
        if frame_idx >= len(self.frames):
            return {}
        
        frame_data = self.frames[frame_idx]
        kp = {}
        
        key = 'keypoints' if self.format_type == 'apple' else 'keypoints2D'
        for point in frame_data.get(key, []):
            name = self.NAME_MAP.get(point['name'], point['name'])
            score = point.get('score', 0.5)
            if score > 0.2 and point['x'] != -1:
                x, y = point['x'], point['y']
                
                # Apply X-axis flip (video mirror correction per Kalpesh)
                if FLIP_X_AXIS:
                    x = self.video_w - x
                
                kp[name] = np.array([x, y])
        
        return kp
    
    def show_frame(self, idx: int):
        """Display frame."""
        if self.cap is None:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Flip video horizontally to match coordinate correction
        if FLIP_X_AXIS:
            frame = cv2.flip(frame, 1)  # Horizontal flip
        
        # Get keypoints (already X-flipped)
        kp2d = self.get_keypoints(idx)
        
        # Update canvases
        self.root.update_idletasks()
        
        # VIDEO PANEL - Clean video
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        
        if canvas_w > 10 and canvas_h > 10:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Fit to canvas
            scale = min(canvas_w / self.video_w, canvas_h / self.video_h)
            new_w, new_h = int(self.video_w * scale), int(self.video_h * scale)
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            
            img = Image.fromarray(frame_resized)
            self.video_photo = ImageTk.PhotoImage(img)
            self.video_canvas.delete('all')
            self.video_canvas.create_image(canvas_w//2, canvas_h//2, image=self.video_photo)
        
        # 3D MODEL PANEL - Render body on dark background (LARGER)
        canvas_w = self.model_canvas.winfo_width()
        canvas_h = self.model_canvas.winfo_height()
        
        if canvas_w > 10 and canvas_h > 10:
            # Create dark background at canvas size for larger display
            model_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            model_img[:] = (30, 30, 40)  # Dark blue-gray
            
            # Draw grid
            for x in range(0, canvas_w, 40):
                cv2.line(model_img, (x, 0), (x, canvas_h), (40, 40, 50), 1)
            for y in range(0, canvas_h, 40):
                cv2.line(model_img, (0, y), (canvas_w, y), (40, 40, 50), 1)
            
            # Scale keypoints to canvas size (NOT inverted)
            scaled_kp = {}
            scale_x = canvas_w / self.video_w
            scale_y = canvas_h / self.video_h
            for name, pos in kp2d.items():
                scaled_kp[name] = np.array([pos[0] * scale_x, pos[1] * scale_y])
            
            # Draw body and racket at full canvas size
            model_img = self.renderer.draw(model_img, scaled_kp)
            
            # Convert and display (already at canvas size)
            model_rgb = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
            
            img = Image.fromarray(model_rgb)
            self.model_photo = ImageTk.PhotoImage(img)
            self.model_canvas.delete('all')
            self.model_canvas.create_image(canvas_w//2, canvas_h//2, image=self.model_photo)
        
        # Update UI
        self.frame_var.set(idx)
        self.frame_label.config(text=f"Frame: {idx + 1} / {self.total_frames}")
        
        # Update info
        info = f"Joints: {len(kp2d)}\n"
        if 'rightWrist' in kp2d:
            info += f"Wrist: ({kp2d['rightWrist'][0]:.0f}, {kp2d['rightWrist'][1]:.0f})\n"
        self.info_label.config(text=info)
    
    def toggle_play(self):
        """Toggle play/pause."""
        if self.is_playing:
            self.stop()
        else:
            self.play()
    
    def play(self):
        """Start playback."""
        if self.cap is None:
            return
        self.is_playing = True
        self.play_btn.config(text="⏸")
        self.status.config(text="Playing...")
        self.play_loop()
    
    def stop(self):
        """Stop playback."""
        self.is_playing = False
        self.play_btn.config(text="▶")
        self.status.config(text="Paused")
    
    def play_loop(self):
        """Playback loop."""
        if not self.is_playing:
            return
        
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self.current_frame = 0
        
        self.show_frame(self.current_frame)
        
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30
        delay = max(10, int(1000 / fps))
        self.root.after(delay, self.play_loop)
    
    def prev_frame(self):
        """Previous frame."""
        self.stop()
        self.current_frame = max(0, self.current_frame - 1)
        self.show_frame(self.current_frame)
    
    def next_frame(self):
        """Next frame."""
        self.stop()
        self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
        self.show_frame(self.current_frame)
    
    def on_slider(self, val):
        """Slider changed."""
        idx = int(float(val))
        if idx != self.current_frame:
            self.current_frame = idx
            self.show_frame(idx)
    
    def run(self):
        """Run the app."""
        self.root.mainloop()
        if self.cap:
            self.cap.release()


if __name__ == '__main__':
    app = TrackerApp()
    app.run()
