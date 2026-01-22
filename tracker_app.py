"""
Tennis Racket 6DoF Tracking - GUI Application
==============================================

User-friendly interface with drag-and-drop support.
Shows video and 3D mannequin in split view.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
import json
import numpy as np
import cv2
from PIL import Image, ImageTk

sys.path.insert(0, 'src')


class TrackerGUI:
    """Main GUI Application for 6DoF Tracking."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Tennis Racket 6DoF Tracker")
        self.root.geometry("1400x800")
        self.root.configure(bg='#1a1a2e')
        
        # State
        self.json_path = None
        self.video_path = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.cap = None
        self.data = None
        self.format_type = None
        
        # Import modules (lazy load to show GUI faster)
        self.tracker = None
        self.mannequin = None
        
        self.setup_ui()
        self.setup_drag_drop()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Title bar
        title_frame = tk.Frame(self.root, bg='#16213e', height=60)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🎾 Tennis Racket 6DoF Tracking System",
            font=('Segoe UI', 18, 'bold'),
            fg='#00d4ff',
            bg='#16213e'
        )
        title_label.pack(pady=15)
        
        # Main content area
        content_frame = tk.Frame(self.root, bg='#1a1a2e')
        content_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        control_frame = tk.Frame(content_frame, bg='#0f3460', width=300)
        control_frame.pack(side='left', fill='y', padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Drop zone
        self.drop_frame = tk.Frame(control_frame, bg='#1a1a2e', highlightbackground='#00d4ff',
                                   highlightthickness=2, height=150)
        self.drop_frame.pack(fill='x', padx=15, pady=15)
        self.drop_frame.pack_propagate(False)
        
        self.drop_label = tk.Label(
            self.drop_frame,
            text="📁 Drop JSON File Here\nor Click to Browse",
            font=('Segoe UI', 12),
            fg='#a0a0a0',
            bg='#1a1a2e',
            cursor='hand2'
        )
        self.drop_label.pack(expand=True)
        self.drop_label.bind('<Button-1>', self.browse_file)
        self.drop_frame.bind('<Button-1>', self.browse_file)
        
        # File info
        self.file_label = tk.Label(
            control_frame,
            text="No file loaded",
            font=('Segoe UI', 10),
            fg='#808080',
            bg='#0f3460',
            wraplength=270
        )
        self.file_label.pack(pady=10)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', padx=15, pady=10)
        
        # Playback controls
        control_label = tk.Label(
            control_frame,
            text="Playback Controls",
            font=('Segoe UI', 12, 'bold'),
            fg='#00d4ff',
            bg='#0f3460'
        )
        control_label.pack(pady=(10, 5))
        
        # Buttons frame
        btn_frame = tk.Frame(control_frame, bg='#0f3460')
        btn_frame.pack(pady=10)
        
        btn_style = {'font': ('Segoe UI', 14), 'width': 3, 'bg': '#16213e', 'fg': 'white',
                    'activebackground': '#00d4ff', 'activeforeground': 'black', 'bd': 0}
        
        self.prev_btn = tk.Button(btn_frame, text="⏮", command=self.prev_frame, **btn_style)
        self.prev_btn.pack(side='left', padx=5)
        
        self.play_btn = tk.Button(btn_frame, text="▶", command=self.toggle_play, **btn_style)
        self.play_btn.pack(side='left', padx=5)
        
        self.next_btn = tk.Button(btn_frame, text="⏭", command=self.next_frame, **btn_style)
        self.next_btn.pack(side='left', padx=5)
        
        # Frame slider
        self.frame_var = tk.IntVar(value=0)
        self.frame_slider = ttk.Scale(
            control_frame,
            from_=0, to=100,
            variable=self.frame_var,
            orient='horizontal',
            command=self.on_slider_change
        )
        self.frame_slider.pack(fill='x', padx=15, pady=10)
        
        # Frame counter
        self.frame_counter = tk.Label(
            control_frame,
            text="Frame: 0 / 0",
            font=('Segoe UI', 10),
            fg='#a0a0a0',
            bg='#0f3460'
        )
        self.frame_counter.pack()
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', padx=15, pady=15)
        
        # Info panel
        info_label = tk.Label(
            control_frame,
            text="Tracking Info",
            font=('Segoe UI', 12, 'bold'),
            fg='#00d4ff',
            bg='#0f3460'
        )
        info_label.pack(pady=(5, 10))
        
        self.info_text = tk.Text(
            control_frame,
            height=10,
            font=('Consolas', 9),
            bg='#1a1a2e',
            fg='#a0a0a0',
            bd=0,
            state='disabled'
        )
        self.info_text.pack(fill='x', padx=15)
        
        # Right panel - Video displays
        display_frame = tk.Frame(content_frame, bg='#1a1a2e')
        display_frame.pack(side='right', fill='both', expand=True)
        
        # Video panel
        video_label = tk.Label(
            display_frame,
            text="Original Video",
            font=('Segoe UI', 11, 'bold'),
            fg='#00d4ff',
            bg='#1a1a2e'
        )
        video_label.pack(anchor='w')
        
        self.video_canvas = tk.Canvas(
            display_frame,
            bg='#0a0a0a',
            highlightthickness=1,
            highlightbackground='#333'
        )
        self.video_canvas.pack(fill='both', expand=True, pady=(5, 10))
        
        # 3D panel
        model_label = tk.Label(
            display_frame,
            text="3D Mannequin",
            font=('Segoe UI', 11, 'bold'),
            fg='#00d4ff',
            bg='#1a1a2e'
        )
        model_label.pack(anchor='w')
        
        self.model_canvas = tk.Canvas(
            display_frame,
            bg='#0a0a0a',
            highlightthickness=1,
            highlightbackground='#333'
        )
        self.model_canvas.pack(fill='both', expand=True, pady=5)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready - Drop a JSON file to begin",
            font=('Segoe UI', 9),
            fg='#808080',
            bg='#16213e',
            anchor='w',
            padx=10
        )
        self.status_bar.pack(fill='x', side='bottom')
        
    def setup_drag_drop(self):
        """Setup drag and drop functionality."""
        # Bind drag events to drop zone
        self.drop_frame.drop_target_register = lambda: None
        
        # For Windows, we'll use a different approach
        # Bind keyboard shortcuts
        self.root.bind('<space>', lambda e: self.toggle_play())
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        self.root.bind('<Escape>', lambda e: self.stop_play())
        
    def browse_file(self, event=None):
        """Open file browser."""
        filetypes = [
            ('JSON files', '*.json'),
            ('All files', '*.*')
        ]
        
        filepath = filedialog.askopenfilename(
            title='Select Tracking JSON File',
            filetypes=filetypes,
            initialdir=os.path.dirname(os.path.abspath(__file__))
        )
        
        if filepath:
            self.load_file(filepath)
    
    def load_file(self, json_path):
        """Load JSON and corresponding video file."""
        self.status_bar.config(text="Loading...")
        self.root.update()
        
        try:
            # Load JSON
            with open(json_path, 'r') as f:
                self.data = json.load(f)
            
            self.json_path = json_path
            
            # Detect format
            if 'original_video_info' in self.data:
                self.format_type = 'apple_vision'
            else:
                self.format_type = 'legacy'
            
            # Find video
            folder = os.path.dirname(json_path)
            self.video_path = None
            
            for f in os.listdir(folder):
                if '_tracked' in f or '_6dof' in f or '_split' in f:
                    continue
                if f.startswith('original_') and f.endswith('.mp4'):
                    self.video_path = os.path.join(folder, f)
                    break
            
            if not self.video_path:
                for f in os.listdir(folder):
                    if '_tracked' in f or '_6dof' in f or '_split' in f:
                        continue
                    if f.endswith('.mp4') or f.endswith('.mov'):
                        self.video_path = os.path.join(folder, f)
                        break
            
            if not self.video_path:
                messagebox.showerror("Error", "No video file found in the same folder!")
                return
            
            # Open video
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open video file!")
                return
            
            # Get frame count
            frames_data = self.data.get('frames', [])
            video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_frames = min(len(frames_data), video_frames)
            
            # Update UI
            self.frame_slider.config(to=self.total_frames - 1)
            self.file_label.config(text=f"✓ {os.path.basename(json_path)}\n({self.total_frames} frames)")
            self.drop_label.config(text="✓ File Loaded\nClick to change", fg='#00ff88')
            
            # Initialize tracker and mannequin
            self.init_tracker()
            
            # Show first frame
            self.current_frame = 0
            self.show_frame(0)
            
            self.status_bar.config(text=f"Loaded: {os.path.basename(json_path)} - Press SPACE to play")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            self.status_bar.config(text="Error loading file")
    
    def init_tracker(self):
        """Initialize the tracker and mannequin renderer."""
        from universal_6dof import create_tracker
        from split_viewer import Mannequin3DRenderer
        
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30.0
        self.tracker = create_tracker(body_height=1.75, fps=fps)
        self.mannequin = Mannequin3DRenderer(width=540, height=350)
    
    def extract_keypoints(self, frame_data):
        """Extract keypoints from frame data."""
        NAME_MAP = {
            'left_shoulder': 'leftShoulder', 'right_shoulder': 'rightShoulder',
            'left_elbow': 'leftElbow', 'right_elbow': 'rightElbow',
            'left_wrist': 'leftWrist', 'right_wrist': 'rightWrist',
            'left_hip': 'leftHip', 'right_hip': 'rightHip',
            'left_knee': 'leftKnee', 'right_knee': 'rightKnee',
            'left_ankle': 'leftAnkle', 'right_ankle': 'rightAnkle',
        }
        
        kp2d, kp3d, conf = {}, {}, {}
        
        kp_key = 'keypoints' if self.format_type == 'apple_vision' else 'keypoints2D'
        for kp in frame_data.get(kp_key, []):
            name = NAME_MAP.get(kp['name'], kp['name'])
            score = kp.get('score', 0.5)
            if score > 0.2 and kp['x'] != -1:
                kp2d[name] = (int(kp['x']), int(kp['y']))
                conf[name] = score
        
        for kp in frame_data.get('keypoints3D', []):
            name = NAME_MAP.get(kp['name'], kp['name'])
            if kp['x'] != -1:
                kp3d[name] = np.array([kp['x'], kp['y'], kp['z']])
        
        return kp2d, kp3d, conf
    
    def draw_skeleton(self, frame, kp2d):
        """Draw skeleton on video frame."""
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
        
        for start, end, color in bones:
            if start in kp2d and end in kp2d:
                cv2.line(frame, kp2d[start], kp2d[end], color, 3, cv2.LINE_AA)
        
        for name, pt in kp2d.items():
            if any(name in b[:2] for b in bones):
                color = (100, 200, 255) if 'right' in name.lower() else (255, 255, 100)
                cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
        
        return frame
    
    def show_frame(self, frame_idx):
        """Display a specific frame."""
        if self.cap is None or self.data is None:
            return
        
        frames_data = self.data.get('frames', [])
        if frame_idx >= len(frames_data):
            return
        
        # Read video frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Get keypoints
        kp2d, kp3d, conf = self.extract_keypoints(frames_data[frame_idx])
        
        # Draw skeleton on video
        frame_with_skeleton = self.draw_skeleton(frame.copy(), kp2d)
        
        # Resize and display video
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        if canvas_w > 1 and canvas_h > 1:
            frame_rgb = cv2.cvtColor(frame_with_skeleton, cv2.COLOR_BGR2RGB)
            
            # Fit to canvas
            h, w = frame_rgb.shape[:2]
            scale = min(canvas_w / w, canvas_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            
            img = Image.fromarray(frame_resized)
            self.video_photo = ImageTk.PhotoImage(img)
            self.video_canvas.delete('all')
            self.video_canvas.create_image(canvas_w//2, canvas_h//2, image=self.video_photo)
        
        # Render 3D mannequin
        if self.mannequin and kp3d:
            # Compute racket pose
            racket_pose = None
            if self.tracker and 'rightShoulder' in kp3d and 'rightWrist' in kp3d:
                shoulder = kp3d['rightShoulder']
                tracker_kp = {
                    'shoulder': np.zeros(3),
                    'wrist': kp3d.get('rightWrist', np.zeros(3)) - shoulder,
                }
                if 'rightElbow' in kp3d:
                    tracker_kp['elbow'] = kp3d['rightElbow'] - shoulder
                
                tracker_conf = {'shoulder': 0.8, 'elbow': 0.7, 'wrist': conf.get('rightWrist', 0.5)}
                results = self.tracker.process_keypoints(tracker_kp, tracker_conf, 
                                                        frame_idx / 30.0, side='right')
                racket_pose = results.get('racket')
            
            mannequin_img = self.mannequin.render(kp3d, racket_pose)
            
            # Display mannequin
            canvas_w = self.model_canvas.winfo_width()
            canvas_h = self.model_canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                mannequin_rgb = cv2.cvtColor(mannequin_img, cv2.COLOR_BGR2RGB)
                
                h, w = mannequin_rgb.shape[:2]
                scale = min(canvas_w / w, canvas_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                mannequin_resized = cv2.resize(mannequin_rgb, (new_w, new_h))
                
                img = Image.fromarray(mannequin_resized)
                self.model_photo = ImageTk.PhotoImage(img)
                self.model_canvas.delete('all')
                self.model_canvas.create_image(canvas_w//2, canvas_h//2, image=self.model_photo)
        
        # Update info panel
        self.update_info(frame_idx, kp3d, conf)
        
        # Update frame counter
        self.frame_counter.config(text=f"Frame: {frame_idx + 1} / {self.total_frames}")
        self.frame_var.set(frame_idx)
    
    def update_info(self, frame_idx, kp3d, conf):
        """Update the info text panel."""
        info_lines = [
            f"Frame: {frame_idx}",
            f"Format: {self.format_type}",
            "",
            "Keypoints Detected:",
        ]
        
        for name in ['rightWrist', 'rightElbow', 'rightShoulder']:
            if name in kp3d:
                pos = kp3d[name]
                c = conf.get(name, 0)
                info_lines.append(f"  {name}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] ({c:.0%})")
        
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', '\n'.join(info_lines))
        self.info_text.config(state='disabled')
    
    def toggle_play(self):
        """Toggle play/pause."""
        if self.is_playing:
            self.stop_play()
        else:
            self.start_play()
    
    def start_play(self):
        """Start playback."""
        if self.cap is None:
            return
        
        self.is_playing = True
        self.play_btn.config(text="⏸")
        self.status_bar.config(text="Playing...")
        self.play_loop()
    
    def stop_play(self):
        """Stop playback."""
        self.is_playing = False
        self.play_btn.config(text="▶")
        self.status_bar.config(text="Paused")
    
    def play_loop(self):
        """Playback loop."""
        if not self.is_playing:
            return
        
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self.current_frame = 0
        
        self.show_frame(self.current_frame)
        
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30
        delay = max(1, int(1000 / fps))
        self.root.after(delay, self.play_loop)
    
    def prev_frame(self):
        """Go to previous frame."""
        self.stop_play()
        self.current_frame = max(0, self.current_frame - 1)
        self.show_frame(self.current_frame)
    
    def next_frame(self):
        """Go to next frame."""
        self.stop_play()
        self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
        self.show_frame(self.current_frame)
    
    def on_slider_change(self, value):
        """Handle slider movement."""
        frame = int(float(value))
        if frame != self.current_frame:
            self.current_frame = frame
            self.show_frame(frame)
    
    def run(self):
        """Start the application."""
        self.root.mainloop()
        
        # Cleanup
        if self.cap:
            self.cap.release()


def main():
    """Entry point."""
    app = TrackerGUI()
    app.run()


if __name__ == '__main__':
    main()

