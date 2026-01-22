"""
Unity Bridge
Sends tracking data to Unity via ZeroMQ for real-time integration.
Handles coordinate system conversion and data serialization.
"""

import numpy as np
import struct
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False
    print("Warning: ZeroMQ not installed. Install with: pip install pyzmq")

from six_dof_tracker import MotionPhase


class TrackingModeInt(IntEnum):
    """Integer codes for tracking mode"""
    VISION = 0
    TEMPLATE = 1
    BLEND = 2
    PREDICTED = 3


class MotionPhaseInt(IntEnum):
    """Integer codes for motion phase"""
    PREPARATION = 0
    TROPHY = 1
    BACKSCRATCH = 2
    ACCELERATION = 3
    IMPACT = 4
    FOLLOW_THROUGH = 5
    UNKNOWN = 6


@dataclass
class TrackingPacket:
    """
    Data packet sent to Unity.
    
    Binary format (little-endian):
    - Header: timestamp (double, 8), frame_id (int, 4), flags (uint, 4)
    - Pose: position (3 floats, 12), quaternion (4 floats, 16)
    - Velocity: linear (3 floats, 12), angular (3 floats, 12)
    - Confidence: (float, 4)
    - [Optional] Impact: normal (3 floats, 12), velocity (3 floats, 12)
    
    Total: 72 bytes (without impact) or 96 bytes (with impact)
    """
    timestamp: float
    frame_id: int
    
    # 6DoF pose
    position: np.ndarray       # [x, y, z]
    rotation: np.ndarray       # [w, x, y, z] quaternion
    
    # Velocities
    velocity: np.ndarray       # [vx, vy, vz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    
    # Metadata
    confidence: float
    tracking_mode: str         # 'vision', 'template', 'blend', 'predicted'
    motion_phase: MotionPhase
    
    # Optional impact data
    impact_detected: bool = False
    impact_normal: Optional[np.ndarray] = None
    impact_velocity: Optional[np.ndarray] = None
    
    def to_bytes(self) -> bytes:
        """Serialize for network transmission"""
        # Build flags
        mode_int = {
            'vision': 0, 'template': 1, 'blend': 2, 'predicted': 3
        }.get(self.tracking_mode, 0)
        
        phase_int = {
            MotionPhase.PREPARATION: 0,
            MotionPhase.TROPHY: 1,
            MotionPhase.BACKSCRATCH: 2,
            MotionPhase.ACCELERATION: 3,
            MotionPhase.IMPACT: 4,
            MotionPhase.FOLLOW_THROUGH: 5,
            MotionPhase.UNKNOWN: 6
        }.get(self.motion_phase, 6)
        
        flags = (
            (1 if self.impact_detected else 0) |
            (mode_int << 1) |
            (phase_int << 4)
        )
        
        # Pack header
        header = struct.pack('<diI', 
            self.timestamp,
            self.frame_id,
            flags
        )
        
        # Pack pose (convert to Unity coordinate system)
        # Unity: left-handed, Y-up
        # Our system: right-handed, Y-up
        # Conversion: flip X and Z rotation components
        pos = self._to_unity_position(self.position)
        rot = self._to_unity_quaternion(self.rotation)
        
        pose = struct.pack('<7f',
            pos[0], pos[1], pos[2],
            rot[0], rot[1], rot[2], rot[3]
        )
        
        # Pack velocities
        vel = self._to_unity_position(self.velocity)
        ang_vel = self._to_unity_angular_velocity(self.angular_velocity)
        
        velocities = struct.pack('<6f',
            vel[0], vel[1], vel[2],
            ang_vel[0], ang_vel[1], ang_vel[2]
        )
        
        # Pack confidence
        conf = struct.pack('<f', self.confidence)
        
        # Pack impact data if present
        if self.impact_detected and self.impact_normal is not None:
            impact_n = self._to_unity_position(self.impact_normal)
            impact_v = self._to_unity_position(self.impact_velocity) if self.impact_velocity is not None else np.zeros(3)
            
            impact = struct.pack('<6f',
                impact_n[0], impact_n[1], impact_n[2],
                impact_v[0], impact_v[1], impact_v[2]
            )
        else:
            impact = b''
        
        return header + pose + velocities + conf + impact
    
    def to_json(self) -> str:
        """Serialize as JSON (for debugging)"""
        return json.dumps({
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'position': self.position.tolist(),
            'rotation': self.rotation.tolist(),
            'velocity': self.velocity.tolist(),
            'angular_velocity': self.angular_velocity.tolist(),
            'confidence': self.confidence,
            'tracking_mode': self.tracking_mode,
            'motion_phase': self.motion_phase.value,
            'impact_detected': self.impact_detected,
            'impact_normal': self.impact_normal.tolist() if self.impact_normal is not None else None,
            'impact_velocity': self.impact_velocity.tolist() if self.impact_velocity is not None else None
        })
    
    def _to_unity_position(self, pos: np.ndarray) -> np.ndarray:
        """Convert position from right-hand to Unity left-hand"""
        # Flip X axis
        return np.array([-pos[0], pos[1], pos[2]])
    
    def _to_unity_quaternion(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion from right-hand to Unity left-hand"""
        # Unity quaternion: (x, y, z, w)
        # Our quaternion: (w, x, y, z)
        # Also need to flip rotation direction for X and Z
        w, x, y, z = q
        return np.array([x, -y, -z, w])  # Unity format with handedness fix
    
    def _to_unity_angular_velocity(self, w: np.ndarray) -> np.ndarray:
        """Convert angular velocity to Unity"""
        return np.array([w[0], -w[1], -w[2]])


class UnityBridge:
    """
    Sends tracking data to Unity via ZeroMQ.
    
    Usage:
        bridge = UnityBridge(port=5555)
        bridge.start()
        
        # In tracking loop:
        packet = TrackingPacket(...)
        bridge.send(packet)
        
        bridge.stop()
    """
    
    def __init__(self, port: int = 5555, protocol: str = 'tcp'):
        """
        Args:
            port: Port to bind to
            protocol: 'tcp' or 'ipc'
        """
        self.port = port
        self.protocol = protocol
        self.address = f"{protocol}://*:{port}"
        
        self.context = None
        self.socket = None
        self.frame_id = 0
        self.is_running = False
        
        # Statistics
        self.packets_sent = 0
        self.bytes_sent = 0
    
    def start(self):
        """Start the bridge"""
        if not HAS_ZMQ:
            raise RuntimeError("ZeroMQ not installed")
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(self.address)
        self.is_running = True
        
        print(f"UnityBridge started on {self.address}")
    
    def stop(self):
        """Stop the bridge"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.is_running = False
        
        print(f"UnityBridge stopped. Sent {self.packets_sent} packets ({self.bytes_sent} bytes)")
    
    def send(self, packet: TrackingPacket):
        """Send a tracking packet"""
        if not self.is_running:
            return
        
        data = packet.to_bytes()
        self.socket.send(data)
        
        self.packets_sent += 1
        self.bytes_sent += len(data)
        self.frame_id += 1
    
    def send_pose(self,
                  timestamp: float,
                  position: np.ndarray,
                  rotation: np.ndarray,
                  velocity: np.ndarray,
                  angular_velocity: np.ndarray,
                  confidence: float,
                  tracking_mode: str,
                  motion_phase: MotionPhase,
                  impact_detected: bool = False,
                  impact_normal: np.ndarray = None,
                  impact_velocity: np.ndarray = None):
        """Convenience method to send pose data"""
        packet = TrackingPacket(
            timestamp=timestamp,
            frame_id=self.frame_id,
            position=position,
            rotation=rotation,
            velocity=velocity,
            angular_velocity=angular_velocity,
            confidence=confidence,
            tracking_mode=tracking_mode,
            motion_phase=motion_phase,
            impact_detected=impact_detected,
            impact_normal=impact_normal,
            impact_velocity=impact_velocity
        )
        self.send(packet)


class UnityBridgeJSON:
    """
    Alternative bridge using JSON over TCP socket.
    Simpler but higher latency than binary ZMQ.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 5556):
        self.host = host
        self.port = port
        self.socket = None
        self.is_connected = False
    
    def connect(self):
        """Connect to Unity server"""
        import socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        self.is_connected = True
        print(f"Connected to Unity at {self.host}:{self.port}")
    
    def disconnect(self):
        """Disconnect from Unity"""
        if self.socket:
            self.socket.close()
        self.is_connected = False
    
    def send(self, packet: TrackingPacket):
        """Send packet as JSON"""
        if not self.is_connected:
            return
        
        json_data = packet.to_json() + '\n'
        self.socket.sendall(json_data.encode('utf-8'))


# C# code generator for Unity side
UNITY_CSHARP_TEMPLATE = '''
// Auto-generated Unity C# code for TrackingBridge
// Save this as TrackingBridge.cs in your Unity project

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

public enum TrackingMode { Vision = 0, Template = 1, Blend = 2, Predicted = 3 }
public enum MotionPhase { Preparation = 0, Trophy = 1, Backscratch = 2, Acceleration = 3, Impact = 4, FollowThrough = 5, Unknown = 6 }

public class TrackingBridge : MonoBehaviour
{
    [SerializeField] private string serverAddress = "tcp://localhost:5555";
    [SerializeField] private Transform racketTransform;
    
    private SubscriberSocket subscriber;
    private TrackingData latestData;
    private bool hasNewData = false;
    
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
        while (subscriber.TryReceiveFrameBytes(out byte[] data))
        {
            latestData = ParseTrackingData(data);
            hasNewData = true;
        }
        
        if (hasNewData && racketTransform != null)
        {
            racketTransform.position = latestData.position;
            racketTransform.rotation = latestData.rotation;
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
        
        // Position
        float px = BitConverter.ToSingle(data, offset); offset += 4;
        float py = BitConverter.ToSingle(data, offset); offset += 4;
        float pz = BitConverter.ToSingle(data, offset); offset += 4;
        result.position = new Vector3(px, py, pz);
        
        // Rotation (already in Unity format: x, y, z, w)
        float qx = BitConverter.ToSingle(data, offset); offset += 4;
        float qy = BitConverter.ToSingle(data, offset); offset += 4;
        float qz = BitConverter.ToSingle(data, offset); offset += 4;
        float qw = BitConverter.ToSingle(data, offset); offset += 4;
        result.rotation = new Quaternion(qx, qy, qz, qw);
        
        // Velocity
        float vx = BitConverter.ToSingle(data, offset); offset += 4;
        float vy = BitConverter.ToSingle(data, offset); offset += 4;
        float vz = BitConverter.ToSingle(data, offset); offset += 4;
        result.velocity = new Vector3(vx, vy, vz);
        
        // Angular velocity
        float wx = BitConverter.ToSingle(data, offset); offset += 4;
        float wy = BitConverter.ToSingle(data, offset); offset += 4;
        float wz = BitConverter.ToSingle(data, offset); offset += 4;
        result.angularVelocity = new Vector3(wx, wy, wz);
        
        // Confidence
        result.confidence = BitConverter.ToSingle(data, offset); offset += 4;
        
        // Impact data (if present)
        if (result.impactDetected && data.Length > offset + 24)
        {
            float nx = BitConverter.ToSingle(data, offset); offset += 4;
            float ny = BitConverter.ToSingle(data, offset); offset += 4;
            float nz = BitConverter.ToSingle(data, offset); offset += 4;
            result.impactNormal = new Vector3(nx, ny, nz);
            
            float ivx = BitConverter.ToSingle(data, offset); offset += 4;
            float ivy = BitConverter.ToSingle(data, offset); offset += 4;
            float ivz = BitConverter.ToSingle(data, offset); offset += 4;
            result.impactVelocity = new Vector3(ivx, ivy, ivz);
        }
        
        return result;
    }
    
    void OnDestroy()
    {
        subscriber?.Close();
        subscriber?.Dispose();
        NetMQConfig.Cleanup();
    }
}
'''


def generate_unity_code(output_path: str = None) -> str:
    """Generate Unity C# code for the bridge"""
    if output_path:
        with open(output_path, 'w') as f:
            f.write(UNITY_CSHARP_TEMPLATE)
        print(f"Generated Unity code: {output_path}")
    return UNITY_CSHARP_TEMPLATE


if __name__ == "__main__":
    import sys
    
    print("Unity Bridge Test")
    print("=" * 50)
    
    # Generate Unity code
    if '--generate' in sys.argv:
        output = sys.argv[sys.argv.index('--generate') + 1] if len(sys.argv) > sys.argv.index('--generate') + 1 else 'TrackingBridge.cs'
        generate_unity_code(output)
        sys.exit(0)
    
    # Test packet serialization
    print("\nTesting packet serialization...")
    
    packet = TrackingPacket(
        timestamp=1.0,
        frame_id=42,
        position=np.array([1.0, 2.0, 3.0]),
        rotation=np.array([1.0, 0.0, 0.0, 0.0]),
        velocity=np.array([5.0, 0.0, 10.0]),
        angular_velocity=np.array([0.0, 15.0, 0.0]),
        confidence=0.85,
        tracking_mode='vision',
        motion_phase=MotionPhase.ACCELERATION,
        impact_detected=False
    )
    
    binary = packet.to_bytes()
    print(f"Binary size: {len(binary)} bytes")
    
    json_str = packet.to_json()
    print(f"JSON size: {len(json_str)} bytes")
    print(f"JSON: {json_str[:100]}...")
    
    # Test with impact
    packet.impact_detected = True
    packet.impact_normal = np.array([0.0, 0.0, 1.0])
    packet.impact_velocity = np.array([0.0, 0.0, 50.0])
    
    binary_with_impact = packet.to_bytes()
    print(f"\nWith impact - Binary size: {len(binary_with_impact)} bytes")
    
    if HAS_ZMQ and '--run' in sys.argv:
        print("\nStarting bridge (Ctrl+C to stop)...")
        bridge = UnityBridge(port=5555)
        bridge.start()
        
        try:
            import time
            t = 0
            while True:
                # Send test packets
                packet = TrackingPacket(
                    timestamp=t,
                    frame_id=int(t * 30),
                    position=np.array([np.sin(t), 1.5, np.cos(t)]),
                    rotation=np.array([1.0, 0.0, 0.0, 0.0]),
                    velocity=np.array([np.cos(t), 0.0, -np.sin(t)]),
                    angular_velocity=np.array([0.0, 0.0, 0.0]),
                    confidence=0.9,
                    tracking_mode='vision',
                    motion_phase=MotionPhase.PREPARATION
                )
                bridge.send(packet)
                
                time.sleep(0.033)  # ~30fps
                t += 0.033
                
        except KeyboardInterrupt:
            pass
        finally:
            bridge.stop()

