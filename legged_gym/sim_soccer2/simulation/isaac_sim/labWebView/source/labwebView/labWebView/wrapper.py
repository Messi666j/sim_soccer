import gymnasium as gym
import numpy as np
from io import BytesIO
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from PIL import Image
import base64
import os
import threading

from isaaclab.envs.ui import ViewportCameraController

from labWebView import LABWEBVIEW_TEMPLATE_DIR

app = Flask(__name__, template_folder=LABWEBVIEW_TEMPLATE_DIR)

socketio = SocketIO(app, cors_allowed_origins="*")

from dataclasses import dataclass

@dataclass
class MsgBuffer:
    next_env: bool
    reset_env: bool = False
    switch_debug: bool = False
    cur_debug = False
    env_id: int = -1
    v_cmd: tuple = None
    viewer_point = None
    viewer_look_at = None
    teleport_cmd = None
    camera_preset = None
    restart_match = None # Added for dynamic restart
    initial_positions: dict = None  # {robot_name: [x, y, theta]}
    apply_initial_positions: bool = False  # Flag to apply after reset
    
msg_buffer = MsgBuffer(False)

@app.route('/')
def index():
    return render_template("index.html")

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    
@socketio.on('next_env')
def handle_next_env():
    print("[Web Viewer] Received next_env trigger from frontend.")
    msg_buffer.next_env = True

@socketio.on('reset_env')
def handle_reset_env():
    print("[Web Viewer] Received reset_env trigger from frontend.")
    msg_buffer.reset_env = True
    # Reset velocity command to 0
    msg_buffer.v_cmd = ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    
    # If initial positions are set, apply them after reset
    if msg_buffer.initial_positions:
        msg_buffer.apply_initial_positions = True

@socketio.on('set_initial_positions')
def handle_set_initial_positions(data):
    """Store spawn positions: {robot_name: [x, y, theta]}"""
    msg_buffer.initial_positions = data
    print(f"[Web Viewer] Stored initial positions for {len(data)} entities")

@socketio.on('restart_match')
def handle_restart_match(data):
    print(f"[Web Viewer] Restart Match Requested with config: {data}")
    msg_buffer.restart_match = data # Store config dict

@socketio.on('switch_debug')
def handle_switch_debug():
    print("[Web Viewer] Received switch_debug trigger from frontend.")
    msg_buffer.switch_debug = True
    
@socketio.on('set_env_id')
def handle_set_env_id(data):
    env_id = data.get('id')
    print(f"[Web Viewer] Received env_id from frontend: {env_id}")
    msg_buffer.env_id = env_id

@socketio.on("set_velocity_range")
def handle_velocity_command(data):
    vx = data.get("vx", [0, 0])
    vy = data.get("vy", [0, 0])
    vz = data.get("vz", [0, 0])
    print(f"Received velocity range:\n"
          f"  X: {vx}\n"
          f"  Y: {vy}\n"
          f"  Z: {vz}")
    msg_buffer.v_cmd = (vx, vy, vz)
    
@socketio.on("set_viewer_point")
def handle_viewer_point(data):
    point = data.get("point", [3, 3, 1])
    msg_buffer.viewer_point = point
    
@socketio.on("set_viewer_look_at")
def handle_viewer_look_at(data):
    point = data.get("point", [3, 3, 1])
    msg_buffer.viewer_look_at = point

@socketio.on("set_camera_preset")
def handle_camera_preset(data):
    preset = data.get("preset", "Top")
    print(f"[Web Viewer] Set camera preset: {preset}")
    msg_buffer.camera_preset = preset

@socketio.on("teleport_entity")
def handle_teleport_entity(data):
    name = data.get("name")
    x = data.get("x")
    y = data.get("y")
    z = data.get("z") # None if not provided
    theta = data.get("theta") # None if not provided
    print(f"[Web Viewer] Teleport {name} to ({x}, {y}, {z}), theta={theta}")
    msg_buffer.teleport_cmd = (name, x, y, z, theta)

def run_flask_server(port):
    socketio.run(app, host='0.0.0.0', port=port, use_reloader=False, debug=False)


class RenderEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RenderEnvWrapper, self).__init__(env)
        self.socketio = socketio
        self.msg_buffer = msg_buffer
        # Safely access viewport_camera_controller
        self.controller = getattr(self.env.unwrapped, "viewport_camera_controller", None)
        if self.controller is not None:
            self.cfg = self.controller.cfg
        else:
            print("[RenderEnvWrapper] Warning: viewport_camera_controller is None. Camera control disabled.")
            self.cfg = None
        
        # Reference to SimBridge for getting robot states (set by SimBridge after creation)
        self.sim_bridge = None
        
        # Frame counter for rate-limiting state broadcasts
        self._frame_count = 0
        self._broadcast_interval = 10  # Broadcast every N frames

    def _web_prev_step(self):
        if msg_buffer.env_id >= 0:
            if msg_buffer.env_id < self.env.num_envs:
                self._set_viewer_env(msg_buffer.env_id)
            msg_buffer.env_id = -1
        elif msg_buffer.next_env:
            print("next env")
            self._set_viewer_env((self.cfg.env_index + 1) % self.env.num_envs)
            msg_buffer.next_env = False
        elif msg_buffer.switch_debug:
            print("switch_debug")
            msg_buffer.cur_debug = not msg_buffer.cur_debug
            self.env.command_manager.set_debug_vis(msg_buffer.cur_debug)
            msg_buffer.switch_debug = False
        elif msg_buffer.v_cmd is not None:
            self._set_v_cmd(msg_buffer.v_cmd)
            msg_buffer.v_cmd = None
        elif msg_buffer.viewer_point is not None:
            self._set_viewer_point(msg_buffer.viewer_point)
            msg_buffer.viewer_point = None
        elif msg_buffer.viewer_look_at is not None:
            self._set_viewer_look_at(msg_buffer.viewer_look_at)
            msg_buffer.viewer_look_at = None
        elif msg_buffer.camera_preset is not None:
            self._apply_camera_preset(msg_buffer.camera_preset)
            msg_buffer.camera_preset = None
        elif msg_buffer.teleport_cmd is not None:
            self._teleport_entity(*msg_buffer.teleport_cmd)
            msg_buffer.teleport_cmd = None
        elif msg_buffer.apply_initial_positions and msg_buffer.initial_positions:
            # Apply stored initial positions after env reset
            print(f"[Web Viewer] Applying initial positions for {len(msg_buffer.initial_positions)} entities")
            for name, pos in msg_buffer.initial_positions.items():
                x, y = pos[0], pos[1]
                theta = pos[2] if len(pos) > 2 else None
                self._teleport_entity(name, x, y, z=None, theta=theta)
            msg_buffer.apply_initial_positions = False

    def step(self, *args, **kwargs):
        self._web_prev_step()
        self.render(mode='rgb_array')
        return self.env.step(*args, **kwargs)

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            try:
                img = self.env.render()
            except Exception as e:
                print(f"Error calling env.render(): {e}")
                return

            if img is None:
                # Limit warning frequency
                if not hasattr(self, "_last_warning_time"): self._last_warning_time = 0
                import time
                if time.time() - self._last_warning_time > 1.0:
                    print(f"Warning: No image data returned from the environment")
                    self._last_warning_time = time.time()
                return
            
            if isinstance(img, list):
                img = img[-1]
            
            # Debug: Print first time we get an image
            if not hasattr(self, "_first_frame_logged"):
                print(f"[RenderEnvWrapper] Got first frame! Shape: {img.shape}, Type: {type(img)}")
                self._first_frame_logged = True

            img = Image.fromarray(img)
            byte_io = BytesIO()
            img.save(byte_io, format='JPEG', quality=80)
            byte_io.seek(0)
            img_data = byte_io.read()

            img_base64 = base64.b64encode(img_data).decode('utf-8')
            self.socketio.emit('new_frame', {'image': img_base64})
            
            # Broadcast robot states periodically
            self._frame_count += 1
            if self._frame_count % self._broadcast_interval == 0:
                self._broadcast_robot_states()
        return

    def _broadcast_robot_states(self):
        """Broadcast robot positions to frontend for minimap sync."""
        if self.sim_bridge is None:
            return
        try:
            states = self.sim_bridge.get_all_robot_states()
            # [NEW] Append last message info to the broadcast
            if hasattr(self.sim_bridge, "last_msg_info"):
                states["_last_msg"] = self.sim_bridge.last_msg_info
            
            self.socketio.emit('robot_states', states)
        except Exception as e:
            # Silently fail to avoid spamming logs
            pass

    def web_run(self, port=5811):
        flask_thread = threading.Thread(target=run_flask_server, args=[port])
        flask_thread.start()

    def _set_viewer_env(self, idx):
        if self.controller is None: return
        self.controller.set_view_env_index(idx)

    def _set_v_cmd(self, v_cmd):
        vx, vy, vz = v_cmd
            
        try:
            for tar in [vx, vy, vz]:
                a, b = tar
                assert isinstance(a, float) or isinstance(a, int), "not valid type"
                assert isinstance(b, float) or isinstance(b, int), "not valid type"
                assert b >= a
                
            cmd_term = self.env.command_manager.get_term("base_velocity")
            cmd_term.cfg.ranges.lin_vel_x = vx
            cmd_term.cfg.ranges.lin_vel_y = vy
            cmd_term.cfg.ranges.ang_vel_z = vz
        except Exception as e:
            print(repr(e))
            
    def _set_viewer_point(self, point):
        x, y, z = point
        if self.controller is not None:
            self.controller.update_view_location(eye=(x, y, z))
        else:
            # Fallback for cameras... omitted for brevity as controller is likely active
            pass
        
    def _set_viewer_look_at(self, point):
        if self.controller is not None:
             x, y, z = point
             self.controller.update_view_location(lookat=(x, y, z))
        else:
             print("[RenderEnvWrapper] 'Look At' not implemented for headless fallback yet.")

    def _apply_camera_preset(self, preset):
        # Presets: eye, lookat
        PRESETS = {
            "Top": ((0.0, 0.0, 18.0), (0.0, 0.0, 0.0)),
            "Side": ((0.0, -12.0, 6.0), (0.0, 0.0, 0.0)),
            "Diagonal": ((-10.0, -10.0, 10.0), (0.0, 0.0, 0.0)),
            "Goal_Left": ((-8.0, 0.0, 3.0), (0.0, 0.0, 1.0)),
            "Goal_Right": ((8.0, 0.0, 3.0), (0.0, 0.0, 1.0)),
        }
        if preset in PRESETS:
            eye, lookat = PRESETS[preset]
            if self.controller is not None:
                self.controller.update_view_location(eye=eye, lookat=lookat)
            else:
                print(f"Warning: No camera controller to set preset {preset}")

    def _teleport_entity(self, name, x, y, z=None, theta=None):
        # Env is usually ManagerBasedRLEnv. Scene is interactive scene.
        # We need to find the asset in env.scene
        # Names are like 'ball', 'robot_rp0', etc.
        try:
            if not hasattr(self.env.unwrapped, "scene"):
                print("Error: No scene found in env")
                return
            
            scene = self.env.unwrapped.scene
            entity = getattr(scene, name, None)
            
            # Check dictionaries if direct attr access failed
            if entity is None:
                if hasattr(scene, "rigid_objects") and name in scene.rigid_objects:
                    entity = scene.rigid_objects[name]
                elif hasattr(scene, "articulations") and name in scene.articulations:
                    entity = scene.articulations[name]
            
            if entity:
                 # Update root state for ENV 0 only (viewer env)
                 # root_state_w is a view. modifying it might not work directly without write back.
                 # Assuming Isaac Lab RigidObject/Articulation
                 # Shape: (num_envs, 13)
                 
                 # Best practice: Get the data, modify, set it back.
                 root_state = entity.data.root_state_w.clone() # (num_envs, 13)
                 
                 # View env index: self.controller.cfg.env_index if available, else 0
                 env_idx = 0
                 if self.controller:
                     env_idx = self.controller.cfg.env_index
                 
                 # Determine Z if not provided
                 if z is None:
                     # Use current Z position (robot is already standing)
                     current_z = root_state[env_idx, 2].item()
                     if 0.5 < current_z < 2.0:
                         z = current_z
                     else:
                         # Robot-type based fallback heights from robot config
                         # Try to get the robot's default init height from its cfg
                         try:
                             default_z = entity.cfg.init_state.pos[2]
                             z = default_z
                         except:
                             # Fallback: K1 ~ 0.57m, G1 ~ 0.76m
                             z = 0.7
                     print(f"Using Z for {name}: {z}")
                 
                 root_state[env_idx, 0] = x
                 root_state[env_idx, 1] = y
                 root_state[env_idx, 2] = z
                 
                 # Apply Rotation (Theta -> Quat) if provided
                 if theta is not None:
                     # Quat (w, x, y, z) for Z-axis rotation
                     half_theta = theta / 2.0
                     w = np.cos(half_theta)
                     z_q = np.sin(half_theta)
                     # root_state indices 3,4,5,6 are w,x,y,z
                     root_state[env_idx, 3] = w
                     root_state[env_idx, 4] = 0.0
                     root_state[env_idx, 5] = 0.0
                     root_state[env_idx, 6] = z_q
                 
                 # Reset velocity
                 root_state[env_idx, 7:] = 0.0
                 
                 entity.write_root_state_to_sim(root_state)
                 entity.reset() # Important to apply
                 print(f"Teleported {name} (env {env_idx}) to {x, y, z, theta}")
            else:
                 print(f"Entity {name} not found in scene.")
                 # Debug info
                 keys = []
                 if hasattr(scene, "rigid_objects"): keys.extend(scene.rigid_objects.keys())
                 if hasattr(scene, "articulations"): keys.extend(scene.articulations.keys())
                 print(f"Available entities: {keys}")
                 
        except Exception as e:
            print(f"Teleport failed: {e}")