# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later


import os
import sys
import torch
import gymnasium as gym
import copy

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic

# Add path to query rsl_rl_utils and cli_args
# Assuming this file is in mos-brain/simulation/soccerlab_bridge/core.py
# and soccerLab is at mos-brain/third_party/soccerLab or configured via env
# We'll handle sys.path in sim_server.py/main setup, so we assume successful import here.

class SimBridge:
    def __init__(self, env_cfg, agent_cfg, task_name, device="cuda", enable_webview=False, webview_port=5811, existing_webview_wrapper=None):
        self.device = device
        # NOTE: self.current_commands will be initialized during policy loading/adaptation when we know num_agents
        # Fallback for now
        self.current_commands = torch.zeros((1, 3), device=self.device)
        self.num_agents = 1
        
        self.last_msg_info = {"timestamp": 0, "id": -1, "source": "unknown"}

        self.reset_needed = False
        
        # Override velocity commands to read from self.current_command
        # Iterate over all policy observations to find velocity_commands terms
        if hasattr(env_cfg.observations, "policy"):
             for attr_name, attr_val in env_cfg.observations.policy.__dict__.items():
                 # ConfigClass attributes might be stored differently, but __dict__ usually works
                 # Check if it looks like a velocity command term
                 if "velocity_commands" in attr_name and hasattr(attr_val, "func"):
                      # Bind the function
                      # We need to wrap it to match signature (env, command_name)
                      # SimBridge._get_velocity_command has (env, command_name) signature.
                      attr_val.func = self._get_velocity_command
                      print(f"[SimBridge] Hooked ZMQ command to {attr_name}")

        # Create Environment
        render_mode = "rgb_array" if enable_webview else None
        self.env = gym.make(task_name, cfg=env_cfg, render_mode=render_mode)

        # Handle MARL
        if isinstance(self.env.unwrapped, DirectMARLEnv):
            self.env = multi_agent_to_single_agent(self.env)

        # Webview Wrapper
        self.webview_wrapper = None
        if enable_webview:
            if existing_webview_wrapper is not None:
                # Reuse existing webview wrapper (for restart scenarios)
                from labWebView.wrapper import RenderEnvWrapper
                # Re-wrap the new env with existing wrapper's Flask/SocketIO
                existing_webview_wrapper.env = self.env
                self.env = existing_webview_wrapper
                self.webview_wrapper = self.env
                self.webview_wrapper.sim_bridge = self  # Connect for pose sync
                print(f"[SimBridge] Reusing existing labWebView wrapper")
            else:
                # Create new webview wrapper
                try:
                    from labWebView.wrapper import RenderEnvWrapper
                    self.env = RenderEnvWrapper(self.env)
                    self.webview_wrapper = self.env
                    self.webview_wrapper.sim_bridge = self  # Connect for pose sync
                    self.env.web_run(port=webview_port)
                    print(f"[SimBridge] labWebView started at http://localhost:{webview_port}")
                except ImportError as e:
                    print(f"[SimBridge] Failed to import labWebView: {e}")
                except Exception as e:
                     print(f"[SimBridge] Failed to start labWebView: {e}")
        
        # RSL-RL Wrapper
        self.env = RslRlVecEnvWrapper(self.env, clip_actions=agent_cfg.clip_actions)
        
        # Load Policy
        # Assume checkpoint path is passed or determined before
        agent_cfg_dict = agent_cfg.to_dict()
        if "algorithm" in agent_cfg_dict and "class_name" not in agent_cfg_dict["algorithm"]:
             agent_cfg_dict["algorithm"]["class_name"] = "PPO"
        if "policy" in agent_cfg_dict and "class_name" not in agent_cfg_dict["policy"]:
             agent_cfg_dict["policy"]["class_name"] = "ActorCritic"

        # Patch broken config values (workaround for configclass to_dict issue)
        # We explicitly enforce defaults if values are missing or invalid (dict/_MISSING_TYPE)
        defaults = {
            "num_steps_per_env": 24,
            "save_interval": 100,
            "empirical_normalization": False,
            "experiment_name": "g1_loco",
            "run_name": "cliped_with_lin"
        }
        
        for key, default_val in defaults.items():
             current = agent_cfg_dict.get(key)
             # Check if current is invalid: empty dict, None, or looks like MISSING
             is_invalid = (
                 isinstance(current, dict) or 
                 current is None or 
                 type(current).__name__ == "_MISSING_TYPE"
             )
             
             if is_invalid:
                  # Try retrieval from object first
                  obj_val = getattr(agent_cfg, key, None)
                  is_obj_valid = (
                      obj_val is not None and 
                      not isinstance(obj_val, dict) and 
                      type(obj_val).__name__ != "_MISSING_TYPE"
                  )
                  
                  if is_obj_valid:
                       agent_cfg_dict[key] = obj_val
                  else:
                       # Fallback to hardcoded default
                       agent_cfg_dict[key] = default_val

        self.agent_cfg_dict = agent_cfg_dict
        self.ppo_runner = OnPolicyRunner(self.env, agent_cfg_dict, log_dir=None, device=device)
        # Checkpoint loading should be handled by runner.load() called externally or here if path provided
        
        self.obs, _ = self.env.get_observations()
        self.policy = None

    def load_checkpoint(self, path):
        print(f"[SimBridge] Loading checkpoint: {path}")
        
        # First, probe the checkpoint to understand its structure
        loaded_dict = None
        ckpt_input_dim = None
        ckpt_output_dim = None
        
        try:
            loaded_dict = torch.load(path, map_location=self.device)
            if "model_state_dict" in loaded_dict:
                loaded_dict = loaded_dict["model_state_dict"]
            
            # Infer dims from checkpoint
            if "actor.0.weight" in loaded_dict:
                ckpt_input_dim = loaded_dict["actor.0.weight"].shape[1]
            if "std" in loaded_dict:
                ckpt_output_dim = loaded_dict["std"].shape[0]
            elif "actor.6.weight" in loaded_dict:
                ckpt_output_dim = loaded_dict["actor.6.weight"].shape[0]
            
            print(f"[SimBridge] Checkpoint dims: Input={ckpt_input_dim}, Output={ckpt_output_dim}")
        except Exception as e:
            print(f"[SimBridge] Could not probe checkpoint: {e}")
        
        # Get environment dimensions from RSL-RL wrapper or observations
        env_obs_dim = self.obs.shape[1] if self.obs is not None else None
        env_act_dim = getattr(self.env, "num_actions", None)
        print(f"[SimBridge] Environment dims: Obs={env_obs_dim}, Act={env_act_dim}")
        
        # Determine if we need adaptation (checkpoint dims != env dims)
        needs_adaptation = (
            ckpt_input_dim is not None and 
            env_obs_dim is not None and 
            ckpt_input_dim != env_obs_dim
        )
        
        # Method 1: Standard Runner Load (only if dims match)
        if not needs_adaptation:
            try:
                self.ppo_runner.load(path)
                self.policy = self.ppo_runner.get_inference_policy(device=self.env.unwrapped.device)
                print(f"[SimBridge] Successfully loaded standard checkpoint.")
                if hasattr(self.policy, "eval"):
                    self.policy.eval()
                return
            except (RuntimeError, NotImplementedError, KeyError) as e:
                print(f"[SimBridge] Standard load failed: {e}")
        else:
            print(f"[SimBridge] Skipping standard load - dimension mismatch detected.")
        
        # Method 2: Adaptation Load - Create model matching checkpoint dims
        if ckpt_input_dim and ckpt_output_dim and loaded_dict:
            print(f"[SimBridge] Attempting Adaptation Load (Single-Agent -> Multi-Agent)...")
            try:
                # Get hidden dims from config or use defaults
                actor_hidden_dims = [512, 256, 128]
                critic_hidden_dims = [512, 256, 128]
                activation = "elu"
                
                if hasattr(self, "agent_cfg_dict"):
                    policy_cfg = self.agent_cfg_dict.get("policy", {})
                    if isinstance(policy_cfg, dict):
                        actor_hidden_dims = policy_cfg.get("actor_hidden_dims", actor_hidden_dims)
                        critic_hidden_dims = policy_cfg.get("critic_hidden_dims", critic_hidden_dims)
                        activation = policy_cfg.get("activation", activation)

                # Instantiate new policy matching checkpoint dimensions
                new_policy = ActorCritic(
                    num_actor_obs=ckpt_input_dim,
                    num_critic_obs=ckpt_input_dim,
                    num_actions=ckpt_output_dim,
                    actor_hidden_dims=actor_hidden_dims,
                    critic_hidden_dims=critic_hidden_dims,
                    activation=activation,
                    init_noise_std=1.0,
                ).to(self.device)
                
                # Filter to actor weights only (ignore critic which may have different dims)
                filtered_dict = {}
                for k, v in loaded_dict.items():
                    if k.startswith("actor.") or k == "std":
                        filtered_dict[k] = v
                
                print(f"[SimBridge] Loading {len(filtered_dict)} actor weights...")
                new_policy.load_state_dict(filtered_dict, strict=False)
                new_policy.eval()
                
                # Store adaptation info for step() to use
                self._checkpoint_input_dim = ckpt_input_dim
                self._checkpoint_output_dim = ckpt_output_dim
                self._num_agents = env_obs_dim // ckpt_input_dim if env_obs_dim and ckpt_input_dim else 1
                
                # Resize current_commands to match num_agents
                self.num_agents = self._num_agents
                self.current_commands = torch.zeros((self.num_agents, 3), device=self.device)
                print(f"[SimBridge] Initialized command buffer for {self.num_agents} agents.")
                
                self.policy = new_policy
                print(f"[SimBridge] Successfully loaded Single-Agent Policy!")
                print(f"[SimBridge] Will adapt obs: {env_obs_dim} -> {self._num_agents} x {ckpt_input_dim}")
                print(f"[SimBridge] Will adapt act: {self._num_agents} x {ckpt_output_dim} -> {env_act_dim}")
                return
                
            except Exception as e:
                print(f"[SimBridge] Adaptation load failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Method 3: JIT Load (Exported .pt) - with dimension adaptation support
        print(f"[SimBridge] Attempting JIT load...")
        try:
            jit_policy = torch.jit.load(path, map_location=self.device)
            
            # Probe JIT model dimensions by inspecting its parameters
            jit_input_dim = None
            jit_output_dim = None
            
            # Try to find the first linear layer's input dim and last linear layer's output dim
            for name, param in jit_policy.named_parameters():
                if 'weight' in name:
                    # First layer we find with 2D weight is likely input layer
                    if jit_input_dim is None and param.dim() == 2:
                        jit_input_dim = param.shape[1]
                    # Keep updating output as we find more layers
                    if param.dim() == 2:
                        jit_output_dim = param.shape[0]
            
            print(f"[SimBridge] JIT policy dims: Input={jit_input_dim}, Output={jit_output_dim}")
            
            # Check if we need adaptation for this JIT policy
            if jit_input_dim and env_obs_dim and jit_input_dim != env_obs_dim:
                if env_obs_dim % jit_input_dim == 0:
                    # Set up adaptation parameters
                    self._checkpoint_input_dim = jit_input_dim
                    self._checkpoint_output_dim = jit_output_dim
                    self._num_agents = env_obs_dim // jit_input_dim
                    self._is_jit_policy = True
                    
                    # Resize current_commands
                    self.num_agents = self._num_agents
                    self.current_commands = torch.zeros((self.num_agents, 3), device=self.device)
                    print(f"[SimBridge] Initialized command buffer for {self.num_agents} agents.")
                    
                    print(f"[SimBridge] JIT policy requires adaptation (Single-Agent -> Multi-Agent)")
                    print(f"[SimBridge] Will adapt obs: {env_obs_dim} -> {self._num_agents} x {jit_input_dim}")
                    print(f"[SimBridge] Will adapt act: {self._num_agents} x {jit_output_dim} -> {env_act_dim}")
                else:
                    print(f"[SimBridge] WARNING: Cannot adapt JIT policy - {env_obs_dim} not divisible by {jit_input_dim}")
            
            self.policy = jit_policy
            print(f"[SimBridge] Successfully loaded JIT policy from {path}")
            if hasattr(self.policy, "eval"):
                self.policy.eval()
            return
        except Exception as e:
            print(f"[SimBridge] JIT load failed: {e}")
        
        raise RuntimeError(f"All checkpoint loading methods failed for {path}")


    def _get_velocity_command(self, env, command_name):
        # Return current commands
        # Shape: (num_envs, 3) where num_envs should equal num_agents * num_parallel_envs
        # Here we assume 1 parallel env multiple agents.
        # If env.num_envs != self.num_agents, we might need repeating or careful handling.
        
        # Simple case: Direct match
        if env.num_envs == self.num_agents:
            return self.current_commands
        
        # If mismatch (e.g. multiple parallel envs of multiple agents?)
        # For now assume repeat
        if env.num_envs > self.num_agents:
             # Repeat the whole block? 
             # Or is it (env1_a1, env1_a2, ..., env2_a1, ...)
             # Let's assume broadcasting if shapes mismatch isn't handled by environment naturally
             pass
             
        return self.current_commands

    def set_command(self, vx, vy, w, robot_id=0, timestamp=0, source="unknown"):
        """Set velocity command for a specific robot."""
        self.last_msg_info = {"timestamp": timestamp, "id": robot_id, "source": source}
        
        if robot_id < 0 or robot_id >= self.num_agents:
            # print(f"[SimBridge Warning] Robot ID {robot_id} out of range [0, {self.num_agents-1}]")
            # Auto-expand if needed? Or just ignore/clamp?
            # For robustness, let's clamp or ignore.
            return

        self.current_commands[robot_id] = torch.tensor([vx, vy, w], device=self.device)

    def step(self):
        try:
            with torch.inference_mode():
                # Reset must be inside inference_mode to avoid tensor inplace update errors
                if self.webview_wrapper and self.webview_wrapper.msg_buffer.reset_env:
                    print("[SimBridge] Resetting environment via Webview...")
                    self.webview_wrapper.msg_buffer.reset_env = False
                    self.obs, _ = self.env.reset()

                # [NEW] Handle Initial Positions (Teleport)
                if self.webview_wrapper and self.webview_wrapper.msg_buffer.apply_initial_positions:
                    print("[SimBridge] Applying Initial Positions (Teleporting)...")
                    self.webview_wrapper.msg_buffer.apply_initial_positions = False
                    positions = self.webview_wrapper.msg_buffer.initial_positions # {name: [x,y,theta]}
                    
                    base_env = self.env.unwrapped
                    import math
                    
                    # Iterate all robots and teleport
                    # InteractiveScene might not mimic dict .items() fully, so use keys
                    for name in base_env.scene.keys():
                        entity = base_env.scene[name]
                        if name.startswith("robot_"):
                            # Read current root state to preserve structure
                            # root_state: (NumEnvs, 13) [pos(3), quat(4), lin_vel(3), ang_vel(3)]
                            # We assume NumEnvs=1 for this logic or broadcast
                            current_state = entity.data.default_root_state.clone() # Use default as base
                            if current_state.shape[0] != self.env.num_envs:
                                 current_state = current_state.repeat(self.env.num_envs, 1)

                            if name in positions:
                                # Active Robot
                                p = positions[name]
                                # Pos
                                current_state[:, 0] = p[0]
                                current_state[:, 1] = p[1]
                                current_state[:, 2] = 0.57 # Default K1 standing height or slight drop
                                
                                # Yaw to Quat (w, x, y, z)
                                half_theta = p[2] * 0.5
                                current_state[:, 3] = math.cos(half_theta)
                                current_state[:, 4] = 0.0
                                current_state[:, 5] = 0.0
                                current_state[:, 6] = math.sin(half_theta)
                            else:
                                # Unused Robot -> Teleport to "Bench" (Far away)
                                current_state[:, 0] = 100.0 + hash(name) % 20
                                current_state[:, 1] = 100.0
                                current_state[:, 2] = -2.0 # Underground
                                
                            # Zero velocities
                            current_state[:, 7:13] = 0.0
                            
                            # Apply
                            entity.write_root_state_to_sim(current_state)
                            entity.reset() # Refresh view
                    
                    # Also reset physics buffers slightly to ensuring stability
                    # self.env.reset() # Loop reset? No, infinite loop.
                    # Just waiting for next step is fine.

                if self.policy is None:
                    # Fallback: Send zero actions for testing purposes
                    num_actions = getattr(self.env, "num_actions", 29)
                    actions = torch.zeros((self.env.num_envs, num_actions), device=self.device)
                else:
                    obs = self.obs
                    num_agents = getattr(self, "_num_agents", 1)
                    expected_dim = getattr(self, "_checkpoint_input_dim", None)
                    
                    # Adapt observations if multi-agent -> single-agent policy
                    if num_agents > 1 and expected_dim:
                        # Reshape: (N, Agents*Dims) -> (N*Agents, Dims)
                        obs = obs.view(-1, num_agents, expected_dim).reshape(-1, expected_dim)
                    
                    # FORCE COMMAND INJECTION: Directly patch observation tensor
                    # Index 9, 10, 11 are velocity commands (lin_x, lin_y, ang_z) for G1
                    if obs.shape[1] >= 12:
                        # Apply commands to specific agents
                        # Obs Shape: (N*Agents, Dims)
                        # We need to broadcast self.current_commands (Agents, 3) to match.
                        
                        # If N=1 (one env, multiple agents), obs is (Agents, Dims)
                        if obs.shape[0] == self.num_agents:
                            obs[:, 9:12] = self.current_commands
                        else:
                            # N > 1, repeat current_commands N times
                            # Order is usually env1_all_agents, env2_all_agents...
                            # So we repeat the block
                            n_envs = obs.shape[0] // self.num_agents
                            obs[:, 9:12] = self.current_commands.repeat(n_envs, 1)

                    # DEBUG: Inspect velocity command in observation (Indices 9,10,11)
                    if torch.rand(1).item() < 0.01: # 1% chance
                        cmd_obs = obs[0, 9:12]
                        # Only print if non-zero to reduce noise
                        if cmd_obs.abs().sum() > 0.001:
                             print(f"[SimBridge DEBUG] Obs[0] Command (idx 9-11): {cmd_obs.cpu().numpy()}")

                    # ActorCritic.forward() raises NotImplementedError
                    # Use act_inference() for ActorCritic, direct call for JIT
                    # Use act_inference() for ActorCritic, direct call for JIT
                    if hasattr(self.policy, 'act_inference'):
                        actions = self.policy.act_inference(obs)
                    else:
                        actions = self.policy(obs)
                    
                if num_agents > 1:
                        # Reshape actions back: (N*Agents, ActDim) -> (N, Agents*ActDim)
                        actions = actions.view(self.env.num_envs, num_agents, -1).reshape(self.env.num_envs, -1)
        
                self.obs, _, _, _ = self.env.step(actions)
                
            return self._extract_state()
        except Exception as e:
            print(f"[SimBridge] ERROR in step(): {e}")
            import traceback
            traceback.print_exc()
            raise

    def _extract_state(self):
        # Extract Ground Truth state from env
        base_env = self.env.unwrapped
        
        # Get match config to determine robot counts
        # Default fallback
        red_count = 1
        blue_count = 1
        
        try:
             import json
             config_str = os.environ.get("SOCCER_MATCH_CONFIG", "{}")
             if config_str:
                 config = json.loads(config_str)
                 red_count = config.get("teams", {}).get("red", {}).get("count", 1)
                 blue_count = config.get("teams", {}).get("blue", {}).get("count", 1)
             else:
                 # Fallback to legacy env var if config missing
                 red_count = int(os.environ.get("SOCCER_NUM_ROBOTS", 1))
                 blue_count = red_count
        except:
             pass
             
        robots_state = []
        
        # Helper to extract state for a named entity
        def get_entity_state(name, team):
            if name in base_env.scene.keys():
                entity = base_env.scene[name]
                try:
                    pos = entity.data.root_pose_w[0, :3]
                    quat = entity.data.root_pose_w[0, 3:]
                    return {
                        "name": name,
                        "x": float(pos[0]),
                        "y": float(pos[1]),
                        "theta": self._quat_to_yaw(quat),
                        "team": team
                    }
                except Exception as e:
                    return None
            return None

        # Check for standard single robot "robot" (legacy / Velocity task)
        if "robot" in base_env.scene.keys():
             data = get_entity_state("robot", "red")
             if data: 
                 robots_state.append(data)
        
        # Check for dynamic players (Soccer task uses 'robot_rp{i}' and 'robot_bp{i}')
        # Red team
        for i in range(red_count):
            name = f"robot_rp{i}"
            data = get_entity_state(name, "red")
            if data: 
                robots_state.append(data)
            
        # Blue team
        for i in range(blue_count):
            name = f"robot_bp{i}"
            data = get_entity_state(name, "blue")
            if data: 
                robots_state.append(data)

        # Ball position
        ball_pos = [0.0, 0.0, 0.0]
        if "ball" in base_env.scene.keys():
             ball = base_env.scene["ball"]
             try:
                 ball_pos = ball.data.root_pose_w[0, :3].tolist()
             except: pass
        
        state = {
            "robots": robots_state,
            "ball": {
                "x": float(ball_pos[0]),
                "y": float(ball_pos[1]),
                "z": float(ball_pos[2])
            }
        }
        return state

    def _quat_to_yaw(self, q):
        # q: (w, x, y, z)
        w, x, y, z = q
        # roll (x-axis rotation)
        # sinr_cosp = 2 * (w * x + y * z)
        # cosr_cosp = 1 - 2 * (x * x + y * y)
        # roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return float(yaw)

    def get_all_robot_states(self):
        """
        Returns position and yaw for all 20 robots.
        Format: {robot_name: {x, y, z, yaw, active}}
        """
        import json
        base_env = self.env.unwrapped
        
        # Get active counts from config
        config_str = os.environ.get("SOCCER_MATCH_CONFIG", "{}")
        config = json.loads(config_str) if config_str else {}
        red_active = config.get("teams", {}).get("red", {}).get("count", 1)
        blue_active = config.get("teams", {}).get("blue", {}).get("count", 1)
        
        states = {}
        
        # Iterate over all possible robot assets
        for team, prefix, active_count in [("red", "rp", red_active), ("blue", "bp", blue_active)]:
            for i in range(10):  # MAX_ROBOTS_PER_TEAM
                asset_name = f"robot_{prefix}{i}"
                try:
                    asset = getattr(base_env.scene, asset_name, None)
                    if asset is None:
                        continue
                    
                    # Get position and orientation
                    pos = asset.data.root_pos_w[0]  # shape: (3,)
                    quat = asset.data.root_quat_w[0]  # shape: (4,) - w, x, y, z
                    
                    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                    yaw = self._quat_to_yaw(quat)
                    
                    # Determine if active (on-field) - off-field robots are at x > 50
                    is_active = i < active_count and x < 50.0
                    
                    states[asset_name] = {
                        "x": x,
                        "y": y,
                        "z": z,
                        "yaw": yaw,
                        "active": is_active,
                        "team": team
                    }
                except Exception as e:
                    # Robot might not exist yet
                    pass
        
        # Also get ball state
        try:
            ball = getattr(base_env.scene, "ball", None)
            if ball is not None:
                pos = ball.data.root_pos_w[0]
                states["ball"] = {
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]),
                    "yaw": 0.0,
                    "active": True,
                    "team": None
                }
        except:
            pass
        
        return states

    def close(self):
        self.env.close()

