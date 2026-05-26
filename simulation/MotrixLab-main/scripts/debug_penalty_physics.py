"""Debug script for k1-penalty-shootout physics stability.

Runs the environment with zero or random actions and prints diagnostic info.
Useful for catching NaN/Inf propagation, velocity explosions, and solver crashes
before they happen in full training.

Usage:
    uv run scripts/debug_penalty_physics.py --env k1-penalty-shootout \
        --num-envs 1 --num-steps 2000 --seed 42 --random-actions

    uv run scripts/debug_penalty_physics.py --env k1-penalty-shootout \
        --num-envs 1 --num-steps 1000 --zero-actions
"""

import logging
import os
import sys
from typing import Optional

import numpy as np
from absl import app, flags

_ENV = flags.DEFINE_string("env", "k1-penalty-shootout", "Environment name")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 1, "Number of parallel environments")
_NUM_STEPS = flags.DEFINE_integer("num-steps", 2000, "Number of steps to run")
_SEED = flags.DEFINE_integer("seed", 42, "Random seed")
_RANDOM_ACTIONS = flags.DEFINE_bool("random-actions", False, "Use random actions instead of zero actions")
_ZERO_ACTIONS = flags.DEFINE_bool("zero-actions", False, "Use zero actions")
_ACTION_SCALE = flags.DEFINE_float("action-scale", 1.0, "Scale factor for actions (overrides env default)")
_DUMP_DIR = flags.DEFINE_string("dump-dir", "debug_dumps", "Directory for crash dump npz files")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def print_state_diagnostics(state, step: int, actions: Optional[np.ndarray] = None):
    """Print per-step diagnostic information."""
    data = state.data
    num_envs = data.shape[0]
    info = state.info

    logger.info(f"=== Step {step} ===")

    # Actions
    if actions is not None:
        logger.info(
            f"  action  min/max/mean/std: {actions.min():.4f}/{actions.max():.4f}/"
            f"{actions.mean():.4f}/{actions.std():.4f}"
        )
        nan_count = int(np.isnan(actions).sum())
        inf_count = int(np.isinf(actions).sum())
        if nan_count or inf_count:
            logger.warning(f"  action  NaN={nan_count} Inf={inf_count}")

    # qpos
    qpos = data.dof_pos
    qpos_finite = np.all(np.isfinite(qpos), axis=1)
    qpos_not_finite = int((~qpos_finite).sum())
    logger.info(f"  qpos    finite={qpos_not_finite == 0} (bad={qpos_not_finite})")

    # qvel
    qvel = data.dof_vel
    qvel_finite = np.all(np.isfinite(qvel), axis=1)
    qvel_not_finite = int((~qvel_finite).sum())
    qvel_max = float(np.max(np.abs(qvel))) if np.any(qvel_finite) else float("nan")
    qvel_mean = float(np.mean(np.abs(qvel))) if np.any(qvel_finite) else float("nan")
    logger.info(f"  qvel    finite={qvel_not_finite == 0} (bad={qvel_not_finite})  max={qvel_max:.4f}  mean={qvel_mean:.4f}")

    # Base height (z position of Trunk)
    if "_robot_pose" in info:
        base_z = info["_robot_pose"][:, 2]
    else:
        base_z = np.full(num_envs, np.nan)
    logger.info(f"  base_z  min={base_z.min():.4f}  max={base_z.max():.4f}  ok={(base_z >= 0.05).all()}")

    # Ball position / velocity
    if "_ball_pos" in info:
        bp = info["_ball_pos"]
        logger.info(f"  ball_pos  min={bp.min(axis=0)}  max={bp.max(axis=0)}  finite={np.all(np.isfinite(bp))}")
    if "_ball_vel" in info:
        bv = info["_ball_vel"]
        bv_norm = np.linalg.norm(bv, axis=-1)
        logger.info(f"  ball_vel  max_norm={bv_norm.max():.4f}  finite={np.all(np.isfinite(bv))}")

    # Reward
    logger.info(f"  reward  min={state.reward.min():.4f}  max={state.reward.max():.4f}")

    # Done
    done_count = int(state.done.sum())
    if done_count:
        logger.info(f"  done    count={done_count}")
        # Show breakdown
        for key in ["goal_scored", "out_of_bounds", "robot_fallen", "base_too_low"]:
            if key in info:
                count = int(info[key].sum())
                if count:
                    logger.info(f"    {key}: {count}")
    else:
        logger.info(f"  done    count=0")

    # Bad env ids
    bad_ids = []
    if qvel_not_finite:
        bad_ids.extend(np.where(~qvel_finite)[0].tolist())
    if qpos_not_finite:
        bad_ids.extend(np.where(~qpos_finite)[0].tolist())
    if bad_ids:
        logger.warning(f"  BAD ENV IDS: {sorted(set(bad_ids))}")

    # Bad state reset counter
    if "_bad_env_reset_count" in info:
        logger.info(f"  bad_reset_count: {info['_bad_env_reset_count']}")


def dump_crash_data(state, actions: np.ndarray, seed: int, step: int, dump_dir: str):
    """Dump state and actions to npz for post-mortem analysis."""
    os.makedirs(dump_dir, exist_ok=True)
    filepath = os.path.join(dump_dir, f"penalty_crash_seed{seed}_step{step:04d}.npz")

    info = state.info
    save_dict = {
        "step": step,
        "seed": seed,
        "dof_pos": state.data.dof_pos,
        "dof_vel": state.data.dof_vel,
        "actions": actions,
        "reward": state.reward,
        "terminated": state.terminated,
        "truncated": state.truncated,
    }
    # Save cached info if available
    for key in ["_robot_pose", "_ball_pos", "_ball_vel", "_dof_pos", "_dof_vel", "_ball_speed"]:
        if key in info and info[key] is not None:
            save_dict[key] = info[key]

    np.savez_compressed(filepath, **save_dict)
    logger.info(f"Crash dump saved to {filepath}")
    return filepath


def main(argv):
    from motrix_envs import registry as env_registry

    env_name = _ENV.value
    num_envs = _NUM_ENVS.value
    num_steps = _NUM_STEPS.value
    seed = _SEED.value
    random_actions = _RANDOM_ACTIONS.value
    zero_actions = _ZERO_ACTIONS.value
    action_scale_override = _ACTION_SCALE.value
    dump_dir = _DUMP_DIR.value

    # Validate flags
    if random_actions and zero_actions:
        logger.error("Cannot use both --random-actions and --zero-actions")
        sys.exit(1)

    if not random_actions and not zero_actions:
        logger.info("Using zero actions (default). Use --random-actions for random actions.")

    use_random = random_actions

    logger.info(f"Creating env: {env_name}, num_envs={num_envs}")
    env = env_registry.make(env_name, sim_backend="np", num_envs=num_envs)

    # Apply action_scale override if provided
    if action_scale_override != 1.0:
        env.cfg.control.action_scale = action_scale_override
        logger.info(f"Overriding action_scale to {action_scale_override}")

    # Print env config summary
    cfg = env.cfg
    logger.info(f"  sim_dt={cfg.sim_dt}, ctrl_dt={cfg.ctrl_dt}, sim_substeps={cfg.sim_substeps}")
    logger.info(f"  action_scale={cfg.control.action_scale}")
    logger.info(f"  max_episode_steps={cfg.max_episode_steps}")
    logger.info(f"  bad_state_reset={getattr(cfg, 'bad_state_reset', 'N/A')}")
    logger.info(f"  max_safe_velocity={getattr(cfg, 'max_safe_velocity', 'N/A')}")
    logger.info(f"  max_safe_ball_speed={getattr(cfg, 'max_safe_ball_speed', 'N/A')}")

    # Set seed
    rng = np.random.default_rng(seed)

    # Initialize env
    state = env.init_state()
    action_dim = env.action_space.shape[0]
    logger.info(f"Env initialized. action_dim={action_dim}, obs_dim={env.observation_space.shape[0]}")

    # Print initial state
    logger.info("=== Initial State ===")
    print_state_diagnostics(state, 0)

    try:
        for step in range(1, num_steps + 1):
            # Generate actions
            if use_random:
                actions = rng.uniform(-1.0, 1.0, (num_envs, action_dim)).astype(np.float32)
            else:
                actions = np.zeros((num_envs, action_dim), dtype=np.float32)

            # Step env
            state = env.step(actions)

            # Print diagnostics every 100 steps
            if step % 100 == 0 or step == 1:
                print_state_diagnostics(state, step, actions)

    except Exception as e:
        logger.error(f"CRASH at step {step}: {type(e).__name__}: {e}")
        try:
            dump_crash_data(state, actions, seed, step, dump_dir)
        except Exception as dump_error:
            logger.error(f"Failed to dump crash data: {dump_error}")
        raise

    logger.info(f"Completed {num_steps} steps without crash.")
    logger.info("=== Final State ===")
    print_state_diagnostics(state, num_steps)


if __name__ == "__main__":
    app.run(main)
