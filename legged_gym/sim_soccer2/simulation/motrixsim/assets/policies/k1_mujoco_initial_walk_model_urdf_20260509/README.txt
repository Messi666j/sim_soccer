K1 MotrixSim initial walk package, generated 2026-05-09.

This is the version used in the latest MotrixSim rerun:
- Teacher replay video: motion/k1_walk_procedural_20s.npz replayed on K1_22dof.urdf.
- Policy video: k1_walk_model_5500_torchscript.pt closed-loop rollout on K1_22dof.urdf.

Files:
- model/k1_walk_model_5500_rslrl_checkpoint.pt: Isaac/RSL-RL checkpoint.
- model/k1_walk_model_5500_torchscript.pt: exported TorchScript policy used by the MotrixSim policy rerun.
- robot/K1_22dof.urdf + robot/meshes/: robot model used by the MotrixSim rerun script.
- motion/k1_walk_procedural_20s.npz: reference motion used by the teacher replay.
- videos/: MotrixSim rerun outputs for quick visual reference.

Note: the teacher replay itself is motion data replay, so its "model" is the motion npz plus URDF, not a policy network.
