# SoccerLab - MOS-Brain Bridge

This directory contains the decoupled ZeroMQ bridge that connects the **SoccerLab** simulation (Isaac Sim) with the **MOS-Brain** decision system directly or via ROS2.

## Architecture

The bridge uses ZeroMQ (TCP) to support direct communication between Simulation and Brain, enabling **ROS-Free** operation on Windows for easy debugging.

1.  **Simulation Server (`sim_server.py`)**:
    *   **Runs on**: Windows / Simulation Machine (Isaac Lab environment).
    *   **Role**: Runs the simulation loop, publishes ground truth state, and accepts velocity commands via ZMQ REP socket.
    *   **Dependencies**: Isaac Lab, `rsl_rl`, `pyzmq`.

2.  **MOS-Brain (`decider.py --simulation`)**:
    *   **Runs on**: Windows (Debug) or Robot (Sim Mode).
    *   **Role**: Connects to Sim Server directly using `SimClient`.
    *   **Dependencies**: `python`, `pyzmq`. **No ROS required in this mode.**

## Prerequisites

### Common
*   `pyzmq` installed:
    ```bash
    pip install pyzmq
    ```

### Simulation Side (Windows)
*   **Isaac Lab** installed and configured.
*   **SoccerLab** repository cloned.

## Usage

### 1. Start Simulation Server (Windows)

Use the provided batch script or run manually.

**Using Batch Script:**
Run `launch_sim.bat` in this directory.

**Manual Launch:**
```cmd
set SOCCERLAB_PATH=path\to\soccerLab
python sim_server.py --task Isaac-Walking-G1-Play-v0 --num_envs 1 --port 5555
```

### 2. Start MOS-Brain (Direct ZMQ Mode)

Navigate to `mos-brain/decider` directory.

```bash
# Direct ZMQ Mode (Simulating on localhost)
python decider.py --simulation --ip 127.0.0.1 --port 5555
```

In this mode, `decider.py` will:
1.  **Skip ROS initialization**.
2.  Connect to `tcp://127.0.0.1:5555`.
3.  Send velocity commands and receive simulated vision/location data synchronously.

## Troubleshooting

*   **Connection Failed**: Ensure the Sim Server is running and "Waiting for client...". Check IP and Firewall.
*   **Missing Dependencies**: Ensure `pyzmq` is installed in both environments.
