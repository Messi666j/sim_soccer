# MOS Brain Simulation

This directory contains the simulation server, configuration, and scripts for the unified soccer environment.

## Directory Structure
- **config/**: Contains configuration files (e.g., `match_config.json`).
- **src/**: Contains the Python source code for the simulation bridge and logic.
- **scripts/**: Contains shell and batch scripts for launching the simulation.
- **logs/**: Stores output logs from simulation runs.

## Usage

To launch the simulation:
```bash
bash simulation/isaac_sim/scripts/launch_sim.sh --headless --webview --task Robocup-Soccer
```

The simulation server will start and listen for ZMQ connections on port 5555.
Web view is available at http://localhost:5811.
