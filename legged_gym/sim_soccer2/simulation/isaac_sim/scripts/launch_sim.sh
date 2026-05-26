#!/bin/bash
# Launch Script for Sim Server (Windows/Linux compatible via git bash or WSL, but primarily for reference)

# Define paths
SCRIPT_DIR=$(dirname "$0")
MOS_BRAIN_DIR=$(realpath "$SCRIPT_DIR/../../")
SOCCERLAB_DIR=$(realpath "$MOS_BRAIN_DIR/../soccerLab")
ISAACLAB_DIR=$(realpath "$MOS_BRAIN_DIR/../../../IsaacLab")

# Check if SOCCERLAB_PATH is set, else use default
if [ -z "$SOCCERLAB_PATH" ]; then
    export SOCCERLAB_PATH="$SOCCERLAB_DIR"
fi

echo "Using SoccerLab at: $SOCCERLAB_PATH"

# Run python. 
# NOTE: User must ensure this python has Isaac Lab installed.
# If running on Windows with standard Isaac Sim, you might need to use the .bat or .ps1 equivalent.
# Python wrapper usage:
# python "$SCRIPT_DIR/sim_server.py" "$@"

# Better approach for Isaac Lab: Use the isaaclab.sh wrapper if it exists and we are on Linux
ISAACLAB_SH="$ISAACLAB_DIR/isaaclab.sh"

if [ -f "$ISAACLAB_SH" ]; then
    echo "Found isaaclab.sh, using it to launch..."
    exec "$ISAACLAB_SH" -p "$SCRIPT_DIR/../src/sim_server.py" "$@"
else
    # Fallback to current python
    echo "isaaclab.sh not found. Using system python..."
    exec python "$SCRIPT_DIR/../src/sim_server.py" "$@"
fi
