#!/bin/bash

# start_team.sh
# Check/Create log dir
mkdir -p logs

# Activate k1 conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate k1

# Paths
DECIDER_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")" # mos-brain/decider
MOS_BRAIN_DIR="$(dirname "$DECIDER_DIR")"
CONFIG_FILE="$MOS_BRAIN_DIR/simulation/config/match_config.json"
DECIDER_SCRIPT="$DECIDER_DIR/decider.py"

echo "Using Match Config: $CONFIG_FILE"

# Function to kill existing sessions
cleanup() {
    echo "Cleaning up existing decider sessions..."
    screen -ls | grep decider_ | cut -d. -f1 | awk '{print $1}' | xargs -r kill
}

# Parse Arguments
RED_OVERRIDE=""
BLUE_OVERRIDE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --kill)
            cleanup
            exit 0
            ;;
        --restart)
            cleanup
            sleep 1
            ;;
        --red)
            RED_OVERRIDE="$2"
            shift
            ;;
        --blue)
            BLUE_OVERRIDE="$2"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

# Get robot counts
if [[ -n "$RED_OVERRIDE" ]]; then
    RED_COUNT=$RED_OVERRIDE
    echo "Red Count (Override): $RED_COUNT"
else
    # Determine based on Field Size Preset
    RED_COUNT=$(python3 -c "
import json
try:
    data = json.load(open('$CONFIG_FILE'))
    preset = data.get('field', {}).get('preset', 'S')
    preset_map = {'L': 5, 'M': 7}
    # Default to config count if preset not in map
    config_count = data.get('teams', {}).get('red', {}).get('count', 1)
    print(preset_map.get(preset, config_count))
except:
    print(1)
")
    echo "Red Count (Auto/Config): $RED_COUNT"
fi

if [[ -n "$BLUE_OVERRIDE" ]]; then
    BLUE_COUNT=$BLUE_OVERRIDE
    echo "Blue Count (Override): $BLUE_COUNT"
else
    # Determine based on Field Size Preset
    BLUE_COUNT=$(python3 -c "
import json
try:
    data = json.load(open('$CONFIG_FILE'))
    preset = data.get('field', {}).get('preset', 'S')
    preset_map = {'L': 5, 'M': 7}
    # Default to config count if preset not in map
    config_count = data.get('teams', {}).get('blue', {}).get('count', 1)
    print(preset_map.get(preset, config_count))
except:
    print(1)
")
    echo "Blue Count (Auto/Config): $BLUE_COUNT"
fi

echo "Launching Red Team: $RED_COUNT robots"
echo "Launching Blue Team: $BLUE_COUNT robots"

# Launch Red Team
for ((i=0; i<RED_COUNT; i++)); do
    SESSION_NAME="decider_red_$i"
    echo "Starting $SESSION_NAME..."
    screen -dmS "$SESSION_NAME" bash -c "python3 $DECIDER_SCRIPT --simulation --color red --id $i; exec bash"
done

# Launch Blue Team
for ((i=0; i<BLUE_COUNT; i++)); do
    SESSION_NAME="decider_blue_$i"
    echo "Starting $SESSION_NAME..."
    screen -dmS "$SESSION_NAME" bash -c "python3 $DECIDER_SCRIPT --simulation --color blue --id $i; exec bash"
done

echo "All agents launched in screen sessions."
echo "Use 'screen -r decider_red_0' etc. to view logs."
echo "Use '$0 --kill' to stop all agents."
