#!/bin/bash

# Configuration
BASE_ZMQ_PORT=5555
BASE_WEB_PORT=5811
NUM_INSTANCES=6
SESSION_PREFIX="sim_instance"

# Kill existing sessions with same prefix?
# echo "Cleaning up old sessions..."
# screen -ls | grep $SESSION_PREFIX | cut -d. -f1 | awk '{print $1}' | xargs kill

echo "Launching $NUM_INSTANCES simulation instances..."

for ((i=0; i<NUM_INSTANCES; i++))
do
    ZMQ_PORT=$((BASE_ZMQ_PORT + i))
    WEB_PORT=$((BASE_WEB_PORT + i))
    SESSION_NAME="${SESSION_PREFIX}_${i}"
    
    CMD="bash $(dirname "$0")/launch_sim.sh --headless --webview --task Robocup-Soccer --port $ZMQ_PORT --webview_port $WEB_PORT"
    
    echo "Starting Instance $i: ZMQ=$ZMQ_PORT, Web=$WEB_PORT (Screen: $SESSION_NAME)"
    # Use -L to enable logging if needed, or just standard detached
    screen -dmS "$SESSION_NAME" bash -c "$CMD; exec bash"
    
    # Stagger launch slightly to avoid resource spikes
    sleep 2
done

echo "All instances launched in detached screen sessions."
echo "Use 'screen -r ${SESSION_PREFIX}_N' to attach to a session."
echo "Web Viewers available at http://localhost:$BASE_WEB_PORT through http://localhost:$((BASE_WEB_PORT + NUM_INSTANCES - 1))"
