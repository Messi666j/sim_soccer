# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import zmq
import time
import sys
import json

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    
    # Default port or from args
    port = "5555"
    if len(sys.argv) > 1:
        port = sys.argv[1]
        
    print(f"Connecting to server on port {port}...")
    socket.connect(f"tcp://localhost:{port}")

    # Send a request
    # We send a dummy command with slight forward velocity
    cmd = {
        "cmd": [0.3, 0.0, 0.0], 
        "timestamp": time.time()
    }
    
    print(f"Sending request: {cmd}")
    socket.send_json(cmd)
    
    # Wait for reply
    print("Waiting for response...")
    # Add a timeout mechanism
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    
    if poller.poll(5000): # 5 seconds timeout
        message = socket.recv_json()
        print("\nReceived reply:")
        print(json.dumps(message, indent=2))
        
        # Specifically check for ball position
        state = message.get("state", {})
        ball = state.get("ball", None)
        
        if ball:
            print(f"\nBall Position: x={ball.get('x')}, y={ball.get('y')}, z={ball.get('z')}")
        else:
            print("\nWARNING: Ball position not found in state!")
    else:
        print("\nTimeout: No response from server. Is the simulation running?")

if __name__ == "__main__":
    main()
