#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import json
import time

import zmq


def main() -> None:
    parser = argparse.ArgumentParser(description="Send cmdvel to sim2sim_runner ZMQ REP server")
    parser.add_argument("--host", default="127.0.0.1", help="ZMQ server host")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ server port")
    parser.add_argument("--vx", type=float, default=0.2, help="Linear velocity x")
    parser.add_argument("--vy", type=float, default=0.0, help="Linear velocity y")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw rate")
    parser.add_argument("--rate", type=float, default=20.0, help="Send rate (Hz)")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration (seconds)")
    parser.add_argument("--robot-id", type=int, default=0, help="Robot id field in request")
    parser.add_argument("--timeout-ms", type=int, default=1000, help="Receive timeout in milliseconds")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print response JSON")
    args = parser.parse_args()

    if args.rate <= 0:
        raise ValueError("--rate must be > 0")
    if args.duration <= 0:
        raise ValueError("--duration must be > 0")

    address = f"tcp://{args.host}:{args.port}"
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(address)

    period = 1.0 / args.rate
    total_steps = int(args.duration * args.rate)

    print(f"[cmdvel-test] connected: {address}")
    print(f"[cmdvel-test] cmd = [{args.vx}, {args.vy}, {args.yaw}], rate={args.rate}Hz, steps={total_steps}")

    try:
        for step in range(total_steps):
            t0 = time.time()
            payload = {
                "cmd": [args.vx, args.vy, args.yaw],
                "timestamp": t0,
                "id": args.robot_id,
            }
            socket.send_json(payload)
            try:
                resp = socket.recv_json()
                if args.pretty:
                    print(f"[{step + 1:04d}] {json.dumps(resp, ensure_ascii=False, indent=2)}")
                else:
                    state = resp.get("state", {})
                    robots = state.get("robots", [])
                    robot = robots[0] if robots else {}
                    ball = state.get("ball", {})
                    print(
                        f"[{step + 1:04d}] "
                        f"sim_t={resp.get('sim_timestamp', 0):.3f} "
                        f"lat={resp.get('step_latency', 0):.4f}s "
                        f"robot=({robot.get('x', 0):.3f},{robot.get('y', 0):.3f},{robot.get('theta', 0):.3f}) "
                        f"ball=({ball.get('x', 0):.3f},{ball.get('y', 0):.3f},{ball.get('z', 0):.3f})"
                    )
            except zmq.error.Again:
                print(f"[{step + 1:04d}] timeout waiting for reply ({args.timeout_ms} ms)")
                break

            sleep_t = period - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
