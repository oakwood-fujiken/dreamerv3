#!/usr/bin/env python3
"""
Gymnasium Bridge for DreamerV3-Rust.

Runs a Gymnasium environment and exposes it via TCP socket with a simple JSON protocol.
This allows the Rust DreamerV3 agent to interact with any Gymnasium-compatible environment.

Usage:
    python gym_bridge.py --task ALE/Pong-v5 --port 9876
    python gym_bridge.py --task CartPole-v1 --port 9876
    python gym_bridge.py --task dm_control/walker-walk --port 9876

Protocol (JSON lines over TCP):
    Request:  {"command": "info"}
    Response: {"obs_image_shape": [64,64,3], "act_type": "discrete", "act_dim": 18, ...}

    Request:  {"command": "reset"}
    Response: {"image": [...], "reward": 0.0, "is_first": true, "is_last": false, ...}

    Request:  {"command": "step", "action_discrete": 3}
    Response: {"image": [...], "reward": 1.0, "is_first": false, "is_last": false, ...}

    Request:  {"command": "close"}
    (connection closed)
"""

import argparse
import json
import socket
import sys
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym


def make_env(task: str, image_size: tuple = (64, 64)):
    """Create a Gymnasium environment with optional image resizing."""
    env = gym.make(task, render_mode="rgb_array")
    return env


def obs_to_image(env, obs, image_size=(64, 64)):
    """Convert observation to an image array [H, W, C]."""
    if isinstance(obs, np.ndarray) and obs.ndim == 3:
        # Already an image
        img = obs
    else:
        # Render the environment
        img = env.render()

    if img is not None and img.shape[:2] != image_size:
        try:
            from PIL import Image
            img = np.array(Image.fromarray(img).resize(
                (image_size[1], image_size[0]), Image.BILINEAR
            ))
        except ImportError:
            # Simple nearest-neighbor resize
            h, w = image_size
            oh, ow = img.shape[:2]
            row_idx = (np.arange(h) * oh // h).astype(int)
            col_idx = (np.arange(w) * ow // w).astype(int)
            img = img[np.ix_(row_idx, col_idx)]

    return img


def get_env_info(env, image_size=(64, 64)):
    """Extract environment space information."""
    info = {
        "reward": 0.0,
        "is_first": False,
        "is_last": False,
        "is_terminal": False,
    }

    # Observation space
    obs_space = env.observation_space
    if hasattr(obs_space, 'shape') and len(obs_space.shape) == 3:
        info["obs_image_shape"] = [image_size[0], image_size[1], obs_space.shape[2]]
    else:
        info["obs_image_shape"] = [image_size[0], image_size[1], 3]

    # Action space
    act_space = env.action_space
    if isinstance(act_space, gym.spaces.Discrete):
        info["act_type"] = "discrete"
        info["act_dim"] = int(act_space.n)
    elif isinstance(act_space, gym.spaces.Box):
        info["act_type"] = "continuous"
        info["act_dim"] = int(np.prod(act_space.shape))
        info["act_low"] = float(act_space.low.min())
        info["act_high"] = float(act_space.high.max())
    else:
        info["act_type"] = "discrete"
        info["act_dim"] = 1

    return info


def handle_client(conn, env, image_size=(64, 64)):
    """Handle a single client connection."""
    buf = b""
    while True:
        try:
            data = conn.recv(4096)
            if not data:
                break
            buf += data

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                msg = json.loads(line.decode())

                command = msg.get("command", "")

                if command == "info":
                    info = get_env_info(env, image_size)
                    response = json.dumps(info) + "\n"
                    conn.sendall(response.encode())

                elif command == "reset":
                    result = env.reset()
                    obs = result[0] if isinstance(result, tuple) else result
                    img = obs_to_image(env, obs, image_size)
                    response = json.dumps({
                        "image": img.flatten().tolist() if img is not None else None,
                        "image_shape": list(img.shape) if img is not None else None,
                        "reward": 0.0,
                        "is_first": True,
                        "is_last": False,
                        "is_terminal": False,
                    }) + "\n"
                    conn.sendall(response.encode())

                elif command == "step":
                    # Parse action
                    act_space = env.action_space
                    if isinstance(act_space, gym.spaces.Discrete):
                        action = msg.get("action_discrete", 0)
                    else:
                        action = np.array(msg.get("action_continuous", [0.0]), dtype=np.float32)

                    result = env.step(action)
                    if len(result) == 5:
                        obs, reward, terminated, truncated, info = result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = result
                        terminated = done
                        truncated = False

                    img = obs_to_image(env, obs, image_size)
                    response = json.dumps({
                        "image": img.flatten().tolist() if img is not None else None,
                        "image_shape": list(img.shape) if img is not None else None,
                        "reward": float(reward),
                        "is_first": False,
                        "is_last": bool(done),
                        "is_terminal": bool(terminated),
                    }) + "\n"
                    conn.sendall(response.encode())

                elif command == "close":
                    conn.close()
                    return

        except (ConnectionResetError, BrokenPipeError):
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            break

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Gymnasium Bridge for DreamerV3-Rust")
    parser.add_argument("--task", type=str, required=True, help="Gymnasium environment ID")
    parser.add_argument("--port", type=int, default=9876, help="TCP port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Listen host")
    parser.add_argument("--image-size", type=int, default=64, help="Image resize resolution")
    args = parser.parse_args()

    image_size = (args.image_size, args.image_size)

    print(f"Creating environment: {args.task}")
    env = make_env(args.task, image_size)

    print(f"Starting bridge server on {args.host}:{args.port}")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)

    print(f"Waiting for connection...")

    try:
        while True:
            conn, addr = server.accept()
            print(f"Connected: {addr}")
            handle_client(conn, env, image_size)
            print(f"Disconnected: {addr}")
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        env.close()
        server.close()


if __name__ == "__main__":
    main()
