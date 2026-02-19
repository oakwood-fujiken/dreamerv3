#!/usr/bin/env python3
"""
Environment Bridge for DreamerV3-Rust.

Runs reinforcement learning environments and exposes them via TCP socket
with a simple JSON-line protocol. Supports Crafter, DMC (DeepMind Control),
Atari, and any Gymnasium-compatible environment.

Usage:
    # Crafter
    python gym_bridge.py --task crafter_reward --port 9876

    # DMC Vision
    python gym_bridge.py --task dmc_walker_walk --port 9876
    python gym_bridge.py --task dmc_cartpole_swingup --port 9876

    # Atari
    python gym_bridge.py --task atari_pong --port 9876
    python gym_bridge.py --task atari_breakout --port 9876

    # Generic Gymnasium
    python gym_bridge.py --task gym_CartPole-v1 --port 9876

Protocol (JSON lines over TCP):
    {"command": "info"}   -> space info
    {"command": "reset"}  -> first observation
    {"command": "step", "action_discrete": 3}  -> next observation
    {"command": "close"}  -> shutdown
"""

import argparse
import json
import os
import socket
import sys
import traceback
import numpy as np


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def resize_image(img, size):
    """Resize image to (H, W) using PIL or nearest-neighbor fallback."""
    if img is None:
        return None
    if img.shape[0] == size[0] and img.shape[1] == size[1]:
        return img
    try:
        from PIL import Image as PILImage
        pil = PILImage.fromarray(img)
        pil = pil.resize((size[1], size[0]), PILImage.BILINEAR)
        return np.array(pil)
    except ImportError:
        h, w = size
        oh, ow = img.shape[:2]
        row_idx = (np.arange(h) * oh // h).astype(int)
        col_idx = (np.arange(w) * ow // w).astype(int)
        return img[np.ix_(row_idx, col_idx)]


def gray_to_rgb(img):
    """Convert grayscale to RGB if needed."""
    if img is not None and img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img is not None and img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img


# ---------------------------------------------------------------------------
# Environment wrappers
# ---------------------------------------------------------------------------

class ActionRepeatWrapper:
    """Repeat each action N times, accumulating reward."""

    def __init__(self, env, repeat):
        self._env = env
        self._repeat = repeat

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._repeat):
            result = self._env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                total_reward += reward
                if terminated or truncated:
                    return obs, total_reward, terminated, truncated, info
            else:
                obs, reward, done, info = result
                total_reward += reward
                if done:
                    return obs, total_reward, done, info
        return result[:-2] + (total_reward,) + result[-2:] if len(result) == 5 \
            else (obs, total_reward, False, False, info) if len(result) == 5 \
            else (obs, total_reward, False, info)

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_crafter(task_name, image_size, seed=None):
    """Create a Crafter environment.

    task_name: 'reward' or 'noreward'
    """
    import crafter
    use_reward = (task_name == "reward")
    env = crafter.Env(size=image_size, reward=use_reward, seed=seed)
    return CrafterBridge(env, image_size)


class CrafterBridge:
    """Bridge Crafter's API to a gym-like interface."""

    def __init__(self, env, image_size):
        self._env = env
        self._image_size = image_size
        self._done = True

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self, **kwargs):
        obs = self._env.reset()
        self._done = False
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._done = done
        return obs, reward, done, False, info

    def render(self):
        return None  # Crafter already returns image observations

    def close(self):
        pass


def make_dmc(domain, task_name, image_size, action_repeat=1, camera=-1):
    """Create a DeepMind Control Suite environment.

    domain: e.g. 'walker', 'cartpole', 'cheetah', 'quadruped'
    task_name: e.g. 'walk', 'swingup', 'run'
    """
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    from dm_control import suite

    default_cameras = {"quadruped": 2, "rodent": 4}
    if camera == -1:
        camera = default_cameras.get(domain, 0)

    actual_domain = domain
    if domain == "cup":
        actual_domain = "ball_in_cup"

    dm_env = suite.load(actual_domain, task_name)
    env = DMCBridge(dm_env, image_size, camera)
    if action_repeat > 1:
        env = ActionRepeatWrapper(env, action_repeat)
    return env


class DMCBridge:
    """Bridge dm_control's API to a gym-like interface with image observations."""

    def __init__(self, dm_env, image_size, camera=0):
        self._env = dm_env
        self._image_size = image_size
        self._camera = camera

    @property
    def observation_space(self):
        class _Space:
            def __init__(self, shape):
                self.shape = shape
                self.dtype = np.uint8
        return _Space((self._image_size[0], self._image_size[1], 3))

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return _BoxSpace(spec.shape, spec.minimum, spec.maximum)

    def _render(self):
        img = self._env.physics.render(
            height=self._image_size[0],
            width=self._image_size[1],
            camera_id=self._camera,
        )
        return img

    def reset(self, **kwargs):
        self._env.reset()
        img = self._render()
        return img, {}

    def step(self, action):
        action = np.clip(action, self._env.action_spec().minimum,
                         self._env.action_spec().maximum)
        ts = self._env.step(action)
        img = self._render()
        reward = ts.reward or 0.0
        terminated = ts.last()
        truncated = False
        return img, reward, terminated, truncated, {}

    def render(self):
        return self._render()

    def close(self):
        pass


class _BoxSpace:
    """Minimal Box-like action space."""

    def __init__(self, shape, low, high):
        self.shape = shape
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.dtype = np.float32

    @property
    def n(self):
        raise AttributeError("Continuous space has no n")


def make_atari(game_name, image_size, action_repeat=4, gray=False):
    """Create an Atari environment via ALE/Gymnasium.

    game_name: e.g. 'pong', 'breakout', 'qbert'
    """
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    # Capitalize first letter of each word for ALE naming
    name = game_name.capitalize()
    env_id = f"ALE/{name}-v5"
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except Exception:
        # Fallback: try without ALE/ prefix
        env_id = f"{name}-v5"
        try:
            env = gym.make(env_id, render_mode="rgb_array")
        except Exception:
            env_id = f"{name}NoFrameskip-v4"
            env = gym.make(env_id, render_mode="rgb_array")

    if action_repeat > 1:
        env = ActionRepeatWrapper(env, action_repeat)
    return AtariBridge(env, image_size, gray)


class AtariBridge:
    """Bridge Atari with image resizing and optional grayscale."""

    def __init__(self, env, image_size, gray=False):
        self._env = env
        self._image_size = image_size
        self._gray = gray

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self, **kwargs):
        result = self._env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        return self._process_obs(obs), {} if isinstance(result, tuple) else self._process_obs(obs)

    def step(self, action):
        result = self._env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return self._process_obs(obs), reward, terminated, truncated, info
        else:
            obs, reward, done, info = result
            return self._process_obs(obs), reward, done, False, info

    def _process_obs(self, obs):
        """Resize and optionally convert to grayscale."""
        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            img = resize_image(obs, self._image_size)
            if self._gray:
                img = np.mean(img, axis=-1, keepdims=True).astype(np.uint8)
            return img
        return obs

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()


def make_gymnasium(env_id, image_size):
    """Create a generic Gymnasium environment."""
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    env = gym.make(env_id, render_mode="rgb_array")
    return env


# ---------------------------------------------------------------------------
# Unified environment creation
# ---------------------------------------------------------------------------

TASK_DEFAULTS = {
    # suite: (action_repeat, image_size, train_ratio, steps)
    "crafter": {"action_repeat": 1, "image_size": 64, "train_ratio": 512},
    "dmc": {"action_repeat": 1, "image_size": 64, "train_ratio": 256},
    "atari": {"action_repeat": 4, "image_size": 64, "train_ratio": 32, "gray": False},
}


def parse_task(task_str):
    """Parse a task string into (suite, task_name).

    Formats:
        crafter_reward       -> ('crafter', 'reward')
        dmc_walker_walk      -> ('dmc', 'walker_walk')
        atari_pong           -> ('atari', 'pong')
        gym_CartPole-v1      -> ('gym', 'CartPole-v1')
    """
    parts = task_str.split("_", 1)
    if len(parts) == 2:
        suite, remainder = parts
        if suite in ("crafter", "dmc", "atari", "atari100k", "dmlab", "gym"):
            return suite, remainder
    # Fallback: treat as generic gymnasium
    return "gym", task_str


def make_env(task_str, image_size=None, action_repeat=None, camera=-1, seed=None):
    """Create an environment from a task string.

    Returns (env, env_info_dict).
    """
    suite, task_name = parse_task(task_str)
    defaults = TASK_DEFAULTS.get(suite, {})

    if image_size is None:
        image_size = defaults.get("image_size", 64)
    size = (image_size, image_size)

    if action_repeat is None:
        action_repeat = defaults.get("action_repeat", 1)

    print(f"Suite: {suite}, Task: {task_name}")
    print(f"Image size: {size}, Action repeat: {action_repeat}")

    if suite == "crafter":
        env = make_crafter(task_name, size, seed=seed)

    elif suite in ("dmc",):
        # dmc_walker_walk -> domain='walker', task='walk'
        dmc_parts = task_name.split("_", 1)
        if len(dmc_parts) == 2:
            domain, dmc_task = dmc_parts
        else:
            domain, dmc_task = task_name, "walk"
        env = make_dmc(domain, dmc_task, size,
                       action_repeat=action_repeat, camera=camera)

    elif suite in ("atari", "atari100k"):
        gray = defaults.get("gray", False)
        env = make_atari(task_name, size,
                         action_repeat=action_repeat, gray=gray)

    elif suite == "gym":
        env = make_gymnasium(task_name, size)

    else:
        # Unknown suite, try as generic gymnasium
        print(f"Unknown suite '{suite}', trying as Gymnasium env: {task_str}")
        env = make_gymnasium(task_str, size)

    return env, size


# ---------------------------------------------------------------------------
# TCP server protocol
# ---------------------------------------------------------------------------

def get_env_info(env, image_size):
    """Extract environment space information for the info handshake."""
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    info = {
        "reward": 0.0,
        "is_first": False,
        "is_last": False,
        "is_terminal": False,
        "obs_image_shape": [image_size[0], image_size[1], 3],
    }

    act_space = env.action_space
    if hasattr(act_space, "n"):
        # Discrete
        info["act_type"] = "discrete"
        info["act_dim"] = int(act_space.n)
    elif hasattr(act_space, "shape"):
        # Box / continuous
        info["act_type"] = "continuous"
        info["act_dim"] = int(np.prod(act_space.shape))
        low = act_space.low if hasattr(act_space, "low") else np.array([-1.0])
        high = act_space.high if hasattr(act_space, "high") else np.array([1.0])
        info["act_low"] = float(np.min(low))
        info["act_high"] = float(np.max(high))
    else:
        info["act_type"] = "discrete"
        info["act_dim"] = 1

    return info


def obs_to_image(env, obs, image_size):
    """Convert observation to a resized RGB image [H, W, C]."""
    if isinstance(obs, np.ndarray) and obs.ndim >= 2:
        img = obs
    else:
        try:
            img = env.render()
        except Exception:
            img = None

    if img is None:
        return np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    img = gray_to_rgb(img)
    img = resize_image(img, image_size)

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    return img


def handle_client(conn, env, image_size):
    """Handle a single client connection."""
    buf = b""
    while True:
        try:
            data = conn.recv(65536)
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
                        "image": img.flatten().tolist(),
                        "image_shape": list(img.shape),
                        "reward": 0.0,
                        "is_first": True,
                        "is_last": False,
                        "is_terminal": False,
                    }) + "\n"
                    conn.sendall(response.encode())

                elif command == "step":
                    act_space = env.action_space
                    if hasattr(act_space, "n"):
                        action = msg.get("action_discrete", 0)
                    else:
                        action = np.array(
                            msg.get("action_continuous", [0.0]),
                            dtype=np.float32,
                        )

                    result = env.step(action)
                    if len(result) == 5:
                        obs, reward, terminated, truncated, info = result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = result
                        terminated = done

                    img = obs_to_image(env, obs, image_size)

                    # Auto-reset on done
                    if done:
                        try:
                            env.reset()
                        except Exception:
                            pass

                    response = json.dumps({
                        "image": img.flatten().tolist(),
                        "image_shape": list(img.shape),
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
            traceback.print_exc()
            print(f"Error handling command: {e}", file=sys.stderr)
            break

    conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Environment Bridge for DreamerV3-Rust",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --task crafter_reward --port 9876
  %(prog)s --task dmc_walker_walk --port 9876
  %(prog)s --task dmc_cartpole_swingup --port 9876
  %(prog)s --task dmc_cheetah_run --port 9876
  %(prog)s --task atari_pong --port 9876
  %(prog)s --task gym_CartPole-v1 --port 9876

Supported task formats:
  crafter_{reward,noreward}
  dmc_{domain}_{task}          (e.g., dmc_walker_walk, dmc_quadruped_run)
  atari_{game}                 (e.g., atari_pong, atari_breakout)
  gym_{env_id}                 (any Gymnasium env ID)
""",
    )
    parser.add_argument(
        "--task", type=str, required=True,
        help="Task string (e.g., crafter_reward, dmc_walker_walk, atari_pong)",
    )
    parser.add_argument("--port", type=int, default=9876, help="TCP port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Listen host")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Image resize resolution (default: per-suite)")
    parser.add_argument("--action-repeat", type=int, default=None,
                        help="Action repeat (default: per-suite)")
    parser.add_argument("--camera", type=int, default=-1,
                        help="DMC camera ID (-1 for auto)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    print(f"Creating environment: {args.task}")
    env, image_size = make_env(
        args.task,
        image_size=args.image_size,
        action_repeat=args.action_repeat,
        camera=args.camera,
        seed=args.seed,
    )

    info = get_env_info(env, image_size)
    print(f"Action space: {info['act_type']}, dim={info['act_dim']}")
    print(f"Observation: image {image_size[0]}x{image_size[1]}x3")

    print(f"\nStarting bridge server on {args.host}:{args.port}")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)

    print("Waiting for connection...")

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
