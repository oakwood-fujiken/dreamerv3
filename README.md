# Mastering Diverse Domains through World Models

A Rust reimplementation of [DreamerV3][paper] using the [Burn][burn] deep learning framework.
DreamerV3 is a scalable and general reinforcement learning algorithm that masters
a wide range of applications with fixed hyperparameters.

![DreamerV3 Tasks](https://user-images.githubusercontent.com/2111293/217647148-cbc522e2-61ad-4553-8e14-1ecdc8d9438b.gif)

If you find this code useful, please reference in your paper:

```
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group}
}
```

To learn more:

- [Research paper][paper]
- [Project website][website]

## DreamerV3

DreamerV3 learns a world model from experiences and uses it to train an actor
critic policy from imagined trajectories. The world model encodes sensory
inputs into categorical representations and predicts future representations and
rewards given actions.

![DreamerV3 Method Diagram](https://user-images.githubusercontent.com/2111293/217355673-4abc0ce5-1a4b-4366-a08d-64754289d659.png)

## Architecture

This implementation is built entirely in Rust using the Burn 0.16 deep learning framework.

```
dreamerv3-rust/
├── src/
│   ├── main.rs              # CLI entrypoint
│   ├── lib.rs               # Library root
│   ├── config/              # YAML-based configuration (DreamerConfig)
│   ├── nn/                  # Neural network primitives
│   │   ├── linear.rs        #   Linear layer with output scaling
│   │   ├── block_linear.rs  #   Block-diagonal linear layer
│   │   ├── conv2d.rs        #   Conv2D / ConvTranspose2D (NHWC)
│   │   ├── mlp.rs           #   Multi-layer perceptron
│   │   ├── norm.rs          #   RMSNorm / LayerNorm
│   │   └── utils.rs         #   symlog, activations
│   ├── models/              # DreamerV3 model components
│   │   ├── agent.rs         #   DreamerV3Agent (top-level module)
│   │   ├── world_model.rs   #   WorldModel (encoder + RSSM + decoder + heads)
│   │   ├── rssm.rs          #   Recurrent State-Space Model (block-wise GRU)
│   │   ├── encoder.rs       #   CNN + MLP observation encoder
│   │   ├── decoder.rs       #   Transposed CNN + MLP decoder
│   │   └── heads.rs         #   Policy, value, reward, continuation heads
│   ├── distributions/       # Probability distributions
│   │   ├── one_hot.rs       #   OneHotCategorical (with straight-through)
│   │   ├── two_hot.rs       #   TwoHotSymexp (for reward/value)
│   │   ├── categorical.rs   #   Categorical distribution
│   │   ├── normal.rs        #   Normal distribution
│   │   └── mse.rs           #   MSE / SymlogMSE loss
│   ├── envs/                # Environment interface
│   │   ├── interface.rs     #   Environment trait, DummyEnv
│   │   └── socket_env.rs    #   SocketEnv (TCP bridge to Python Gymnasium)
│   ├── replay/              # Experience replay
│   │   └── buffer.rs        #   Uniform replay buffer with sequence sampling
│   └── training/            # Training loop
│       └── trainer.rs       #   Trainer + AutodiffTrainer (gradient training)
└── scripts/
    └── gym_bridge.py        # Python bridge for Gymnasium environments
```

## Requirements

- **Rust** 1.75+ (2021 edition)
- **Burn** 0.16 with `ndarray` (CPU) or `wgpu` (GPU) backend

For connecting to Gymnasium environments:
- **Python** 3.9+ with `gymnasium` installed

## Build

```sh
cd dreamerv3-rust
cargo build --release
```

## Quick Start

### Training with DummyEnv (no Python required)

```sh
cargo run --release -- --task dummy --steps 100000 --backend ndarray
```

### Training with Gymnasium environment

1. Start the Python bridge server:

```sh
python scripts/gym_bridge.py --task CartPole-v1 --port 9876
```

2. Run the Rust agent:

```sh
cargo run --release -- \
  --task CartPole-v1 \
  --env-addr 127.0.0.1:9876 \
  --steps 1000000 \
  --backend ndarray
```

### Using GPU (WGPU backend)

```sh
cargo run --release -- \
  --task dummy \
  --backend wgpu \
  --steps 1000000
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | `dummy` | Environment name |
| `--size` | `12m` | Model size: `1m`, `12m`, `25m`, `50m`, `200m` |
| `--steps` | `1000000` | Total environment steps |
| `--batch-size` | `16` | Training batch size |
| `--batch-length` | `64` | Sequence length per batch |
| `--lr` | `4e-5` | Learning rate |
| `--backend` | `ndarray` | Compute backend: `ndarray` (CPU) or `wgpu` (GPU) |
| `--image-size` | `64` | Observation image resolution |
| `--n-actions` | `18` | Number of discrete actions |
| `--logdir` | `logdir` | Log output directory |
| `--config` | — | Path to YAML config file |
| `--env-addr` | — | TCP address for SocketEnv (e.g., `127.0.0.1:9876`) |
| `--checkpoint` | — | Checkpoint directory |
| `--resume` | `false` | Resume training from checkpoint |
| `--seed` | `0` | Random seed |

## Training

The `AutodiffTrainer` implements the full DreamerV3 training loop using Burn's autodiff backend:

1. **Collect experience** — interact with the environment using the current policy
2. **World model training** — encode observations through CNN, process with RSSM, compute KL divergence and reconstruction losses, update via Adam optimizer
3. **Imagination** — roll out trajectories in the learned world model using the current policy
4. **Actor-critic training** — compute lambda-returns from imagined trajectories, train the policy to maximize returns, train the value network to predict returns

## Checkpointing

Checkpoints are saved using Burn's `NamedMpkFileRecorder` (MessagePack format).

```sh
# Train with periodic checkpointing
cargo run --release -- --task dummy --checkpoint ./checkpoints

# Resume from checkpoint
cargo run --release -- --task dummy --checkpoint ./checkpoints --resume
```

Checkpoint contents:
- `agent.mpk` — full model weights (world model + policy + value networks)
- `metrics.json` — training progress (total steps, train steps, episodes)

## Connecting to Gymnasium Environments

The `SocketEnv` communicates with Python Gymnasium via a TCP socket using a JSON-line protocol.
The included `scripts/gym_bridge.py` bridge supports any Gymnasium-compatible environment.

```sh
# Atari
python scripts/gym_bridge.py --task ALE/Pong-v5 --port 9876

# Classic control
python scripts/gym_bridge.py --task CartPole-v1 --port 9876

# Custom image size
python scripts/gym_bridge.py --task ALE/Breakout-v5 --port 9876 --image-size 64
```

Protocol:
- `{"command": "info"}` — returns observation/action space info
- `{"command": "reset"}` — resets the environment
- `{"command": "step", "action_discrete": 3}` — steps with a discrete action
- `{"command": "step", "action_continuous": [0.1, -0.5]}` — steps with continuous actions
- `{"command": "close"}` — closes the connection

## Configuration

All hyperparameters can be set via a YAML file:

```sh
cargo run --release -- --config my_config.yaml --task dummy
```

See `src/config/dreamer_config.rs` for the full list of configurable parameters.
CLI flags override YAML values.

## Model Sizes

| Size | RSSM Deter | Hidden | Parameters (approx.) |
|------|-----------|--------|---------------------|
| `1m` | 512 | 256 | ~1M |
| `12m` | 4096 | 2048 | ~12M |
| `25m` | 8192 | 2048 | ~25M |
| `50m` | 8192 | 4096 | ~50M |
| `200m` | 8192 | 4096 | ~200M |

## Disclaimer

This repository contains a Rust reimplementation of DreamerV3 based on the
[original Python/JAX implementation](https://github.com/danijar/dreamerv3).
It is unrelated to Google or DeepMind.

[paper]: https://arxiv.org/pdf/2301.04104
[website]: https://danijar.com/dreamerv3
[burn]: https://burn.dev/
