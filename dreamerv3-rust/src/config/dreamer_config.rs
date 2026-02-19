use serde::{Deserialize, Serialize};

/// Top-level DreamerV3 configuration.
///
/// Corresponds to the YAML configuration system in `dreamerv3/configs.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamerConfig {
    /// Run configuration
    pub run: RunConfig,
    /// Model architecture
    pub model: ModelConfig,
    /// Training hyperparameters
    pub training: TrainingConfig,
    /// Replay buffer settings
    pub replay: ReplayConfig,
    /// Loss scales
    pub loss_scales: LossScales,
    /// Environment configuration
    pub env: EnvConfig,
}

impl Default for DreamerConfig {
    fn default() -> Self {
        Self {
            run: RunConfig::default(),
            model: ModelConfig::default(),
            training: TrainingConfig::default(),
            replay: ReplayConfig::default(),
            loss_scales: LossScales::default(),
            env: EnvConfig::default(),
        }
    }
}

/// Run configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    /// Total environment steps
    pub steps: usize,
    /// Number of parallel environments
    pub envs: usize,
    /// Training steps per environment step
    pub train_ratio: f64,
    /// Logging interval (in env steps)
    pub log_every: usize,
    /// Report/video interval (in env steps)
    pub report_every: usize,
    /// Random seed
    pub seed: u64,
    /// Log directory
    pub logdir: String,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            steps: 100_000_000,
            envs: 16,
            train_ratio: 32.0,
            log_every: 1000,
            report_every: 10000,
            seed: 0,
            logdir: "logdir".to_string(),
        }
    }
}

/// Model architecture configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// RSSM configuration
    pub rssm: RSSMModelConfig,
    /// Encoder configuration
    pub encoder: EncoderModelConfig,
    /// Decoder configuration
    pub decoder: DecoderModelConfig,
    /// Policy head
    pub policy: HeadConfig,
    /// Value head
    pub value: HeadConfig,
    /// Reward head
    pub reward: HeadConfig,
    /// Continuation head
    pub continue_head: HeadConfig,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            rssm: RSSMModelConfig::default(),
            encoder: EncoderModelConfig::default(),
            decoder: DecoderModelConfig::default(),
            policy: HeadConfig::default(),
            value: HeadConfig::default(),
            reward: HeadConfig::default(),
            continue_head: HeadConfig::default(),
        }
    }
}

/// RSSM model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RSSMModelConfig {
    pub deter: usize,
    pub hidden: usize,
    pub stoch: usize,
    pub classes: usize,
    pub blocks: usize,
    pub act: String,
    pub norm: String,
    pub unimix: f64,
    pub outscale: f64,
    pub imglayers: usize,
    pub obslayers: usize,
    pub dynlayers: usize,
    pub absolute: bool,
    pub free_nats: f64,
}

impl Default for RSSMModelConfig {
    fn default() -> Self {
        Self {
            deter: 4096,
            hidden: 2048,
            stoch: 32,
            classes: 32,
            blocks: 8,
            act: "gelu".to_string(),
            norm: "rms".to_string(),
            unimix: 0.01,
            outscale: 1.0,
            imglayers: 2,
            obslayers: 1,
            dynlayers: 1,
            absolute: false,
            free_nats: 1.0,
        }
    }
}

/// Encoder model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderModelConfig {
    pub depth: usize,
    pub mults: Vec<usize>,
    pub units: usize,
    pub layers: usize,
    pub kernel: usize,
    pub act: String,
    pub norm: String,
    pub symlog: bool,
}

impl Default for EncoderModelConfig {
    fn default() -> Self {
        Self {
            depth: 64,
            mults: vec![2, 3, 4, 4],
            units: 1024,
            layers: 3,
            kernel: 5,
            act: "gelu".to_string(),
            norm: "rms".to_string(),
            symlog: true,
        }
    }
}

/// Decoder model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderModelConfig {
    pub depth: usize,
    pub mults: Vec<usize>,
    pub units: usize,
    pub layers: usize,
    pub kernel: usize,
    pub act: String,
    pub norm: String,
    pub symlog: bool,
    pub bspace: usize,
}

impl Default for DecoderModelConfig {
    fn default() -> Self {
        Self {
            depth: 64,
            mults: vec![2, 3, 4, 4],
            units: 1024,
            layers: 3,
            kernel: 5,
            act: "gelu".to_string(),
            norm: "rms".to_string(),
            symlog: true,
            bspace: 8,
        }
    }
}

/// Head (MLP) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadConfig {
    pub layers: usize,
    pub units: usize,
    pub act: String,
    pub norm: String,
    pub outscale: f64,
}

impl Default for HeadConfig {
    fn default() -> Self {
        Self {
            layers: 3,
            units: 1024,
            act: "gelu".to_string(),
            norm: "rms".to_string(),
            outscale: 1.0,
        }
    }
}

/// Training hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Batch size (number of parallel sequences)
    pub batch_size: usize,
    /// Sequence length per batch
    pub batch_length: usize,
    /// Learning rate
    pub lr: f64,
    /// Adaptive gradient clipping threshold
    pub agc: f64,
    /// Optimizer epsilon
    pub eps: f64,
    /// Adam beta1
    pub beta1: f64,
    /// Adam beta2
    pub beta2: f64,
    /// Weight decay
    pub wd: f64,
    /// LR schedule warmup steps
    pub warmup: usize,
    /// Imagination horizon
    pub imag_length: usize,
    /// Discount horizon
    pub horizon: f64,
    /// Lambda for returns
    pub lambda: f64,
    /// Entropy regularization for policy
    pub actent: f64,
    /// Use slow target for value
    pub slowtar: bool,
    /// Slow target regularization weight
    pub slowreg: f64,
    /// Slow target update rate (polyak)
    pub slow_update: f64,
    /// Whether to use continuous discounting
    pub contdisc: bool,
    /// Replay context length
    pub replay_context: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 16,
            batch_length: 64,
            lr: 4e-5,
            agc: 0.3,
            eps: 1e-20,
            beta1: 0.9,
            beta2: 0.999,
            wd: 0.0,
            warmup: 1000,
            imag_length: 15,
            horizon: 333.0,
            lambda: 0.95,
            actent: 3e-4,
            slowtar: true,
            slowreg: 1.0,
            slow_update: 0.02,
            contdisc: true,
            replay_context: 1,
        }
    }
}

/// Replay buffer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayConfig {
    /// Maximum buffer size (in transitions)
    pub size: usize,
    /// Minimum length before sampling starts
    pub min_length: usize,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            size: 5_000_000,
            min_length: 1000,
        }
    }
}

/// Loss scaling factors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossScales {
    pub rec: f64,
    pub rew: f64,
    pub con: f64,
    pub dyn_loss: f64,
    pub rep: f64,
    pub policy: f64,
    pub value: f64,
    pub repval: f64,
}

impl Default for LossScales {
    fn default() -> Self {
        Self {
            rec: 1.0,
            rew: 1.0,
            con: 1.0,
            dyn_loss: 1.0,
            rep: 0.1,
            policy: 1.0,
            value: 1.0,
            repval: 0.3,
        }
    }
}

/// Environment configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvConfig {
    /// Environment type (atari, dmc, crafter, etc.)
    pub task: String,
    /// Image resolution [H, W]
    pub image_size: [usize; 2],
    /// Number of action repeat
    pub action_repeat: usize,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            task: "atari_pong".to_string(),
            image_size: [64, 64],
            action_repeat: 4,
        }
    }
}

/// Apply per-task preset configuration overrides matching the Python configs.yaml.
///
/// Task format: `{suite}_{task}` (e.g., `crafter_reward`, `dmc_walker_walk`, `atari_pong`)
pub fn task_config(config: &mut DreamerConfig) {
    let suite = config.env.task.split('_').next().unwrap_or("").to_string();
    match suite.as_str() {
        "crafter" => {
            config.env.image_size = [64, 64];
            config.env.action_repeat = 1;
            config.run.steps = 1_100_000;
            config.run.train_ratio = 512.0;
            config.run.envs = 1;
        }
        "dmc" => {
            config.env.image_size = [64, 64];
            config.env.action_repeat = 1;
            config.run.steps = 1_100_000;
            config.run.train_ratio = 256.0;
        }
        "atari" => {
            config.env.image_size = [64, 64];
            config.env.action_repeat = 4;
            config.run.steps = 51_000_000;
            config.run.train_ratio = 32.0;
        }
        "atari100k" => {
            config.env.image_size = [64, 64];
            config.env.action_repeat = 4;
            config.run.steps = 110_000;
            config.run.envs = 1;
            config.run.train_ratio = 256.0;
        }
        _ => {}
    }
}

/// Predefined model size configurations matching the Python implementation.
pub fn size_config(size: &str) -> ModelConfig {
    match size {
        "1m" => ModelConfig {
            rssm: RSSMModelConfig {
                deter: 512,
                hidden: 256,
                ..Default::default()
            },
            encoder: EncoderModelConfig {
                depth: 16,
                units: 256,
                ..Default::default()
            },
            decoder: DecoderModelConfig {
                depth: 16,
                units: 256,
                ..Default::default()
            },
            policy: HeadConfig {
                units: 256,
                ..Default::default()
            },
            value: HeadConfig {
                units: 256,
                ..Default::default()
            },
            reward: HeadConfig {
                units: 256,
                ..Default::default()
            },
            continue_head: HeadConfig {
                units: 256,
                ..Default::default()
            },
        },
        "12m" | "default" => ModelConfig::default(),
        "25m" => ModelConfig {
            rssm: RSSMModelConfig {
                deter: 8192,
                hidden: 2048,
                stoch: 32,
                classes: 64,
                ..Default::default()
            },
            ..Default::default()
        },
        "50m" => ModelConfig {
            rssm: RSSMModelConfig {
                deter: 8192,
                hidden: 4096,
                stoch: 32,
                classes: 64,
                ..Default::default()
            },
            encoder: EncoderModelConfig {
                depth: 96,
                units: 1536,
                ..Default::default()
            },
            decoder: DecoderModelConfig {
                depth: 96,
                units: 1536,
                ..Default::default()
            },
            policy: HeadConfig {
                units: 1536,
                ..Default::default()
            },
            value: HeadConfig {
                units: 1536,
                ..Default::default()
            },
            reward: HeadConfig {
                units: 1536,
                ..Default::default()
            },
            continue_head: HeadConfig {
                units: 1536,
                ..Default::default()
            },
        },
        "200m" => ModelConfig {
            rssm: RSSMModelConfig {
                deter: 8192,
                hidden: 4096,
                stoch: 64,
                classes: 64,
                ..Default::default()
            },
            encoder: EncoderModelConfig {
                depth: 128,
                units: 2048,
                ..Default::default()
            },
            decoder: DecoderModelConfig {
                depth: 128,
                units: 2048,
                ..Default::default()
            },
            policy: HeadConfig {
                units: 2048,
                ..Default::default()
            },
            value: HeadConfig {
                units: 2048,
                ..Default::default()
            },
            reward: HeadConfig {
                units: 2048,
                ..Default::default()
            },
            continue_head: HeadConfig {
                units: 2048,
                ..Default::default()
            },
        },
        _ => {
            log::warn!("Unknown size '{}', using default", size);
            ModelConfig::default()
        }
    }
}
