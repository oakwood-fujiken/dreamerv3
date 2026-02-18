#![recursion_limit = "256"]

use burn::backend::Autodiff;
use burn::prelude::Backend;
use clap::Parser;
use log::info;

use dreamerv3::config::{DreamerConfig, size_config};
use dreamerv3::envs::{DummyEnv, SocketEnv};
use dreamerv3::models::agent::DreamerV3AgentConfig;
use dreamerv3::training::AutodiffTrainer;

/// DreamerV3: Mastering Diverse Domains through World Models
///
/// Rust/Burn implementation of DreamerV3, a model-based reinforcement learning
/// algorithm that learns a world model and trains a policy via imagination.
#[derive(Parser, Debug)]
#[command(name = "dreamerv3")]
#[command(about = "DreamerV3 - World Model RL Agent (Rust/Burn)")]
struct Cli {
    /// Task/environment to run (e.g., atari_pong, dmc_walker_walk)
    #[arg(short, long, default_value = "dummy")]
    task: String,

    /// Model size: 1m, 12m, 25m, 50m, 200m
    #[arg(short, long, default_value = "12m")]
    size: String,

    /// Total environment steps
    #[arg(long, default_value_t = 1_000_000)]
    steps: usize,

    /// Batch size
    #[arg(long, default_value_t = 16)]
    batch_size: usize,

    /// Sequence length
    #[arg(long, default_value_t = 64)]
    batch_length: usize,

    /// Learning rate
    #[arg(long, default_value_t = 4e-5)]
    lr: f64,

    /// Log directory
    #[arg(long, default_value = "logdir")]
    logdir: String,

    /// Random seed
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Backend: wgpu, ndarray
    #[arg(long, default_value = "ndarray")]
    backend: String,

    /// Image resolution
    #[arg(long, default_value_t = 64)]
    image_size: usize,

    /// Number of actions (for dummy env)
    #[arg(long, default_value_t = 18)]
    n_actions: usize,

    /// Config YAML file path (optional)
    #[arg(long)]
    config: Option<String>,

    /// Environment bridge address (host:port) for SocketEnv
    #[arg(long)]
    env_addr: Option<String>,

    /// Checkpoint directory for saving/loading
    #[arg(long)]
    checkpoint: Option<String>,

    /// Resume from checkpoint if available
    #[arg(long, default_value_t = false)]
    resume: bool,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    println!(r"---  ___                           __   ______ ---");
    println!(r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---");
    println!(r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---");
    println!(r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---");
    println!();
    println!("DreamerV3 - Rust/Burn Implementation");
    println!("====================================");
    println!("Task:    {}", cli.task);
    println!("Size:    {}", cli.size);
    println!("Steps:   {}", cli.steps);
    println!("Backend: {}", cli.backend);
    println!();

    // Load or create config
    let mut config = if let Some(config_path) = &cli.config {
        let yaml = std::fs::read_to_string(config_path)
            .unwrap_or_else(|e| panic!("Failed to read config file {}: {}", config_path, e));
        serde_yaml::from_str(&yaml)
            .unwrap_or_else(|e| panic!("Failed to parse config YAML: {}", e))
    } else {
        DreamerConfig::default()
    };

    // Override with CLI arguments
    config.run.steps = cli.steps;
    config.run.logdir = cli.logdir.clone();
    config.run.seed = cli.seed;
    config.training.batch_size = cli.batch_size;
    config.training.batch_length = cli.batch_length;
    config.training.lr = cli.lr;
    config.model = size_config(&cli.size);
    config.env.task = cli.task.clone();
    config.env.image_size = [cli.image_size, cli.image_size];

    match cli.backend.as_str() {
        "ndarray" => run_with_ndarray(config, &cli),
        "wgpu" => run_with_wgpu(config, &cli),
        _ => {
            eprintln!("Unknown backend: {}. Use 'ndarray' or 'wgpu'.", cli.backend);
            std::process::exit(1);
        }
    }
}

fn run_with_ndarray(config: DreamerConfig, cli: &Cli) {
    use burn::backend::NdArray;
    type InnerB = NdArray;

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    info!("Using NdArray (CPU) backend with Autodiff");

    run_training::<InnerB>(config, cli, device);
}

fn run_with_wgpu(config: DreamerConfig, cli: &Cli) {
    use burn::backend::Wgpu;
    type InnerB = Wgpu;

    let device = burn::backend::wgpu::WgpuDevice::default();
    info!("Using WGPU (GPU) backend with Autodiff");

    run_training::<InnerB>(config, cli, device);
}

fn run_training<InnerB: Backend>(config: DreamerConfig, cli: &Cli, device: InnerB::Device) {
    let image_size = config.env.image_size;
    let n_actions = cli.n_actions;

    // Create agent with Autodiff backend
    let agent_config = DreamerV3AgentConfig::for_discrete_actions(
        image_size,
        3, // RGB
        n_actions,
    );
    let agent = agent_config.init::<Autodiff<InnerB>>(&device);

    info!("Agent created with {} discrete actions", n_actions);
    info!(
        "Feature dim: {}",
        config.model.rssm.deter + config.model.rssm.stoch * config.model.rssm.classes
    );

    // Determine checkpoint directory
    let checkpoint_dir = cli
        .checkpoint
        .clone()
        .unwrap_or_else(|| format!("{}/checkpoints", cli.logdir));

    // Create environment and run training with autodiff
    if let Some(ref addr) = cli.env_addr {
        // Connect to external Gymnasium environment via TCP socket
        info!("Connecting to environment at {}...", addr);
        let mut env = SocketEnv::connect(addr)
            .unwrap_or_else(|e| panic!("Failed to connect to env at {}: {}", addr, e));
        info!("Connected to SocketEnv at {}", addr);

        let mut trainer = AutodiffTrainer::<InnerB>::new(config, agent, device);
        if cli.resume {
            trainer.maybe_load_checkpoint(&checkpoint_dir);
        }
        trainer.set_checkpoint_dir(&checkpoint_dir);
        trainer.train_autodiff(&mut env);
    } else {
        match cli.task.as_str() {
            "dummy" => {
                let mut env = DummyEnv::new(
                    [image_size[0], image_size[1], 3],
                    n_actions,
                    1000,
                );
                info!("Created DummyEnv");

                let mut trainer = AutodiffTrainer::<InnerB>::new(config, agent, device);
                if cli.resume {
                    trainer.maybe_load_checkpoint(&checkpoint_dir);
                }
                trainer.set_checkpoint_dir(&checkpoint_dir);
                trainer.train_autodiff(&mut env);
            }
            _ => {
                eprintln!(
                    "Environment '{}' not implemented. Use --env-addr to connect to a Python bridge, or use 'dummy'.",
                    cli.task
                );
                eprintln!(
                    "Example: python scripts/gym_bridge.py --task {} --port 9876",
                    cli.task
                );
                eprintln!(
                    "Then run: dreamerv3 --task {} --env-addr 127.0.0.1:9876",
                    cli.task
                );
                std::process::exit(1);
            }
        }
    }
}
