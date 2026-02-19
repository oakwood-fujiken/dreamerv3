#![recursion_limit = "256"]

use burn::backend::Autodiff;
use burn::prelude::Backend;
use clap::Parser;
use log::info;
use std::process::{Child, Command, Stdio};

use dreamerv3::config::{DreamerConfig, size_config, task_config};
use dreamerv3::envs::{ActionSpace, DummyEnv, Environment, SocketEnv};
use dreamerv3::models::agent::DreamerV3AgentConfig;
use dreamerv3::training::AutodiffTrainer;

/// DreamerV3: Mastering Diverse Domains through World Models
///
/// Rust/Burn implementation of DreamerV3, a model-based reinforcement learning
/// algorithm that learns a world model and trains a policy via imagination.
///
/// Supported task formats:
///   crafter_reward, crafter_noreward
///   dmc_walker_walk, dmc_cartpole_swingup, dmc_cheetah_run, ...
///   atari_pong, atari_breakout, ...
///   dummy (built-in test environment)
#[derive(Parser, Debug)]
#[command(name = "dreamerv3")]
#[command(about = "DreamerV3 - World Model RL Agent (Rust/Burn)")]
struct Cli {
    /// Task/environment (e.g., crafter_reward, dmc_walker_walk, atari_pong, dummy)
    #[arg(short, long, default_value = "dummy")]
    task: String,

    /// Model size: 1m, 12m, 25m, 50m, 200m
    #[arg(short, long, default_value = "12m")]
    size: String,

    /// Total environment steps (0 = use task preset)
    #[arg(long, default_value_t = 0)]
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

    /// Image resolution (0 = use task preset)
    #[arg(long, default_value_t = 0)]
    image_size: usize,

    /// Number of actions (for dummy env; 0 = auto-detect from bridge)
    #[arg(long, default_value_t = 0)]
    n_actions: usize,

    /// Config YAML file path (optional)
    #[arg(long)]
    config: Option<String>,

    /// Environment bridge address (host:port) for SocketEnv.
    /// If not specified, the bridge is auto-launched for known tasks.
    #[arg(long)]
    env_addr: Option<String>,

    /// Bridge port for auto-launched Python bridge
    #[arg(long, default_value_t = 9876)]
    bridge_port: u16,

    /// Python executable for auto-launching the bridge
    #[arg(long, default_value = "python3")]
    python: String,

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

    // Set task and apply per-task preset defaults
    config.env.task = cli.task.clone();
    task_config(&mut config);

    // Override with CLI arguments (non-zero values override presets)
    if cli.steps > 0 {
        config.run.steps = cli.steps;
    }
    if cli.image_size > 0 {
        config.env.image_size = [cli.image_size, cli.image_size];
    }
    config.run.logdir = cli.logdir.clone();
    config.run.seed = cli.seed;
    config.training.batch_size = cli.batch_size;
    config.training.batch_length = cli.batch_length;
    config.training.lr = cli.lr;
    config.model = size_config(&cli.size);

    println!("Steps:   {}", config.run.steps);
    println!("Image:   {}x{}", config.env.image_size[0], config.env.image_size[1]);
    println!("Train ratio: {}", config.run.train_ratio);
    println!();

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

/// Try to find the gym_bridge.py script relative to the executable.
fn find_bridge_script() -> Option<String> {
    // Try relative to current dir
    let candidates = [
        "scripts/gym_bridge.py",
        "dreamerv3-rust/scripts/gym_bridge.py",
        "../scripts/gym_bridge.py",
    ];
    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }
    // Try relative to executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("../scripts/gym_bridge.py");
            if p.exists() {
                return Some(p.to_string_lossy().to_string());
            }
        }
    }
    None
}

/// Launch the Python bridge subprocess and return (child, address).
fn launch_bridge(python: &str, task: &str, port: u16) -> (Child, String) {
    let script = find_bridge_script().unwrap_or_else(|| {
        eprintln!("Could not find scripts/gym_bridge.py.");
        eprintln!("Either run from the dreamerv3-rust directory, or use --env-addr to connect to a manually started bridge.");
        std::process::exit(1);
    });

    info!("Launching Python bridge: {} {} --task {} --port {}", python, script, task, port);

    let child = Command::new(python)
        .arg(&script)
        .arg("--task")
        .arg(task)
        .arg("--port")
        .arg(port.to_string())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap_or_else(|e| {
            eprintln!("Failed to launch Python bridge: {}", e);
            eprintln!("Make sure '{}' is installed and the required packages (crafter, dm_control, gymnasium, etc.) are available.", python);
            std::process::exit(1);
        });

    // Wait for the bridge to start listening
    let addr = format!("127.0.0.1:{}", port);
    info!("Waiting for bridge to start on {}...", addr);
    for attempt in 0..20 {
        std::thread::sleep(std::time::Duration::from_millis(500));
        if std::net::TcpStream::connect(&addr).is_ok() {
            info!("Bridge is ready (attempt {})", attempt + 1);
            return (child, addr);
        }
    }

    eprintln!("Bridge did not start within 10 seconds. Check Python output above for errors.");
    std::process::exit(1);
}

fn run_training<InnerB: Backend>(config: DreamerConfig, cli: &Cli, device: InnerB::Device) {
    let image_size = config.env.image_size;

    // Determine checkpoint directory
    let checkpoint_dir = cli
        .checkpoint
        .clone()
        .unwrap_or_else(|| format!("{}/checkpoints", cli.logdir));

    if cli.task == "dummy" {
        // Built-in dummy environment
        let n_actions = if cli.n_actions > 0 { cli.n_actions } else { 18 };
        let agent_config = DreamerV3AgentConfig::for_discrete_actions(
            image_size,
            3,
            n_actions,
        );
        let agent = agent_config.init::<Autodiff<InnerB>>(&device);

        info!("Created DummyEnv with {} discrete actions", n_actions);

        let mut env = DummyEnv::new([image_size[0], image_size[1], 3], n_actions, 1000);
        let mut trainer = AutodiffTrainer::<InnerB>::new(config, agent, device);
        if cli.resume {
            trainer.maybe_load_checkpoint(&checkpoint_dir);
        }
        trainer.set_checkpoint_dir(&checkpoint_dir);
        trainer.train_autodiff(&mut env);
        return;
    }

    // For all other tasks, connect to (or launch) the Python bridge
    let mut bridge_child: Option<Child> = None;

    let addr = if let Some(ref addr) = cli.env_addr {
        addr.clone()
    } else {
        // Auto-launch bridge
        let (child, addr) = launch_bridge(&cli.python, &cli.task, cli.bridge_port);
        bridge_child = Some(child);
        addr
    };

    info!("Connecting to environment at {}...", addr);
    let mut env = SocketEnv::connect(&addr)
        .unwrap_or_else(|e| panic!("Failed to connect to env at {}: {}", addr, e));
    info!("Connected to SocketEnv at {}", addr);

    // Detect action space from the environment
    let act_space = env.act_space();
    let action_dim = act_space.dim();

    let agent_config = match &act_space {
        ActionSpace::Discrete { n } => {
            info!("Discrete action space: {} actions", n);
            DreamerV3AgentConfig::for_discrete_actions(image_size, 3, *n)
        }
        ActionSpace::Continuous { dim, low, high } => {
            info!("Continuous action space: dim={}, range=[{}, {}]", dim, low, high);
            DreamerV3AgentConfig::for_continuous_actions(image_size, 3, *dim)
        }
    };
    let agent = agent_config.init::<Autodiff<InnerB>>(&device);

    info!("Agent created | action_dim={} | feat_dim={}",
        action_dim,
        config.model.rssm.deter + config.model.rssm.stoch * config.model.rssm.classes,
    );

    let mut trainer = AutodiffTrainer::<InnerB>::new(config, agent, device);
    if cli.resume {
        trainer.maybe_load_checkpoint(&checkpoint_dir);
    }
    trainer.set_checkpoint_dir(&checkpoint_dir);
    trainer.train_autodiff(&mut env);

    // Cleanup bridge subprocess
    if let Some(mut child) = bridge_child {
        info!("Shutting down bridge subprocess...");
        let _ = child.kill();
        let _ = child.wait();
    }
}
