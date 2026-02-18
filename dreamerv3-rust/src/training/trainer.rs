use burn::backend::Autodiff;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{NamedMpkFileRecorder, FullPrecisionSettings};
use burn::tensor::ElementConversion;

use crate::config::DreamerConfig;
use crate::envs::{Action, ActionSpace, Environment, Observation};
use crate::models::agent::DreamerV3Agent;
use crate::replay::{BatchData, ReplayBuffer, Transition};

/// Training metrics collected during training.
#[derive(Debug, Clone, Default)]
pub struct TrainMetrics {
    pub total_steps: usize,
    pub train_steps: usize,
    pub episodes: usize,
    pub episode_return: f64,
    pub episode_length: usize,
    pub wm_loss: f64,
    pub actor_loss: f64,
    pub value_loss: f64,
}

/// DreamerV3 Trainer with checkpointing support.
///
/// Supports:
/// - Environment interaction with policy inference
/// - Periodic checkpointing with save/resume
/// - Replay buffer management
pub struct Trainer<B: Backend> {
    config: DreamerConfig,
    agent: DreamerV3Agent<B>,
    replay: ReplayBuffer,
    metrics: TrainMetrics,
    device: B::Device,
    checkpoint_dir: Option<String>,
    checkpoint_every: usize,
}

impl<B: Backend> Trainer<B> {
    /// Create a new trainer.
    pub fn new(
        config: DreamerConfig,
        agent: DreamerV3Agent<B>,
        device: B::Device,
    ) -> Self {
        let replay = ReplayBuffer::new(
            config.replay.size,
            config.training.batch_length,
        );

        Self {
            config,
            agent,
            replay,
            metrics: TrainMetrics::default(),
            device,
            checkpoint_dir: None,
            checkpoint_every: 10000,
        }
    }

    /// Set checkpoint directory.
    pub fn set_checkpoint_dir(&mut self, dir: &str) {
        self.checkpoint_dir = Some(dir.to_string());
    }

    /// Save agent checkpoint.
    pub fn save_checkpoint(&self, path: &str) {
        std::fs::create_dir_all(path).unwrap_or_else(|e| {
            log::warn!("Failed to create checkpoint dir {}: {}", path, e);
        });

        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        let model_path = format!("{}/agent", path);
        match self.agent.clone().save_file(&model_path, &recorder) {
            Ok(()) => log::info!("Checkpoint saved to {}", path),
            Err(e) => log::error!("Failed to save checkpoint: {}", e),
        }

        // Save metrics as JSON
        let metrics_path = format!("{}/metrics.json", path);
        let metrics_json = format!(
            "{{\"total_steps\":{},\"train_steps\":{},\"episodes\":{}}}",
            self.metrics.total_steps, self.metrics.train_steps, self.metrics.episodes
        );
        std::fs::write(&metrics_path, metrics_json).unwrap_or_else(|e| {
            log::warn!("Failed to save metrics: {}", e);
        });
    }

    /// Try to load checkpoint if available.
    pub fn maybe_load_checkpoint(&mut self, path: &str) {
        let model_path = format!("{}/agent", path);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        match self.agent.clone().load_file(&model_path, &recorder, &self.device) {
            Ok(loaded) => {
                self.agent = loaded;
                log::info!("Loaded checkpoint from {}", path);

                // Load metrics
                let metrics_path = format!("{}/metrics.json", path);
                if let Ok(json) = std::fs::read_to_string(&metrics_path) {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json) {
                        self.metrics.total_steps =
                            parsed["total_steps"].as_u64().unwrap_or(0) as usize;
                        self.metrics.train_steps =
                            parsed["train_steps"].as_u64().unwrap_or(0) as usize;
                        self.metrics.episodes =
                            parsed["episodes"].as_u64().unwrap_or(0) as usize;
                        log::info!(
                            "Resumed from step {}, train step {}, episode {}",
                            self.metrics.total_steps,
                            self.metrics.train_steps,
                            self.metrics.episodes
                        );
                    }
                }
            }
            Err(_) => {
                log::info!("No checkpoint found at {}, starting fresh", path);
            }
        }
    }

    /// Get current training metrics.
    pub fn metrics(&self) -> &TrainMetrics {
        &self.metrics
    }

    /// Run the full training loop using only the base Backend (inference-only, no gradient training).
    pub fn train<E: Environment>(&mut self, env: &mut E) {
        log::info!("Starting DreamerV3 training (inference-only mode, no autodiff)");
        log::info!("Config: {:?}", self.config.run);

        let total_steps = self.config.run.steps;

        // Initialize environment
        let mut obs = env.reset();
        let act_space = env.act_space();
        let action_dim = act_space.dim();

        // Initialize RSSM carry state
        let mut carry = self.agent.world_model.rssm.initial(1, &self.device);
        let mut prev_action = Tensor::<B, 2>::zeros([1, action_dim], &self.device);
        let mut episode_return = 0.0f64;
        let mut episode_length = 0usize;

        log::info!(
            "Prefilling replay buffer (min {} transitions)...",
            self.config.replay.min_length
        );

        while self.metrics.total_steps < total_steps {
            // Store transition
            let transition = obs_to_transition(&obs, &prev_action_to_vec(action_dim));
            self.replay.add(transition);
            self.metrics.total_steps += 1;

            // Select action
            let action = if self.replay.len() < self.config.replay.min_length {
                random_action(&act_space)
            } else {
                let image = obs_to_image_tensor::<B>(&obs, &self.device);
                let is_first_val = if obs.is_first { 1i32 } else { 0i32 };
                let is_first = Tensor::<B, 1, Int>::from_ints(
                    [is_first_val].as_slice(),
                    &self.device,
                )
                .equal_elem(1);

                let (new_carry, action_tensor) = self.agent.policy_step(
                    &carry,
                    image,
                    None,
                    prev_action.clone(),
                    is_first,
                );
                carry = new_carry;
                prev_action = action_tensor.clone();
                tensor_to_action::<B>(&action_tensor, &act_space)
            };

            // Step environment
            let next_obs = env.step(&action);
            episode_return += next_obs.reward as f64;
            episode_length += 1;

            // Handle episode end
            if next_obs.is_last {
                self.metrics.episodes += 1;
                self.metrics.episode_return = episode_return;
                self.metrics.episode_length = episode_length;

                if self.metrics.episodes % 10 == 0 {
                    log::info!(
                        "Episode {} | Steps {} | Return {:.2} | Length {}",
                        self.metrics.episodes,
                        self.metrics.total_steps,
                        episode_return,
                        episode_length,
                    );
                }

                episode_return = 0.0;
                episode_length = 0;
                obs = env.reset();
                carry = self.agent.world_model.rssm.initial(1, &self.device);
                prev_action = Tensor::zeros([1, action_dim], &self.device);
            } else {
                obs = next_obs;
            }

            // Checkpointing
            if let Some(ref dir) = self.checkpoint_dir {
                if self.metrics.total_steps % self.checkpoint_every == 0 {
                    self.save_checkpoint(dir);
                }
            }

            // Logging
            if self.metrics.total_steps % self.config.run.log_every == 0 {
                log::info!(
                    "Step {} | Buffer size {} | Episodes {}",
                    self.metrics.total_steps,
                    self.replay.len(),
                    self.metrics.episodes,
                );
            }
        }

        // Final checkpoint
        if let Some(ref dir) = self.checkpoint_dir {
            self.save_checkpoint(dir);
        }

        log::info!("Training complete after {} steps", self.metrics.total_steps);
    }
}

/// Autodiff-enabled trainer that performs actual gradient-based training.
///
/// This wraps a base `Trainer` with autodiff capabilities, using an Adam optimizer
/// for all DreamerV3 components (world model, actor, critic).
///
/// Training flow per step:
/// 1. World model loss: encode observations, RSSM observe, compute KL + reconstruction losses
/// 2. Imagination: roll out trajectories in the learned world model
/// 3. Actor loss: maximize lambda-returns from imagined trajectories
/// 4. Value loss: minimize prediction error on lambda-returns
pub struct AutodiffTrainer<InnerB: Backend> {
    trainer: Trainer<Autodiff<InnerB>>,
    wm_optimizer: burn::optim::adaptor::OptimizerAdaptor<
        burn::optim::Adam,
        DreamerV3Agent<Autodiff<InnerB>>,
        Autodiff<InnerB>,
    >,
    lr: f64,
}

impl<InnerB: Backend> AutodiffTrainer<InnerB> {
    /// Create a new autodiff trainer.
    pub fn new(
        config: DreamerConfig,
        agent: DreamerV3Agent<Autodiff<InnerB>>,
        device: <Autodiff<InnerB> as Backend>::Device,
    ) -> Self {
        let lr = config.training.lr;

        let optimizer = AdamConfig::new()
            .with_epsilon(config.training.eps as f32)
            .with_beta_1(config.training.beta1 as f32)
            .with_beta_2(config.training.beta2 as f32)
            .init();

        let trainer = Trainer::new(config, agent, device);

        Self {
            trainer,
            wm_optimizer: optimizer,
            lr,
        }
    }

    /// Set checkpoint directory.
    pub fn set_checkpoint_dir(&mut self, dir: &str) {
        self.trainer.set_checkpoint_dir(dir);
    }

    /// Try to load checkpoint if available.
    pub fn maybe_load_checkpoint(&mut self, path: &str) {
        self.trainer.maybe_load_checkpoint(path);
    }

    /// Run the full training loop with gradient-based learning.
    pub fn train_autodiff<E: Environment>(&mut self, env: &mut E) {
        log::info!("Starting DreamerV3 training (autodiff mode)");
        log::info!("Config: {:?}", self.trainer.config.run);

        let total_steps = self.trainer.config.run.steps;
        let batch_size = self.trainer.config.training.batch_size;
        let batch_length = self.trainer.config.training.batch_length;
        let train_ratio = batch_size as f64
            * batch_length as f64
            * self.trainer.config.run.train_ratio;
        let image_size = self.trainer.config.env.image_size;

        // Initialize environment
        let mut obs = env.reset();
        let act_space = env.act_space();
        let action_dim = act_space.dim();

        // Initialize RSSM carry state
        let mut carry = self
            .trainer
            .agent
            .world_model
            .rssm
            .initial(1, &self.trainer.device);
        let mut prev_action =
            Tensor::<Autodiff<InnerB>, 2>::zeros([1, action_dim], &self.trainer.device);
        let mut episode_return = 0.0f64;
        let mut episode_length = 0usize;
        let mut train_step_counter = 0usize;

        log::info!(
            "Prefilling replay buffer (min {} transitions)...",
            self.trainer.config.replay.min_length
        );

        while self.trainer.metrics.total_steps < total_steps {
            // Store transition
            let transition = obs_to_transition(&obs, &prev_action_to_vec(action_dim));
            self.trainer.replay.add(transition);
            self.trainer.metrics.total_steps += 1;

            // Select action
            let action = if self.trainer.replay.len() < self.trainer.config.replay.min_length {
                random_action(&act_space)
            } else {
                let image =
                    obs_to_image_tensor::<Autodiff<InnerB>>(&obs, &self.trainer.device);
                let is_first_val = if obs.is_first { 1i32 } else { 0i32 };
                let is_first = Tensor::<Autodiff<InnerB>, 1, Int>::from_ints(
                    [is_first_val].as_slice(),
                    &self.trainer.device,
                )
                .equal_elem(1);

                let (new_carry, action_tensor) = self.trainer.agent.policy_step(
                    &carry,
                    image,
                    None,
                    prev_action.clone(),
                    is_first,
                );
                carry = new_carry;
                prev_action = action_tensor.clone();
                tensor_to_action::<Autodiff<InnerB>>(&action_tensor, &act_space)
            };

            // Step environment
            let next_obs = env.step(&action);
            episode_return += next_obs.reward as f64;
            episode_length += 1;

            // Handle episode end
            if next_obs.is_last {
                self.trainer.metrics.episodes += 1;
                self.trainer.metrics.episode_return = episode_return;
                self.trainer.metrics.episode_length = episode_length;

                if self.trainer.metrics.episodes % 10 == 0 {
                    log::info!(
                        "Episode {} | Steps {} | Return {:.2} | Length {}",
                        self.trainer.metrics.episodes,
                        self.trainer.metrics.total_steps,
                        episode_return,
                        episode_length,
                    );
                }

                episode_return = 0.0;
                episode_length = 0;
                obs = env.reset();
                carry = self
                    .trainer
                    .agent
                    .world_model
                    .rssm
                    .initial(1, &self.trainer.device);
                prev_action = Tensor::zeros([1, action_dim], &self.trainer.device);
            } else {
                obs = next_obs;
            }

            // Training
            if self.trainer.replay.can_sample(batch_length)
                && self.trainer.replay.len() >= self.trainer.config.replay.min_length
            {
                let n_train_steps = (train_ratio / self.trainer.metrics.total_steps as f64)
                    .max(1.0) as usize;
                for _ in 0..n_train_steps.min(1) {
                    self.train_step(batch_size, batch_length, image_size, action_dim);
                    train_step_counter += 1;
                }
            }

            // Checkpointing
            if let Some(ref dir) = self.trainer.checkpoint_dir.clone() {
                if self.trainer.metrics.total_steps % self.trainer.checkpoint_every == 0 {
                    self.trainer.save_checkpoint(dir);
                }
            }

            // Logging
            if self.trainer.metrics.total_steps % self.trainer.config.run.log_every == 0 {
                log::info!(
                    "Step {} | Train steps {} | Buffer {} | Ep {} | WM {:.4} | Act {:.4} | Val {:.4}",
                    self.trainer.metrics.total_steps,
                    train_step_counter,
                    self.trainer.replay.len(),
                    self.trainer.metrics.episodes,
                    self.trainer.metrics.wm_loss,
                    self.trainer.metrics.actor_loss,
                    self.trainer.metrics.value_loss,
                );
            }
        }

        // Final checkpoint
        if let Some(ref dir) = self.trainer.checkpoint_dir.clone() {
            self.trainer.save_checkpoint(dir);
        }

        log::info!(
            "Training complete after {} steps",
            self.trainer.metrics.total_steps
        );
    }

    /// Execute a single training step with gradient computation.
    fn train_step(
        &mut self,
        batch_size: usize,
        batch_length: usize,
        image_size: [usize; 2],
        action_dim: usize,
    ) {
        let sequences = self.trainer.replay.sample(batch_size, batch_length);
        if sequences.is_empty() {
            return;
        }

        self.trainer.metrics.train_steps += 1;

        // Convert batch data to tensors
        let batch = BatchData::from_sequences(&sequences);
        let device = &self.trainer.device;

        // Build observation images: [B, T, H, W, C]
        let obs_images = batch_to_image_tensor::<Autodiff<InnerB>>(
            &batch.observations,
            batch_size,
            batch_length,
            image_size,
            device,
        );

        // Build actions: [B, T, action_dim]
        let actions = batch_to_action_tensor::<Autodiff<InnerB>>(
            &batch.actions,
            batch_size,
            batch_length,
            action_dim,
            device,
        );

        // Build rewards: [B, T]
        let rewards = batch_to_reward_tensor::<Autodiff<InnerB>>(
            &batch.rewards,
            batch_size,
            batch_length,
            device,
        );

        // Build is_first: [B, T]
        let is_first = batch_to_bool_tensor::<Autodiff<InnerB>>(
            &batch.is_first,
            batch_size,
            batch_length,
            device,
        );

        // Build is_terminal: [B, T]
        let is_terminal = batch_to_bool_tensor::<Autodiff<InnerB>>(
            &batch.is_terminal,
            batch_size,
            batch_length,
            device,
        );

        // Initial RSSM state
        let carry = self
            .trainer
            .agent
            .world_model
            .rssm
            .initial(batch_size, device);

        // ---- World Model Training ----
        let (_new_carry, losses, feat) = self.trainer.agent.world_model_loss(
            carry,
            Some(obs_images),
            None,
            actions,
            rewards,
            is_first,
            is_terminal,
        );

        // Compute total world model loss
        let dyn_loss_mean = losses.dyn_loss.clone().mean();
        let rep_loss_mean = losses.rep_loss.clone().mean();
        let scales = &self.trainer.config.loss_scales;
        let wm_loss = dyn_loss_mean * scales.dyn_loss + rep_loss_mean * scales.rep;

        let wm_loss_val: f32 = wm_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
        self.trainer.metrics.wm_loss = wm_loss_val as f64;

        // Backward pass and optimizer step for world model
        let grads = wm_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.trainer.agent);
        self.trainer.agent =
            self.wm_optimizer
                .step(self.lr, self.trainer.agent.clone(), grads);

        // ---- Actor-Critic Training via Imagination ----
        let feat_tensor = feat.to_tensor();
        let feat_dims = feat_tensor.dims();
        let b = feat_dims[0];
        let t = feat_dims[1];

        // Use last timestep states as imagination starts
        let deter_flat = feat
            .deter
            .clone()
            .reshape([b * t, feat.deter.dims()[2]]);
        let stoch_flat = {
            let sd = feat.stoch.dims();
            feat.stoch.clone().reshape([b * t, sd[2], sd[3]])
        };

        let starts = crate::models::rssm::RSSMState {
            deter: deter_flat,
            stoch: stoch_flat,
        };

        // Imagine trajectories
        let imag_horizon = self.trainer.config.training.imag_length;
        let imag_losses = self.trainer.agent.imagination_loss(starts, imag_horizon);

        // Compute returns
        let rew = imag_losses.reward_pred.clone().squeeze::<2>(2);
        let cont = imag_losses.continue_pred.clone().squeeze::<2>(2);
        let val = imag_losses.value_pred.clone().squeeze::<2>(2);

        let discount = 1.0 - 1.0 / self.trainer.config.training.horizon;
        let lambda = self.trainer.config.training.lambda;
        let returns = crate::models::agent::lambda_return(rew, cont, val.clone(), discount, lambda);

        // Actor loss: maximize returns
        if returns.dims()[1] > 0 {
            let actor_loss = -(returns.clone().mean());
            let actor_loss_val: f32 =
                actor_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            self.trainer.metrics.actor_loss = actor_loss_val as f64;

            let actor_grads = actor_loss.backward();
            let actor_grads = GradientsParams::from_grads(actor_grads, &self.trainer.agent);
            self.trainer.agent =
                self.wm_optimizer
                    .step(self.lr, self.trainer.agent.clone(), actor_grads);
        }

        // Value loss: minimize squared error
        if returns.dims()[1] > 0 {
            let returns_len = returns.dims()[1];
            let val_truncated = val.slice([0..b * t, 0..returns_len]);
            let value_loss = (val_truncated - returns).powf_scalar(2.0).mean();

            let value_loss_val: f32 =
                value_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            self.trainer.metrics.value_loss = value_loss_val as f64;

            let value_grads = value_loss.backward();
            let value_grads = GradientsParams::from_grads(value_grads, &self.trainer.agent);
            self.trainer.agent =
                self.wm_optimizer
                    .step(self.lr, self.trainer.agent.clone(), value_grads);
        }

        log::debug!(
            "Train step {} | WM {:.4} | Act {:.4} | Val {:.4}",
            self.trainer.metrics.train_steps,
            self.trainer.metrics.wm_loss,
            self.trainer.metrics.actor_loss,
            self.trainer.metrics.value_loss,
        );
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn obs_to_transition(obs: &Observation, prev_action: &[f32]) -> Transition {
    let observation = if let Some(ref img) = obs.image {
        img.iter().map(|&b| b as f32).collect()
    } else if let Some(ref vec) = obs.vector {
        vec.clone()
    } else {
        vec![]
    };

    Transition {
        observation,
        action: prev_action.to_vec(),
        reward: obs.reward,
        is_first: obs.is_first,
        is_last: obs.is_last,
        is_terminal: obs.is_terminal,
    }
}

fn prev_action_to_vec(action_dim: usize) -> Vec<f32> {
    vec![0.0; action_dim]
}

fn obs_to_image_tensor<B: Backend>(
    obs: &Observation,
    device: &B::Device,
) -> Option<Tensor<B, 4>> {
    obs.image.as_ref().map(|img| {
        let shape = obs.image_shape.unwrap();
        let h = shape[0];
        let w = shape[1];
        let c = shape[2];
        let data: Vec<f32> = img.iter().map(|&b| b as f32).collect();
        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([1, h, w, c])
    })
}

fn random_action(act_space: &ActionSpace) -> Action {
    match act_space {
        ActionSpace::Discrete { n } => {
            let idx = rand::Rng::gen_range(&mut rand::thread_rng(), 0..*n);
            Action {
                discrete: Some(idx),
                continuous: None,
            }
        }
        ActionSpace::Continuous { dim, low, high } => {
            let actions: Vec<f32> = (0..*dim)
                .map(|_| rand::Rng::gen_range(&mut rand::thread_rng(), *low..*high))
                .collect();
            Action {
                discrete: None,
                continuous: Some(actions),
            }
        }
    }
}

fn tensor_to_action<B: Backend>(
    tensor: &Tensor<B, 2>,
    act_space: &ActionSpace,
) -> Action {
    match act_space {
        ActionSpace::Discrete { n: _ } => {
            let idx: i64 = tensor
                .clone()
                .argmax(1)
                .squeeze::<1>(1)
                .into_scalar()
                .elem();
            Action {
                discrete: Some(idx as usize),
                continuous: None,
            }
        }
        ActionSpace::Continuous { dim, .. } => {
            let data = tensor.clone().into_data();
            let all_values: Vec<f32> = data.to_vec::<f32>().unwrap();
            let actions: Vec<f32> = (0..*dim).map(|i| all_values[i]).collect();
            Action {
                discrete: None,
                continuous: Some(actions),
            }
        }
    }
}

// ============================================================================
// Batch conversion helpers for training
// ============================================================================

/// Convert batch observations to image tensor [B, T, H, W, C].
fn batch_to_image_tensor<B: Backend>(
    observations: &[Vec<Vec<f32>>],
    batch_size: usize,
    seq_len: usize,
    image_size: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 5> {
    let h = image_size[0];
    let w = image_size[1];
    let c = 3;

    let mut all_data = Vec::with_capacity(batch_size * seq_len * h * w * c);
    for seq in observations {
        for obs in seq {
            if obs.len() >= h * w * c {
                all_data.extend_from_slice(&obs[..h * w * c]);
            } else {
                all_data.extend_from_slice(obs);
                all_data.extend(std::iter::repeat(0.0f32).take(h * w * c - obs.len()));
            }
        }
    }

    Tensor::<B, 1>::from_floats(all_data.as_slice(), device)
        .reshape([batch_size, seq_len, h, w, c])
}

/// Convert batch actions to tensor [B, T, action_dim].
fn batch_to_action_tensor<B: Backend>(
    actions: &[Vec<Vec<f32>>],
    batch_size: usize,
    seq_len: usize,
    action_dim: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mut all_data = Vec::with_capacity(batch_size * seq_len * action_dim);
    for seq in actions {
        for act in seq {
            if act.len() >= action_dim {
                all_data.extend_from_slice(&act[..action_dim]);
            } else {
                all_data.extend_from_slice(act);
                all_data.extend(std::iter::repeat(0.0f32).take(action_dim - act.len()));
            }
        }
    }

    Tensor::<B, 1>::from_floats(all_data.as_slice(), device)
        .reshape([batch_size, seq_len, action_dim])
}

/// Convert batch rewards to tensor [B, T].
fn batch_to_reward_tensor<B: Backend>(
    rewards: &[Vec<f32>],
    batch_size: usize,
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut all_data = Vec::with_capacity(batch_size * seq_len);
    for seq in rewards {
        all_data.extend_from_slice(seq);
    }

    Tensor::<B, 1>::from_floats(all_data.as_slice(), device).reshape([batch_size, seq_len])
}

/// Convert batch boolean flags to Bool tensor [B, T].
fn batch_to_bool_tensor<B: Backend>(
    flags: &[Vec<bool>],
    batch_size: usize,
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let mut all_data = Vec::with_capacity(batch_size * seq_len);
    for seq in flags {
        for &flag in seq {
            all_data.push(if flag { 1i32 } else { 0i32 });
        }
    }

    Tensor::<B, 1, Int>::from_ints(all_data.as_slice(), device)
        .reshape([batch_size, seq_len])
        .equal_elem(1)
}
