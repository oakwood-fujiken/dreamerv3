use burn::prelude::*;
use burn::tensor::ElementConversion;

use crate::config::DreamerConfig;
use crate::envs::{Action, ActionSpace, Environment, Observation};
use crate::models::agent::DreamerV3Agent;
use crate::replay::{ReplayBuffer, Transition};

/// Training metrics collected during training.
#[derive(Debug, Clone, Default)]
pub struct TrainMetrics {
    pub total_steps: usize,
    pub train_steps: usize,
    pub episodes: usize,
    pub episode_return: f64,
    pub episode_length: usize,
}

/// DreamerV3 Trainer orchestrating the full training loop.
///
/// Corresponds to the training flow in `embodied/run/train.py`.
///
/// Training loop:
/// 1. Collect experience from environment(s) using current policy
/// 2. Store transitions in replay buffer
/// 3. Sample batches and train the world model
/// 4. Train actor-critic via imagination in the learned world model
pub struct Trainer<B: Backend> {
    config: DreamerConfig,
    agent: DreamerV3Agent<B>,
    replay: ReplayBuffer,
    metrics: TrainMetrics,
    device: B::Device,
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
        }
    }

    /// Run the full training loop.
    pub fn train<E: Environment>(&mut self, env: &mut E) {
        log::info!("Starting DreamerV3 training");
        log::info!("Config: {:?}", self.config.run);

        let total_steps = self.config.run.steps;
        let train_ratio = self.config.training.batch_size as f64
            * self.config.training.batch_length as f64
            * self.config.run.train_ratio;

        // Initialize environment
        let mut obs = env.reset();
        let act_space = env.act_space();
        let action_dim = act_space.dim();

        // Initialize RSSM carry state
        let mut carry = self.agent.world_model.rssm.initial(1, &self.device);
        let mut prev_action = Tensor::<B, 2>::zeros([1, action_dim], &self.device);
        let mut episode_return = 0.0f64;
        let mut episode_length = 0usize;
        let mut train_step_counter = 0usize;

        // Prefill replay buffer
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
                // Random action during prefill
                random_action(&act_space)
            } else {
                // Policy action
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

            // Training
            if self.replay.can_sample(self.config.training.batch_length)
                && self.replay.len() >= self.config.replay.min_length
            {
                let n_train_steps = (train_ratio / self.metrics.total_steps as f64)
                    .max(1.0) as usize;
                for _ in 0..n_train_steps.min(1) {
                    self.train_step();
                    train_step_counter += 1;
                }
            }

            // Logging
            if self.metrics.total_steps % self.config.run.log_every == 0 {
                log::info!(
                    "Step {} | Train steps {} | Buffer size {} | Episodes {}",
                    self.metrics.total_steps,
                    train_step_counter,
                    self.replay.len(),
                    self.metrics.episodes,
                );
            }
        }

        log::info!("Training complete after {} steps", self.metrics.total_steps);
    }

    /// Execute a single training step.
    fn train_step(&mut self) {
        let sequences = self.replay.sample(
            self.config.training.batch_size,
            self.config.training.batch_length,
        );

        if sequences.is_empty() {
            return;
        }

        self.metrics.train_steps += 1;

        // Note: Full training step implementation requires AutodiffBackend.
        // This is a placeholder showing the data flow.
        // In production, you would:
        // 1. Convert sequences to tensors
        // 2. Forward pass through world model
        // 3. Compute world model loss (reconstruction + KL)
        // 4. Imagine trajectories
        // 5. Compute actor-critic loss (policy + value)
        // 6. Backward pass and optimizer step

        log::debug!("Train step {} completed", self.metrics.train_steps);
    }

    /// Get current training metrics.
    pub fn metrics(&self) -> &TrainMetrics {
        &self.metrics
    }
}

// Helper functions for converting between environment data and tensors.

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
            // Get argmax
            let idx: i64 = tensor.clone().argmax(1).squeeze::<1>(1).into_scalar().elem();
            Action {
                discrete: Some(idx as usize),
                continuous: None,
            }
        }
        ActionSpace::Continuous { dim, .. } => {
            let data = tensor.clone().into_data();
            let all_values: Vec<f32> = data.to_vec::<f32>().unwrap();
            let actions: Vec<f32> = (0..*dim)
                .map(|i| all_values[i])
                .collect();
            Action {
                discrete: None,
                continuous: Some(actions),
            }
        }
    }
}
