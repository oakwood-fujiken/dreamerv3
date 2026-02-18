use burn::prelude::*;

use super::heads::{ActionType, MLPHead, MLPHeadConfig, PolicyHead, PolicyHeadConfig};
use super::rssm::{RSSMFeat, RSSMSeqFeat, RSSMState};
use super::world_model::{WorldModel, WorldModelConfig};

/// DreamerV3 Agent combining world model, policy, and value networks.
///
/// Corresponds to `agent.Agent` in the Python implementation.
///
/// Training flow:
/// 1. Encode observations -> tokens
/// 2. RSSM observe -> posterior (world model training)
/// 3. RSSM imagine -> imagined trajectories (actor-critic training)
/// 4. Policy and value heads predict actions and values
#[derive(Module, Debug)]
pub struct DreamerV3Agent<B: Backend> {
    pub world_model: WorldModel<B>,
    pub policy: PolicyHead<B>,
    pub value: MLPHead<B>,
    pub slow_value: MLPHead<B>,
}

#[derive(Debug, Clone)]
pub struct DreamerV3AgentConfig {
    pub world_model: WorldModelConfig,
    pub policy: PolicyHeadConfig,
    pub value: MLPHeadConfig,
    pub slow_value: MLPHeadConfig,
}

impl DreamerV3AgentConfig {
    /// Create a config for discrete action spaces (e.g., Atari).
    pub fn for_discrete_actions(
        image_res: [usize; 2],
        image_channels: usize,
        n_actions: usize,
    ) -> Self {
        let wm = WorldModelConfig::for_image_task(image_res, image_channels, n_actions);
        let feat_dim = wm.rssm.deter + wm.rssm.stoch * wm.rssm.classes;

        Self {
            world_model: wm,
            policy: PolicyHeadConfig::new(
                feat_dim,
                ActionType::Discrete {
                    n_classes: n_actions,
                },
                n_actions,
            ),
            value: MLPHeadConfig::scalar(feat_dim),
            slow_value: MLPHeadConfig::scalar(feat_dim),
        }
    }

    /// Create a config for continuous action spaces (e.g., DMC).
    pub fn for_continuous_actions(
        image_res: [usize; 2],
        image_channels: usize,
        action_dim: usize,
    ) -> Self {
        let wm = WorldModelConfig::for_image_task(image_res, image_channels, action_dim);
        let feat_dim = wm.rssm.deter + wm.rssm.stoch * wm.rssm.classes;

        Self {
            world_model: wm,
            policy: PolicyHeadConfig::new(feat_dim, ActionType::Continuous, action_dim),
            value: MLPHeadConfig::scalar(feat_dim),
            slow_value: MLPHeadConfig::scalar(feat_dim),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> DreamerV3Agent<B> {
        DreamerV3Agent {
            world_model: self.world_model.init(device),
            policy: self.policy.init(device),
            value: self.value.init(device),
            slow_value: self.slow_value.init(device),
        }
    }
}

impl<B: Backend> DreamerV3Agent<B> {
    /// Run a single policy step (for environment interaction).
    ///
    /// # Arguments
    /// * `carry` - Current RSSM state
    /// * `image` - Current observation image [1, H, W, C]
    /// * `prev_action` - Previous action [1, action_dim]
    /// * `is_first` - Whether this is the first step of an episode
    ///
    /// # Returns
    /// * New carry state
    /// * Sampled action [1, action_dim]
    pub fn policy_step(
        &self,
        carry: &RSSMState<B>,
        image: Option<Tensor<B, 4>>,
        vector: Option<Tensor<B, 2>>,
        prev_action: Tensor<B, 2>,
        is_first: Tensor<B, 1, Bool>,
    ) -> (RSSMState<B>, Tensor<B, 2>) {
        // Encode observation
        let tokens = self.world_model.encoder.forward(image, vector);

        // RSSM observe step
        let (new_carry, feat) = self.world_model.rssm.observe_step(
            carry,
            tokens,
            prev_action,
            is_first,
        );

        // Sample action from policy
        let feat_tensor = feat.to_tensor();
        let action = self.policy.sample(feat_tensor);

        (new_carry, action)
    }

    /// Compute the world model loss for a batch of sequences.
    ///
    /// # Arguments
    /// * `carry` - Initial RSSM state
    /// * `image` - Observation images [B, T, H, W, C]
    /// * `vector` - Observation vectors [B, T, vec_dim] (optional)
    /// * `prev_action` - Previous actions [B, T, action_dim]
    /// * `reward` - Rewards [B, T]
    /// * `is_first` - Episode start flags [B, T]
    /// * `is_terminal` - Terminal flags [B, T]
    ///
    /// # Returns
    /// * Updated carry
    /// * Dictionary of losses
    /// * Features for imagination
    pub fn world_model_loss(
        &self,
        carry: RSSMState<B>,
        image: Option<Tensor<B, 5>>,
        vector: Option<Tensor<B, 3>>,
        prev_action: Tensor<B, 3>,
        reward: Tensor<B, 2>,
        is_first: Tensor<B, 2, Bool>,
        is_terminal: Tensor<B, 2, Bool>,
    ) -> (RSSMState<B>, WorldModelLosses<B>, RSSMSeqFeat<B>) {
        // Encode observations
        let tokens = self.world_model.encoder.forward_seq(image, vector);

        // RSSM observe
        let (carry, _entries, feat) =
            self.world_model.rssm.observe_seq(carry, tokens, prev_action, is_first);

        // KL losses
        let (dyn_loss, rep_loss) = self.world_model.rssm.compute_kl_loss(&feat);

        // Reconstruction loss (image + vector)
        let (_img_recon, _vec_recon) = self.world_model.decoder.forward_seq(
            feat.deter.clone(),
            feat.stoch.clone(),
        );

        // Reward and continuation prediction
        let feat_tensor = feat.to_tensor();
        let feat_flat = {
            let dims = feat_tensor.dims();
            feat_tensor.clone().reshape([dims[0] * dims[1], dims[2]])
        };
        let _rew_pred = self.world_model.reward_head.forward(feat_flat.clone());
        let _con_pred = self.world_model.continue_head.forward(feat_flat);

        let losses = WorldModelLosses {
            dyn_loss,
            rep_loss,
            reward,
            is_terminal,
        };

        (carry, losses, feat)
    }

    /// Compute imagination loss for actor-critic training.
    ///
    /// # Arguments
    /// * `starts` - Starting states for imagination [B*K, ...]
    /// * `horizon` - Number of imagination steps
    pub fn imagination_loss(
        &self,
        starts: RSSMState<B>,
        horizon: usize,
    ) -> ImaginationLosses<B> {
        // Define policy function for imagination
        let _feat_dim = self.world_model.feat_dim();
        let policy = |feat: &RSSMFeat<B>| -> Tensor<B, 2> {
            let feat_tensor = feat.to_tensor();
            self.policy.sample(feat_tensor)
        };

        // Imagine forward
        let (_final_carry, img_feat, img_actions) =
            self.world_model.rssm.imagine(starts, &policy, horizon);

        // Predict rewards and continuations from imagined features
        let feat_tensor = img_feat.to_tensor();
        let dims = feat_tensor.dims();
        let b = dims[0];
        let t = dims[1];
        let f = dims[2];
        let feat_flat = feat_tensor.clone().reshape([b * t, f]);

        let rew_pred = self.world_model.reward_head.pred(feat_flat.clone());
        let con_pred = self.world_model.continue_head.forward(feat_flat.clone());

        // Value predictions
        let val_pred = self.value.pred(feat_flat.clone());
        let slow_val_pred = self.slow_value.pred(feat_flat);

        ImaginationLosses {
            feat: img_feat,
            actions: img_actions,
            reward_pred: rew_pred.reshape([b, t, 1]),
            continue_pred: con_pred.reshape([b, t, 1]),
            value_pred: val_pred.reshape([b, t, 1]),
            slow_value_pred: slow_val_pred.reshape([b, t, 1]),
        }
    }
}

/// World model loss components.
#[derive(Debug)]
pub struct WorldModelLosses<B: Backend> {
    pub dyn_loss: Tensor<B, 2>,
    pub rep_loss: Tensor<B, 2>,
    pub reward: Tensor<B, 2>,
    pub is_terminal: Tensor<B, 2, Bool>,
}

/// Imagination loss components.
#[derive(Debug)]
pub struct ImaginationLosses<B: Backend> {
    pub feat: RSSMSeqFeat<B>,
    pub actions: Vec<Tensor<B, 2>>,
    pub reward_pred: Tensor<B, 3>,
    pub continue_pred: Tensor<B, 3>,
    pub value_pred: Tensor<B, 3>,
    pub slow_value_pred: Tensor<B, 3>,
}

/// Lambda return computation for actor-critic training.
///
/// Corresponds to `lambda_return` in the Python implementation.
///
/// V_t = r_t + gamma * [lambda * V_{t+1} + (1-lambda) * boot_{t+1}]
pub fn lambda_return<B: Backend>(
    reward: Tensor<B, 2>,
    continue_prob: Tensor<B, 2>,
    value: Tensor<B, 2>,
    discount: f64,
    lambda: f64,
) -> Tensor<B, 2> {
    let dims = reward.dims();
    let b = dims[0];
    let t = dims[1];
    let device = reward.device();

    // Compute returns backwards through time
    let mut returns = Vec::with_capacity(t);

    // Bootstrap from last value
    let mut next_return: Tensor<B, 1> = value.clone().slice([0..b, t - 1..t]).squeeze(1);

    for step in (0..t - 1).rev() {
        let r: Tensor<B, 1> = reward.clone().slice([0..b, step + 1..step + 2]).squeeze(1);
        let c: Tensor<B, 1> = continue_prob.clone().slice([0..b, step + 1..step + 2]).squeeze(1);
        let v: Tensor<B, 1> = value.clone().slice([0..b, step + 1..step + 2]).squeeze(1);

        let live = c * discount;
        let cont = live.clone() * lambda;
        let interm = r + (cont.clone().neg() + 1.0) * live.clone() * v;
        next_return = interm + live * cont * next_return;
        returns.push(next_return.clone());
    }

    returns.reverse();

    // Stack returns: [B, T-1]
    let returns: Vec<Tensor<B, 2>> = returns.into_iter().map(|r| r.unsqueeze_dim::<2>(1)).collect();
    if returns.is_empty() {
        Tensor::zeros([b, 0], &device)
    } else {
        Tensor::cat(returns, 1)
    }
}
