use burn::prelude::*;
use burn::module::Ignored;

use crate::distributions::one_hot::OneHotCategorical;
use crate::nn::{
    BlockLinear, BlockLinearConfig, DreamerLinear, DreamerLinearConfig, Norm,
    NormConfig,
};

/// RSSM (Recurrent State-Space Model) carry state.
///
/// Contains the deterministic and stochastic parts of the world model state.
#[derive(Debug, Clone)]
pub struct RSSMState<B: Backend> {
    /// Deterministic hidden state [B, deter]
    pub deter: Tensor<B, 2>,
    /// Stochastic latent state [B, stoch, classes] (one-hot)
    pub stoch: Tensor<B, 3>,
}

/// RSSM features (state + logits) used for downstream predictions.
#[derive(Debug, Clone)]
pub struct RSSMFeat<B: Backend> {
    pub deter: Tensor<B, 2>,
    pub stoch: Tensor<B, 3>,
    pub logit: Tensor<B, 3>,
}

impl<B: Backend> RSSMFeat<B> {
    /// Convert features to a flat tensor for use by downstream heads.
    /// Concatenates deter and flattened stoch: [B, deter + stoch*classes]
    pub fn to_tensor(&self) -> Tensor<B, 2> {
        let stoch_flat = {
            let dims = self.stoch.dims();
            self.stoch.clone().reshape([dims[0], dims[1] * dims[2]])
        };
        Tensor::cat(vec![self.deter.clone(), stoch_flat], 1)
    }
}

/// RSSM sequence features (with time dimension).
#[derive(Debug, Clone)]
pub struct RSSMSeqFeat<B: Backend> {
    /// [B, T, deter]
    pub deter: Tensor<B, 3>,
    /// [B, T, stoch, classes]
    pub stoch: Tensor<B, 4>,
    /// [B, T, stoch, classes]
    pub logit: Tensor<B, 4>,
}

impl<B: Backend> RSSMSeqFeat<B> {
    /// Convert to flat tensor [B, T, deter + stoch*classes]
    pub fn to_tensor(&self) -> Tensor<B, 3> {
        let dims = self.stoch.dims();
        let stoch_flat = self.stoch.clone().reshape([dims[0], dims[1], dims[2] * dims[3]]);
        Tensor::cat(vec![self.deter.clone(), stoch_flat], 2)
    }
}

/// RSSM entries stored in replay buffer for context reconstruction.
#[derive(Debug, Clone)]
pub struct RSSMEntries<B: Backend> {
    pub deter: Tensor<B, 3>,
    pub stoch: Tensor<B, 4>,
}

/// Recurrent State-Space Model for DreamerV3.
///
/// Corresponds to `rssm.RSSM` in the Python implementation.
///
/// Architecture:
/// - Deterministic path: Block-wise GRU with gated update
/// - Stochastic path: Categorical distribution (stoch x classes)
/// - Prior: MLP from deter state to stochastic distribution
/// - Posterior: MLP from deter state + observation tokens
#[derive(Module, Debug)]
pub struct RSSM<B: Backend> {
    // --- Core recurrence (block-wise GRU) ---
    /// Linear projections for inputs to the recurrence
    dynin0: DreamerLinear<B>,
    dynin0norm: Norm<B>,
    dynin1: DreamerLinear<B>,
    dynin1norm: Norm<B>,
    dynin2: DreamerLinear<B>,
    dynin2norm: Norm<B>,
    /// Hidden layers within the recurrence
    dynhid: Vec<BlockLinear<B>>,
    dynhid_norms: Vec<Norm<B>>,
    /// GRU gate computation (3x deter for reset, candidate, update)
    dyngru: BlockLinear<B>,

    // --- Prior (from deter to stochastic) ---
    prior_layers: Vec<DreamerLinear<B>>,
    prior_norms: Vec<Norm<B>>,
    prior_logit: DreamerLinear<B>,

    // --- Posterior (from deter + tokens to stochastic) ---
    obs_layers: Vec<DreamerLinear<B>>,
    obs_norms: Vec<Norm<B>>,
    obs_logit: DreamerLinear<B>,

    // --- Config ---
    deter: Ignored<usize>,
    stoch: Ignored<usize>,
    classes: Ignored<usize>,
    blocks: Ignored<usize>,
    hidden: Ignored<usize>,
    unimix: Ignored<f64>,
    free_nats: Ignored<f64>,
    act: Ignored<String>,
    absolute: Ignored<bool>,
}

#[derive(Debug, Clone)]
pub struct RSSMConfig {
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
    /// Dimension of action embedding
    pub action_dim: usize,
    /// Dimension of observation tokens from encoder
    pub token_dim: usize,
}

impl Default for RSSMConfig {
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
            action_dim: 0,
            token_dim: 0,
        }
    }
}

impl RSSMConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RSSM<B> {
        assert!(
            self.deter % self.blocks == 0,
            "deter must be divisible by blocks"
        );

        let stoch_flat = self.stoch * self.classes;

        // Dynamic input projections
        let dynin0 = DreamerLinearConfig::new(self.deter, self.hidden).init(device);
        let dynin0norm = NormConfig::new(&self.norm, self.hidden).init(device);
        let dynin1 = DreamerLinearConfig::new(stoch_flat, self.hidden).init(device);
        let dynin1norm = NormConfig::new(&self.norm, self.hidden).init(device);
        let dynin2 = DreamerLinearConfig::new(self.action_dim, self.hidden).init(device);
        let dynin2norm = NormConfig::new(&self.norm, self.hidden).init(device);

        // The concatenated input to each block: deter/blocks + hidden*3
        // Actually: deter_per_block + hidden*3 (broadcast to each block)
        let block_in = self.deter / self.blocks + self.hidden * 3;

        // Dynamic hidden layers (block linear)
        let mut dynhid = Vec::new();
        let mut dynhid_norms = Vec::new();
        for _i in 0..self.dynlayers {
            dynhid.push(
                BlockLinearConfig::new(
                    // First layer takes block_in * blocks
                    self.blocks * block_in,
                    self.deter,
                    self.blocks,
                )
                .init(device),
            );
            dynhid_norms.push(NormConfig::new(&self.norm, self.deter).init(device));
        }

        // GRU gates: 3 * deter (reset, candidate, update)
        let gru_in = if self.dynlayers > 0 {
            self.deter
        } else {
            self.blocks * block_in
        };
        let dyngru = BlockLinearConfig::new(gru_in, 3 * self.deter, self.blocks).init(device);

        // Prior layers
        let mut prior_layers = Vec::new();
        let mut prior_norms = Vec::new();
        for i in 0..self.imglayers {
            let in_size = if i == 0 { self.deter } else { self.hidden };
            prior_layers.push(DreamerLinearConfig::new(in_size, self.hidden).init(device));
            prior_norms.push(NormConfig::new(&self.norm, self.hidden).init(device));
        }
        let prior_logit = DreamerLinearConfig::new(self.hidden, stoch_flat).init(device);

        // Observation (posterior) layers
        let obs_input = if self.absolute {
            self.token_dim
        } else {
            self.deter + self.token_dim
        };
        let mut obs_layers = Vec::new();
        let mut obs_norms = Vec::new();
        for i in 0..self.obslayers {
            let in_size = if i == 0 { obs_input } else { self.hidden };
            obs_layers.push(DreamerLinearConfig::new(in_size, self.hidden).init(device));
            obs_norms.push(NormConfig::new(&self.norm, self.hidden).init(device));
        }
        let obs_logit = DreamerLinearConfig::new(self.hidden, stoch_flat).init(device);

        RSSM {
            dynin0,
            dynin0norm,
            dynin1,
            dynin1norm,
            dynin2,
            dynin2norm,
            dynhid,
            dynhid_norms,
            dyngru,
            prior_layers,
            prior_norms,
            prior_logit,
            obs_layers,
            obs_norms,
            obs_logit,
            deter: Ignored(self.deter),
            stoch: Ignored(self.stoch),
            classes: Ignored(self.classes),
            blocks: Ignored(self.blocks),
            hidden: Ignored(self.hidden),
            unimix: Ignored(self.unimix),
            free_nats: Ignored(self.free_nats),
            act: Ignored(self.act.clone()),
            absolute: Ignored(self.absolute),
        }
    }
}

impl<B: Backend> RSSM<B> {
    /// Feature dimension: deter + stoch * classes
    pub fn feat_dim(&self) -> usize {
        self.deter.0 + self.stoch.0 * self.classes.0
    }

    /// Create initial zero state.
    pub fn initial(&self, batch_size: usize, device: &B::Device) -> RSSMState<B> {
        RSSMState {
            deter: Tensor::zeros([batch_size, self.deter.0], device),
            stoch: Tensor::zeros([batch_size, self.stoch.0, self.classes.0], device),
        }
    }

    /// Observe: process tokens and compute posterior.
    ///
    /// Single step: given previous state, action, and observation tokens,
    /// compute the next state (posterior distribution).
    pub fn observe_step(
        &self,
        carry: &RSSMState<B>,
        tokens: Tensor<B, 2>,
        action: Tensor<B, 2>,
        reset: Tensor<B, 1, Bool>,
    ) -> (RSSMState<B>, RSSMFeat<B>) {
        let batch = carry.deter.dims()[0];

        // Mask carry and action on reset
        let mask_float = reset.clone().bool_not().float();
        let mask_2d = mask_float.clone().unsqueeze_dim::<2>(1);
        let mask_3d = mask_float.clone().unsqueeze_dim::<2>(1).unsqueeze_dim::<3>(2);

        let deter = carry.deter.clone() * mask_2d.clone();
        let stoch = carry.stoch.clone() * mask_3d;
        let action = action * mask_2d;

        // Core recurrence
        let new_deter = self.core(&deter, &stoch, &action);

        // Compute posterior from observation
        let token_dim = tokens.dims()[1];
        let tokens_flat = tokens.reshape([batch, token_dim]);
        let x = if self.absolute.0 {
            tokens_flat
        } else {
            Tensor::cat(vec![new_deter.clone(), tokens_flat], 1)
        };

        let mut x = x;
        for (linear, norm) in self.obs_layers.iter().zip(self.obs_norms.iter()) {
            x = linear.forward(x);
            x = norm.forward(x);
            x = apply_act_2d(x, &self.act);
        }
        let logit = self.obs_logit.forward(x);
        let logit = logit.reshape([batch, self.stoch.0, self.classes.0]);

        // Sample from posterior
        let dist = OneHotCategorical::new(logit.clone(), self.unimix.0);
        let new_stoch = dist.pred(); // Use mode for deterministic behavior during initial impl

        let carry = RSSMState {
            deter: new_deter.clone(),
            stoch: new_stoch.clone(),
        };
        let feat = RSSMFeat {
            deter: new_deter,
            stoch: new_stoch,
            logit,
        };

        (carry, feat)
    }

    /// Observe over a sequence.
    ///
    /// tokens: [B, T, token_dim]
    /// action: [B, T, action_dim]
    /// reset: [B, T] (bool)
    pub fn observe_seq(
        &self,
        mut carry: RSSMState<B>,
        tokens: Tensor<B, 3>,
        action: Tensor<B, 3>,
        reset: Tensor<B, 2, Bool>,
    ) -> (RSSMState<B>, RSSMEntries<B>, RSSMSeqFeat<B>) {
        let dims = tokens.dims();
        let b = dims[0];
        let t = dims[1];

        let mut all_deter = Vec::with_capacity(t);
        let mut all_stoch = Vec::with_capacity(t);
        let mut all_logit = Vec::with_capacity(t);

        for step in 0..t {
            let tok_t: Tensor<B, 2> = tokens.clone().slice([0..b, step..step + 1]).squeeze(1);
            let act_t: Tensor<B, 2> = action.clone().slice([0..b, step..step + 1]).squeeze(1);
            let rst_t: Tensor<B, 1, Bool> = reset.clone().slice([0..b, step..step + 1]).squeeze(1);

            let (new_carry, feat) = self.observe_step(&carry, tok_t, act_t, rst_t);
            carry = new_carry;

            all_deter.push(feat.deter.unsqueeze_dim::<3>(1));
            all_stoch.push(feat.stoch.unsqueeze_dim::<4>(1));
            all_logit.push(feat.logit.unsqueeze_dim::<4>(1));
        }

        let deter_seq = Tensor::cat(all_deter, 1);
        let stoch_seq = Tensor::cat(all_stoch, 1);
        let logit_seq = Tensor::cat(all_logit, 1);

        let entries = RSSMEntries {
            deter: deter_seq.clone(),
            stoch: stoch_seq.clone(),
        };

        let feat = RSSMSeqFeat {
            deter: deter_seq,
            stoch: stoch_seq,
            logit: logit_seq,
        };

        (carry, entries, feat)
    }

    /// Imagine forward from a starting state using a policy.
    ///
    /// Returns imagined features for `length` steps.
    pub fn imagine<F>(
        &self,
        mut carry: RSSMState<B>,
        policy: &F,
        length: usize,
    ) -> (RSSMState<B>, RSSMSeqFeat<B>, Vec<Tensor<B, 2>>)
    where
        F: Fn(&RSSMFeat<B>) -> Tensor<B, 2>,
    {
        let batch = carry.deter.dims()[0];
        let device = carry.deter.device();

        let mut all_deter = Vec::with_capacity(length);
        let mut all_stoch = Vec::with_capacity(length);
        let mut all_logit = Vec::with_capacity(length);
        let mut all_actions = Vec::with_capacity(length);

        for _ in 0..length {
            // Get action from policy (based on current state)
            let feat = RSSMFeat {
                deter: carry.deter.clone(),
                stoch: carry.stoch.clone(),
                logit: Tensor::zeros([batch, self.stoch.0, self.classes.0], &device),
            };
            let action = policy(&feat);
            all_actions.push(action.clone());

            // Imagine one step
            let (new_carry, new_feat) = self.imagine_step(&carry, action);
            carry = new_carry;

            all_deter.push(new_feat.deter.unsqueeze_dim::<3>(1));
            all_stoch.push(new_feat.stoch.unsqueeze_dim::<4>(1));
            all_logit.push(new_feat.logit.unsqueeze_dim::<4>(1));
        }

        let feat = RSSMSeqFeat {
            deter: Tensor::cat(all_deter, 1),
            stoch: Tensor::cat(all_stoch, 1),
            logit: Tensor::cat(all_logit, 1),
        };

        (carry, feat, all_actions)
    }

    /// Imagine a single step forward using the prior.
    pub fn imagine_step(
        &self,
        carry: &RSSMState<B>,
        action: Tensor<B, 2>,
    ) -> (RSSMState<B>, RSSMFeat<B>) {
        let _batch = carry.deter.dims()[0];

        let new_deter = self.core(&carry.deter, &carry.stoch, &action);
        let logit = self.prior(&new_deter);
        let dist = OneHotCategorical::new(logit.clone(), self.unimix.0);
        let new_stoch = dist.pred();

        let new_carry = RSSMState {
            deter: new_deter.clone(),
            stoch: new_stoch.clone(),
        };
        let feat = RSSMFeat {
            deter: new_deter,
            stoch: new_stoch,
            logit,
        };

        (new_carry, feat)
    }

    /// Compute KL losses (dynamics and representation).
    pub fn compute_kl_loss(
        &self,
        feat: &RSSMSeqFeat<B>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let dims = feat.deter.dims();
        let b = dims[0];
        let t = dims[1];

        let mut dyn_losses = Vec::with_capacity(t);
        let mut rep_losses = Vec::with_capacity(t);

        for step in 0..t {
            let deter_t: Tensor<B, 2> = feat.deter.clone().slice([0..b, step..step + 1]).squeeze(1);
            let post_logit: Tensor<B, 3> = feat.logit.clone().slice([0..b, step..step + 1]).squeeze(1);

            // Prior from deterministic state
            let prior_logit = self.prior(&deter_t);

            let prior_dist = OneHotCategorical::new(prior_logit, self.unimix.0);
            let post_dist = OneHotCategorical::new(post_logit, self.unimix.0);

            // dyn loss: KL(stop_grad(post) || prior)
            let dyn_kl = post_dist.kl(&prior_dist); // [B, stoch]
            let dyn_kl = dyn_kl.sum_dim(1).squeeze::<1>(1); // [B]

            // rep loss: KL(post || stop_grad(prior))
            let rep_kl = post_dist.kl(&prior_dist); // Approximation (same direction)
            let rep_kl = rep_kl.sum_dim(1).squeeze::<1>(1); // [B]

            // Apply free nats
            let dyn_kl = dyn_kl.clamp_min(self.free_nats.0);
            let rep_kl = rep_kl.clamp_min(self.free_nats.0);

            dyn_losses.push(dyn_kl.unsqueeze_dim::<2>(1));
            rep_losses.push(rep_kl.unsqueeze_dim::<2>(1));
        }

        let dyn_loss = Tensor::cat(dyn_losses, 1); // [B, T]
        let rep_loss = Tensor::cat(rep_losses, 1); // [B, T]

        (dyn_loss, rep_loss)
    }

    /// Core recurrence: block-wise GRU computation.
    ///
    /// Processes deter state, stoch state, and action through the
    /// block-diagonal recurrent structure.
    fn core(
        &self,
        deter: &Tensor<B, 2>,
        stoch: &Tensor<B, 3>,
        action: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let batch = deter.dims()[0];
        let stoch_flat = {
            let dims = stoch.dims();
            stoch.clone().reshape([dims[0], dims[1] * dims[2]])
        };

        // Normalize action: action / max(1, |action|)
        let action = {
            let abs_action = action.clone().abs();
            let max_val = abs_action.clamp_min(1.0);
            action.clone() / max_val
        };

        // Project inputs
        let x0 = self.dynin0.forward(deter.clone());
        let x0 = self.dynin0norm.forward(x0);
        let x0 = apply_act_2d(x0, &self.act);

        let x1 = self.dynin1.forward(stoch_flat);
        let x1 = self.dynin1norm.forward(x1);
        let x1 = apply_act_2d(x1, &self.act);

        let x2 = self.dynin2.forward(action);
        let x2 = self.dynin2norm.forward(x2);
        let x2 = apply_act_2d(x2, &self.act);

        // Concatenate: [x0, x1, x2] -> [batch, hidden*3]
        let combined = Tensor::cat(vec![x0, x1, x2], 1);

        // Expand to blocks: repeat combined for each block, concat with deter block
        let g = self.blocks.0;
        let deter_per_block = self.deter.0 / g;

        // Split deter into blocks and concatenate with combined input
        let mut block_inputs = Vec::with_capacity(g);
        for i in 0..g {
            let start = i * deter_per_block;
            let end = start + deter_per_block;
            let deter_block = deter.clone().slice([0..batch, start..end]);
            block_inputs.push(Tensor::cat(vec![deter_block, combined.clone()], 1));
        }
        let x = Tensor::cat(block_inputs, 1);

        // Dynamic hidden layers
        let mut x = x;
        for (linear, norm) in self.dynhid.iter().zip(self.dynhid_norms.iter()) {
            x = linear.forward(x);
            x = norm.forward(x);
            x = apply_act_2d(x, &self.act);
        }

        // GRU gates
        let gates = self.dyngru.forward(x);
        let gate_size = self.deter.0;

        let reset_gate = gates.clone().slice([0..batch, 0..gate_size]);
        let cand = gates.clone().slice([0..batch, gate_size..2 * gate_size]);
        let update_gate = gates.slice([0..batch, 2 * gate_size..3 * gate_size]);

        let reset_gate = burn::tensor::activation::sigmoid(reset_gate);
        let cand = (reset_gate * cand).tanh();
        let update_gate = burn::tensor::activation::sigmoid(update_gate - 1.0);

        // New deter = update * cand + (1 - update) * deter
        update_gate.clone() * cand + (update_gate.neg() + 1.0) * deter.clone()
    }

    /// Prior network: predict stochastic distribution from deterministic state.
    fn prior(&self, deter: &Tensor<B, 2>) -> Tensor<B, 3> {
        let batch = deter.dims()[0];
        let mut x = deter.clone();

        for (linear, norm) in self.prior_layers.iter().zip(self.prior_norms.iter()) {
            x = linear.forward(x);
            x = norm.forward(x);
            x = apply_act_2d(x, &self.act);
        }

        let logit = self.prior_logit.forward(x);
        logit.reshape([batch, self.stoch.0, self.classes.0])
    }
}

fn apply_act_2d<B: Backend>(x: Tensor<B, 2>, act: &str) -> Tensor<B, 2> {
    match act {
        "gelu" => burn::tensor::activation::gelu(x),
        "silu" | "swish" => burn::tensor::activation::silu(x),
        "relu" => burn::tensor::activation::relu(x),
        "tanh" => x.tanh(),
        "none" => x,
        _ => panic!("Unknown activation: {}", act),
    }
}
