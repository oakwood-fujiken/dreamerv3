use burn::prelude::*;
use burn::module::Ignored;

use crate::distributions::two_hot::TwoHotSymexp;
use crate::nn::{DreamerLinear, DreamerLinearConfig, MLP, MLPConfig};

/// MLP Head for scalar predictions (reward, continuation, value).
///
/// Corresponds to `embodied.jax.MLPHead` in the Python implementation.
/// Supports different output distributions (TwoHot for value/reward,
/// Binary for continuation).
#[derive(Module, Debug)]
pub struct MLPHead<B: Backend> {
    mlp: MLP<B>,
    output: DreamerLinear<B>,
    output_type: Ignored<String>,
    num_bins: Ignored<usize>,
}

#[derive(Debug, Clone)]
pub struct MLPHeadConfig {
    pub input_dim: usize,
    pub layers: usize,
    pub units: usize,
    pub act: String,
    pub norm: String,
    pub output_type: String,
    pub num_bins: usize,
    pub outscale: f64,
}

impl MLPHeadConfig {
    pub fn scalar(input_dim: usize) -> Self {
        Self {
            input_dim,
            layers: 3,
            units: 1024,
            act: "gelu".to_string(),
            norm: "rms".to_string(),
            output_type: "twohot_symexp".to_string(),
            num_bins: 255,
            outscale: 1.0,
        }
    }

    pub fn binary(input_dim: usize) -> Self {
        Self {
            input_dim,
            layers: 3,
            units: 1024,
            act: "gelu".to_string(),
            norm: "rms".to_string(),
            output_type: "binary".to_string(),
            num_bins: 1,
            outscale: 1.0,
        }
    }

    pub fn with_output_type(mut self, output_type: &str) -> Self {
        self.output_type = output_type.to_string();
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> MLPHead<B> {
        let mlp = MLPConfig::new(self.input_dim, self.layers, self.units)
            .with_act(&self.act)
            .with_norm(&self.norm)
            .init(device);

        let out_size = match self.output_type.as_str() {
            "twohot_symexp" => self.num_bins,
            "binary" => 1,
            "mse" | "symlog_mse" => 1,
            _ => self.num_bins,
        };

        let output = DreamerLinearConfig::new(self.units, out_size)
            .with_outscale(self.outscale)
            .init(device);

        MLPHead {
            mlp,
            output,
            output_type: Ignored(self.output_type.clone()),
            num_bins: Ignored(self.num_bins),
        }
    }
}

impl<B: Backend> MLPHead<B> {
    /// Forward pass: features -> distribution parameters.
    ///
    /// Input: [B, feat_dim]
    /// Output: [B, output_size]
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.mlp.forward(x);
        self.output.forward(x)
    }

    /// Forward for 3D input [B, T, feat_dim].
    pub fn forward3d(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dims = x.dims();
        let flat = x.reshape([dims[0] * dims[1], dims[2]]);
        let out = self.forward(flat);
        let out_dim = out.dims()[1];
        out.reshape([dims[0], dims[1], out_dim])
    }

    /// Get predicted scalar value.
    pub fn pred(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let logits = self.forward(x);
        match self.output_type.as_str() {
            "twohot_symexp" => {
                let bins =
                    TwoHotSymexp::<B>::create_bins(self.num_bins.0, 20.0, &logits.device());
                let dist = TwoHotSymexp::new(logits.unsqueeze_dim::<3>(1), bins, true);
                dist.pred().squeeze::<1>(1).unsqueeze_dim::<2>(1)
            }
            "binary" => burn::tensor::activation::sigmoid(logits),
            "mse" | "symlog_mse" => logits,
            _ => logits,
        }
    }

    /// Compute loss against target.
    pub fn loss(&self, x: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
        let logits = self.forward(x);
        match self.output_type.as_str() {
            "twohot_symexp" => {
                let bins =
                    TwoHotSymexp::<B>::create_bins(self.num_bins.0, 20.0, &logits.device());
                let logits_3d = logits.unsqueeze_dim::<3>(1);
                let dist = TwoHotSymexp::new(logits_3d, bins, true);
                dist.loss(target).mean_dim(1).squeeze::<1>(1)
            }
            "binary" => {
                // Binary cross-entropy
                let logp = burn::tensor::activation::log_sigmoid(logits.clone());
                let lognotp = burn::tensor::activation::log_sigmoid(-logits);
                let loss = -(target.clone() * logp + (target.neg() + 1.0) * lognotp);
                loss.mean_dim(1).squeeze::<1>(1)
            }
            "mse" => {
                let diff = logits - target;
                (diff.clone() * diff).mean_dim(1).squeeze::<1>(1)
            }
            "symlog_mse" => {
                let pred = logits;
                let target = crate::nn::utils::symlog(target);
                let diff = pred - target;
                (diff.clone() * diff).mean_dim(1).squeeze::<1>(1)
            }
            _ => {
                let diff = logits - target;
                (diff.clone() * diff).mean_dim(1).squeeze::<1>(1)
            }
        }
    }
}

/// Policy head for action distribution.
///
/// For discrete actions: outputs logits for Categorical distribution.
/// For continuous actions: outputs mean and std for Normal distribution.
#[derive(Module, Debug)]
pub struct PolicyHead<B: Backend> {
    mlp: MLP<B>,
    output: DreamerLinear<B>,
    action_type: Ignored<ActionType>,
    action_dim: Ignored<usize>,
}

#[derive(Debug, Clone, Copy)]
pub enum ActionType {
    Discrete { n_classes: usize },
    Continuous,
}

// Default needed for 
impl Default for ActionType {
    fn default() -> Self {
        ActionType::Discrete { n_classes: 1 }
    }
}

#[derive(Debug, Clone)]
pub struct PolicyHeadConfig {
    pub input_dim: usize,
    pub layers: usize,
    pub units: usize,
    pub act: String,
    pub norm: String,
    pub action_type: ActionType,
    pub action_dim: usize,
}

impl PolicyHeadConfig {
    pub fn new(input_dim: usize, action_type: ActionType, action_dim: usize) -> Self {
        Self {
            input_dim,
            layers: 3,
            units: 1024,
            act: "gelu".to_string(),
            norm: "rms".to_string(),
            action_type,
            action_dim,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> PolicyHead<B> {
        let mlp = MLPConfig::new(self.input_dim, self.layers, self.units)
            .with_act(&self.act)
            .with_norm(&self.norm)
            .init(device);

        let out_size = match self.action_type {
            ActionType::Discrete { n_classes } => n_classes,
            ActionType::Continuous => self.action_dim * 2, // mean + std
        };

        let output = DreamerLinearConfig::new(self.units, out_size).init(device);

        PolicyHead {
            mlp,
            output,
            action_type: Ignored(self.action_type),
            action_dim: Ignored(self.action_dim),
        }
    }
}

impl<B: Backend> PolicyHead<B> {
    /// Forward pass: features -> action distribution parameters.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.mlp.forward(x);
        self.output.forward(x)
    }

    /// Sample an action.
    pub fn sample(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let logits = self.forward(x.clone());
        let device = logits.device();

        match self.action_type.0 {
            ActionType::Discrete { n_classes: _ } => {
                // Gumbel-softmax sampling
                let shape = logits.dims();
                let uniform = Tensor::<B, 2>::random(
                    shape,
                    burn::tensor::Distribution::Uniform(1e-7, 1.0 - 1e-7),
                    &device,
                );
                let gumbel = -((-uniform.log()).log());
                let perturbed = logits + gumbel;

                // One-hot of argmax (straight-through)
                let probs = burn::tensor::activation::softmax(perturbed, 1);
                probs
            }
            ActionType::Continuous => {
                let batch = logits.dims()[0];
                let mean = logits.clone().slice([0..batch, 0..self.action_dim.0]);
                let raw_std = logits.slice([0..batch, self.action_dim.0..2 * self.action_dim.0]);
                let std = burn::tensor::activation::softplus(raw_std, 1.0) + 0.1;

                let noise = Tensor::<B, 2>::random(
                    mean.dims(),
                    burn::tensor::Distribution::Normal(0.0, 1.0),
                    &device,
                );
                let action = mean + noise * std;
                // Clip to [-1, 1]
                action.clamp(-1.0, 1.0)
            }
        }
    }

    /// Log-probability of an action.
    pub fn logp(&self, x: Tensor<B, 2>, action: Tensor<B, 2>) -> Tensor<B, 1> {
        let logits = self.forward(x);

        match self.action_type.0 {
            ActionType::Discrete { .. } => {
                let log_probs = burn::tensor::activation::log_softmax(logits, 1);
                (log_probs * action).sum_dim(1).squeeze::<1>(1)
            }
            ActionType::Continuous => {
                let batch = logits.dims()[0];
                let mean = logits.clone().slice([0..batch, 0..self.action_dim.0]);
                let raw_std = logits.slice([0..batch, self.action_dim.0..2 * self.action_dim.0]);
                let std = burn::tensor::activation::softplus(raw_std, 1.0) + 0.1;

                let var = std.clone().powf_scalar(2.0);
                let diff = action - mean;
                let log_2pi = (2.0 * std::f64::consts::PI).ln();
                let logp = -(diff.clone() * diff) / (var * 2.0) - std.log() - log_2pi / 2.0;
                logp.sum_dim(1).squeeze::<1>(1)
            }
        }
    }

    /// Entropy of the action distribution.
    pub fn entropy(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        let logits = self.forward(x);

        match self.action_type.0 {
            ActionType::Discrete { .. } => {
                let log_probs = burn::tensor::activation::log_softmax(logits.clone(), 1);
                let probs = burn::tensor::activation::softmax(logits, 1);
                -(probs * log_probs).sum_dim(1).squeeze::<1>(1)
            }
            ActionType::Continuous => {
                let batch = logits.dims()[0];
                let raw_std = logits.slice([0..batch, self.action_dim.0..2 * self.action_dim.0]);
                let std = burn::tensor::activation::softplus(raw_std, 1.0) + 0.1;
                let log_2pi = (2.0 * std::f64::consts::PI).ln();
                (std.powf_scalar(2.0).log() * 0.5 + (log_2pi + 1.0) / 2.0)
                    .sum_dim(1)
                    .squeeze::<1>(1)
            }
        }
    }
}
