use burn::prelude::*;
use burn::module::Ignored;

use super::linear::{DreamerLinear, DreamerLinearConfig};
use super::norm::{Norm, NormConfig};

/// Multi-Layer Perceptron with configurable activation and normalization.
///
/// Corresponds to `nn.MLP` in the Python implementation.
/// Architecture per layer: Linear -> Norm -> Activation
#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    layers: Vec<DreamerLinear<B>>,
    norms: Vec<Norm<B>>,
    act: Ignored<String>,
}

#[derive(Debug, Clone)]
pub struct MLPConfig {
    pub input_size: usize,
    pub n_layers: usize,
    pub units: usize,
    pub act: String,
    pub norm: String,
    pub bias: bool,
}

impl MLPConfig {
    pub fn new(input_size: usize, n_layers: usize, units: usize) -> Self {
        Self {
            input_size,
            n_layers,
            units,
            act: "gelu".to_string(),
            norm: "rms".to_string(),
            bias: true,
        }
    }

    pub fn with_act(mut self, act: &str) -> Self {
        self.act = act.to_string();
        self
    }

    pub fn with_norm(mut self, norm: &str) -> Self {
        self.norm = norm.to_string();
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> MLP<B> {
        let mut layers = Vec::with_capacity(self.n_layers);
        let mut norms = Vec::with_capacity(self.n_layers);

        for i in 0..self.n_layers {
            let in_size = if i == 0 { self.input_size } else { self.units };
            layers.push(
                DreamerLinearConfig::new(in_size, self.units)
                    .with_bias(self.bias)
                    .init(device),
            );
            norms.push(NormConfig::new(&self.norm, self.units).init(device));
        }

        MLP {
            layers,
            norms,
            act: Ignored(self.act.clone()),
        }
    }
}

impl<B: Backend> MLP<B> {
    /// Forward pass: for each layer, apply Linear -> Norm -> Activation.
    /// Input: [batch, features]
    /// Output: [batch, units]
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = x;
        for (linear, norm) in self.layers.iter().zip(self.norms.iter()) {
            x = linear.forward(x);
            x = norm.forward(x);
            x = apply_act(&x, &self.act);
        }
        x
    }

    /// Forward pass for 3D input [batch, seq, features].
    pub fn forward3d(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dims = x.dims();
        let flat = x.reshape([dims[0] * dims[1], dims[2]]);
        let out = self.forward(flat);
        let out_dim = out.dims()[1];
        out.reshape([dims[0], dims[1], out_dim])
    }
}

fn apply_act<B: Backend>(x: &Tensor<B, 2>, act: &str) -> Tensor<B, 2> {
    match act {
        "gelu" => burn::tensor::activation::gelu(x.clone()),
        "silu" | "swish" => burn::tensor::activation::silu(x.clone()),
        "relu" => burn::tensor::activation::relu(x.clone()),
        "tanh" => x.clone().tanh(),
        "sigmoid" => burn::tensor::activation::sigmoid(x.clone()),
        "none" => x.clone(),
        _ => panic!("Unknown activation: {}", act),
    }
}
