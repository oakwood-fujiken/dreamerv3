use burn::prelude::*;
use burn::nn::{Linear as BurnLinear, LinearConfig as BurnLinearConfig};
use burn::module::Ignored;

/// Linear layer with optional output scaling.
///
/// Corresponds to `nn.Linear` in the Python implementation.
/// Supports `outscale` parameter for output initialization scaling.
#[derive(Module, Debug)]
pub struct DreamerLinear<B: Backend> {
    linear: BurnLinear<B>,
    outscale: Ignored<f64>,
}

#[derive(Debug, Clone)]
pub struct DreamerLinearConfig {
    pub input_size: usize,
    pub output_size: usize,
    pub bias: bool,
    pub outscale: f64,
}

impl DreamerLinearConfig {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            output_size,
            bias: true,
            outscale: 1.0,
        }
    }

    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    pub fn with_outscale(mut self, outscale: f64) -> Self {
        self.outscale = outscale;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> DreamerLinear<B> {
        let config = BurnLinearConfig::new(self.input_size, self.output_size)
            .with_bias(self.bias);
        DreamerLinear {
            linear: config.init(device),
            outscale: Ignored(self.outscale),
        }
    }
}

impl<B: Backend> DreamerLinear<B> {
    /// Forward pass: y = x @ W^T + b
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }

    /// Forward pass for 3D input (batch, seq, features).
    pub fn forward3d(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dims = x.dims();
        let flat = x.reshape([dims[0] * dims[1], dims[2]]);
        let out = self.forward(flat);
        let out_dim = out.dims()[1];
        out.reshape([dims[0], dims[1], out_dim])
    }
}
