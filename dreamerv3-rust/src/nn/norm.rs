use burn::prelude::*;
use burn::nn::{RmsNorm, RmsNormConfig, LayerNorm, LayerNormConfig};

/// Normalization layer supporting RMSNorm and LayerNorm.
///
/// Corresponds to `nn.Norm` in the Python implementation.
/// DreamerV3 primarily uses RMSNorm ('rms').
///
/// We use a struct with optional fields instead of an enum so that
/// `#[derive(Module)]` works (all fields must implement Module or be skipped).
#[derive(Module, Debug)]
pub struct Norm<B: Backend> {
    rms: Option<RmsNorm<B>>,
    layer: Option<LayerNorm<B>>,
}

/// Configuration for Norm.
#[derive(Debug, Clone)]
pub struct NormConfig {
    pub impl_type: String,
    pub d_model: usize,
    pub eps: f64,
}

impl NormConfig {
    pub fn new(impl_type: &str, d_model: usize) -> Self {
        Self {
            impl_type: impl_type.to_string(),
            d_model,
            eps: 1e-4,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Norm<B> {
        match self.impl_type.as_str() {
            "rms" => {
                let config = RmsNormConfig::new(self.d_model).with_epsilon(self.eps);
                Norm {
                    rms: Some(config.init(device)),
                    layer: None,
                }
            }
            "layer" => {
                let config = LayerNormConfig::new(self.d_model).with_epsilon(self.eps);
                Norm {
                    rms: None,
                    layer: Some(config.init(device)),
                }
            }
            "none" => Norm {
                rms: None,
                layer: None,
            },
            _ => panic!("Unknown norm type: {}", self.impl_type),
        }
    }
}

impl<B: Backend> Norm<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        if let Some(ref norm) = self.rms {
            norm.forward(x)
        } else if let Some(ref norm) = self.layer {
            norm.forward(x)
        } else {
            x
        }
    }

    pub fn forward3d(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Apply normalization on the last dimension
        let dims = x.dims();
        let batch = dims[0];
        let seq = dims[1];
        let feat = dims[2];
        let flat = x.reshape([batch * seq, feat]);
        let normed = self.forward(flat);
        normed.reshape([batch, seq, feat])
    }

    pub fn forward4d(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Apply normalization on the last dimension (channel dim for NHWC)
        let dims = x.dims();
        let batch = dims[0];
        let h = dims[1];
        let w = dims[2];
        let c = dims[3];
        let flat = x.reshape([batch * h * w, c]);
        let normed = self.forward(flat);
        normed.reshape([batch, h, w, c])
    }
}
