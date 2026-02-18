use burn::prelude::*;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::PaddingConfig2d;

/// 2D Convolution layer with DreamerV3-style configuration.
///
/// Corresponds to `nn.Conv2D` in the Python implementation.
/// Supports both regular and transposed convolutions.
/// Uses NHWC (channels-last) format internally, matching the Python code.
/// Burn uses NCHW, so we transpose on input/output.
#[derive(Module, Debug)]
pub enum DreamerConv2d<B: Backend> {
    Regular(Conv2d<B>),
    Transposed(ConvTranspose2d<B>),
}

#[derive(Debug, Clone)]
pub struct DreamerConv2dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub transposed: bool,
    pub bias: bool,
}

impl DreamerConv2dConfig {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            transposed: false,
            bias: true,
        }
    }

    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    pub fn with_transposed(mut self, transposed: bool) -> Self {
        self.transposed = transposed;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> DreamerConv2d<B> {
        if self.transposed {
            // ConvTranspose2d in Burn 0.16 uses [usize; 2] for padding, not PaddingConfig2d.
            // Compute "same" padding manually: pad = (kernel_size - 1) / 2
            let pad = (self.kernel_size - 1) / 2;
            let config = ConvTranspose2dConfig::new(
                [self.in_channels, self.out_channels],
                [self.kernel_size, self.kernel_size],
            )
            .with_stride([self.stride, self.stride])
            .with_padding([pad, pad])
            .with_bias(self.bias);
            DreamerConv2d::Transposed(config.init(device))
        } else {
            let config = Conv2dConfig::new(
                [self.in_channels, self.out_channels],
                [self.kernel_size, self.kernel_size],
            )
            .with_stride([self.stride, self.stride])
            .with_padding(PaddingConfig2d::Same)
            .with_bias(self.bias);
            DreamerConv2d::Regular(config.init(device))
        }
    }
}

impl<B: Backend> DreamerConv2d<B> {
    /// Forward pass.
    /// Input: [B, H, W, C] (NHWC format, matching Python code)
    /// Output: [B, H', W', C_out] (NHWC format)
    ///
    /// Internally converts to NCHW for Burn's conv layers.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // NHWC -> NCHW
        let x = x.permute([0, 3, 1, 2]);

        let out = match self {
            DreamerConv2d::Regular(conv) => conv.forward(x),
            DreamerConv2d::Transposed(conv) => conv.forward(x),
        };

        // NCHW -> NHWC
        out.permute([0, 2, 3, 1])
    }
}

/// Max pool 2D with stride 2, applied in NHWC format.
pub fn max_pool2d_nhwc<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let dims = x.dims();
    let b = dims[0];
    let h = dims[1];
    let w = dims[2];
    let c = dims[3];

    // Reshape to [B, H/2, 2, W/2, 2, C] and take max over pool dims
    let x = x.reshape([b, h / 2, 2, w / 2, 2, c]);
    // Max over dim 2 then dim 3 (after squeeze)
    let x = x.max_dim(2).squeeze::<5>(2); // [B, H/2, W/2, 2, C]
    x.max_dim(3).squeeze::<4>(3) // [B, H/2, W/2, C]
}

/// Upsample 2x by repeating (nearest neighbor) in NHWC format.
pub fn upsample2x_nhwc<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let dims = x.dims();
    let b = dims[0];
    let h = dims[1];
    let w = dims[2];
    let c = dims[3];

    // Repeat along height: [B, H, W, C] -> [B, H, 2, W, C] -> [B, H*2, W, C]
    let x = x.unsqueeze_dim::<5>(2); // [B, H, 1, W, C]
    let x = x.repeat_dim(2, 2); // [B, H, 2, W, C]
    let x = x.reshape([b, h * 2, w, c]); // [B, H*2, W, C]

    // Repeat along width
    let x = x.unsqueeze_dim::<5>(3); // [B, H*2, W, 1, C]
    let x = x.repeat_dim(3, 2); // [B, H*2, W, 2, C]
    x.reshape([b, h * 2, w * 2, c]) // [B, H*2, W*2, C]
}
