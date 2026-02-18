use burn::prelude::*;
use burn::module::Ignored;

use crate::nn::{
    BlockLinear, BlockLinearConfig, DreamerConv2d, DreamerConv2dConfig,
    DreamerLinear, DreamerLinearConfig, MLP, MLPConfig, Norm, NormConfig, upsample2x_nhwc,
};
use crate::nn::utils::apply_activation4d;

/// Decoder that reconstructs observations from world model features.
///
/// Corresponds to `rssm.Decoder` in the Python implementation.
///
/// Architecture:
/// - Image path: Transposed Conv2D pyramid with upsampling
/// - Vector path: MLP with separate output heads
/// - Features input: concatenation of deter and flattened stoch
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    // Image decoder
    /// Spatial projection from features to initial spatial grid
    sp0: Option<BlockLinear<B>>,
    sp1: Option<DreamerLinear<B>>,
    sp1norm: Option<Norm<B>>,
    sp2: Option<DreamerLinear<B>>,
    spnorm: Option<Norm<B>>,
    /// Upsampling conv layers
    conv_layers: Vec<DreamerConv2d<B>>,
    conv_norms: Vec<Norm<B>>,
    /// Final output conv
    img_out: Option<DreamerConv2d<B>>,

    // Vector decoder
    vec_mlp: Option<MLP<B>>,
    vec_head: Option<DreamerLinear<B>>,

    // Config
    depths: Ignored<Vec<usize>>,
    act: Ignored<String>,
    bspace: Ignored<usize>,
    min_res: Ignored<[usize; 2]>,
    image_channels: Ignored<usize>,
    has_image: Ignored<bool>,
    has_vector: Ignored<bool>,
    vector_dim: Ignored<usize>,
}

#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// Feature dimension (deter + stoch * classes)
    pub feat_dim: usize,
    /// Deterministic state dimension
    pub deter_dim: usize,
    /// Stochastic dimension (stoch * classes, flattened)
    pub stoch_dim: usize,
    /// Whether to decode images
    pub has_image: bool,
    /// Image resolution [H, W]
    pub image_res: [usize; 2],
    /// Image output channels
    pub image_channels: usize,
    /// Whether to decode vectors
    pub has_vector: bool,
    /// Vector output dimension
    pub vector_dim: usize,
    /// Base depth for CNN
    pub depth: usize,
    /// Depth multipliers
    pub mults: Vec<usize>,
    /// MLP hidden units
    pub units: usize,
    /// Number of MLP layers for vectors
    pub layers: usize,
    /// Conv kernel size
    pub kernel: usize,
    /// Activation function
    pub act: String,
    /// Normalization type
    pub norm: String,
    /// Block spatial dimension
    pub bspace: usize,
    /// Whether to use symlog for vectors
    pub symlog: bool,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            feat_dim: 5120,
            deter_dim: 4096,
            stoch_dim: 1024,
            has_image: true,
            image_res: [64, 64],
            image_channels: 3,
            has_vector: false,
            vector_dim: 0,
            depth: 64,
            mults: vec![2, 3, 4, 4],
            units: 1024,
            layers: 3,
            kernel: 5,
            act: "gelu".to_string(),
            norm: "rms".to_string(),
            bspace: 8,
            symlog: true,
        }
    }
}

impl DecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Decoder<B> {
        let depths: Vec<usize> = self.mults.iter().map(|m| self.depth * m).collect();
        let n_layers = depths.len();
        let factor = 2usize.pow(n_layers as u32);
        let min_res = [self.image_res[0] / factor, self.image_res[1] / factor];

        // Image decoder components
        let (sp0, sp1, sp1norm, sp2, spnorm) = if self.has_image && self.bspace > 0 {
            let g = self.bspace;
            let u = min_res[0] * min_res[1] * depths[n_layers - 1];
            let sp0 = BlockLinearConfig::new(self.deter_dim, u * g / g, g).init(device);
            let sp1 = DreamerLinearConfig::new(self.stoch_dim, 2 * self.units).init(device);
            let sp1norm = NormConfig::new(&self.norm, 2 * self.units).init(device);
            let sp2_out = min_res[0] * min_res[1] * depths[n_layers - 1];
            let sp2 = DreamerLinearConfig::new(2 * self.units, sp2_out).init(device);
            let spnorm = NormConfig::new(&self.norm, depths[n_layers - 1]).init(device);
            (
                Some(sp0),
                Some(sp1),
                Some(sp1norm),
                Some(sp2),
                Some(spnorm),
            )
        } else {
            (None, None, None, None, None)
        };

        // Upsampling conv layers (reversed order, excluding last depth)
        let mut conv_layers = Vec::new();
        let mut conv_norms = Vec::new();
        if self.has_image {
            for i in (0..n_layers - 1).rev() {
                let in_ch = if i == n_layers - 2 {
                    depths[n_layers - 1]
                } else {
                    depths[i + 1]
                };
                let conv = DreamerConv2dConfig::new(in_ch, depths[i], self.kernel).init(device);
                let norm = NormConfig::new(&self.norm, depths[i]).init(device);
                conv_layers.push(conv);
                conv_norms.push(norm);
            }
        }

        // Final image output conv
        let img_out = if self.has_image {
            Some(
                DreamerConv2dConfig::new(depths[0], self.image_channels, self.kernel).init(device),
            )
        } else {
            None
        };

        // Vector decoder
        let vec_mlp = if self.has_vector && self.vector_dim > 0 {
            Some(
                MLPConfig::new(self.feat_dim, self.layers, self.units)
                    .with_act(&self.act)
                    .with_norm(&self.norm)
                    .init(device),
            )
        } else {
            None
        };

        let vec_head = if self.has_vector && self.vector_dim > 0 {
            Some(DreamerLinearConfig::new(self.units, self.vector_dim).init(device))
        } else {
            None
        };

        Decoder {
            sp0,
            sp1,
            sp1norm,
            sp2,
            spnorm,
            conv_layers,
            conv_norms,
            img_out,
            vec_mlp,
            vec_head,
            depths: Ignored(depths),
            act: Ignored(self.act.clone()),
            bspace: Ignored(self.bspace),
            min_res: Ignored(min_res),
            image_channels: Ignored(self.image_channels),
            has_image: Ignored(self.has_image),
            has_vector: Ignored(self.has_vector),
            vector_dim: Ignored(self.vector_dim),
        }
    }
}

impl<B: Backend> Decoder<B> {
    /// Decode features into reconstructed observations.
    ///
    /// # Arguments
    /// * `deter` - Deterministic state [B, deter_dim]
    /// * `stoch` - Stochastic state [B, stoch, classes] (flattened internally)
    ///
    /// # Returns
    /// * `image` - Optional reconstructed image [B, H, W, C] (sigmoid, 0-1)
    /// * `vector` - Optional reconstructed vector [B, vec_dim]
    pub fn forward(
        &self,
        deter: Tensor<B, 2>,
        stoch: Tensor<B, 3>,
    ) -> (Option<Tensor<B, 4>>, Option<Tensor<B, 2>>) {
        let batch = deter.dims()[0];
        let stoch_flat = {
            let dims = stoch.dims();
            stoch.reshape([dims[0], dims[1] * dims[2]])
        };
        let feat = Tensor::cat(vec![deter.clone(), stoch_flat.clone()], 1);

        // Vector reconstruction
        let vec_out = if self.has_vector.0 {
            if let (Some(mlp), Some(head)) = (&self.vec_mlp, &self.vec_head) {
                let x = mlp.forward(feat.clone());
                Some(head.forward(x))
            } else {
                None
            }
        } else {
            None
        };

        // Image reconstruction
        let img_out = if self.has_image.0 {
            let h = self.min_res[0];
            let w = self.min_res[1];
            let c = *self.depths.last().unwrap();

            // Spatial projection
            let x = if let (Some(sp0), Some(sp1), Some(sp1norm), Some(sp2), Some(spnorm)) = (
                &self.sp0,
                &self.sp1,
                &self.sp1norm,
                &self.sp2,
                &self.spnorm,
            ) {
                // Block-wise spatial projection from deter
                let x0 = sp0.forward(deter.clone());
                let x0 = x0.reshape([batch, h, w, c]);

                // MLP projection from stoch
                let x1 = sp1.forward(stoch_flat);
                let x1 = sp1norm.forward(x1);
                let x1 = apply_act_2d(x1, &self.act);
                let x1 = sp2.forward(x1);
                let x1 = x1.reshape([batch, h, w, c]);

                // Combine and normalize
                let x = x0 + x1;
                let x = spnorm.forward4d(x);
                apply_activation4d(x, &self.act)
            } else {
                // Simple linear projection
                Tensor::zeros([batch, h, w, c], &feat.device())
            };

            // Upsample + conv layers
            let mut x = x;
            for (conv, norm) in self.conv_layers.iter().zip(self.conv_norms.iter()) {
                x = upsample2x_nhwc(x);
                x = conv.forward(x);
                x = norm.forward4d(x);
                x = apply_activation4d(x, &self.act);
            }

            // Final output conv + sigmoid
            if let Some(img_out_conv) = &self.img_out {
                x = upsample2x_nhwc(x);
                x = img_out_conv.forward(x);
                x = burn::tensor::activation::sigmoid(x);
            }

            Some(x)
        } else {
            None
        };

        (img_out, vec_out)
    }

    /// Decode sequential features.
    ///
    /// # Arguments
    /// * `deter` - [B, T, deter_dim]
    /// * `stoch` - [B, T, stoch, classes]
    pub fn forward_seq(
        &self,
        deter: Tensor<B, 3>,
        stoch: Tensor<B, 4>,
    ) -> (Option<Tensor<B, 5>>, Option<Tensor<B, 3>>) {
        let deter_dims = deter.dims();
        let b = deter_dims[0];
        let t = deter_dims[1];

        let deter_flat = deter.reshape([b * t, deter_dims[2]]);
        let stoch_flat = {
            let dims = stoch.dims();
            stoch.reshape([b * t, dims[2], dims[3]])
        };

        let (img, vec) = self.forward(deter_flat, stoch_flat);

        let img_out = img.map(|x| {
            let dims = x.dims();
            x.reshape([b, t, dims[1], dims[2], dims[3]])
        });

        let vec_out = vec.map(|x| {
            let dims = x.dims();
            x.reshape([b, t, dims[1]])
        });

        (img_out, vec_out)
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
