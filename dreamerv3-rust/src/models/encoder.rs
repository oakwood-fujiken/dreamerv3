use burn::prelude::*;
use burn::module::Ignored;

use crate::nn::{
    DreamerConv2d, DreamerConv2dConfig, MLP, MLPConfig, Norm,
    NormConfig, max_pool2d_nhwc,
};
use crate::nn::utils::apply_activation4d;

/// Encoder that processes observations (images + vectors) into token representations.
///
/// Corresponds to `rssm.Encoder` in the Python implementation.
///
/// Architecture:
/// - Image path: Conv2D pyramid with max-pooling (4 layers, depth * mults channels)
/// - Vector path: MLP with symlog preprocessing
/// - Outputs: concatenated token vector
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    /// CNN layers for image encoding
    cnn_layers: Vec<DreamerConv2d<B>>,
    cnn_norms: Vec<Norm<B>>,
    /// MLP for vector encoding
    vec_mlp: Option<MLP<B>>,
    /// Activation function name
    act: Ignored<String>,
    /// Whether to apply symlog to vector inputs
    symlog: Ignored<bool>,
    /// Depth multipliers for CNN
    depths: Ignored<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Whether the encoder has image inputs
    pub has_image: bool,
    /// Image channels (e.g., 3 for RGB)
    pub image_channels: usize,
    /// Whether the encoder has vector inputs
    pub has_vector: bool,
    /// Vector input dimension
    pub vector_dim: usize,
    /// Base depth for CNN channels
    pub depth: usize,
    /// Depth multipliers per CNN layer (e.g., [2, 3, 4, 4])
    pub mults: Vec<usize>,
    /// MLP hidden units
    pub units: usize,
    /// Number of MLP layers for vectors
    pub layers: usize,
    /// Convolution kernel size
    pub kernel: usize,
    /// Activation function
    pub act: String,
    /// Normalization type
    pub norm: String,
    /// Whether to apply symlog to vector inputs
    pub symlog: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            has_image: true,
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
            symlog: true,
        }
    }
}

impl EncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
        let depths: Vec<usize> = self.mults.iter().map(|m| self.depth * m).collect();

        // CNN layers
        let mut cnn_layers = Vec::new();
        let mut cnn_norms = Vec::new();

        if self.has_image {
            let mut in_channels = self.image_channels;
            for &depth in &depths {
                let conv = DreamerConv2dConfig::new(in_channels, depth, self.kernel).init(device);
                let norm = NormConfig::new(&self.norm, depth).init(device);
                cnn_layers.push(conv);
                cnn_norms.push(norm);
                in_channels = depth;
            }
        }

        // Vector MLP
        let vec_mlp = if self.has_vector && self.vector_dim > 0 {
            Some(
                MLPConfig::new(self.vector_dim, self.layers, self.units)
                    .with_act(&self.act)
                    .with_norm(&self.norm)
                    .init(device),
            )
        } else {
            None
        };

        Encoder {
            cnn_layers,
            cnn_norms,
            vec_mlp,
            act: Ignored(self.act.clone()),
            symlog: Ignored(self.symlog),
            depths: Ignored(depths),
        }
    }
}

impl<B: Backend> Encoder<B> {
    /// Encode observations into token vectors.
    ///
    /// # Arguments
    /// * `image` - Optional image tensor [B, H, W, C] in uint8 (0-255)
    /// * `vector` - Optional vector tensor [B, vec_dim]
    ///
    /// # Returns
    /// Token tensor [B, token_dim]
    pub fn forward(
        &self,
        image: Option<Tensor<B, 4>>,
        vector: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 2> {
        let mut outputs = Vec::new();

        // Vector path
        if let (Some(mlp), Some(vec)) = (&self.vec_mlp, vector) {
            let x = if self.symlog.0 {
                crate::nn::utils::symlog(vec)
            } else {
                vec
            };
            outputs.push(mlp.forward(x));
        }

        // Image path
        if let Some(img) = image {
            // Normalize from uint8 [0, 255] to [-0.5, 0.5]
            let mut x = img / 255.0 - 0.5;

            for (conv, norm) in self.cnn_layers.iter().zip(self.cnn_norms.iter()) {
                x = conv.forward(x);
                // Max pool 2x2
                x = max_pool2d_nhwc(x);
                x = norm.forward4d(x);
                x = apply_activation4d(x, &self.act);
            }

            // Flatten spatial dimensions: [B, H', W', C] -> [B, H'*W'*C]
            let dims = x.dims();
            let flat = x.reshape([dims[0], dims[1] * dims[2] * dims[3]]);
            outputs.push(flat);
        }

        // Concatenate all outputs
        if outputs.len() == 1 {
            outputs.pop().unwrap()
        } else {
            Tensor::cat(outputs, 1)
        }
    }

    /// Encode a batch of sequential observations.
    ///
    /// # Arguments
    /// * `image` - Optional [B, T, H, W, C]
    /// * `vector` - Optional [B, T, vec_dim]
    ///
    /// # Returns
    /// [B, T, token_dim]
    pub fn forward_seq(
        &self,
        image: Option<Tensor<B, 5>>,
        vector: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        // Flatten batch and time, process, reshape back
        match (&image, &vector) {
            (Some(img), Some(vec)) => {
                let img_dims = img.dims();
                let vec_dims = vec.dims();
                let b = img_dims[0];
                let t = img_dims[1];

                let img_flat =
                    img.clone()
                        .reshape([b * t, img_dims[2], img_dims[3], img_dims[4]]);
                let vec_flat = vec.clone().reshape([b * t, vec_dims[2]]);

                let tokens = self.forward(Some(img_flat), Some(vec_flat));
                let token_dim = tokens.dims()[1];
                tokens.reshape([b, t, token_dim])
            }
            (Some(img), None) => {
                let dims = img.dims();
                let b = dims[0];
                let t = dims[1];
                let img_flat =
                    img.clone().reshape([b * t, dims[2], dims[3], dims[4]]);
                let tokens = self.forward(Some(img_flat), None);
                let token_dim = tokens.dims()[1];
                tokens.reshape([b, t, token_dim])
            }
            (None, Some(vec)) => {
                let dims = vec.dims();
                let b = dims[0];
                let t = dims[1];
                let vec_flat = vec.clone().reshape([b * t, dims[2]]);
                let tokens = self.forward(None, Some(vec_flat));
                let token_dim = tokens.dims()[1];
                tokens.reshape([b, t, token_dim])
            }
            (None, None) => panic!("Encoder requires at least one input (image or vector)"),
        }
    }
}
