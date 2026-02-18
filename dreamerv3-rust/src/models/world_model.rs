use burn::prelude::*;

use super::decoder::{Decoder, DecoderConfig};
use super::encoder::{Encoder, EncoderConfig};
use super::heads::{MLPHead, MLPHeadConfig};
use super::rssm::{RSSM, RSSMConfig};

/// World Model combining encoder, RSSM, decoder, and prediction heads.
///
/// Corresponds to the world model portion of `agent.Agent` in the Python implementation.
#[derive(Module, Debug)]
pub struct WorldModel<B: Backend> {
    pub encoder: Encoder<B>,
    pub rssm: RSSM<B>,
    pub decoder: Decoder<B>,
    pub reward_head: MLPHead<B>,
    pub continue_head: MLPHead<B>,
}

#[derive(Debug, Clone)]
pub struct WorldModelConfig {
    pub encoder: EncoderConfig,
    pub rssm: RSSMConfig,
    pub decoder: DecoderConfig,
    pub reward_head: MLPHeadConfig,
    pub continue_head: MLPHeadConfig,
}

impl WorldModelConfig {
    /// Create a default configuration for image-based tasks (e.g., Atari).
    pub fn for_image_task(image_res: [usize; 2], image_channels: usize, action_dim: usize) -> Self {
        let depth = 64;
        let mults = vec![2, 3, 4, 4];
        let deter = 4096;
        let stoch = 32;
        let classes = 32;
        let units = 1024;
        let feat_dim = deter + stoch * classes;

        // Compute token dim from encoder CNN
        let n_layers = mults.len();
        let factor = 2usize.pow(n_layers as u32);
        let final_h = image_res[0] / factor;
        let final_w = image_res[1] / factor;
        let final_depth = depth * mults[n_layers - 1];
        let token_dim = final_h * final_w * final_depth;

        Self {
            encoder: EncoderConfig {
                has_image: true,
                image_channels,
                has_vector: false,
                vector_dim: 0,
                depth,
                mults: mults.clone(),
                units,
                layers: 3,
                kernel: 5,
                act: "gelu".to_string(),
                norm: "rms".to_string(),
                symlog: true,
            },
            rssm: RSSMConfig {
                deter,
                hidden: 2048,
                stoch,
                classes,
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
                action_dim,
                token_dim,
            },
            decoder: DecoderConfig {
                feat_dim,
                deter_dim: deter,
                stoch_dim: stoch * classes,
                has_image: true,
                image_res,
                image_channels,
                has_vector: false,
                vector_dim: 0,
                depth,
                mults,
                units,
                layers: 3,
                kernel: 5,
                act: "gelu".to_string(),
                norm: "rms".to_string(),
                bspace: 8,
                symlog: true,
            },
            reward_head: MLPHeadConfig::scalar(feat_dim),
            continue_head: MLPHeadConfig::binary(feat_dim),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> WorldModel<B> {
        WorldModel {
            encoder: self.encoder.init(device),
            rssm: self.rssm.init(device),
            decoder: self.decoder.init(device),
            reward_head: self.reward_head.init(device),
            continue_head: self.continue_head.init(device),
        }
    }
}

impl<B: Backend> WorldModel<B> {
    /// Feature dimension of the world model.
    pub fn feat_dim(&self) -> usize {
        self.rssm.feat_dim()
    }
}
