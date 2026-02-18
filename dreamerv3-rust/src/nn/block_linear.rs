use burn::prelude::*;
use burn::nn::{Linear as BurnLinear, LinearConfig as BurnLinearConfig};
use burn::module::Ignored;

/// Block-diagonal linear layer for efficient grouped transformations.
///
/// Corresponds to `nn.BlockLinear` in the Python implementation.
/// The weight matrix is block-diagonal: each block processes its portion
/// of the input independently. This is used in the RSSM core for
/// block-wise recurrent processing.
///
/// For `blocks=8, units=4096`:
///   input is split into 8 groups of input_size/8
///   each group has its own (input_size/8, 512) weight matrix
///   outputs are concatenated back to 4096 dims
#[derive(Module, Debug)]
pub struct BlockLinear<B: Backend> {
    /// One linear layer per block
    blocks: Vec<BurnLinear<B>>,
    n_blocks: Ignored<usize>,
    units: Ignored<usize>,
}

#[derive(Debug, Clone)]
pub struct BlockLinearConfig {
    pub units: usize,
    pub n_blocks: usize,
    pub input_size: usize,
    pub bias: bool,
}

impl BlockLinearConfig {
    pub fn new(input_size: usize, units: usize, n_blocks: usize) -> Self {
        assert!(
            units % n_blocks == 0,
            "units ({}) must be divisible by n_blocks ({})",
            units,
            n_blocks
        );
        assert!(
            input_size % n_blocks == 0,
            "input_size ({}) must be divisible by n_blocks ({})",
            input_size,
            n_blocks
        );
        Self {
            units,
            n_blocks,
            input_size,
            bias: true,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> BlockLinear<B> {
        let in_per_block = self.input_size / self.n_blocks;
        let out_per_block = self.units / self.n_blocks;

        let blocks: Vec<BurnLinear<B>> = (0..self.n_blocks)
            .map(|_| {
                BurnLinearConfig::new(in_per_block, out_per_block)
                    .with_bias(self.bias)
                    .init(device)
            })
            .collect();

        BlockLinear {
            blocks,
            n_blocks: Ignored(self.n_blocks),
            units: Ignored(self.units),
        }
    }
}

impl<B: Backend> BlockLinear<B> {
    /// Forward pass with block-diagonal structure.
    ///
    /// Input shape: [batch, input_size] where input_size = n_blocks * in_per_block
    /// Output shape: [batch, units] where units = n_blocks * out_per_block
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch = x.dims()[0];
        let in_per_block = x.dims()[1] / *self.n_blocks;
        let _out_per_block = *self.units / *self.n_blocks;

        let mut outputs = Vec::with_capacity(*self.n_blocks);
        for (i, block) in self.blocks.iter().enumerate() {
            let start = i * in_per_block;
            let end = start + in_per_block;
            let x_block = x.clone().slice([0..batch, start..end]);
            outputs.push(block.forward(x_block));
        }
        Tensor::cat(outputs, 1)
    }

    /// Forward pass for 3D input.
    pub fn forward3d(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dims = x.dims();
        let flat = x.reshape([dims[0] * dims[1], dims[2]]);
        let out = self.forward(flat);
        let out_dim = out.dims()[1];
        out.reshape([dims[0], dims[1], out_dim])
    }
}
