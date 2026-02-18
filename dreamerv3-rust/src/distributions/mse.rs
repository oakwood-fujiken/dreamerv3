use burn::prelude::*;

/// MSE (Mean Squared Error) distribution for regression outputs.
///
/// Corresponds to `embodied.jax.outs.MSE` in the Python implementation.
/// Used for image and continuous observation reconstruction.
#[derive(Debug, Clone)]
pub struct MseDist<B: Backend> {
    pub mean: Tensor<B, 4>,
}

impl<B: Backend> MseDist<B> {
    pub fn new(mean: Tensor<B, 4>) -> Self {
        Self { mean }
    }

    /// Predicted value.
    pub fn pred(&self) -> Tensor<B, 4> {
        self.mean.clone()
    }

    /// MSE loss: (mean - target)^2, summed over last `agg_dims` dimensions.
    pub fn loss(&self, target: Tensor<B, 4>) -> Tensor<B, 2> {
        let diff = self.mean.clone() - target;
        let sq = diff.clone() * diff;
        // Sum over the last 2 dims (spatial) for images
        sq.sum_dim(3)
            .squeeze::<3>(3)
            .sum_dim(2)
            .squeeze::<2>(2)
    }
}

/// MSE distribution for 2D output (vectors).
#[derive(Debug, Clone)]
pub struct MseDist2d<B: Backend> {
    pub mean: Tensor<B, 3>,
}

impl<B: Backend> MseDist2d<B> {
    pub fn new(mean: Tensor<B, 3>) -> Self {
        Self { mean }
    }

    pub fn pred(&self) -> Tensor<B, 3> {
        self.mean.clone()
    }

    /// MSE loss summed over the last dimension.
    pub fn loss(&self, target: Tensor<B, 3>) -> Tensor<B, 2> {
        let diff = self.mean.clone() - target;
        let sq = diff.clone() * diff;
        sq.sum_dim(2).squeeze::<2>(2)
    }
}
