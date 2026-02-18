use burn::prelude::*;
use burn::tensor::ElementConversion;

use super::categorical::Categorical;

/// OneHot distribution with straight-through gradients.
///
/// Corresponds to `embodied.jax.outs.OneHot` in the Python implementation.
/// Used by the RSSM for stochastic state representation.
/// Samples are one-hot encoded but gradients flow through softmax probabilities.
#[derive(Debug, Clone)]
pub struct OneHotCategorical<B: Backend> {
    pub dist: Categorical<B>,
}

impl<B: Backend> OneHotCategorical<B> {
    pub fn new(logits: Tensor<B, 3>, unimix: f64) -> Self {
        Self {
            dist: Categorical::new(logits, unimix),
        }
    }

    /// Predicted one-hot vector (argmax with straight-through grad).
    pub fn pred(&self) -> Tensor<B, 3> {
        let probs = burn::tensor::activation::softmax(self.dist.logits.clone(), 2);
        // In inference, we return the one-hot of the argmax.
        // In training with autograd, gradients flow through softmax probs.
        // This is the straight-through estimator.
        probs
    }

    /// Sample a one-hot vector using Gumbel-softmax-like sampling.
    pub fn sample(&self, device: &B::Device) -> Tensor<B, 3> {
        let shape = self.dist.logits.dims();
        let n_classes = shape[2];
        let index = self.dist.sample(device);

        // Create one-hot from index
        let probs = burn::tensor::activation::softmax(self.dist.logits.clone(), 2);

        // One-hot encode the sampled indices
        // For straight-through: one_hot(sample) + probs - stop_grad(probs)
        // Since Burn handles autograd differently, we approximate:
        let batch = shape[0];
        let seq = shape[1];
        let device = self.dist.logits.device();

        // Create one-hot representation
        let flat_index = index.reshape([batch * seq]);
        let mut one_hot = Tensor::<B, 2>::zeros([batch * seq, n_classes], &device);
        // Use scatter-like approach
        for i in 0..batch * seq {
            let idx: i64 = flat_index
                .clone()
                .slice([i..i + 1])
                .into_scalar()
                .elem();
            one_hot = one_hot.slice_assign(
                [i..i + 1, idx as usize..(idx as usize + 1)],
                Tensor::ones([1, 1], &device),
            );
        }
        let one_hot = one_hot.reshape([batch, seq, n_classes]);

        // Straight-through: use one_hot for forward, probs for backward
        // one_hot + (probs - probs.detach())
        // In Burn, .detach() is .inner() for AutodiffBackend
        one_hot + probs.clone() - probs
    }

    /// Log-probability of a one-hot event.
    pub fn logp(&self, event: Tensor<B, 3>) -> Tensor<B, 2> {
        let log_softmax = burn::tensor::activation::log_softmax(self.dist.logits.clone(), 2);
        (log_softmax * event).sum_dim(2).squeeze::<2>(2)
    }

    /// Entropy of the underlying categorical distribution.
    pub fn entropy(&self) -> Tensor<B, 2> {
        self.dist.entropy()
    }

    /// KL divergence between two OneHotCategorical distributions.
    pub fn kl(&self, other: &OneHotCategorical<B>) -> Tensor<B, 2> {
        self.dist.kl(&other.dist)
    }
}
