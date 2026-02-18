use burn::prelude::*;

/// Categorical distribution over discrete classes.
///
/// Corresponds to `embodied.jax.outs.Categorical` in the Python implementation.
/// Supports uniform mixing (unimix) for exploration.
#[derive(Debug, Clone)]
pub struct Categorical<B: Backend> {
    /// Log-probabilities after softmax (with optional unimix).
    pub logits: Tensor<B, 3>,
}

impl<B: Backend> Categorical<B> {
    /// Create a new Categorical distribution from logits.
    ///
    /// If `unimix > 0`, mixes softmax probabilities with a uniform distribution:
    ///   probs = (1 - unimix) * softmax(logits) + unimix * uniform
    ///   logits = log(probs)
    pub fn new(logits: Tensor<B, 3>, unimix: f64) -> Self {
        let logits = if unimix > 0.0 {
            let probs = burn::tensor::activation::softmax(logits, 2);
            let n_classes = probs.dims()[2];
            let uniform = probs.clone().ones_like() / (n_classes as f64);
            let mixed = probs * (1.0 - unimix) + uniform * unimix;
            mixed.log()
        } else {
            logits
        };
        Self { logits }
    }

    /// Predicted class (argmax).
    pub fn pred(&self) -> Tensor<B, 2, Int> {
        self.logits.clone().argmax(2).squeeze::<2>(2)
    }

    /// Sample from the distribution using Gumbel-max trick.
    pub fn sample(&self, device: &B::Device) -> Tensor<B, 2, Int> {
        let shape = self.logits.dims();
        let uniform = Tensor::<B, 3>::random(shape, burn::tensor::Distribution::Uniform(1e-7, 1.0 - 1e-7), device);
        let gumbel = -((-uniform.log()).log());
        let perturbed = self.logits.clone() + gumbel;
        perturbed.argmax(2).squeeze::<2>(2)
    }

    /// Log-probability of an event (integer class index).
    pub fn logp(&self, event: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let log_softmax = burn::tensor::activation::log_softmax(self.logits.clone(), 2);
        let n_classes = log_softmax.dims()[2];
        let one_hot = one_hot_encode::<B>(event, n_classes);
        (log_softmax * one_hot).sum_dim(2).squeeze::<2>(2)
    }

    /// Entropy of the distribution: -sum(p * log(p)).
    pub fn entropy(&self) -> Tensor<B, 2> {
        let log_probs = burn::tensor::activation::log_softmax(self.logits.clone(), 2);
        let probs = burn::tensor::activation::softmax(self.logits.clone(), 2);
        -(probs * log_probs).sum_dim(2).squeeze::<2>(2)
    }

    /// KL divergence: KL(self || other) = sum(p * (log(p) - log(q))).
    pub fn kl(&self, other: &Categorical<B>) -> Tensor<B, 2> {
        let log_p = burn::tensor::activation::log_softmax(self.logits.clone(), 2);
        let log_q = burn::tensor::activation::log_softmax(other.logits.clone(), 2);
        let p = burn::tensor::activation::softmax(self.logits.clone(), 2);
        (p * (log_p - log_q)).sum_dim(2).squeeze::<2>(2)
    }
}

/// One-hot encode integer indices into float tensor.
pub fn one_hot_encode<B: Backend>(indices: Tensor<B, 2, Int>, n_classes: usize) -> Tensor<B, 3> {
    let dims = indices.dims();
    let batch = dims[0];
    let seq = dims[1];
    let device = indices.device();

    let flat = indices.clone().reshape([batch * seq]);
    let zeros = Tensor::<B, 2>::zeros([batch * seq, n_classes], &device);

    let ones = Tensor::<B, 1>::ones([batch * seq], &device);
    let row_indices = Tensor::<B, 1, Int>::arange(0..(batch * seq) as i64, &device);
    let col_indices = flat;

    // Build one-hot by scatter: zeros[row, col] = 1.0
    let result = zeros.select_assign(
        0,
        row_indices.clone(),
        Tensor::<B, 2>::zeros([batch * seq, n_classes], &device)
            .select_assign(1, col_indices, ones.unsqueeze_dim::<2>(1)),
    );
    result.reshape([batch, seq, n_classes])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::ElementConversion;

    type TestBackend = NdArray;

    #[test]
    fn test_categorical_entropy() {
        let device = Default::default();
        // Uniform logits should give max entropy = ln(n_classes)
        let logits = Tensor::<TestBackend, 3>::zeros([1, 1, 4], &device);
        let dist = Categorical::new(logits, 0.0);
        let entropy = dist.entropy();
        let expected = (4.0_f32).ln();
        let val: f32 = entropy.into_scalar().elem();
        assert!((val - expected).abs() < 1e-5);
    }
}
