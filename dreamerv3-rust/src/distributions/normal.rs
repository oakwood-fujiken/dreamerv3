use burn::prelude::*;

/// Normal (Gaussian) distribution.
///
/// Corresponds to `embodied.jax.outs.Normal` in the Python implementation.
#[derive(Debug, Clone)]
pub struct Normal<B: Backend> {
    pub mean: Tensor<B, 3>,
    pub stddev: Tensor<B, 3>,
}

impl<B: Backend> Normal<B> {
    pub fn new(mean: Tensor<B, 3>, stddev: Tensor<B, 3>) -> Self {
        Self { mean, stddev }
    }

    pub fn new_with_scalar_std(mean: Tensor<B, 3>, stddev: f64) -> Self {
        let stddev_tensor = mean.clone().ones_like() * stddev;
        Self {
            mean,
            stddev: stddev_tensor,
        }
    }

    /// Predicted value (mean).
    pub fn pred(&self) -> Tensor<B, 3> {
        self.mean.clone()
    }

    /// Sample from Normal(mean, stddev).
    pub fn sample(&self, device: &B::Device) -> Tensor<B, 3> {
        let noise = Tensor::<B, 3>::random(
            self.mean.dims(),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        self.mean.clone() + noise * self.stddev.clone()
    }

    /// Log-probability: log N(event; mean, stddev).
    pub fn logp(&self, event: Tensor<B, 3>) -> Tensor<B, 3> {
        let var = self.stddev.clone().powf_scalar(2.0);
        let log_std = self.stddev.clone().log();
        let diff = event - self.mean.clone();
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        -(diff.clone() * diff) / (var * 2.0) - log_std - log_2pi / 2.0
    }

    /// Entropy: 0.5 * log(2 * pi * stddev^2) + 0.5
    pub fn entropy(&self) -> Tensor<B, 3> {
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        self.stddev.clone().powf_scalar(2.0).log() * 0.5 + (log_2pi + 1.0) / 2.0
    }

    /// KL divergence: KL(self || other).
    pub fn kl(&self, other: &Normal<B>) -> Tensor<B, 3> {
        let var_self = self.stddev.clone().powf_scalar(2.0);
        let var_other = other.stddev.clone().powf_scalar(2.0);
        let diff = other.mean.clone() - self.mean.clone();
        (var_self.clone() / var_other.clone()
            + diff.clone() * diff / var_other.clone()
            + other.stddev.clone().log() * 2.0
            - self.stddev.clone().log() * 2.0
            - 1.0)
            * 0.5
    }
}
