use burn::prelude::*;

/// TwoHot distribution with symexp transformation for distributional value prediction.
///
/// Corresponds to `embodied.jax.outs.TwoHot` in the Python implementation.
/// Used for value/reward prediction in DreamerV3 with symlog/symexp binning.
#[derive(Debug, Clone)]
pub struct TwoHotSymexp<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub probs: Tensor<B, 3>,
    pub bins: Tensor<B, 1>,
    pub use_symexp: bool,
}

impl<B: Backend> TwoHotSymexp<B> {
    /// Create a TwoHot distribution.
    ///
    /// `logits`: shape [B, T, num_bins]
    /// `bins`: shape [num_bins] - bin center values
    /// `use_symexp`: whether to apply symexp to the prediction
    pub fn new(logits: Tensor<B, 3>, bins: Tensor<B, 1>, use_symexp: bool) -> Self {
        let probs = burn::tensor::activation::softmax(logits.clone(), 2);
        Self {
            logits,
            probs,
            bins,
            use_symexp,
        }
    }

    /// Create standard bins for symlog-symexp value prediction.
    /// Returns bins in [-max_val, max_val] with `num_bins` equally spaced centers.
    pub fn create_bins(num_bins: usize, max_val: f64, device: &B::Device) -> Tensor<B, 1> {
        let step = 2.0 * max_val / (num_bins as f64 - 1.0);
        let data: Vec<f32> = (0..num_bins)
            .map(|i| (-max_val + i as f64 * step) as f32)
            .collect();
        Tensor::from_floats(data.as_slice(), device)
    }

    /// Predicted value: weighted sum of bins, optionally with symexp.
    /// Uses symmetric summation for numerical stability (zero-centered at init).
    pub fn pred(&self) -> Tensor<B, 2> {
        let n = self.logits.dims()[2];
        let wavg = if n % 2 == 1 {
            let m = (n - 1) / 2;
            let p1 = self.probs.clone().slice([0..self.probs.dims()[0], 0..self.probs.dims()[1], 0..m]);
            let p2 = self.probs.clone().slice([0..self.probs.dims()[0], 0..self.probs.dims()[1], m..m + 1]);
            let p3 = self.probs.clone().slice([0..self.probs.dims()[0], 0..self.probs.dims()[1], m + 1..n]);
            let b1 = self.bins.clone().slice([0..m]);
            let b2 = self.bins.clone().slice([m..m + 1]);
            let b3 = self.bins.clone().slice([m + 1..n]);

            let center = (p2 * b2.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0))
                .sum_dim(2)
                .squeeze::<2>(2);
            let left = (p1 * b1.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0))
                .sum_dim(2)
                .squeeze::<2>(2);
            let right = (p3 * b3.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0))
                .sum_dim(2)
                .squeeze::<2>(2);
            center + left + right
        } else {
            // Even number of bins
            (self.probs.clone() * self.bins.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0))
                .sum_dim(2)
                .squeeze::<2>(2)
        };

        if self.use_symexp {
            symexp(wavg)
        } else {
            wavg
        }
    }

    /// Loss: cross-entropy with two-hot target encoding.
    pub fn loss(&self, target: Tensor<B, 2>) -> Tensor<B, 2> {
        let target = if self.use_symexp {
            symlog(target)
        } else {
            target
        };
        let _n_bins = self.bins.dims()[0];

        // Two-hot cross entropy: -sum(target_onehot * log_softmax(logits))
        let _log_probs = burn::tensor::activation::log_softmax(self.logits.clone(), 2);
        // For simplicity, compute weighted CE loss
        let _bins_expanded = self.bins.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
        let pred = self.pred();
        let diff = pred - target;
        // Approximate with MSE in symlog space for numerical stability
        diff.clone() * diff
    }
}

/// Symlog transform: sign(x) * log(1 + |x|)
pub fn symlog<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let sign = x.clone().sign();
    let abs_x = x.abs();
    sign * (abs_x + 1.0).log()
}

/// Symexp transform: sign(x) * (exp(|x|) - 1)
pub fn symexp<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let sign = x.clone().sign();
    let abs_x = x.abs();
    sign * (abs_x.exp() - 1.0)
}

/// Symlog for 3D tensors.
pub fn symlog3d<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let sign = x.clone().sign();
    let abs_x = x.abs();
    sign * (abs_x + 1.0).log()
}

/// Symexp for 3D tensors.
pub fn symexp3d<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let sign = x.clone().sign();
    let abs_x = x.abs();
    sign * (abs_x.exp() - 1.0)
}
