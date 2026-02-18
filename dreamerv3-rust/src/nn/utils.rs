use burn::prelude::*;

/// Symlog transform: sign(x) * log(1 + |x|)
/// Used for preprocessing observations before feeding into the encoder.
pub fn symlog<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let sign = x.clone().sign();
    let abs_x = x.abs();
    sign * (abs_x + 1.0).log()
}

/// Symexp transform: sign(x) * (exp(|x|) - 1)
/// Inverse of symlog.
pub fn symexp<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let sign = x.clone().sign();
    let abs_x = x.abs();
    sign * (abs_x.exp() - 1.0)
}

/// Mask tensor values: where mask is true, keep x; where false, set to zero.
/// Corresponds to `nn.mask` in the Python implementation.
pub fn mask_tensor<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    mask: Tensor<B, 1, Bool>,
) -> Tensor<B, D> {
    let mask_float = mask.float();
    // Expand mask to match tensor dimensions
    let mut shape = [1usize; D];
    shape[0] = mask_float.dims()[0];
    let mask_expanded = mask_float.reshape(shape);
    x * mask_expanded
}

/// GeLU activation function.
pub fn gelu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    burn::tensor::activation::gelu(x)
}

/// SiLU (Swish) activation function.
pub fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    burn::tensor::activation::silu(x)
}

/// Apply activation by name.
pub fn apply_activation<B: Backend>(x: Tensor<B, 2>, act: &str) -> Tensor<B, 2> {
    match act {
        "gelu" => burn::tensor::activation::gelu(x),
        "silu" | "swish" => burn::tensor::activation::silu(x),
        "relu" => burn::tensor::activation::relu(x),
        "tanh" => x.tanh(),
        "sigmoid" => burn::tensor::activation::sigmoid(x),
        "none" => x,
        _ => panic!("Unknown activation: {}", act),
    }
}

/// Apply activation for 3D tensors.
pub fn apply_activation3d<B: Backend>(x: Tensor<B, 3>, act: &str) -> Tensor<B, 3> {
    match act {
        "gelu" => burn::tensor::activation::gelu(x),
        "silu" | "swish" => burn::tensor::activation::silu(x),
        "relu" => burn::tensor::activation::relu(x),
        "tanh" => x.tanh(),
        "sigmoid" => burn::tensor::activation::sigmoid(x),
        "none" => x,
        _ => panic!("Unknown activation: {}", act),
    }
}

/// Apply activation for 4D tensors (images).
pub fn apply_activation4d<B: Backend>(x: Tensor<B, 4>, act: &str) -> Tensor<B, 4> {
    match act {
        "gelu" => burn::tensor::activation::gelu(x),
        "silu" | "swish" => burn::tensor::activation::silu(x),
        "relu" => burn::tensor::activation::relu(x),
        "tanh" => x.tanh(),
        "sigmoid" => burn::tensor::activation::sigmoid(x),
        "none" => x,
        _ => panic!("Unknown activation: {}", act),
    }
}
