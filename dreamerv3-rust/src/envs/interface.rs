/// Observation from the environment.
#[derive(Debug, Clone)]
pub struct Observation {
    /// Image observation [H, W, C] as flattened bytes (0-255)
    pub image: Option<Vec<u8>>,
    /// Image dimensions [H, W, C]
    pub image_shape: Option<[usize; 3]>,
    /// Vector observation (proprioception, etc.)
    pub vector: Option<Vec<f32>>,
    /// Scalar reward
    pub reward: f32,
    /// Whether this is the first step of an episode
    pub is_first: bool,
    /// Whether this is the last step of an episode
    pub is_last: bool,
    /// Whether the episode terminated (vs truncated)
    pub is_terminal: bool,
}

/// Action to send to the environment.
#[derive(Debug, Clone)]
pub struct Action {
    /// For discrete actions: action index
    pub discrete: Option<usize>,
    /// For continuous actions: action vector
    pub continuous: Option<Vec<f32>>,
}

/// Observation space descriptor.
#[derive(Debug, Clone)]
pub struct ObsSpace {
    pub image_shape: Option<[usize; 3]>,
    pub vector_dim: Option<usize>,
}

/// Action space descriptor.
#[derive(Debug, Clone)]
pub enum ActionSpace {
    Discrete { n: usize },
    Continuous { dim: usize, low: f32, high: f32 },
}

impl ActionSpace {
    pub fn dim(&self) -> usize {
        match self {
            ActionSpace::Discrete { n } => *n,
            ActionSpace::Continuous { dim, .. } => *dim,
        }
    }
}

/// Environment trait that all environments must implement.
///
/// Corresponds to `embodied.core.base.Env` in the Python implementation.
pub trait Environment {
    /// Get the observation space.
    fn obs_space(&self) -> ObsSpace;

    /// Get the action space.
    fn act_space(&self) -> ActionSpace;

    /// Reset the environment and return the initial observation.
    fn reset(&mut self) -> Observation;

    /// Take a step in the environment.
    fn step(&mut self, action: &Action) -> Observation;

    /// Close the environment.
    fn close(&mut self) {}
}

/// A dummy environment for testing.
pub struct DummyEnv {
    obs_space: ObsSpace,
    act_space: ActionSpace,
    step_count: usize,
    episode_length: usize,
}

impl DummyEnv {
    pub fn new(image_shape: [usize; 3], n_actions: usize, episode_length: usize) -> Self {
        Self {
            obs_space: ObsSpace {
                image_shape: Some(image_shape),
                vector_dim: None,
            },
            act_space: ActionSpace::Discrete { n: n_actions },
            step_count: 0,
            episode_length,
        }
    }
}

impl Environment for DummyEnv {
    fn obs_space(&self) -> ObsSpace {
        self.obs_space.clone()
    }

    fn act_space(&self) -> ActionSpace {
        self.act_space.clone()
    }

    fn reset(&mut self) -> Observation {
        self.step_count = 0;
        let shape = self.obs_space.image_shape.unwrap();
        let n_pixels = shape[0] * shape[1] * shape[2];
        Observation {
            image: Some(vec![128u8; n_pixels]),
            image_shape: Some(shape),
            vector: None,
            reward: 0.0,
            is_first: true,
            is_last: false,
            is_terminal: false,
        }
    }

    fn step(&mut self, _action: &Action) -> Observation {
        self.step_count += 1;
        let is_last = self.step_count >= self.episode_length;
        let shape = self.obs_space.image_shape.unwrap();
        let n_pixels = shape[0] * shape[1] * shape[2];

        Observation {
            image: Some(vec![128u8; n_pixels]),
            image_shape: Some(shape),
            vector: None,
            reward: if is_last { 1.0 } else { 0.0 },
            is_first: false,
            is_last,
            is_terminal: is_last,
        }
    }
}
