use std::collections::VecDeque;

/// A single transition stored in the replay buffer.
#[derive(Debug, Clone)]
pub struct Transition {
    /// Flattened observation (image bytes or float vector)
    pub observation: Vec<f32>,
    /// Action taken (one-hot for discrete, raw for continuous)
    pub action: Vec<f32>,
    /// Reward received
    pub reward: f32,
    /// Whether this is the first step of an episode
    pub is_first: bool,
    /// Whether this is the last step of an episode
    pub is_last: bool,
    /// Whether the episode terminated (vs truncated)
    pub is_terminal: bool,
}

/// Consecutive sequence sampled from the replay buffer.
#[derive(Debug, Clone)]
pub struct Sequence {
    pub transitions: Vec<Transition>,
}

/// Uniform replay buffer with circular storage and consecutive sampling.
///
/// Corresponds to `embodied.core.replay.Replay` in the Python implementation.
///
/// Features:
/// - Circular buffer with configurable capacity
/// - Samples consecutive sequences of fixed length
/// - Tracks episode boundaries for proper sequence construction
pub struct ReplayBuffer {
    /// Storage for transitions
    storage: VecDeque<Transition>,
    /// Maximum capacity
    capacity: usize,
    /// Episode start indices within the buffer
    episode_starts: Vec<usize>,
    /// Current total number of transitions added
    total_added: usize,
    /// Minimum sequence length for sampling
    min_seq_len: usize,
}

impl ReplayBuffer {
    /// Create a new replay buffer.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of transitions to store
    /// * `min_seq_len` - Minimum length of sampled sequences
    pub fn new(capacity: usize, min_seq_len: usize) -> Self {
        Self {
            storage: VecDeque::with_capacity(capacity),
            capacity,
            episode_starts: Vec::new(),
            total_added: 0,
            min_seq_len,
        }
    }

    /// Add a transition to the buffer.
    pub fn add(&mut self, transition: Transition) {
        if transition.is_first {
            self.episode_starts.push(self.total_added);
        }

        if self.storage.len() >= self.capacity {
            self.storage.pop_front();
            // Adjust episode starts
            self.episode_starts
                .retain(|&start| start > self.total_added - self.capacity);
        }

        self.storage.push_back(transition);
        self.total_added += 1;
    }

    /// Sample a batch of consecutive sequences.
    ///
    /// # Arguments
    /// * `batch_size` - Number of sequences to sample
    /// * `seq_len` - Length of each sequence
    ///
    /// # Returns
    /// Vector of Sequence, each containing `seq_len` consecutive transitions.
    pub fn sample(&self, batch_size: usize, seq_len: usize) -> Vec<Sequence> {
        let mut rng = rand::thread_rng();
        let mut sequences = Vec::with_capacity(batch_size);

        let buffer_len = self.storage.len();
        if buffer_len < seq_len {
            return sequences;
        }

        let max_start = buffer_len - seq_len;

        for _ in 0..batch_size {
            let start = rand::Rng::gen_range(&mut rng, 0..=max_start);
            let transitions: Vec<Transition> = (start..start + seq_len)
                .map(|i| self.storage[i].clone())
                .collect();
            sequences.push(Sequence { transitions });
        }

        sequences
    }

    /// Current number of transitions in the buffer.
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Whether the buffer has enough data for sampling.
    pub fn can_sample(&self, seq_len: usize) -> bool {
        self.storage.len() >= seq_len
    }

    /// Total number of transitions ever added.
    pub fn total_steps(&self) -> usize {
        self.total_added
    }
}

/// Convert a batch of sequences into tensors for training.
///
/// Returns (observations, actions, rewards, is_first, is_last, is_terminal)
/// each with shape [batch_size, seq_len, ...].
pub struct BatchData {
    pub observations: Vec<Vec<Vec<f32>>>,
    pub actions: Vec<Vec<Vec<f32>>>,
    pub rewards: Vec<Vec<f32>>,
    pub is_first: Vec<Vec<bool>>,
    pub is_last: Vec<Vec<bool>>,
    pub is_terminal: Vec<Vec<bool>>,
}

impl BatchData {
    pub fn from_sequences(sequences: &[Sequence]) -> Self {
        let batch_size = sequences.len();
        if batch_size == 0 {
            return Self {
                observations: vec![],
                actions: vec![],
                rewards: vec![],
                is_first: vec![],
                is_last: vec![],
                is_terminal: vec![],
            };
        }

        let seq_len = sequences[0].transitions.len();

        let mut observations = Vec::with_capacity(batch_size);
        let mut actions = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut is_first = Vec::with_capacity(batch_size);
        let mut is_last = Vec::with_capacity(batch_size);
        let mut is_terminal = Vec::with_capacity(batch_size);

        for seq in sequences {
            let mut obs_seq = Vec::with_capacity(seq_len);
            let mut act_seq = Vec::with_capacity(seq_len);
            let mut rew_seq = Vec::with_capacity(seq_len);
            let mut first_seq = Vec::with_capacity(seq_len);
            let mut last_seq = Vec::with_capacity(seq_len);
            let mut term_seq = Vec::with_capacity(seq_len);

            for t in &seq.transitions {
                obs_seq.push(t.observation.clone());
                act_seq.push(t.action.clone());
                rew_seq.push(t.reward);
                first_seq.push(t.is_first);
                last_seq.push(t.is_last);
                term_seq.push(t.is_terminal);
            }

            observations.push(obs_seq);
            actions.push(act_seq);
            rewards.push(rew_seq);
            is_first.push(first_seq);
            is_last.push(last_seq);
            is_terminal.push(term_seq);
        }

        Self {
            observations,
            actions,
            rewards,
            is_first,
            is_last,
            is_terminal,
        }
    }
}
