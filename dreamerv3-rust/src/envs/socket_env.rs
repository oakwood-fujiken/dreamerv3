use std::io::{Write, BufReader, BufRead};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};

use super::interface::{Action, ActionSpace, Environment, Observation, ObsSpace};

/// Environment that communicates with a Python Gymnasium process via TCP socket.
///
/// This enables DreamerV3 Rust to interact with any Gymnasium-compatible environment
/// through a lightweight JSON protocol over TCP.
///
/// Protocol (JSON lines):
/// - Rust -> Python: {"command": "reset"} or {"command": "step", "action": ...}
/// - Python -> Rust: {"image": [...], "reward": 0.0, "is_first": true, ...}
pub struct SocketEnv {
    stream: TcpStream,
    child: Option<Child>,
    obs_space: ObsSpace,
    act_space: ActionSpace,
    image_shape: Option<[usize; 3]>,
}

/// Message sent from Rust to the Python bridge.
#[derive(serde::Serialize)]
struct EnvCommand {
    command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    action_discrete: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    action_continuous: Option<Vec<f32>>,
}

/// Message received from the Python bridge.
#[derive(serde::Deserialize)]
struct EnvResponse {
    image: Option<Vec<u8>>,
    image_shape: Option<[usize; 3]>,
    vector: Option<Vec<f32>>,
    reward: f32,
    is_first: bool,
    is_last: bool,
    is_terminal: bool,
    // Metadata from initial handshake
    obs_image_shape: Option<[usize; 3]>,
    act_type: Option<String>,
    act_dim: Option<usize>,
    act_low: Option<f32>,
    act_high: Option<f32>,
}

impl SocketEnv {
    /// Connect to an already-running Python bridge at the given address.
    pub fn connect(addr: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;

        let mut env = Self {
            stream,
            child: None,
            obs_space: ObsSpace {
                image_shape: None,
                vector_dim: None,
            },
            act_space: ActionSpace::Discrete { n: 1 },
            image_shape: None,
        };

        // Handshake: request environment info
        env.send_command(&EnvCommand {
            command: "info".to_string(),
            action_discrete: None,
            action_continuous: None,
        })?;
        let info = env.recv_response()?;

        if let Some(shape) = info.obs_image_shape {
            env.obs_space.image_shape = Some(shape);
            env.image_shape = Some(shape);
        }

        if let Some(act_type) = &info.act_type {
            let dim = info.act_dim.unwrap_or(1);
            env.act_space = match act_type.as_str() {
                "discrete" => ActionSpace::Discrete { n: dim },
                "continuous" => ActionSpace::Continuous {
                    dim,
                    low: info.act_low.unwrap_or(-1.0),
                    high: info.act_high.unwrap_or(1.0),
                },
                _ => ActionSpace::Discrete { n: dim },
            };
        }

        log::info!(
            "SocketEnv connected to {}: obs_space={:?}, act_space={:?}",
            addr,
            env.obs_space,
            env.act_space
        );

        Ok(env)
    }

    /// Launch a Python bridge subprocess and connect to it.
    ///
    /// # Arguments
    /// * `python` - Path to Python executable (e.g., "python3")
    /// * `script` - Path to the bridge script
    /// * `task` - Gymnasium environment ID (e.g., "ALE/Pong-v5")
    /// * `port` - TCP port to use
    pub fn launch(
        python: &str,
        script: &str,
        task: &str,
        port: u16,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let child = Command::new(python)
            .arg(script)
            .arg("--task")
            .arg(task)
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        // Wait for the server to start
        std::thread::sleep(std::time::Duration::from_secs(2));

        let addr = format!("127.0.0.1:{}", port);
        let mut env = Self::connect(&addr)?;
        env.child = Some(child);
        Ok(env)
    }

    fn send_command(&mut self, cmd: &EnvCommand) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(cmd)?;
        writeln!(self.stream, "{}", json)?;
        self.stream.flush()?;
        Ok(())
    }

    fn recv_response(&mut self) -> Result<EnvResponse, Box<dyn std::error::Error>> {
        let mut reader = BufReader::new(&self.stream);
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let resp: EnvResponse = serde_json::from_str(line.trim())?;
        Ok(resp)
    }

    fn response_to_observation(&self, resp: EnvResponse) -> Observation {
        Observation {
            image: resp.image,
            image_shape: resp.image_shape.or(self.image_shape),
            vector: resp.vector,
            reward: resp.reward,
            is_first: resp.is_first,
            is_last: resp.is_last,
            is_terminal: resp.is_terminal,
        }
    }
}

impl Environment for SocketEnv {
    fn obs_space(&self) -> ObsSpace {
        self.obs_space.clone()
    }

    fn act_space(&self) -> ActionSpace {
        self.act_space.clone()
    }

    fn reset(&mut self) -> Observation {
        self.send_command(&EnvCommand {
            command: "reset".to_string(),
            action_discrete: None,
            action_continuous: None,
        })
        .expect("Failed to send reset command");

        let resp = self.recv_response().expect("Failed to receive reset response");
        self.response_to_observation(resp)
    }

    fn step(&mut self, action: &Action) -> Observation {
        self.send_command(&EnvCommand {
            command: "step".to_string(),
            action_discrete: action.discrete,
            action_continuous: action.continuous.clone(),
        })
        .expect("Failed to send step command");

        let resp = self.recv_response().expect("Failed to receive step response");
        self.response_to_observation(resp)
    }

    fn close(&mut self) {
        let _ = self.send_command(&EnvCommand {
            command: "close".to_string(),
            action_discrete: None,
            action_continuous: None,
        });
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for SocketEnv {
    fn drop(&mut self) {
        self.close();
    }
}
