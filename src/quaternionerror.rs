use std::fmt;

#[derive(Debug)]
pub struct QuaternionError {
    msg: String,
}

impl fmt::Display for QuaternionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quaternion Operation Error: {}", self.msg)
    }
}

impl std::error::Error for QuaternionError {
    fn description(&self) -> &str {
        self.msg.as_str()
    }
}

impl QuaternionError {
    pub fn new(msg: &str) -> QuaternionError {
        QuaternionError {
            msg: String::from(msg),
        }
    }
}
