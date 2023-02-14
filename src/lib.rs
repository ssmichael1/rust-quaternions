#[cfg(feature = "ndarrbindings")]
mod qndarray;
mod qoperators;
///
/// Define rotations
///
mod quaternion;

pub use quaternion::Quaternion;
pub use quaternion::Vec3;
pub use quaternion::DCM;
pub use quaternion::{QuaternionD, QuaternionS};
pub use quaternion::{Vec3D, Vec3S};
