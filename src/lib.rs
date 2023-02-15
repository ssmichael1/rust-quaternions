///
/// Define rotations
///
mod quaternion;

mod qoperators;

#[cfg(any(feature = "ndarray", feature = "python"))]
mod qndarray;

#[cfg(feature = "python")]
pub mod qnumpy;

pub use quaternion::Quaternion;
pub use quaternion::Vec3;
pub use quaternion::DCM;
pub use quaternion::{QuaternionD, QuaternionS};
pub use quaternion::{Vec3D, Vec3S};
