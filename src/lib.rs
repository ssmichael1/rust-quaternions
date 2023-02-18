///
/// The qrotate crate provides Quaternion representations of rotations
/// of 3-element vectors representing points in 3-dimensional space.
/// 3-element vectors can use rust standard library types,
/// or vectors from the ndarray crate.
///
///
mod quaternion;

mod qoperators;

mod quaternionerror;

#[cfg(any(feature = "ndarray", feature = "python"))]
pub mod qndarray;

#[cfg(feature = "python")]
pub mod qnumpy;

pub use quaternionerror::QuaternionError;
pub use quaternionerror::QuaternionResult;

pub use quaternion::Quaternion;
pub use quaternion::Vec3;
pub use quaternion::DCM;
pub use quaternion::{QuaternionD, QuaternionS};
pub use quaternion::{Vec3D, Vec3S};
