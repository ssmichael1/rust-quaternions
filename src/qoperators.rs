use crate::{Quaternion, Vec3};
use num_traits::{Float, Zero};
use std::ops::{Add, Mul};

use crate::QuaternionError;

#[inline(always)]
pub(crate) fn raw_qmult<T>(q1: [T; 4], q2: [T; 4]) -> [T; 4]
where
    T: Float,
{
    [
        q1[3] * q2[0] + q2[3] * q1[0] + q1[1] * q2[2] - q2[1] * q1[2],
        q1[3] * q2[1] + q2[3] * q1[1] + q1[2] * q2[0] - q2[2] * q1[0],
        q1[3] * q2[2] + q2[3] * q1[2] + q1[0] * q2[1] - q2[0] * q1[1],
        q1[3] * q2[3] - q2[0] * q1[0] - q1[1] * q2[1] - q2[2] * q1[2],
    ]
}

#[inline(always)]
fn raw_qadd<T>(q1: [T; 4], q2: [T; 4]) -> [T; 4]
where
    T: Float,
{
    [q1[0] + q2[0], q1[1] + q2[1], q1[2] + q2[2], q1[3] + q2[3]]
}

/// Quaternion multiply by quaternion
impl<T> Mul<Quaternion<T>> for Quaternion<T>
where
    T: Float,
{
    type Output = Quaternion<T>;
    #[inline(always)]
    fn mul(self, q2: Quaternion<T>) -> Self::Output {
        Quaternion::<T> {
            raw: raw_qmult(self.raw, q2.raw),
        }
    }
}

// Quaternion multiply by quaternion,  RHS as reference
impl<'a, T> Mul<&'a Quaternion<T>> for Quaternion<T>
where
    T: Float,
{
    type Output = Quaternion<T>;
    #[inline(always)]
    fn mul(self, q2: &'a Quaternion<T>) -> Self::Output {
        Quaternion::<T> {
            raw: raw_qmult(self.raw, q2.raw),
        }
    }
}

/// Quaternion multiply by quaternion, LHS as reference
impl<'a, T> Mul<Quaternion<T>> for &'a Quaternion<T>
where
    T: Float,
{
    type Output = Quaternion<T>;
    #[inline(always)]
    fn mul(self, q2: Quaternion<T>) -> Self::Output {
        Quaternion::<T> {
            raw: raw_qmult(self.raw, q2.raw),
        }
    }
}

/// Quaternion multiply by scalar
impl<T> Mul<T> for Quaternion<T>
where
    T: Float,
{
    type Output = Quaternion<T>;
    #[inline(always)]
    fn mul(self, s: T) -> Self::Output {
        Quaternion::<T> {
            raw: [
                self.raw[0] * s,
                self.raw[1] * s,
                self.raw[2] * s,
                self.raw[3] * s,
            ],
        }
    }
}

/// Reference Quaternion multiply by scalar
impl<'a, T> Mul<T> for &'a Quaternion<T>
where
    T: Float,
{
    type Output = Quaternion<T>;
    #[inline(always)]
    fn mul(self, s: T) -> Self::Output {
        Quaternion::<T> {
            raw: [
                self.raw[0] * s,
                self.raw[1] * s,
                self.raw[2] * s,
                self.raw[3] * s,
            ],
        }
    }
}

/// Quaternion multiply by quaternion
impl<T> Add<Quaternion<T>> for Quaternion<T>
where
    T: Float,
{
    type Output = Quaternion<T>;
    #[inline(always)]
    fn add(self, other: Quaternion<T>) -> Quaternion<T> {
        Quaternion::<T> {
            raw: raw_qadd(self.raw, other.raw),
        }
    }
}

/// Quaternion multiply by quaternion, LHS & RHS as reference
impl<'a, 'b, T> Add<&'b Quaternion<T>> for &'a Quaternion<T>
where
    T: Float,
{
    type Output = Quaternion<T>;

    #[inline(always)]
    fn add(self, other: &'b Quaternion<T>) -> Quaternion<T> {
        Quaternion::<T> {
            raw: raw_qadd(self.raw, other.raw),
        }
    }
}

// Quaternion rotation of 3D vector represented as slice
impl<T> Mul<&[T]> for Quaternion<T>
where
    T: Float,
{
    type Output = [T; 3];
    fn mul(self, rhs: &[T]) -> Self::Output {
        assert!(rhs.len() == 3);
        let qv = Quaternion::<T> {
            raw: [rhs[0], rhs[1], rhs[2], Zero::zero()],
        };
        let r = self * qv * self.conjugate();
        r.vector()
    }
}

// Reference quaternion rotation of 3D vector represented as slice
impl<'a, T> Mul<&[T]> for &'a Quaternion<T>
where
    T: Float,
{
    type Output = [T; 3];
    fn mul(self, rhs: &[T]) -> Self::Output {
        assert!(rhs.len() == 3);
        let qv = Quaternion::<T> {
            raw: [rhs[0], rhs[1], rhs[2], Zero::zero()],
        };
        let r = self * qv * self.conjugate();
        r.vector()
    }
}

// Quaternion rotation of 3D vector represented as std::vec::Vec
impl<T> Mul<std::vec::Vec<T>> for Quaternion<T>
where
    T: Float,
{
    type Output = std::vec::Vec<T>;
    fn mul(self, rhs: std::vec::Vec<T>) -> Self::Output {
        assert!(rhs.len() == 3);
        std::vec::Vec::<T>::from(self * rhs.as_slice())
    }
}

// Reference quaternion rotation of 3D vector represented as std::vec::Vec
impl<'a, T> Mul<std::vec::Vec<T>> for &'a Quaternion<T>
where
    T: Float,
{
    type Output = std::vec::Vec<T>;
    fn mul(self, rhs: std::vec::Vec<T>) -> Self::Output {
        assert!(rhs.len() == 3);
        std::vec::Vec::<T>::from(self * rhs.as_slice())
    }
}

///
/// Rotate 3-element by left-multiplying quaternion
///
/// # Arguments:
///
/// * `rhs` - the [T ; 3] vector that will be rotated
///
///
/// ```
/// // Example: Rotate unit x-axis unit vector about z axis by
/// // pi/2 to get a y axis unit vector
/// use rotations::Quaternion;
///
/// let xhat = [1.0, 0.0, 0.0];
/// let q = Quaternion::<f64>::rotz(std::f64::consts::PI/2.0);
/// let yhat = q * xhat;
/// println!("yhat = {yhat:?}");
/// // yhat should be [0.0, 1.0, 0.0]
/// ```
///
impl<T> Mul<Vec3<T>> for Quaternion<T>
where
    T: Float,
{
    type Output = Vec3<T>;
    fn mul(self, rhs: Vec3<T>) -> Self::Output {
        let qv = Quaternion::<T> {
            raw: [rhs[0], rhs[1], rhs[2], Zero::zero()],
        };
        let q = self * qv * self.conjugate();
        q.vector()
    }
}

/// Rotate reference Vec3 by left-multiplying quaternion
impl<'a, T> Mul<&'a Vec3<T>> for Quaternion<T>
where
    T: Float,
{
    type Output = Vec3<T>;
    fn mul(self, rhs: &'a Vec3<T>) -> Self::Output {
        let qv = Quaternion::<T> {
            raw: [rhs[0], rhs[1], rhs[2], Zero::zero()],
        };
        let q = self * qv * self.conjugate();
        q.vector()
    }
}

/// Rotate Vec3 by left-multiplying quaternion reference
impl<'a, T> Mul<Vec3<T>> for &'a Quaternion<T>
where
    T: Float,
{
    type Output = Vec3<T>;
    fn mul(self, rhs: Vec3<T>) -> Self::Output {
        let qv = Quaternion::<T> {
            raw: [rhs[0], rhs[1], rhs[2], Zero::zero()],
        };
        let q = self * qv * self.conjugate();
        q.vector()
    }
}

/// Rotate Vec3 reference by left-multiplying quaternion reference
impl<'a, 'b, T> Mul<&'b Vec3<T>> for &'a Quaternion<T>
where
    T: Float,
{
    type Output = Vec3<T>;
    fn mul(self, rhs: &'b Vec3<T>) -> Self::Output {
        let qv = Quaternion::<T> {
            raw: [rhs[0], rhs[1], rhs[2], Zero::zero()],
        };
        let q = self * qv * self.conjugate();
        q.vector()
    }
}

// Non-operator rotations
impl<T> Quaternion<T>
where
    T: Float,
{
    /// Rotate slice with error checking
    pub fn rotate_slice(&self, s: &[T]) -> Result<[T; 3], QuaternionError> {
        if s.len() != 3 {
            Err(QuaternionError::new("Slice to rotate must have 3 elements"))
        } else {
            Ok(self * s)
        }
    }

    /// Rotate 3-element array
    pub fn rotate_array(&self, s: &[T; 3]) -> [T; 3] {
        self * s
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused)]
    use super::*;

    #[test]
    fn simple_tests() {
        use crate::QuaternionD;
        let q1 = QuaternionD::identity();
        println!("q1 = {q1}");
        let q2 = q1.clone();
        println!("q1 + q2 = {}", q1 + q2);
        println!("q1 + q2 = {}", q1 + q2);
        let q3 = q1 + q2;
        println!("q3 = {q3}");
        println!("q3 normalized == {}", q3.normalized());
    }

    #[test]
    fn rotate() {
        use crate::{QuaternionD, Vec3D};
        use std::f64::consts::PI;
        let q = QuaternionD::rotz(PI / 2.0);
        println!("q = {q}");
        println!("DCM = {:?}", q.ldcm());
        let dcm = q.ldcm();
        let q2 = QuaternionD::from_ldcm(&dcm);
        println!("q2 = {q2}");

        let v: Vec3D = [1.0, 0.0, 0.0];
        println!("v = {:?}", v);

        let v2 = q * v;
        println!("v2 = {:?}", v2);

        let v3 = q2 * v;
        println!("v3 = {:?}", v3);

        let q = QuaternionD::qv1tov2(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        println!("q xhat to yhat = {q}");
        println!("q * xhat = {:?}", q * [1.0, 0.0, 0.0]);
    }
}
