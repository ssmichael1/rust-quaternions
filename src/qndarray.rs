use ndarray::{Array1, ArrayView1};
use num_traits::Float;
use std::ops::Mul;

use crate::Quaternion;

/// Quaternion multiply by Array1.
/// Array1 must have 3 elements or function will panic
///
/// Returns new Array1 representing rotation of input Array1
/// by quaternion
///
/// Example:
/// ```
/// use rotations::QuaternionD;
/// use ndarray::array;
///
/// let xhat = array![1.0, 0.0, 0.0];
/// let q = QuaternionD::rotx(std::f64::consts::PI / 2.0);
/// let yhat = q * xhat;
/// // yhat should be [0.0, 1.0, 0.0]
/// ```
///
impl<T> Mul<Array1<T>> for Quaternion<T>
where
    T: Float,
{
    type Output = Array1<T>;
    #[inline(always)]
    fn mul(self, rhs: Array1<T>) -> Self::Output {
        assert!(rhs.len() == 3);
        let t = self * rhs.as_slice().unwrap();
        let av = ArrayView1::<T>::from(t.as_ref());
        av.to_owned()
    }
}

/// Reference quaternion multiply by Array1
impl<'a, T> Mul<Array1<T>> for &'a Quaternion<T>
where
    T: Float,
{
    type Output = Array1<T>;
    #[inline(always)]
    fn mul(self, rhs: Array1<T>) -> Self::Output {
        assert!(rhs.len() == 3);
        let t = self * rhs.as_slice().unwrap();
        let av = ArrayView1::<T>::from(t.as_ref());
        av.to_owned()
    }
}

/// Quaternion multiply by ArrayView1, a read-only slice of array
impl<'a, T> Mul<ArrayView1<'a, T>> for Quaternion<T>
where
    T: Float,
{
    type Output = Array1<T>;
    #[inline(always)]
    fn mul(self, rhs: ArrayView1<'a, T>) -> Self::Output {
        assert!(rhs.len() == 3);
        let t = self * rhs.as_slice().unwrap();
        let av = ArrayView1::<T>::from(t.as_ref());
        av.to_owned()
    }
}

/// Reference quaternion multiply by ArrayView1, a read-only slice of array
impl<'a, 'b, T> Mul<ArrayView1<'a, T>> for &'b Quaternion<T>
where
    T: Float,
{
    type Output = Array1<T>;
    #[inline(always)]
    fn mul(self, rhs: ArrayView1<'a, T>) -> Self::Output {
        assert!(rhs.len() == 3);
        let t = self * rhs.as_slice().unwrap();
        let av = ArrayView1::<T>::from(t.as_ref());
        av.to_owned()
    }
}

impl<T> Quaternion<T>
where
    T: Float,
{
    /// Return Direction-Cosine-Matric (DCM) which when
    /// left multiplied by 3x1 row-vector matches rotation of
    /// input quaternion.  Return type is Array2 from
    /// ndarray crate
    pub fn ldcm_ndarr(&self) -> ndarray::Array2<T> {
        ndarray::arr2(self.ldcm().as_slice())
    }

    /// Return Direction-Cosine-Matric (DCM) which when
    /// right multiplied by 1x3 column-vector matches rotation of
    /// input quaternion.  Return type is Array2 from
    /// ndarray crate
    pub fn rdcm_ndarr(&self) -> ndarray::Array2<T> {
        ndarray::arr2(self.rdcm().as_slice())
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused)]
    use super::*;
    use crate::quaternion::*;
    use ndarray::array;

    #[test]
    // Show that quaternion rotations are equivalent to DCM rotations
    fn rotation_tests() {
        let mut idx = 0;
        while idx < 1000 {
            // Create a random 3d array
            let arr = array![
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5
            ];

            // Create a random quaternion
            let mut q = QuaternionD::from_axis_angle(
                [
                    rand::random::<f64>() - 0.5,
                    rand::random::<f64>() - 0.5,
                    rand::random::<f64>() - 0.5,
                ],
                rand::random::<f64>() * std::f64::consts::PI,
            );
            q.normalize();

            // Rotate with quaternion, and left, and right multiply of
            // direction cosine matrix
            let r1 = q * arr.clone();
            let r2 = q.ldcm_ndarr().dot(&arr);
            let r3 = arr.dot(&q.rdcm_ndarr());

            // Verify that results are the same
            (r1.clone() - r2)
                .iter()
                .for_each(|x| assert!(x.abs() < 1.0e-8));
            (r1.clone() - r3)
                .iter()
                .for_each(|x| assert!(x.abs() < 1.0e-8));

            idx = idx + 1;
        }
    }
}
