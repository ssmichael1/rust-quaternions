use num_traits::{Float, One, Zero};

use std::fmt;

/// Quaternion to represent rotations of 3D vectors
#[derive(Copy, Clone)]
pub struct Quaternion<T>
where
    T: Float,
{
    pub raw: [T; 4],
}

/// 3D vector
pub type Vec3<T> = [T; 3];

/// Direction Cosine Matrix (DCM)
pub type DCM<T> = [[T; 3]; 3];

/// Double-precision quaternion
pub type QuaternionD = Quaternion<f64>;

/// Single-precision quaternion
pub type QuaternionS = Quaternion<f32>;

// Double-precision 3D vector
pub type Vec3D = Vec3<f64>;

// Single-precision 3D vector
pub type Vec3S = Vec3<f32>;

impl<T> Quaternion<T>
where
    T: Float,
{
    /// Return identity quaternion
    pub fn new() -> Quaternion<T> {
        Quaternion::<T> {
            raw: [Zero::zero(), Zero::zero(), Zero::zero(), One::one()],
        }
    }

    /// Quaternion from input axis & angle
    pub fn from_axis_angle(axis: Vec3<T>, angle: T) -> Quaternion<T> {
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if n == Zero::zero() {
            return Quaternion::<T>::identity();
        }
        let halfangle = angle.mul(T::from(0.5).unwrap());
        let sinhalfangle = halfangle.sin();
        let coshalfangle = halfangle.cos();
        Quaternion::<T> {
            raw: [
                sinhalfangle * axis[0] / n,
                sinhalfangle * axis[1] / n,
                sinhalfangle * axis[2] / n,
                coshalfangle,
            ],
        }
    }

    /// Quaternion from input Direction-Cosine Matrix (DCM)
    /// The DCM left multiplies a 3x1 row vector to produce a rotation
    /// The resulting quaternion multipled by a 3-element vector
    /// will have the same output
    pub fn from_ldcm(m: &DCM<T>) -> Quaternion<T> {
        Quaternion::<T>::from_rdcm(m).conjugate()
    }

    /// Quaternion from input Direction-Cosine Matrix (DCM)
    /// The DCM right multiplies a 1x3 column vector to produce a rotation
    /// The resulting quaternion multipled by a 3-element vector
    /// will have the same output
    pub fn from_rdcm(m: &DCM<T>) -> Quaternion<T> {
        // https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
        let t = m[0][0] + m[1][1] + m[2][2];
        let one: T = T::one();
        let onehalf: T = T::from(0.5).unwrap();
        let onequarter: T = T::from(0.25).unwrap();

        if t > Zero::zero() {
            let k = onehalf / (one + t).sqrt();
            Quaternion::<T> {
                raw: [
                    k * (m[1][2] - m[2][1]),
                    k * (m[2][0] - m[0][2]),
                    k * (m[0][1] - m[1][0]),
                    onequarter / k,
                ],
            }
        } else if (m[0][0] > m[1][1]) && (m[0][0] > m[2][2]) {
            let k = onehalf / (one + m[0][0] - m[1][1] - m[2][2]).sqrt();
            Quaternion::<T> {
                raw: [
                    onequarter / k,
                    k * (m[1][0] + m[0][1]),
                    k * (m[2][0] + m[0][2]),
                    k * (m[1][2] - m[2][1]),
                ],
            }
        } else if m[1][1] > m[2][2] {
            let k = onehalf / (one + m[1][1] - m[0][0] - m[2][2]).sqrt();
            Quaternion::<T> {
                raw: [
                    k * (m[1][0] + m[0][1]),
                    onequarter / k,
                    k * (m[2][1] + m[1][2]),
                    k * (m[2][0] - m[0][2]),
                ],
            }
        } else {
            let k = onehalf / (one + m[2][2] - m[0][0] - m[1][1]).sqrt();
            Quaternion::<T> {
                raw: [
                    k * (m[2][0] + m[0][2]),
                    k * (m[2][1] + m[1][2]),
                    onequarter / k,
                    k * (m[0][1] - m[1][0]),
                ],
            }
        }
    }

    /// Return conjugate of quaternion
    #[inline(always)]
    pub fn conjugate(&self) -> Quaternion<T> {
        Quaternion::<T> {
            raw: [-self.raw[0], -self.raw[1], -self.raw[2], self.raw[3]],
        }
    }

    /// Shortened syntax for quaternion conjugate
    /// equivalent to self.conjugate()
    #[inline(always)]
    pub fn conj(&self) -> Quaternion<T> {
        self.conjugate()
    }

    /// Return Quaternion that will rotate vector "v1" to vector "v2"
    pub fn qv1tov2(v1: &Vec3<T>, v2: &Vec3<T>) -> Quaternion<T> {
        // Cross product gives axis of rotation
        let vc: Vec3<T> = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ];
        // Dot product is cosine of angle between vectors
        let vd: T = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
        // norm of cross product is sin of angle between vectors
        let cn = (vc[0] * vc[0] + vc[1] * vc[1] + vc[2] * vc[2]).sqrt();
        // Return identity if cross product magnitude is zero
        // (vectors are parallel)
        if cn == Zero::zero() {
            return Quaternion::<T>::identity();
        }
        let theta = (cn / vd).atan();
        let thetaover2 = T::from(0.5).unwrap() * theta;
        let ct = thetaover2.cos();
        let st = thetaover2.sin();
        Quaternion::<T> {
            raw: [vc[0] / cn * st, vc[1] / cn * st, vc[2] / cn * st, ct],
        }
    }

    /// Return vector component of quaternion
    pub fn vector(&self) -> Vec3<T> {
        [self.raw[0], self.raw[1], self.raw[2]]
    }

    /// Return scalar component of quaternion
    pub fn scalar(&self) -> T {
        self.raw[3]
    }

    /// Quaternion with every element as zero
    pub fn zero() -> Quaternion<T> {
        Quaternion::<T> {
            raw: [Zero::zero(), Zero::zero(), Zero::zero(), Zero::zero()],
        }
    }

    /// Quaternion identity (unit rotation)
    #[inline(always)]
    pub fn identity() -> Quaternion<T> {
        Quaternion::new()
    }

    /// Quaternion Norm
    #[inline(always)]
    pub fn norm(&self) -> T {
        (self.raw[0] * self.raw[0]
            + self.raw[1] * self.raw[1]
            + self.raw[2] * self.raw[2]
            + self.raw[3] * self.raw[3])
            .sqrt()
    }

    /// Normalize quaternion in-place
    #[inline(always)]
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n != Zero::zero() {
            self.raw[0] = self.raw[0] / n;
            self.raw[1] = self.raw[1] / n;
            self.raw[2] = self.raw[2] / n;
            self.raw[3] = self.raw[3] / n;
        }
    }

    /// Return normalized verseion of Quaternion
    /// or if norm is zero, return "zero" quaternion
    #[inline(always)]
    pub fn normalized(&self) -> Quaternion<T> {
        let n = self.norm();
        if n != Zero::zero() {
            Quaternion::<T> {
                raw: [
                    self.raw[0] / n,
                    self.raw[1] / n,
                    self.raw[2] / n,
                    self.raw[3] / n,
                ],
            }
        } else {
            Quaternion::<T>::zero()
        }
    }

    /// Quaternion representing right-handed rotation of vector
    /// about x axis by input radians
    #[inline(always)]
    pub fn rotx(angle: T) -> Quaternion<T>
    where
        T: Float,
    {
        let halfangle = angle.mul(T::from(0.5).unwrap());
        Quaternion::<T> {
            raw: [halfangle.sin(), Zero::zero(), Zero::zero(), halfangle.cos()],
        }
    }

    /// Quaternion representing right-handed rotation of vector
    /// about y axis by input radians
    #[inline(always)]
    pub fn roty(angle: T) -> Quaternion<T>
    where
        T: Float,
    {
        let halfangle = angle.mul(T::from(0.5).unwrap());
        Quaternion::<T> {
            raw: [Zero::zero(), halfangle.sin(), Zero::zero(), halfangle.cos()],
        }
    }

    /// Quaternion derivative given input angle rates in rad/s
    pub fn qdot(&self, omega: Vec3<T>) -> Quaternion<T> {
        let qomega = Quaternion::<T> {
            raw: [omega[0], omega[1], omega[2], Zero::zero()],
        };

        // qd = -0.5 * qomega * q
        // Then normalize by adding (1.0 - qd.norm())*qd
        // -0.5 * qdot + (1.0 - qdot.norm()) * qdot
        let qd = qomega * self;
        qd * (T::from(0.5).unwrap() - qd.norm())
    }

    /// Quaternion representing right-handed rotation of vector
    /// about z axis by input radians
    #[inline(always)]
    pub fn rotz(angle: T) -> Quaternion<T>
    where
        T: Float,
    {
        let halfangle = angle.mul(T::from(0.5).unwrap());
        Quaternion::<T> {
            raw: [Zero::zero(), Zero::zero(), halfangle.sin(), halfangle.cos()],
        }
    }

    /// Return normalized axis of rotation of quaternion
    #[inline(always)]
    pub fn axis(&self) -> Vec3<T> {
        let mut v: Vec3<T> = [self.raw[0], self.raw[1], self.raw[2]];
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if n != Zero::zero() {
            v[0] = v[0] / n;
            v[1] = v[1] / n;
            v[2] = v[2] / n;
            v
        } else {
            [One::one(), Zero::zero(), Zero::zero()]
        }
    }

    /// Return angle of rotation of quaternion, in radians
    #[inline(always)]
    pub fn angle(&self) -> T {
        if self.raw[3].abs() < One::one() {
            let mut ang = self.raw[3].acos() * T::from(2.0).unwrap();
            if ang > T::from(std::f64::consts::PI).unwrap() {
                ang = T::from(std::f64::consts::PI * 2.0).unwrap() - ang;
            }
            ang
        } else {
            Zero::zero()
        }
    }

    // Return axis and angle of rotation of quaternion
    #[inline(always)]
    pub fn axis_angle(&self) -> (Vec3<T>, T) {
        (self.axis(), self.angle())
    }

    /// Represent quaternion as Direction-Cosine Matrix (DCM)
    /// The DCM, when left-multiplied against a 3x1 row
    /// matrix, will match the rotation of the quaternion
    ///
    /// For Example, the following are equivalent
    /// (uses ndarray feature with ldcm_ndarr)
    ///
    /// ```
    /// use rotations::QuaternionD;
    /// use ndarray::array;
    /// let xhat = array![1.0, 0.0, 0.0];
    /// let qz = QuaternionD::rotz(std::f64::consts::PI/2.0);
    ///
    /// // The following 3 calculations yield the same result
    /// // Quaternion rotation
    /// let yhat_q = qz * xhat.clone();
    /// // Quaternion to matrix, them left multiply
    /// let yhat_ldcm = qz.ldcm_ndarr().dot(&xhat);
    /// // Quaternion to matrix, then right  multiply
    /// let yhat_rdcm = xhat.dot(&qz.ldcm_ndarr());
    ///```
    ///
    ///
    #[inline(always)]
    pub fn ldcm(&self) -> DCM<T> {
        let r = &self.raw;
        let one: T = One::one();
        let two: T = T::from(2.0).unwrap();

        [
            [
                one - two * (r[1] * r[1] + r[2] * r[2]),
                two * (r[0] * r[1] - r[2] * r[3]),
                two * (r[0] * r[2] + r[1] * r[3]),
            ],
            [
                two * (r[0] * r[1] + r[2] * r[3]),
                one - two * (r[0] * r[0] + r[2] * r[2]),
                two * (r[1] * r[2] - r[0] * r[3]),
            ],
            [
                two * (r[0] * r[2] - r[1] * r[3]),
                two * (r[1] * r[2] + r[0] * r[3]),
                one - two * (r[0] * r[0] + r[1] * r[1]),
            ],
        ]
    }

    /// Represent quaternion as Direction-Cosine Matrix (DCM)
    /// The DCM, when right-multiplied against a 1x3 column
    /// matrix, will match the rotation of the quaternion
    #[inline(always)]
    pub fn rdcm(&self) -> DCM<T> {
        self.conjugate().ldcm()
    }
}

impl<T> fmt::Display for Quaternion<T>
where
    T: Float,
{
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (ax, angle) = self.axis_angle();
        write!(
            f,
            "Quaternion: Axis = [{:.3}, {:.3}, {:.3}], Angle = {:.3}] rad",
            ax[0].to_f64().unwrap(),
            ax[1].to_f64().unwrap(),
            ax[2].to_f64().unwrap(),
            angle.to_f64().unwrap()
        )
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused)]
    use super::*;

    #[test]
    fn q2dcm2q() {
        // Create random quaternion

        let mut idx = 0;
        while idx < 1000 {
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

            // Convert to direction cosine matrix
            let dcm = q.ldcm();
            // Convert back to quaternion
            let q2 = QuaternionD::from_ldcm(&dcm);
            // Check angle beteween quaternions
            let q0 = q * q2.conjugate();
            assert!(q0.angle().abs() < 1.0e-7);

            idx = idx + 1;
        }
    }
}
