/// Quaternion bindings for NumPy (creates a python library)
use crate::QuaternionD as Quat;

use np::convert::ToPyArray;
use numpy as np;
use pyo3::prelude::*;

///
///
/// Python class representing Quaternions
///
/// Quaternions can be used in place of rotation matrices
/// They are more computationally efficient, and can be
/// concatenated many times without risk of losing unitary
/// nature of rotation matrix
///
/// Here, a quaterinon can rotate numpy vector of floating
/// point values via the "*" or "__mul__" operator
///
/// Quaternions can be concatenated together in the same way
///
/// A good description of quaternions is at:
/// https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
///
#[pyclass]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct Quaternion {
    pub inner: Quat,
}

#[pymethods]
impl Quaternion {
    ///
    /// Default identity quaternion:
    /// Rotation about xhat axis by 0 radians
    ///
    #[new]
    fn py_new() -> PyResult<Self> {
        Ok(Quaternion {
            inner: Quat::from_vals(0.0, 0.0, 0.0, 1.0),
        })
    }

    ///
    /// Default identity quaternion:
    /// Rotation about xhat axis by 0 radians
    ///
    #[staticmethod]
    fn identity() -> PyResult<Self> {
        Ok(Quaternion {
            inner: Quat::from_vals(0.0, 0.0, 0.0, 1.0),
        })
    }

    ///
    /// Return quaternion representing rotation about xhat axis
    /// by input number of radians
    ///
    #[staticmethod]
    fn rotx(theta_rad: f64) -> PyResult<Self> {
        Ok(Quaternion {
            inner: Quat::rotx(theta_rad),
        })
    }

    ///
    /// Return quaternion representing rotation about yhat axis
    /// by input number of radians
    ///
    #[staticmethod]
    fn roty(theta_rad: f64) -> PyResult<Self> {
        Ok(Quaternion {
            inner: Quat::roty(theta_rad),
        })
    }

    ///
    /// Return quaternion representing rotation about zhat axis
    /// by input number of radians
    ///
    #[staticmethod]
    fn rotz(theta_rad: f64) -> PyResult<Self> {
        Ok(Quaternion {
            inner: Quat::rotz(theta_rad),
        })
    }

    ///
    /// Quaternion representing rotation about given axis by
    /// given angle in radians
    ///
    /// Inputs:
    ///
    ///    axis:   3-element numpy.float64 numpy array representing axis
    ///
    ///    angle:  Floating point representing angle in radians
    ///
    ///
    #[staticmethod]
    fn from_axis_angle(axis: np::PyReadonlyArray1<f64>, angle: f64) -> PyResult<Self> {
        if axis.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Input axis is not 3 elements",
            ));
        }
        let ax = axis.as_slice().unwrap();

        Ok(Quaternion {
            inner: Quat::from_axis_angle(&[ax[0], ax[1], ax[2]], angle),
        })
    }

    #[staticmethod]
    fn from_dcm(dcm: np::PyReadonlyArray2<f64>) -> PyResult<Self> {
        let dims = dcm.dims();
        if dims[0] != 3 || dims[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err("Input must be 3x3"));
        }
        let raw = dcm.as_slice().unwrap();

        Ok(Quaternion {
            inner: Quat::from_ldcm(&[
                [raw[0], raw[1], raw[2]],
                [raw[3], raw[4], raw[5]],
                [raw[6], raw[7], raw[8]],
            ]),
        })
    }

    /// Return Quaternion that rotates  from input vector v1 to input vector v2
    #[staticmethod]
    fn qv1tov2(v1: np::PyReadonlyArray1<f64>, v2: np::PyReadonlyArray1<f64>) -> PyResult<Self> {
        if v1.len() != 3 || v2.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Input vectors must each have 3 elements",
            ));
        }

        let mut vv1: [f64; 3] = Default::default();
        let mut vv2: [f64; 3] = Default::default();
        vv1.copy_from_slice(v1.as_slice().unwrap());
        vv2.copy_from_slice(v2.as_slice().unwrap());

        Ok(Quaternion {
            inner: Quat::qv1tov2(&vv1, &vv2),
        })
    }

    fn __str__(&self) -> PyResult<String> {
        let ax = self.inner.axis();
        let angle = self.inner.angle();
        Ok(format!(
            "Quaternion(Axis = [{:6.4}, {:6.4}, {:6.4}], Angle = {:6.4} rad)",
            ax[0], ax[1], ax[2], angle
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }

    ///
    /// Angle of rotation of Quaternion, in radians
    ///
    #[getter]
    fn angle(&self) -> PyResult<f64> {
        Ok(self.inner.angle())
    }

    ///
    /// Axis of rotation of Quaternion
    ///
    #[getter]
    fn axis(&self) -> PyResult<PyObject> {
        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            let a = self.inner.axis();
            let pyarr = np::PyArray1::<f64>::from_slice(py, a.as_ref());
            Ok(pyarr.to_object(py))
        })
    }

    /// Direction cosine matrix that when left multiplied
    /// against a 3-element vector produces the same rotation
    /// as the quaternion multiplied by the 3-element vector
    #[getter]
    fn dcm(&self) -> PyResult<PyObject> {
        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            Ok(self.inner.ldcm_ndarr().to_pyarray(py).to_object(py))
        })
    }

    /// Quaternion conjugate
    #[getter]
    fn conj(&self) -> PyResult<Quaternion> {
        Ok(Quaternion {
            inner: self.inner.conjugate(),
        })
    }

    /// Quaternion conjugate
    #[getter]
    fn conjugate(&self) -> PyResult<Quaternion> {
        Ok(Quaternion {
            inner: self.inner.conjugate(),
        })
    }

    ///
    /// Multiply operator
    ///
    /// Possible Right-Hand Side (RHS) values:
    ///
    /// 1. Other quaternion:  
    ///    Return self * other, a quaternion representing
    ///    a concatenation of rotations
    ///
    /// 2. 3-element numpy array
    ///    Return 3-element numpy array represenging rotation
    ///    of input array by the LHS quaternion
    ///
    /// 3. Nx3 2D numpy array
    ///    Return Nx3 2D numpy array where each column is
    ///    rotated by the LHS quaternion
    ///
    /// 4. 3-element list of integers or floats
    ///    Return 3-element list of floating point representing
    ///    rotation of input list by LHS quaternion
    ///
    /// 5. Floating point scalar
    ///    Return LHS quaternion multiplied by scalar
    ///    (generally this is not done...)
    ///
    fn __mul__(&self, other: &PyAny) -> PyResult<PyObject> {
        // Multiply quaternion by quaternion
        if other.is_instance_of::<Quaternion>().unwrap() {
            let q: Quat = Quat::try_from(other)?;
            pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                return Ok(Quaternion {
                    inner: self.inner * q,
                }
                .into_py(py));
            })
        }
        // This incorrectly matches for all PyArray types
        else if other.is_instance_of::<np::PyArray2<f64>>().unwrap() {
            // So, check for 2D condition
            match other.extract::<np::PyReadonlyArray2<f64>>() {
                Ok(v) => {
                    if v.dims()[1] != 3 {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "Invalid rhs. 2nd dimension must be 3 in size",
                        ));
                    }
                    let qmat = self.inner.ldcm_ndarr();

                    pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                        let res = v.as_array().dot(&qmat).to_pyarray(py);
                        Ok(res.into_py(py))
                    })
                }
                // If not, check for 1D condition
                Err(_) => match other.extract::<np::PyReadonlyArray1<f64>>() {
                    Ok(v1) => {
                        if v1.len() != 3 {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "rhs 1D array must have 3 elements",
                            ));
                        }
                        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                            let vout = self.inner * v1.as_array();
                            Ok(vout.to_pyarray(py).into_py(py))
                        })
                    }
                    // Input is incorrect size...
                    Err(_) => {
                        return Err(pyo3::exceptions::PyIndexError::new_err(
                            "RHS must be 1x3 or nx3",
                        ));
                    }
                },
            }
        } else if other.is_instance_of::<pyo3::types::PyFloat>().unwrap() {
            let m = other.extract::<f64>().unwrap();
            return pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                return Ok(Quaternion {
                    inner: self.inner * m,
                }
                .into_py(py));
            });
        } else if other.is_instance_of::<pyo3::types::PyList>().unwrap() {
            let l = other.downcast::<pyo3::types::PyList>().unwrap();
            if l.len() != 3 {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "RHS list must have 3 elements",
                ));
            }
            let mut v = [0.0, 0.0, 0.0];
            v[0] = l.get_item(0).unwrap().extract::<f64>().unwrap();
            v[1] = l.get_item(1).unwrap().extract::<f64>().unwrap();
            v[2] = l.get_item(2).unwrap().extract::<f64>().unwrap();
            let v2 = self.inner * v;
            pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                Ok(pyo3::types::PyList::new(py, v2).into_py(py))
            })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err("Invalid rhs"))
        }
    }
}

#[pymodule]
pub fn qrotate(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Quaternion>()?;
    Ok(())
}

impl TryFrom<&PyAny> for Quat {
    type Error = PyErr;
    fn try_from(p: &PyAny) -> Result<Quat, PyErr> {
        if !p.is_instance_of::<Quaternion>().unwrap() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Input must be Quaternion",
            ));
        } else {
            Ok(p.extract::<Quaternion>().unwrap().inner)
        }
    }
}

impl From<Quat> for Quaternion {
    fn from(q: Quat) -> Quaternion {
        Quaternion { inner: q }
    }
}

impl From<&Quat> for Quaternion {
    fn from(q: &Quat) -> Quaternion {
        Quaternion { inner: q.clone() }
    }
}

impl From<Quaternion> for Quat {
    fn from(q: Quaternion) -> Quat {
        q.inner
    }
}

impl From<&Quaternion> for Quat {
    fn from(q: &Quaternion) -> Quat {
        q.inner
    }
}
