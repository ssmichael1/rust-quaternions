# <b>qrotate</b>

The ``qrotate`` crate provides Quaternion representations of rotations of 3-element vectors representing points in 3-dimensional space.  3-element vectors can use rust standard library types, or vectors from the ``ndarray`` crate.  

<br><br>


<a href="https://opensource.org/licenses/Apache-2.0">
<img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
</a>
<br>
<br>
Copyright (c) 2023 Steven Michael (ssmichael@gmail.com)


------

## <b>Overview</b>

<br>

Quaternion rotations are represented as multiply (``*``) operations, and  possible on the following repersentations of 3D vectors, where ``T`` is a template parameter matching a floating-point type, ``f32`` or ``f64``.

<br>

|||
| ------------: |:-------------| 
| ``[T; 3]``    | 3-element fixed-size array | 
| ``&[T; 3]``   | Reference to 3-element fixed size array      |   
| ``&[T]`` | Slice containing 3 elements      | 
| ``std::vec::Vec<T>`` | standard library vector with 3 elements |
| ``&std::vec::Vec<T>`` | Reference to standard library vector with 3 elements |
| ``ndarray::Array1<'a, T>`` | 1D array from ``ndarray`` crate (with ``ndarray`` feature enabled) |
| ``ndarray::ArrayView1<'a, T>`` | View into 1D array from ``ndarray`` crate (with ``ndarray`` feature enabled)|

<br>

Quaternions can also be multiplied with other quaternions via the ``*`` operator to represent a concatenation of rotations.

The above operations are also defined to work when used by a reference to a quaternion.

<b>Note</b>: this quaternion represetntation uses the "conventional" definition of a quaternion, which when constructed from an axis and and angle, defines a <it>right-handed</it> rotation of a vector about the given axis by the given angle.

<br><br>

-------

<br>

## <b>Examples</b>

<br>

### <b>Simple Rotation</b>

Rotation of the x-axis unit vector about the z axis by π/2
```
use std::f64::consts::PI;
let xhat = [1.0, 0.0, 0.0];
let yhat = qrotate::Quaternion::<f64>>rotz(PI / 2.0) * xhat
// yhat = [0.0, 1.0, 0.0];
```
<br>

### <b>Concatenated Rotations</b>
Rotation of the x-axis unit vector about z axis by π/2 then y axis by π/3
```
use std::f64::consts::PI;
let xhat = [1.0, 0.0, 0.0];
let q1 = qrotate::Quaternion::<f64>::rotz(PI / 2.0);
let q2 = qrotate::Quaternion::<f64>::roty(PI / 3.0);
let result q2 * q1 * xhat
```
<br>

### <b>Find quaternion to rotate between two vectors</b>
Find quaterion that rotates from xhat vector to yhat vector
```
let xhat = [1.0, 0.0, 0.0];
let yhat = [0.0, 1.0, 0.0];
let q = qrotation::qv1tov2(&xhat, &yhat);
```

<br>

### <b>Direction-Cosine Matrix</b>
Represent quaternion as a direction-cosine matrix  (DCM) that left-multiplies a vector to perform a rotation or a direction-cosine matrix that right-multiples a 2D (Nx3) column matrix to perform a rotation
```
use std::f64::consts::PI;
let q = qrotate::Quaternion::<f64>::rotz(PI / 2.0);

// get [[f64; 3]; 3] representation of quaternion as DCM
// that left-multiples row vector
let dcm = q.ldcm();

// Same thing, excpet DCM as ndarray::Array2<f64> 
let dcm_ndarr = q.ldcm_ndarr();

// Same thing, except DCM as ndarray::Array2<f64>
// that right-multiplies column vector
let dcm_ndarr = q.rdcm_ndarr();
```

Convert Direction-Cosine Matrix (DCM) to quaternion
```
use std::f64::consts::PI;
let q = qrotate::Quaternion::<f64>::rotz(PI / 2.0);

// get [[f64; 3]; 3] representation of quaternion as DCM
// that left-multiples row vector
let dcm = q.ldcm();

// Convert back to quaternion
let q2 = Quaternion::<f64>::from_ldcm(dcm);
```

<br>

### <b> Axis, Angle</b>
Convert quaternion to and from axis and angle of rotation representations
```
// Construct quaternion representing rotation about zhat axis by PI/2
let zhat = [0.0, 1.0, 0.0]
let theta = std::f64::consts::PI / 2.0;
let q = Quaternion::<f64>::from_axis_angle(zhat, theta);
// Result is same as zhat
let axis: [f64; 3] = q.axis();
// result is PI / 2.0
let angle: f64 = q.angle();
```
<br>

### <b> Other Operations</b>
Miscelaneous operations below
```
// Construct identity quaternion (no rotation)
let mut q = Quaternion::<f64>::identity();

// now rotate about y axis
q = Quaternion::<f64>::qroty(0.1) * q;

// Get conjugate
let qc = q.conjugate();

// Get quaternion norm (should be 1.0 most of the time)
let n = q.norm();

// Scale quaternion
q = q * 2.0;

// Get normalized version of quaternion
let qn = q.normalized();

// Normalize quaternion in place
q.normalize();

// Get vector elements of quaternion
let v: [f64; 3] = q.vector();

// Get scalar elements of quaternion
let w: f64 = q.scalar();
```





----
## <b>Python Bindings</b>

By enabling the ``python`` feature, the crate also has the option to compile into a python library. The library makes available a "Quaternion" class which can be used to rotate 3-element NumPy vectors (and each element of a Nx3 NumPy "matrix").  Many of the functions described above are available in python