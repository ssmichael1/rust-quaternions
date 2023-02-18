from qrotate import Quaternion
import math as m
import numpy as np

import pytest


class TestQuaternion(object):

    # Simple initial test
    def test_rotz(self):
        # Initial rotation
        xhat = [1.0, 0.0, 0.0]
        print(f"xhat = {xhat}")
        qz = Quaternion.rotz(m.pi / 2.0)
        yhat = qz * xhat
        assert yhat[0] == pytest.approx(0.0)
        assert yhat[1] == pytest.approx(1.0)
        assert yhat[2] == pytest.approx(0.0)

    # Compare quaternion rotation with
    # rotation by equivalent DCM
    def test_dcm(self):
        for idx in range(1000):
            testvec = np.random.rand(3)
            axis = np.random.rand(
                3,
            )
            # Create quaternion from random axis and angle
            axis = axis / np.sqrt(np.sum(axis**2))
            angle = np.random.rand(1) * m.pi
            q = Quaternion.from_axis_angle(axis, angle)

            # Rotate by test vector
            r1 = q * testvec
            # Compare against equivalent dcm
            r2 = q.dcm @ testvec.reshape((3, 1))

            # Convert from DCM to quaternion
            q2 = Quaternion.from_dcm(q.dcm)
            r3 = q2 * testvec

            # Verify results are equivalent
            for idx in range(3):
                assert r1[idx] == pytest.approx(r2[idx])
                assert r1[idx] == pytest.approx(r3[idx])
