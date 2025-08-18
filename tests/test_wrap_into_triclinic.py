import numpy as np
import pytest
from numpy.testing import assert_allclose

from ocellar.utils.pkdtree import wrap_into_triclinic

@pytest.fixture(scope="module")
def cubic_cell():
    """10*10*10 orthorhombic cell."""
    return np.diag([10.0, 10.0, 10.0])

@pytest.fixture(scope="module")
def center():
    """Origin at (0, 0, 0)."""
    return np.zeros(3)

# single points
@pytest.mark.parametrize(
    "point, expected",
    [
        pytest.param(np.array([3.2, 5.5, 1.1]),
                     np.array([3.2, 5.5, 1.1]),
                     id="inside"),
        pytest.param(np.array([12.5, -1.0, 25.3]),
                     np.array([2.5,  9.0,  5.3]),
                     id="outside"),
        pytest.param(np.array([10.0, 0.0, 3.0]),
                     np.array([0.0,  0.0,  3.0]),
                     id="on_face"),
    ],
)
def test_single_point(point, expected, center, cubic_cell):
    out = wrap_into_triclinic(point, center, cubic_cell)
    assert_allclose(out, expected, atol=1e-12)

# multiple points
@pytest.mark.parametrize(
    "points",
    [
        pytest.param(
            np.array([
                [3.2, 5.5, 1.1],      # inside
                [12.5, -1.0, 25.3],   # outside
                [10.0, 0.0, 3.0],     # on_face
            ]),
            id="batch_inside_outside_face",
        ),
    ],
)
def test_multiple_points(points, center, cubic_cell):
    # Expected wrap result via modulo 10
    expected = np.mod(points, 10.0)
    out = wrap_into_triclinic(points, center, cubic_cell)
    assert_allclose(out, expected, atol=1e-12)
