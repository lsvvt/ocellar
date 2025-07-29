import pytest
import numpy as np
from ocellar.utils import pkdtree

test_bounds = [
    np.array([0, 4, 0, 4, 0, 5, 90, 90, 120]),
    np.array([0, 10, 0, 10, 0, 10, 90, 90, 90]),
    np.array([0, 10, 0, 10, 0, 10, 80, 85, 75]),
    np.array([0, 10, 0, 10, 0, 10, 80, 85, 75]),
    np.array([0, 4, 0, 5, 0, 4, 90, 120, 90]),
    np.array([0, 4, 0, 5, 0, 4, 90, 120, 90]),
    np.array([0, 4, 0, 5, 0, 4, 90, 120, 90])
]

test_x = [
    np.array([1.0, 1.5, 2.0]),
    np.array([1, 4, 5]),
    np.array([2, 3, 1]),
    np.array([2, 3, 1]),
    np.array([1, 4, 3]),
    np.array([1, 4, 3]),
    np.array([1, 4, 3])
]

test_distance_upper_bound = [5, 3, 5, 5, 5, 4, 2]

@pytest.mark.parametrize("bounds, x, distance_upper_bound", test_bounds, test_x, test_distance_upper_bound)
def test_duplicate(bounds, x, distance_upper_bound):
    bounds_matrix = pkdtree.cell_matrix_from_bounds(bounds)
    real_x = pkdtree.map_x_onto_canonical_unit_cell(x, bounds, bounds_matrix)
    mirror_images = pkdtree._gen_relevant_images_triclinic(real_x, bounds_matrix, distance_upper_bound)

    for mirror_image in mirror_images:
        assert eval(mirror_image) == x