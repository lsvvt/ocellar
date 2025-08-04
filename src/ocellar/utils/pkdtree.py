"""The periodicKDTree class initially written by Patrick Varilly.

Modified to work with triclinic cells by Angelica Yakubinskaya and Ilya Ivanov.
See https://github.com/patvarilly/periodic_kdtree.
"""

import numpy as np
from scipy.spatial import KDTree


def cell_matrix_from_bounds(bounds: np.typing.ArrayLike) -> np.ndarray:
    """Build a matrix representation of the cell bounds.

    Parameters
    ----------
    bounds : np.typing.ArrayLike
        3 values of the cell boundary lengths and 3 angles between the edges

    Returns
    -------
    cell_matrix : np.ndarray
        A matrix representation of the cell bounds of shape `(3,3)`,
        where `cell_matrix[0]` corrsponds to first cell vector.

    """
    lx, ly, lz, alpha, beta, gamma = map(float, bounds)

    if (lx <= 0) or (ly <= 0) or (lz <= 0):
        raise ValueError("Lenghts along all axes must be > 0")
    if not (0 < alpha < 180) or not (0 < beta < 180) or not (0 < gamma < 180):
        raise ValueError("Angles between the edges must be in the range from 0 to 180")

    cos_alpha = 0.0 if np.isclose(alpha, 90.0) else np.cos(np.radians(alpha))
    cos_beta = 0.0 if np.isclose(beta, 90.0) else np.cos(np.radians(beta))
    if np.isclose(gamma, 90.0):
        cos_gamma = 0.0
        sin_gamma = 1.0
    else:
        gamma = np.radians(gamma)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)

    cell_matrix = np.zeros((3, 3))
    cell_matrix[0, 0] = lx
    cell_matrix[1, 0] = ly * cos_gamma
    cell_matrix[1, 1] = ly * sin_gamma
    cell_matrix[2, 0] = lz * cos_beta
    cell_matrix[2, 1] = lz * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cell_matrix[2, 2] = np.sqrt(
        lz * lz - cell_matrix[2, 0] ** 2 - cell_matrix[2, 1] ** 2
    )
    return cell_matrix


def wrap_into_triclinic(
    x: np.typing.ArrayLike, cell_center: np.typing.ArrayLike, cell_matrix: np.ndarray
) -> np.ndarray:
    """Wrap x into the canonical unit triclinic cell.

    Parameters
    ----------
    x : np.typing.ArrayLike
        An array of points.
    cell_center : np.typing.ArrayLike
        Cell center coordinates.
    cell_matrix : np.ndarray
        A matrix representation of the cell bounds.

    Returns
    -------
    x_wrapped : np.ndarray
        Coordinates of the x point into the triclinic cell

    """
    x = np.asarray(x)
    cell_col = np.asarray(cell_matrix)  # convert to columns
    cell_inv = np.linalg.inv(cell_col)
    frac = (x - cell_center) @ cell_inv
    frac -= np.floor(frac)
    x_wrapped = frac @ cell_col + cell_center

    return x_wrapped


def _gen_relevant_images_triclinic(
    x: np.typing.ArrayLike,
    cell_center: np.typing.ArrayLike,
    cell_matrix: np.ndarray,
    distance_upper_bound: float = np.inf,
) -> np.ndarray:
    """Produce the mirror images of x coordinates.

    Parameters
    ----------
    x : np.typing.ArrayLike
        An array of points.
    cell_center : np.typing.ArrayLike
        Cell center coordinates.
    cell_matrix : np.ndarray
        A matrix representation of the cell bounds.
    distance_upper_bound : float, optional
        Distance of x mirror images generation. Must be >= 0

    Returns
    -------
    xs_to_try : np.ndarray
        Coordinates of the mirror images.

    """
    # Calculate shifts for each axis
    shift_x = cell_matrix[0]
    shift_y = cell_matrix[1]
    shift_z = cell_matrix[2]
    end = cell_matrix[0] + cell_matrix[1] + cell_matrix[2]

    # Calculate plane norm vectors
    plane_norms = np.zeros((3, 3))
    plane_norms[0] = np.cross(cell_matrix[1], cell_matrix[2])
    plane_norms[1] = np.cross(cell_matrix[2], cell_matrix[0])
    plane_norms[2] = np.cross(cell_matrix[0], cell_matrix[1])

    # Normalize
    for i in range(3):
        norm = np.linalg.norm(plane_norms[i])
        if norm > 0:
            plane_norms[i] = plane_norms[i] / norm

    # Map x into given cell
    real_x = wrap_into_triclinic(x, cell_center, cell_matrix)
    xs_to_try = [real_x]

    if distance_upper_bound <= 0:
        raise ValueError("Distance upper bound must be > 0")

    if distance_upper_bound == np.inf:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    mirror_image = real_x + dx * shift_x + dy * shift_y + dz * shift_z
                    xs_to_try.append(mirror_image)

    else:
        coord = real_x
        other = end - coord

        # identify the condition
        lo_x = np.dot(coord, plane_norms[0]) <= distance_upper_bound
        hi_x = np.dot(other, plane_norms[0]) <= distance_upper_bound
        lo_y = np.dot(coord, plane_norms[1]) <= distance_upper_bound
        hi_y = np.dot(other, plane_norms[1]) <= distance_upper_bound
        lo_z = np.dot(coord, plane_norms[2]) <= distance_upper_bound
        hi_z = np.dot(other, plane_norms[2]) <= distance_upper_bound

        # Calculate mirror images coordinates depending on proximity to the edge
        if lo_x:
            xs_to_try.append(coord + shift_x)

            if lo_y:
                xs_to_try.append(coord + shift_x + shift_y)

                if lo_z:
                    xs_to_try.append(coord + shift_x + shift_y + shift_z)
                elif hi_z:
                    xs_to_try.append(coord + shift_x + shift_y - shift_z)

            elif hi_y:
                xs_to_try.append(coord + shift_x - shift_y)

                if lo_z:
                    xs_to_try.append(coord + shift_x - shift_y + shift_z)
                elif hi_z:
                    xs_to_try.append(coord + shift_x - shift_y - shift_z)

            if lo_z:
                xs_to_try.append(coord + shift_x + shift_z)

            elif hi_z:
                xs_to_try.append(coord + shift_x - shift_z)

        elif hi_x:
            xs_to_try.append(coord - shift_x)

            if lo_y:
                xs_to_try.append(coord - shift_x + shift_y)

                if lo_z:
                    xs_to_try.append(coord - shift_x + shift_y + shift_z)
                elif hi_z:
                    xs_to_try.append(coord - shift_x + shift_y - shift_z)

            elif hi_y:
                xs_to_try.append(coord - shift_x - shift_y)

                if lo_z:
                    xs_to_try.append(coord - shift_x - shift_y + shift_z)
                elif hi_z:
                    xs_to_try.append(coord - shift_x - shift_y - shift_z)

            if lo_z:
                xs_to_try.append(coord - shift_x + shift_z)

            elif hi_z:
                xs_to_try.append(coord - shift_x - shift_z)

        if lo_y:
            xs_to_try.append(coord + shift_y)

            if lo_z:
                xs_to_try.append(coord + shift_y + shift_z)
            elif hi_z:
                xs_to_try.append(coord + shift_y - shift_z)

        elif hi_y:
            xs_to_try.append(coord - shift_y)

            if lo_z:
                xs_to_try.append(coord - shift_y + shift_z)
            elif hi_z:
                xs_to_try.append(coord - shift_y - shift_z)

        if lo_z:
            xs_to_try.append(coord + shift_z)

        elif hi_z:
            xs_to_try.append(coord - shift_z)

    return xs_to_try


class PeriodicKDTree(KDTree):
    """Cython kd-tree for quick nearest-neighbor lookup with periodic boundaries.

    See :class:`scipy.spatial.KDTree` for details on kd-trees.

    Searches with periodic boundaries are implemented by mapping all
    initial data points to one canonical periodic image, building an
    ordinary kd-tree with these points, then querying this kd-tree multiple
    times, if necessary, with all the relevant periodic images of the
    query point.

    Note that to ensure that no two distinct images of the same point
    appear in the results, it is essential to restrict the maximum
    distance between a query point and a data point to half the smallest
    box dimension.

    Attributes
    ----------
    cell_bounds : np.typing.ArrayLike
        Lengths along each axis and angles between the edges.
    cell_center : np.typing.ArrayLike
        Cell center coordinates.
    data : np.typing.ArrayLike
        The n data points of dimension m to index before wrapping.
        The array is not copied unless necessary.
    leafsize : int, optional
        The number of points at which the algorithm switches over to brute-force,
        by default 10

    """

    def __init__(
        self,
        cell_bounds: np.typing.ArrayLike,
        cell_center: np.typing.ArrayLike,
        data: np.typing.ArrayLike,
        leafsize: int = 10,
    ) -> None:
        """Initialize PeriodicKDTree.

        Parameters
        ----------
        cell_bounds : np.typing.ArrayLike
            Lengths along each axis and angles between the edges.
        cell_center : np.typing.ArrayLike
            Cell center coordinates.
        data : np.typing.ArrayLike
            The n data points of dimension m to index before wrapping.
            The array is not copied unless necessary.
        leafsize : int, optional
            The number of points at which the algorithm switches over to brute-force,
            by default 10

        """
        self.cell_bounds = np.asarray(cell_bounds)
        self.cell_center = np.asarray(cell_center)
        self._data = np.asarray(data)
        cell_matrix = cell_matrix_from_bounds(self.cell_bounds)
        # Map all points to canonical periodic image
        wrapped_data = wrap_into_triclinic(self._data, self.cell_center, cell_matrix)

        # Calculate maximum distance_upper_bound
        self.max_distance_upper_bound = np.min(
            np.where(self.cell_bounds > 0, 0.5 * self.cell_bounds, np.inf)
        )

        # Set up underlying kd-tree
        super().__init__(wrapped_data, leafsize)

    def __query_ball_point(
        self,
        x: np.typing.ArrayLike,
        r: float,
        p: float = 2.0,
        eps: float = 0.0,
        workers: int = 1,
    ) -> list[int]:
        """Internal query method, which guarantees that x
        is a single point, not an array of points.
        """  # noqa: D205, D401
        # Cap r
        r = min(r, self.max_distance_upper_bound)

        # Run queries over all relevant images of x
        results = []
        cell_matrix = cell_matrix_from_bounds(self.cell_bounds)
        for real_x in _gen_relevant_images_triclinic(
            x, self.cell_center, cell_matrix, r
        ):
            results.extend(super().query_ball_point(real_x, r, p, eps, workers=workers))
        return results

    def query_ball_point(
        self,
        x: np.typing.ArrayLike,
        r: float,
        p: float = 2.0,
        eps: float = 0.0,
        workers: int = 1,
    ) -> list[int] | np.ndarray:
        """Find all points within distance r of point(s) x.

        Parameters
        ----------
        x : ArrayLike, shape (..., m)
            The point or points to search for neighbours of; last dimension
            must match dimensions of bounds.
        r : float
            Positive radius within which to return points.
        p : float, optional
            Minkowski norm p, in range [1, ∞].  Defaults to 2.0 (Euclidean).
        eps : float, optional
            Approximate search tolerance: branches are pruned if nearer than
            r/(1 + eps) or added if further than r*(1 + eps).
        workers : int, optional
            The number of parallel workers to use. Defaults to 1.

        Returns
        -------
        neighbours : list of int or ndarray of lists
            If `x` is a single point (shape (m,)), returns a list of neighbour
            indices. If `x` has shape (..., m), returns an object array of
            shape `x.shape[:-1]` where each element is a list of neighbour
            indices.

        """
        x = np.asarray(x).astype(np.float32)
        if x.shape[-1] != self.m:
            raise ValueError(
                "Searching for a %d-dimensional point in a "
                "%d-dimensional KDTree" % (x.shape[-1], self.m)
            )
        if len(x.shape) == 1:
            return self.__query_ball_point(x, r, p, eps, workers)
        else:
            retshape = x.shape[:-1]
            result = np.empty(retshape, dtype=np.object)
            for c in np.ndindex(retshape):
                result[c] = self.__query_ball_point(x[c], r, p, eps, workers)
            return result
