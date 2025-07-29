"""The periodicKDTree class written by Patrick Varilly.

Some minor modifications were done to make it work with Python 3

See https://github.com/patvarilly/periodic_kdtree
"""

import heapq
import itertools

import numpy as np
from scipy.spatial import KDTree


def _gen_relevant_images(x, bounds, distance_upper_bound):
    # Map x onto the canonical unit cell, then produce the relevant
    # mirror images
    real_x = x - np.where(bounds > 0.0, np.floor(x / bounds) * bounds, 0.0)
    m = len(x)

    xs_to_try = [real_x]
    for i in range(m):
        if bounds[i] > 0.0:
            disp = np.zeros(m)
            disp[i] = bounds[i]

            if distance_upper_bound == np.inf:
                xs_to_try = list(
                    itertools.chain.from_iterable(
                        (_ + disp, _, _ - disp) for _ in xs_to_try
                    )
                )
            else:
                extra_xs = []

                # Point near lower boundary, include image on upper side
                if abs(real_x[i]) < distance_upper_bound:
                    extra_xs.extend(_ + disp for _ in xs_to_try)

                # Point near upper boundary, include image on lower side
                if abs(bounds[i] - real_x[i]) < distance_upper_bound:
                    extra_xs.extend(_ - disp for _ in xs_to_try)

                xs_to_try.extend(extra_xs)

    return xs_to_try


def build_matrix_bounds(bounds: np.typing.ArrayLike) -> np.ndarray:
    """Build a matrix representation of the cell bounds.

    Parameters
    ----------
        bounds : np.typing.ArrayLike
            6 values of the cell boundary coordinates and 3 angles between the edges

    Returns
    -------
        cell_matrix : np.ndarray
            A matrix representation of the cell bounds.

    """
    lx = bounds[0]
    ly = bounds[1]
    lz = bounds[2]
    if lx <= 0:
        raise ValueError("Lenght along X axis must be > 0")
    if ly <= 0:
        raise ValueError("Lenght along Y axis must be > 0")
    if lz <= 0:
        raise ValueError("Lenght along Z axis must be > 0")
    alpha = bounds[3]
    beta = bounds[4]
    gamma = bounds[5]

    if not (0 <= alpha <= 180) or not (0 <= beta <= 180) or not (0 <= gamma <= 180):
        raise ValueError("Angles between the edges must be in the range from 0 to 180")

    cell_matrix = np.zeros((3, 3))
    cell_matrix[0, 0] = lx
    if alpha == 90.0:
        cos_alpha = 0.0
    else:
        cos_alpha = np.cos(np.radians(alpha))
    if beta == 90.0:
        cos_beta = 0.0
    else:
        cos_beta = np.cos(np.radians(beta))
    if gamma == 90.0:
        cos_gamma = 0.0
        sin_gamma = 1.0
    else:
        gamma = np.radians(gamma)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)
    cell_matrix[1, 0] = ly * cos_gamma
    cell_matrix[1, 1] = ly * sin_gamma
    cell_matrix[2, 0] = lz * cos_beta
    cell_matrix[2, 1] = lz * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cell_matrix[2, 2] = np.sqrt(
        lz * lz - cell_matrix[2, 0] ** 2 - cell_matrix[2, 1] ** 2
    )

    return cell_matrix


def wrap_into_triclinic(
    x: np.typing.ArrayLike, center: np.typing.ArrayLike, cell_matrix: np.ndarray
) -> np.ndarray:
    """Wrap x into the canonical unit triclinic cell.

    Parameters
    ----------
    x : np.typing.ArrayLike
        An array of points.
    center : np.typing.ArrayLike
        A cell cetner coordinates.
    cell_matrix : np.ndarray
        A matrix representation of the cell bounds.

    Returns
    -------
    x_canonical : np.ndarray
        Coordinates of the x point into the triclinic cell

    """
    center_x = x - center
    box_vectors_matrix = cell_matrix.T
    box_vectors_matrix_inv = np.linalg.inv(box_vectors_matrix)
    r = center_x.T

    f = np.dot(box_vectors_matrix_inv, r)
    g = np.zeros(3)
    for i in range(3):
        g[i] = f.T[i] - np.floor(f.T[i])

    x_canonical_center = np.dot(box_vectors_matrix, g.T)
    x_canonical = x_canonical_center.T + center

    return x_canonical


def gen_relevant_images_for_triclinic_cell(
    x: np.typing.ArrayLike,
    center: np.typing.ArrayLike,
    cell_matrix: np.ndarray,
    distance_upper_bound: float = np.inf,
) -> np.ndarray:
    """Produce the mirror images of x coordinates.

    Parameters
    ----------
    x : np.typing.ArrayLike
        An array of points.
    center : np.typing.ArrayLike
        A cell cetner coordinates.
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

    shiftX = cell_matrix[0]
    shiftY = cell_matrix[1]
    shiftZ = cell_matrix[2]
    end = cell_matrix[0] + cell_matrix[1] + cell_matrix[2]
    # Calculate reciprocal vectors
    reciprocal = np.zeros((3, 3))
    reciprocal[0] = np.cross(cell_matrix[1], cell_matrix[2])
    reciprocal[1] = np.cross(cell_matrix[2], cell_matrix[0])
    reciprocal[2] = np.cross(cell_matrix[0], cell_matrix[1])

    # Normalize
    for i in range(3):
        norm = np.linalg.norm(reciprocal[i])
        if norm > 0:
            reciprocal[i] = reciprocal[i] / norm

    # Map x onto the canonical unit cell
    real_x = wrap_into_triclinic(x, center, cell_matrix)
    xs_to_try = [real_x]

    if distance_upper_bound <= 0:
        raise ValueError("Distance upper bound must be > 0")

    if distance_upper_bound == np.inf:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    mirror_image = real_x + dx * shiftX + dy * shiftY + dz * shiftZ
                    xs_to_try.append(mirror_image)

    else:
        coord = real_x
        other = end - coord

        # identify the condition
        lo_x = np.dot(coord, reciprocal[0]) <= distance_upper_bound
        hi_x = np.dot(other, reciprocal[0]) <= distance_upper_bound
        lo_y = np.dot(coord, reciprocal[1]) <= distance_upper_bound
        hi_y = np.dot(other, reciprocal[1]) <= distance_upper_bound
        lo_z = np.dot(coord, reciprocal[2]) <= distance_upper_bound
        hi_z = np.dot(other, reciprocal[2]) <= distance_upper_bound

        # Calculate mirror images coordinates depending on proximity to the edge
        if lo_x:
            xs_to_try.append(coord + shiftX)

            if lo_y:
                xs_to_try.append(coord + shiftX + shiftY)

                if lo_z:
                    xs_to_try.append(coord + shiftX + shiftY + shiftZ)

                elif hi_z:
                    xs_to_try.append(coord + shiftX + shiftY - shiftZ)

            elif hi_y:
                xs_to_try.append(coord + shiftX - shiftY)

                if lo_z:
                    xs_to_try.append(coord + shiftX - shiftY + shiftZ)

                elif hi_z:
                    xs_to_try.append(coord + shiftX - shiftY - shiftZ)

            if lo_z:
                xs_to_try.append(coord + shiftX + shiftZ)

            elif hi_z:
                xs_to_try.append(coord + shiftX - shiftZ)

        elif hi_x:
            xs_to_try.append(coord - shiftX)

            if lo_y:
                xs_to_try.append(coord - shiftX + shiftY)

                if lo_z:
                    xs_to_try.append(coord - shiftX + shiftY + shiftZ)

                elif hi_z:
                    xs_to_try.append(coord - shiftX + shiftY - shiftZ)

            elif hi_y:
                xs_to_try.append(coord - shiftX - shiftY)

                if lo_z:
                    xs_to_try.append(coord - shiftX - shiftY + shiftZ)

                elif hi_z:
                    xs_to_try.append(coord - shiftX - shiftY - shiftZ)

            if lo_z:
                xs_to_try.append(coord - shiftX + shiftZ)

            elif hi_z:
                xs_to_try.append(coord - shiftX - shiftZ)

        if lo_y:
            xs_to_try.append(coord + shiftY)

            if lo_z:
                xs_to_try.append(coord + shiftY + shiftZ)

            elif hi_z:
                xs_to_try.append(coord + shiftY - shiftZ)

        elif hi_y:
            xs_to_try.append(coord - shiftY)

            if lo_z:
                xs_to_try.append(coord - shiftY + shiftZ)

            elif hi_z:
                xs_to_try.append(coord - shiftY - shiftZ)

        if lo_z:
            xs_to_try.append(coord + shiftZ)

        elif hi_z:
            xs_to_try.append(coord - shiftZ)

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
    bounds : np.typing.ArrayLike
        Lengths along each axis and angles between the edges.
    center : np.typing.ArrayLike
        A cell center coordinates.
    data : np.typing.ArrayLike
        The n data points of dimension m to index before wrapping.
        The array is not copied unless necessary.
    leafsize : int, optional
        The number of points at which the algorithm switches over to brute-force,
        by default 10

    """

    def __init__(
        self,
        bounds: np.typing.ArrayLike,
        center: np.typing.ArrayLike,
        data: np.typing.ArrayLike,
        leafsize: int = 10,
    ) -> None:
        """Initialize PeriodicKDTree.

        Parameters
        ----------
        bounds : np.typing.ArrayLike
            Lengths along each axis and angles between the edges.
        center : np.typing.ArrayLike
            A cell cetner coordinates.
        data : np.typing.ArrayLike
            The n data points of dimension m to index before wrapping.
            The array is not copied unless necessary.
        leafsize : int, optional
            The number of points at which the algorithm switches over to brute-force,
            by default 10

        """
        # Map all points to canonical periodic image
        self.bounds = np.array(bounds)
        self.center = np.array(center)
        self.real_data = np.asarray(data)
        cell_matrix = build_matrix_bounds(self.bounds)
        wrapped_data = wrap_into_triclinic(self.real_data, self.center, cell_matrix)

        # Calculate maximum distance_upper_bound
        self.max_distance_upper_bound = np.min(
            np.where(self.bounds > 0, 0.5 * self.bounds, np.inf)
        )

        # Set up underlying kd-tree
        super().__init__(wrapped_data, leafsize)

    # Ideally, KDTree and cKDTree would expose identical query and __query
    # interfaces.  But they don't, and cKDTree.__query is also inaccessible
    # from Python.  We do our best here to cope.
    def __query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf, workers=1):
        # This is the internal query method, which guarantees that x
        # is a single point, not an array of points
        #
        # A slight complication: k could be "None", which means "return
        # all neighbors within the given distance_upper_bound".

        # Cap distance_upper_bound
        distance_upper_bound = np.min(
            [distance_upper_bound, self.max_distance_upper_bound]
        )

        # Run queries over all relevant images of x
        hits_list = []
        for real_x in _gen_relevant_images(x, self.bounds, distance_upper_bound):
            d, i = super().query(
                real_x, k, eps, p, distance_upper_bound, workers=workers
            )
            if k > 1:
                hits_list.append(list(zip(d, i, strict=True)))
            else:
                hits_list.append([(d, i)])

        # Now merge results
        if k > 1:
            return heapq.nsmallest(k, itertools.chain(*hits_list))
        elif k == 1:
            return [min(itertools.chain(*hits_list))]
        else:
            raise ValueError("Invalid k in periodic_kdtree._KDTree__query")

    def query(
        self,
        x: np.typing.ArrayLike,
        k: int = 1,
        eps: float = 0,
        p: float = 2,
        distance_upper_bound: float = np.inf,
    ):
        r"""Query the kd-tree for nearest neighbors.

        Parameters
        ----------
        x : array_like, last dimension self.m
            An array of points to query.
        k : int or Sequence[int], optional
            Either the number of nearest neighbors to return, or a list of the
            k-th nearest neighbors to return, starting from 1.
        eps : nonnegative float, optional
            Return approximate nearest neighbors; the kth returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real kth nearest neighbor.
        p : float, 1<=p<=infinity, optional
            Which Minkowski p-norm to use.
            1 is the sum-of-absolute-values distance ("Manhattan" distance).
            2 is the usual Euclidean distance.
            infinity is the maximum-coordinate-difference distance.
            A large, finite p may cause a ValueError if overflow can occur.
        distance_upper_bound : nonnegative float, optional
            Return only neighbors within this distance. This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.


        Returns
        -------
        d : float or array of floats
            The distances to the nearest neighbors.
            If ``x`` has shape ``tuple+(self.m,)``, then ``d`` has shape
            ``tuple+(k,)``.
            When k == 1, the last dimension of the output is squeezed.
            Missing neighbors are indicated with infinite distances.
            Hits are sorted by distance (nearest first).

        i : integer or array of integers
            The index of each neighbor in ``self.data``.
            ``i`` is the same shape as d.
            Missing neighbors are indicated with ``self.n``.

        """
        x = np.asarray(x)
        if np.shape(x)[-1] != self.m:
            raise ValueError(
                "x must consist of vectors of length %d "
                "but has shape %s" % (self.m, np.shape(x))
            )
        if p < 1:
            raise ValueError("Only p-norms with 1<=p<=infinity permitted")
        retshape = np.shape(x)[:-1]
        if retshape != ():
            if k > 1:
                dd = np.empty(retshape + (k,), dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape + (k,), dtype=np.int)
                ii.fill(self.n)
            elif k == 1:
                dd = np.empty(retshape, dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape, dtype=np.int)
                ii.fill(self.n)
            else:
                raise ValueError(
                    "Requested %s nearest neighbors; "
                    "acceptable numbers are "
                    "integers greater than or equal "
                    "to one, or None"
                )
            for c in np.ndindex(retshape):
                hits = self.__query(
                    x[c], k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound
                )
                if k > 1:
                    for j in range(len(hits)):
                        dd[c + (j,)], ii[c + (j,)] = hits[j]
                elif k == 1:
                    if len(hits) > 0:
                        dd[c], ii[c] = hits[0]
                    else:
                        dd[c] = np.inf
                        ii[c] = self.n
            return dd, ii
        else:
            hits = self.__query(
                x, k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound
            )
            if k == 1:
                if len(hits) > 0:
                    return hits[0]
                else:
                    return np.inf, self.n
            elif k > 1:
                dd = np.empty(k, dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(k, dtype=np.int)
                ii.fill(self.n)
                for j in range(len(hits)):
                    dd[j], ii[j] = hits[j]
                return dd, ii
            else:
                raise ValueError(
                    "Requested %s nearest neighbors; "
                    "acceptable numbers are integers greater "
                    "than or equal to one, or None"
                )

    # Ideally, KDTree and cKDTree would expose identical __query_ball_point
    # interfaces.  But they don't, and cKDTree.__query_ball_point is also
    # inaccessible from Python.  We do our best here to cope.
    def __query_ball_point(self, x, r, p=2.0, eps=0, workers=1):
        # This is the internal query method, which guarantees that x
        # is a single point, not an array of points

        # Cap r
        r = min(r, self.max_distance_upper_bound)

        # Run queries over all relevant images of x
        results = []
        cell_matrix = build_matrix_bounds(self.bounds)
        for real_x in gen_relevant_images_for_triclinic_cell(
            x, self.center, cell_matrix, r
        ):
            results.extend(super().query_ball_point(real_x, r, p, eps, workers=workers))
        return results

    def query_ball_point(self, x, r, p=2.0, eps=0, workers=1):
        """Find all points within distance r of point(s) x.

        :arg x: array_like, shape tuple + (self.m,)
            The point or points to search for neighbors of.
        :arg r: positive float
            The radius of points to return.
        :arg p: float, optional
            Which Minkowski p-norm to use.  Should be in the range [1, inf].
        :arg eps: nonnegative float, optional
            Approximate search. Branches of the tree are not explored if their
            nearest points are further than ``r / (1 + eps)``, and branches are
            added in bulk if their furthest points are nearer than
            ``r * (1 + eps)``.

        :returns: list or array of lists
            If `x` is a single point, returns a list of the indices of the
            neighbors of `x`. If `x` is an array of points, returns an object
            array of shape tuple containing lists of neighbors.


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

    def query_ball_tree(self, other, r, p=2.0, eps=0):
        """"""
        raise NotImplementedError()

    def query_pairs(self, r, p=2.0, eps=0):
        """"""
        raise NotImplementedError()

    def count_neighbors(self, other, r, p=2.0):
        """"""
        raise NotImplementedError()

    def sparse_distance_matrix(self, other, max_distance, p=2.0):
        """"""
        raise NotImplementedError()
