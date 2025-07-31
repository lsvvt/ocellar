"""Main ocellar package module, containing Molecule class."""

import warnings
from pathlib import Path

import networkx
import numpy as np
from scipy.spatial import KDTree

from ocellar import io
from ocellar.utils.pkdtree import (
    PeriodicKDTree,
    cell_matrix_from_bounds,
    wrap_into_triclinic,
)


class Molecule:
    """A class to represent a molecule and its properties.

    Attributes
    ----------
    input_geometry : str or None
        Path to the input geometry file.
    element_types : list[str] or None
        Element symbols corresponding to atom types in certain file formats
        (CFG, LAMMPS dump). Required when atoms are specified by numeric
        type identifiers rather than element symbols.
    cell_center : np.typing.ArrayLike or None
        List of cell center coordinates in format `[x_0, y_0, z_0]`.
    cell_bounds : np.typing.ArrayLike or None
        List of cell parameters in format `[L_x, L_y, L_z, alpha, beta, gamma]`.
    geometry : tuple[list, np.ndarray] or None
        A tuple containing:
        - list: A list of element symbols.
        - np.ndarray: An array of atomic coordinates.
    graph : networkx.Graph or None
        Graph representation of the molecular structure.
    subgraphs : list or None
        list of connected components in the molecular graph.
    unwrapped: bool
        True if Molecule was unwrapped by :py:meth:`Molecule.unwrap` method

    """

    def __init__(
        self,
        input_geometry: str | Path | None = None,
        element_types: np.typing.ArrayLike | None = None,
        cell_center: np.typing.ArrayLike | None = None,
        cell_bounds: np.typing.ArrayLike | None = None,
    ) -> None:
        """Initialize the Molecule object.

        Parameters
        ----------
        input_geometry : str or Path or None
            Path to the input geometry file.
        element_types : np.typing.ArrayLike or None
            Element symbols corresponding to atom types in certain file formats
            (CFG, LAMMPS dump). Required when atoms are specified by numeric
            type identifiers rather than element symbols.
        cell_center : np.typing.ArrayLike or None
            List of cell center coordinates in format `[x_0, y_0, z_0]`.
        cell_bounds : np.typing.ArrayLike or None
            List of cell parameters in format `[L_x, L_y, L_z, alpha, beta, gamma]`.

        """
        self.input_geometry = input_geometry
        self.element_types = element_types
        if cell_bounds is not None:
            self.cell_bounds = np.asarray(cell_bounds, dtype=float)
            self.cell_matrix = cell_matrix_from_bounds(self.cell_bounds)
        self.cell_center = (
            np.asarray(cell_center, dtype=float) if cell_center is not None else None
        )
        self.geometry = None
        self.graph: networkx.Graph | None = None
        self.graph_pbc: networkx.Graph | None = None
        self.subgraphs = None
        self.unwrapped = None

    def build_geometry(self, backend: str = "cclib") -> None:
        """Build molecular geometry from input file.

        Reads atomic coordinates and element information from the specified
        input geometry file using the selected computational backend. The
        method populates the geometry attribute with processed molecular data.

        Parameters
        ----------
        backend : str, optional
            Computational backend for file processing. Options include:
            - "cclib": Quantum chemistry formats - .xyz (default)
            - "MDAnalysis": Molecular dynamics trajectories - .dump
            - "internal": Custom MLIP .cfg and LAMMPS .dump parsers

        Raises
        ------
        ValueError
            If `input_geometry` is not defined or
            if `element_types` is required but not provided
            for the selected backend or if `bounds` is required but not provided.

        """
        if self.input_geometry is None:
            raise ValueError("input_geometry is not defined")

        driver = io.Driver(backend)
        if backend in ["cclib", "openbabel"]:
            self.geometry = driver._build_geometry(self.input_geometry)
        else:
            if self.element_types is None:
                raise ValueError("element_types is not defined")
            if self.cell_bounds is None:
                warnings.warn(
                    "Cell bounds are not defined, extracting will be non-periodic",
                    stacklevel=2,
                )
            self.geometry = driver._build_geometry(
                self.input_geometry, self.element_types
            )
            self.geometry = (
                self.geometry[0],
                wrap_into_triclinic(
                    self.geometry[1],
                    self.cell_center,
                    self.cell_matrix,
                ),
            )

    def build_graph(self, backend: str = "openbabel") -> None:
        """Build the graph representation of the molecule.

        Parameters
        ----------
        backend : str, optional
            The backend to use for building the graph (default is "openbabel").

        """
        if self.geometry is None:
            raise ValueError("Geometry is not built. Call build_geometry() first.")

        driver = io.Driver(backend)
        self.graph, self.graph_pbc = driver._build_bonds(self)

    def save_xyz(self, file_name: str, backend: str = "cclib") -> None:
        """Save the molecule geometry in XYZ format.

        Parameters
        ----------
        file_name : str
            The name of the file to save the XYZ data.
        backend : str, optional
            The backend to use for saving XYZ (default is "cclib").

        """
        driver = io.Driver(backend)
        driver._save_xyz(file_name, self.geometry)

    def save_pdb(self, file_name: str, backend: str = "openbabel") -> None:
        """Save the molecule geometry in PDB format.

        Parameters
        ----------
        file_name : str
            The name of the file to save the PDB data.
        backend : str, optional
            The backend to use for saving PDB (default is "openbabel").

        """
        driver = io.Driver(backend)
        driver._save_pdb(file_name, self.geometry)

    def save_dump(
        self,
        file_name: str,
        input_geometry: str,
        idxs: list[int],
        backend: str = "internal",
    ) -> None:
        """Save the molecule geometry in LAMMPS dump format.

        Parameters
        ----------
        file_name : str
            The name of the file to save the dump data.
        input_geometry : str or None
            Path to the input geometry file.
        idxs : list[int]
            Indices of atoms.
        backend : str, optional
            The backend to use for saving dump (default is "internal").

        """
        driver = io.Driver(backend)
        driver._save_dump(file_name, input_geometry, idxs)

    def save_cfg(
        self,
        file_name: str,
        input_geometry: str,
        idxs: list[int],
        backend: str = "internal",
    ) -> None:
        """Save the molecule geometry in cfg format.

        Parameters
        ----------
        file_name : str
            The name of the file to save the cfg data.
        input_geometry : str or None
            Path to the input geometry file.
        idxs : list[int]
            Indices of atoms.
        backend : str, optional
            The backend to use for saving cfg (default is "internal").

        """
        driver = io.Driver(backend)
        driver._save_cfg(file_name, input_geometry, idxs)

    def build_structure(self, *, cut_molecule: bool) -> None:
        """Build the substructure of the molecule by identifying connected components.

        Parameters
        ----------
        cut_molecule : bool
            Whether to break single bonds between heavy atoms.
            When True, molecules are fragmented at single bonds connecting
            atoms with multiple neighbors. When False, all bonds are preserved.

        """
        if self.graph is None:
            raise ValueError("Graph is not built. Call build_graph() first.")

        this_graph = self.graph if self.cell_bounds is None else self.graph_pbc

        cutted_graph = this_graph.copy()

        if cut_molecule:
            single_bonds = [
                (u, v) for u, v, d in cutted_graph.edges(data=True) if d["order"] == 1
            ]
            for u, v in single_bonds:
                # if "H" not in [self.geometry[0][u], self.geometry[0][v]]:
                if (
                    len(list(this_graph.neighbors(u))) > 1
                    and len(list(this_graph.neighbors(v))) > 1
                ):
                    cutted_graph.remove_edge(u, v)

        self.subgraphs = [c for c in networkx.connected_components(cutted_graph)]

    def select_r(self, center: list[float], r: float) -> list[int]:
        """Select atoms within a given radius of a point.

        Parameters
        ----------
        center : list[float]
            The center point coordinates [x, y, z].
        r : float
            The radius within which to select atoms.

        Returns
        -------
        idx : list[int]
            Indices of atoms within the specified radius,
            with respect to PBC if cell_bounds is defined.

        """
        if self.geometry is None:
            raise ValueError("Geometry is not built. Call build_geometry() first.")
        center = np.array(center)
        if self.cell_bounds is None:
            tree = KDTree(data=self.geometry[1])
            idx = tree.query_ball_point(center, r, workers=-1)
        else:
            tree = PeriodicKDTree(
                cell_bounds=self.cell_bounds,
                cell_center=self.cell_center,
                data=self.geometry[1],
            )
            idx = tree.query_ball_point(center, r, workers=-1)
        return idx

    def select(self, selected_atoms: list[int]) -> tuple["Molecule", list[int]]:
        """Select a subset of the molecule based on atom indices.

        Parameters
        ----------
        selected_atoms : list[int]
            list of selected atom indices.

        Returns
        -------
        tuple: [Molecule, list[int]]
            A tuple containing:
            - Molecule: A new Molecule object containing the selected atoms
            and necessary hydrogens.
            - list[int]: An array of selected atoms index.

        """

        def norm(xyz: np.typing.ArrayLike) -> np.ndarray:
            """Normalize a vector.

            Parameters
            ----------
            xyz : np.typing.ArrayLike
                Vector coordinates

            Returns
            -------
            np.ndarray
                Normalized coordinates of given vector

            """
            return np.array(xyz) / np.linalg.norm(xyz)

        if self.geometry is None or self.graph is None or self.subgraphs is None:
            raise ValueError(
                "Molecule structure is not fully built."
                "Call build_geometry(), build_graph(), and build_structure() first."
            )
        if self.cell_bounds is not None:
            self.unwrap_graph_based([0, 0, 0])

        this_graph = self.graph if self.cell_bounds is None else self.graph_pbc

        new_hydrogens = []
        for atom in selected_atoms:
            for neighbor in this_graph.neighbors(atom):
                if neighbor not in selected_atoms:
                    # Add hydrogen at standard C-H distance along bond direction
                    bond_vector = self.geometry[1][neighbor] - self.geometry[1][atom]
                    print(np.linalg.norm(bond_vector))
                    hydrogen_position = (
                        norm(bond_vector) * 1.09 + self.geometry[1][atom]
                    )
                    new_hydrogens.append(hydrogen_position)

        new_molecule = Molecule(cell_bounds=self.cell_bounds)
        selected_atoms.sort()

        new_molecule.geometry = (
            [self.geometry[0][i] for i in selected_atoms],
            self.geometry[1][selected_atoms],
        )

        if len(new_hydrogens) > 0:
            new_molecule.geometry = (
                new_molecule.geometry[0] + ["H"] * len(new_hydrogens),
                np.append(new_molecule.geometry[1], np.array(new_hydrogens), axis=0),
            )

        return new_molecule, selected_atoms

    def expand_selection(self, idxs: list[int]) -> list[int]:
        """Select electronegative atoms/functional group near the edge of selection.

        Parameters
        ----------
        idxs : list[int]
            List of atom indices of initial selection by radius.

        Returns
        -------
        list[int]
            Indices of selected atoms.

        """
        if self.geometry is None or self.graph is None or self.subgraphs is None:
            raise ValueError(
                "Molecule structure is not fully built."
                "Call build_geometry(), build_graph(), and build_structure() first."
            )

        this_graph = self.graph if self.cell_bounds is None else self.graph_pbc

        # Include complete molecular fragments containing selected atoms
        selected_atoms = []
        for subgraph in self.subgraphs:
            if any(idx in idxs for idx in subgraph):
                selected_atoms.extend(list(subgraph))

        electronegative_atoms = {"O", "S", "N", "P"}
        functional_group_atoms = {"O", "N", "F", "Cl", "Br", "I"}

        for atom in selected_atoms:
            for neighbor in this_graph.neighbors(atom):
                if (
                    self.geometry[0][neighbor] in electronegative_atoms
                    and neighbor not in selected_atoms
                ):
                    selected_atoms.append(neighbor)
                if self.geometry[0][neighbor] in {"C", "N"}:
                    if any(
                        self.geometry[0][c_n_neighbor] in functional_group_atoms
                        for c_n_neighbor in this_graph.neighbors(neighbor)
                    ):
                        if neighbor not in selected_atoms:
                            selected_atoms.append(neighbor)
                        for next_neighbor in this_graph.neighbors(neighbor):
                            if next_neighbor not in selected_atoms:
                                selected_atoms.append(next_neighbor)

        return selected_atoms

    def unwrap_graph_based(
        self,
        reference_coord: list[float],
    ) -> None:
        """Unwrap atomic coordinates by comparing periodic and non-periodic bond graphs.

        This method adjusts coordinates in-place so that all bonded fragments
        are contiguous, removing jumps across periodic boundaries.

        Parameters
        ----------
        reference_coord : list[float]
            xyz coordinates of reference atom.

        Raises
        ------
        ValueError
            If geometry or cell_bounds has not been initialized.

        """
        if self.geometry is None or self.cell_bounds is None:
            raise ValueError(
                "Both geometry and cell_bounds must be set before unwrapping."
            )

        # copy coordinates and prepare cell matrices
        coords = self.geometry[1].copy()
        inv_cell = np.linalg.inv(self.cell_matrix)

        # identify connected components in the non-PBC graph
        components = list(networkx.connected_components(self.graph))
        node_to_comp = {
            node: comp_id for comp_id, comp in enumerate(components) for node in comp
        }
        n_comps = len(components)

        # Arrays to accumulate component shifts and track which are set
        comp_shifts = np.zeros((n_comps, 3), dtype=int)
        shift_assigned = np.zeros(n_comps, dtype=bool)
        dependents = [[] for _ in range(n_comps)]

        def propagate_shift(root_comp: int) -> None:
            """Cascade the shift of root_comp to all its dependents.

            When a component shift is determined, all downstream dependent
            components must inherit that shift addition.
            """
            stack = [root_comp]
            while stack:
                curr = stack.pop()
                for dep in dependents[curr]:
                    comp_shifts[dep] += comp_shifts[curr]
                    stack.append(dep)
                # Clear to avoid reprocessing
                dependents[curr].clear()

        # find edges that exist only under PBC
        pbc_edges = set(self.graph_pbc.edges())
        non_pbc_edges = set(self.graph.edges())
        boundary_edges = pbc_edges - non_pbc_edges

        for u, v in boundary_edges:
            # Determine which atom is farther from reference
            dist_u = np.linalg.norm(coords[u] - reference_coord)
            dist_v = np.linalg.norm(coords[v] - reference_coord)
            farther, closer = (v, u) if dist_v > dist_u else (u, v)

            # Compute fractional shift between fragments
            delta_frac = (coords[farther] - coords[closer]) @ inv_cell
            shift_vec = np.round(delta_frac).astype(int)
            if not shift_vec.any():
                continue  # no periodic shift

            comp_far = node_to_comp[farther]
            comp_close = node_to_comp[closer]

            if not shift_assigned[comp_far]:
                # Base shift on closer component if it was already shifted
                base_shift = (
                    comp_shifts[comp_close] if shift_assigned[comp_close] else 0
                )
                comp_shifts[comp_far] = shift_vec + base_shift
                shift_assigned[comp_far] = True
                # Mark dependency and propagate
                dependents[comp_close].append(comp_far)
                propagate_shift(comp_far)

        # apply computed shifts to coordinates
        for comp_id, shift_vec in enumerate(comp_shifts):
            if shift_assigned[comp_id] and shift_vec.any():
                atom_indices = list(components[comp_id])
                coords[atom_indices] -= shift_vec @ self.cell_matrix

        # update geometry and mark as unwrapped
        self.geometry = (self.geometry[0], coords)
        self.unwrapped = True
