"""Main ocellar package module, containing Molecule class."""

import warnings
from pathlib import Path

import networkx
import numpy as np
from scipy.spatial import KDTree

from ocellar import io
from ocellar.utils.pkdtree import PeriodicKDTree, cell_matrix_from_bounds


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
        self.cell_bounds = np.array(cell_bounds) if cell_bounds is not None else None
        self.cell_center = np.array(cell_center) if cell_center is not None else None
        self.geometry = None
        self.graph = None
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
        self.graph: networkx.Graph = driver._build_bonds(self)

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

        cutted_graph = self.graph.copy()
        if cut_molecule:
            single_bonds = [
                (u, v) for u, v, d in cutted_graph.edges(data=True) if d["order"] == 1
            ]
            for u, v in single_bonds:
                # if "H" not in [self.geometry[0][u], self.geometry[0][v]]:
                if (
                    len(list(self.graph.neighbors(u))) > 1
                    and len(list(self.graph.neighbors(v))) > 1
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

        new_hydrogens = []
        for atom in selected_atoms:
            for neighbor in self.graph.neighbors(atom):
                if neighbor not in selected_atoms:
                    # Add hydrogen at standard C-H distance along bond direction
                    bond_vector = self.geometry[1][neighbor] - self.geometry[1][atom]
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

        # Include complete molecular fragments containing selected atoms
        selected_atoms = []
        for subgraph in self.subgraphs:
            if any(idx in idxs for idx in subgraph):
                selected_atoms.extend(list(subgraph))

        electronegative_atoms = {"O", "S", "N", "P"}
        functional_group_atoms = {"O", "N", "F", "Cl", "Br", "I"}

        for atom in selected_atoms:
            for neighbor in self.graph.neighbors(atom):
                if (
                    self.geometry[0][neighbor] in electronegative_atoms
                    and neighbor not in selected_atoms
                ):
                    selected_atoms.append(neighbor)
                if self.geometry[0][neighbor] in {"C", "N"}:
                    if any(
                        self.geometry[0][c_n_neighbor] in functional_group_atoms
                        for c_n_neighbor in self.graph.neighbors(neighbor)
                    ):
                        for next_neighbor in self.graph.neighbors(neighbor):
                            if next_neighbor not in selected_atoms:
                                selected_atoms.append(next_neighbor)

        return selected_atoms

    def unwrap(self, ref_atom: list[float]) -> None:
        """Unwrap Molecule atomic coordinates relative to a reference atom.

        PBC in all axis are assumed.

        Parameters
        ----------
        ref_atom : list[float]
            Reference atom coordinates [x_ref, y_ref, z_ref].

        Returns
        -------
        None
            Unwraps coordinates in the same order as in `self.geometry` **inplace**.

        """
        ref_arr = np.asarray(ref_atom, dtype=float)
        coords_arr = np.asarray(self.geometry[1], dtype=float)
        cell_col = np.asarray(cell_matrix_from_bounds(self.cell_bounds)).T
        cell_inv = np.linalg.inv(cell_col)

        delta_frac = (coords_arr - ref_arr) @ cell_inv  # fractional separation
        shift = np.round(delta_frac)  # nearest lattice vector (-1, 0, 1)
        coords_unwrapped = coords_arr - shift @ cell_col

        self.geometry = (self.geometry[0], coords_unwrapped)
        self.unwrapped = True
