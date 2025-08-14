"""Main ocellar package module, containing Molecule class."""

from pathlib import Path

import networkx
import numpy as np
from scipy.spatial import KDTree

from ocellar import io


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
    geometry : tuple[list, np.ndarray] or None
        A tuple containing:
        - list: A list of element symbols.
        - np.ndarray: An array of atomic coordinates.
    graph : networkx.Graph or None
        Graph representation of the molecular structure.
    subgraphs : list or None
        list of connected components in the molecular graph.
    cell : np.ndarray or None
        Cell parameters of shape [3, 4] in ovito format, set externally.
    charge : int
        Charge of this molecule object, possibly inferred by
        meth::Molecule:_infer_fromal_charge

    """

    def __init__(
        self,
        input_geometry: str | Path | None = None,
        element_types: np.typing.ArrayLike | None = None,
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

        """
        self.input_geometry = input_geometry
        self.element_types = element_types
        self.geometry = None
        self.graph: networkx.Graph | None = None
        self.subgraphs = None
        self.cell = None
        self.charge = 0

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
            - "openbabel" Quantum chemistry formats - .xyz
            - "ovito" Molecular dynamics (LAMMPS) trajectories - .dump
            - "MDAnalysis": Molecular dynamics (LAMMPS) trajectories - .dump
            - "internal": MLIP file format - .cfg

        Raises
        ------
        ValueError
            If `input_geometry` is not defined or
            if `element_types` is required but not provided
            for the selected backend.

        """
        if self.input_geometry is None:
            raise ValueError("input_geometry is not defined")

        driver = io.Driver(backend)
        if backend in ["cclib", "openbabel"]:
            self.geometry = driver._build_geometry(self.input_geometry)
        else:
            if self.element_types is None:
                raise ValueError("element_types is not defined")
            self.geometry = driver._build_geometry(
                self.input_geometry, self.element_types
            )

    def replicate_geometry(self) -> None:
        """Replicate current geometry into a 3x3x3 supercell.

        The replication order follows ovito's logic,
        iterating a, b, c vectors from negative to positive.

        This function is necessary to correctly work with periodic boundary conditions.

        Raises
        ------
        ValueError
            If `self.geometry` is not set.
            If `self.cell` is not set.

        Returns
        -------
        None
            Replicates `self.geometry` inplace.

        Notes
        -----
        `self.cell` must be a NumPy array where each column is a lattice vector:
        `a = self.cell[:, 0]`, `b = self.cell[:, 1]`, `c = self.cell[:, 2]`.

        """
        if self.geometry is None:
            raise ValueError("Geometry is not built. Call build_geometry() first.")
        if self.cell is None:
            raise ValueError("Cell is not set. Set self.cell before replication.")

        elements, coords = self.geometry
        a = np.asarray(self.cell[:, 0], dtype=float)
        b = np.asarray(self.cell[:, 1], dtype=float)
        c = np.asarray(self.cell[:, 2], dtype=float)

        shifts = []
        for ai in (-1, 0, 1):
            for bi in (-1, 0, 1):
                for ci in (-1, 0, 1):
                    shifts.append(ai * a + bi * b + ci * c)

        replicated_coords = np.concatenate([coords + s for s in shifts], axis=0)
        replicated_elements = elements * 27

        self.geometry = (replicated_elements, replicated_coords)

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
        self.graph = driver._build_bonds(self)

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
                if self.graph.degree[u] > 1 and self.graph.degree[v] > 1:
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
            Indices of atoms within the specified radius.

        """
        if self.geometry is None:
            raise ValueError("Geometry is not built. Call build_geometry() first.")
        center = np.asarray(center, dtype=float)
        tree = KDTree(data=self.geometry[1])
        idx = tree.query_ball_point(center, r, workers=-1)
        return idx

    def select(self, selected_atoms: set[int]) -> tuple["Molecule", list[int]]:
        """Select a subset of the molecule based on atom indices.

        Parameters
        ----------
        selected_atoms : set[int]
            set of selected atom indices.

        Returns
        -------
        tuple: [Molecule, list[int]]
            A tuple containing:
            - Molecule: A new Molecule object containing the selected atoms
            and necessary hydrogens with inferred charge.
            - list[int]: A sorted array of selected atoms index.

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

        selected_atoms = sorted(list(selected_atoms))

        new_molecule = Molecule()
        new_molecule.charge = self.charge  # assign charge to new_mol
        self.charge = 0  # restore charge for further usage

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

    def expand_selection(self, idxs: list[int]) -> set[int]:
        """Select electronegative atoms/functional group near the edge of selection.

        Additionally, the function infers the total formal charge of the
        expanded selection (quaternary ammonium, alkoxide, thiolate,
        carboxylate) and stores it in `self.charge`.

        Parameters
        ----------
        idxs : list[int]
            List of atom indices of initial selection by radius.

        Returns
        -------
        selected_atoms : set[int]
            Indices of selected atoms.

        """
        if self.geometry is None or self.graph is None or self.subgraphs is None:
            raise ValueError(
                "Molecule structure is not fully built."
                "Call build_geometry(), build_graph(), and build_structure() first."
            )

        # Include complete molecular fragments containing selected atoms
        selected_atoms = set()
        for subgraph in self.subgraphs:
            if any(idx in idxs for idx in subgraph):
                selected_atoms.update(subgraph)

        electronegative_atoms = {"O", "S", "N", "P"}
        functional_group_atoms = {"O", "N", "F", "Cl", "Br", "I"}
        new_selected = set()
        self._processed_atoms = set()
        total_charge = 0

        for atom in selected_atoms:
            for neighbor in self.graph.neighbors(atom):
                # do not replace CX2 fragment with two H's
                if self.graph.degree[neighbor] >= 2:
                    new_selected.add(neighbor)
                # do not leave electronegative atoms on border
                if self.geometry[0][neighbor] in electronegative_atoms:
                    new_selected.add(neighbor)
                # do not leave functional groups on border
                if self.geometry[0][neighbor] in {"C", "N"}:
                    if any(
                        self.geometry[0][c_n_neighbor] in functional_group_atoms
                        for c_n_neighbor in self.graph.neighbors(neighbor)
                    ):
                        new_selected.add(neighbor)
                        new_selected.update(self.graph.neighbors(neighbor))

        selected_atoms.update(new_selected)

        # infer charges for final selection
        for atom in selected_atoms:
            if (self.geometry[0][atom] in electronegative_atoms) and (
                atom not in self._processed_atoms
            ):
                total_charge += self._infer_formal_charge(atom)

        self.charge = total_charge
        return selected_atoms

    def _infer_formal_charge(self, atom_id: int) -> int:
        """Infer formal charge for common functional groups.

        Parameters
        ----------
        atom_id : int
            Index of electronegative atom carrying possible charge.

        Returns
        -------
        int
            Formal charge of atom with `atom_id`

        Notes
        -----
        +1 : quaternary ammonium - N with 4 neighbours
        -1 : alkoxide - O with 1 neighbour C and C has 4 neighbours
        -1 : thiolate - S with 1 neighbour C and C has 4 neighbours
        -1 : carboxylate oxygen - O with 1 neighbour C,
            C has 3 neighbours and two O neighbours

        """
        element = self.geometry[0][atom_id]
        degree = self.graph.degree(atom_id)

        # R4N+, charge +1
        if element == "N" and degree == 4:
            return +1

        # RO- or RS-, charge -1
        if element in {"O", "S"} and degree == 1:
            c = next(iter(self.graph.neighbors(atom_id)))
            if self.geometry[0][c] == "C":
                c_deg = self.graph.degree(c)
                if c_deg == 4:  # saturated carbon
                    return -1
                if c_deg == 3 and element == "O":  # unsaturated carbon
                    o_cnt = sum(
                        self.geometry[0][nbr] == "O" for nbr in self.graph.neighbors(c)
                    )
                    if o_cnt == 2:
                        self._processed_atoms.update(self.graph.neighbors(c))
                        return -1
        return 0
