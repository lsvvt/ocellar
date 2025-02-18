import networkx
import numpy as np
from scipy.spatial import cKDTree

from ocellar import io


class Molecule:
    """A class to represent a molecule and its properties.

    Attributes
    ----------
    input_geometry : str or None
        Path to the input geometry file.
    geometry : tuple[list, np.ndarray] or None
        A tuple containing:
        - list: A list of element symbols.
        - np.ndarray: An array of atomic coordinates.
    graph : networkx.Graph or None
        Graph representation of the molecular structure.
    subgraphs : list or None
        list of connected components in the molecular graph.

    """

    def __init__(self, **kwargs) -> None:
        """Initialize the Molecule object.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments to set as attributes.

        """
        self.input_geometry = None
        self.geometry = None
        for key, val in kwargs.items():
            setattr(self, key, val)

    def build_geometry(self, backend: str = "cclib") -> None:
        """Build the geometry of the molecule.

        Parameters
        ----------
        backend : str, optional
            The backend to use for building the geometry (default is "cclib").

        Raises
        ------
        ValueError
            If input_geometry is not defined.

        """
        if self.input_geometry is None:
            raise ValueError("input_geometry is not defined")

        driver = io.Driver(backend)
        self.geometry = driver._build_geometry(self.input_geometry)

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
        self.graph = driver._build_bonds(self.geometry)

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

    def build_structure(self, cut_molecule: bool = True) -> None:
        """Build the substructure of the molecule by identifying connected components."""
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

    def select_r(self, x: list[float], r: float) -> list[int]:
        """Select atoms within a given radius of a point.

        Parameters
        ----------
        x : list[float]
            The center point coordinates.
        r : float
            The radius within which to select atoms.

        Returns
        -------
        list[int]
            Indices of atoms within the specified radius.

        """
        if self.geometry is None:
            raise ValueError("Geometry is not built. Call build_geometry() first.")

        tree = cKDTree(self.geometry[1])
        idx = tree.query_ball_point(x, r)
        return idx

    def select(self, idxs: list[int]) -> tuple["Molecule", list[int]]:
        """Select a subset of the molecule based on atom indices.

        Parameters
        ----------
        idxs : list[int]
            list of atom indices to select.

        Returns
        -------
        Molecule
            A new Molecule object containing the selected atoms and necessary hydrogens.
        tuple
            A tuple containing:
            - Molecule: A new Molecule object containing the selected atoms and necessary hydrogens.
            - list[int]: An array of selected atoms index.

        """

        def norm(xyz) -> np.ndarray:
            """Normalize a vector."""
            return np.array(xyz) / np.linalg.norm(xyz)

        if self.geometry is None or self.graph is None or self.subgraphs is None:
            raise ValueError(
                "Molecule structure is not fully built. Call build_geometry(), build_graph(), and build_structure() first."
            )

        selected_n = []
        for graph in self.subgraphs:
            if any(idx in idxs for idx in graph):
                selected_n += list(graph)

        new_at = []
        for atom in selected_n:
            for neighbor in self.graph.neighbors(atom):
                if neighbor not in selected_n:
                    new_at.append(
                        norm(self.geometry[1][neighbor] - self.geometry[1][atom]) * 1.09
                        + self.geometry[1][atom]
                    )

        new_molecule = Molecule()

        selected_n.sort()
        new_molecule.geometry = (
            [self.geometry[0][i] for i in selected_n],
            self.geometry[1][selected_n],
        )
        if len(new_at) > 0:
            new_molecule.geometry = (
                new_molecule.geometry[0] + ["H"] * len(new_at),
                np.append(new_molecule.geometry[1], np.array(new_at), axis=0),
            )

        return new_molecule, selected_n
