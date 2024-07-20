from ocellar import io
from ocellar import utils
import networkx
from scipy.spatial import cKDTree
import numpy
from typing import List


class Molecule:
    """
    A class to represent a molecule and its properties.

    Attributes
    ----------
    input_geometry : str or None
        Path to the input geometry file.
    geometry : Tuple[list, numpy.ndarray] or None
        A tuple containing:
        - list: A list of element symbols.
        - numpy.ndarray: An array of atomic coordinates.
    graph : networkx.Graph or None
        Graph representation of the molecular structure.
    subgraphs : list or None
        List of connected components in the molecular graph.

    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the Molecule object.


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
        """
        Build the geometry of the molecule.

        Parameters
        ----------
        backend : str, optional
            The backend to use for building the geometry (default is "cclib").

        Raises
        ------
        ValueError
            If input_geometry is not defined.
        """
        if self.input_geometry:
            driver = io.Driver(backend)
            self.geometry = driver._build_geometry(self.input_geometry)
        else:
            raise ValueError("input_geometry is not defined")


    def build_graph(self, backend: str = "openbabel") -> None:
        """
        Build the graph representation of the molecule.

        Parameters
        ----------
        backend : str, optional
            The backend to use for building the graph (default is "openbabel").
        """
        if self.geometry:
            driver = io.Driver(backend)
            self.graph = driver._build_bonds(self.geometry)
        else:
            self.build_geometry()


    def save_xyz(self, file_name: str, backend: str = "cclib") -> None:
        """
        Save the molecule geometry in XYZ format.

        Parameters
        ----------
        file_name : str
            The name of the file to save the XYZ data.
        backend : str, optional
            The backend to use for saving XYZ (default is "cclib").
        """
        driver = io.Driver(backend)
        driver._save_xyz(file_name, self.geometry)


    def build_structure(self) -> None:
        """
        Build the substructure of the molecule by identifying connected components.
        """
        if self.graph is None:
            raise ValueError("Graph is not built. Call build_graph() first.")
        
        cutted_graph = self.graph.copy()
        single_bonds = [(u, v) for u, v, d in cutted_graph.edges(data=True) if d['order'] == 1]
        for u, v in single_bonds:
            cutted_graph.remove_edge(u, v)
            
        self.subgraphs = [c for c in networkx.connected_components(cutted_graph)]


    def select_r(self, x: List[float], r: float) -> List[int]:
        """
        Select atoms within a given radius of a point.

        Parameters
        ----------
        x : List[float]
            The center point coordinates.
        r : float
            The radius within which to select atoms.

        Returns
        -------
        List[int]
            Indices of atoms within the specified radius.
        """
        if self.geometry is None:
            raise ValueError("Geometry is not built. Call build_geometry() first.")
        
        tree = cKDTree(self.geometry[1])
        idx = tree.query_ball_point(x, r)
        return idx


    def select(self, idxs: List[int]) -> 'Molecule':
        """
        Select a subset of the molecule based on atom indices.

        Parameters
        ----------
        idxs : List[int]
            List of atom indices to select.

        Returns
        -------
        Molecule
            A new Molecule object containing the selected atoms and necessary hydrogens.
        """
        if self.geometry is None or self.graph is None or self.subgraphs is None:
            raise ValueError("Molecule structure is not fully built. Call build_geometry(), build_graph(), and build_structure() first.")
        
        selected_n = []
        for graph in self.subgraphs:
            if any(idx in idxs for idx in graph):
                selected_n += list(graph)

        new_at = []
        for atom in selected_n:
            for neighbor in self.graph.neighbors(atom):
                if neighbor not in selected_n:
                    new_at.append(utils.norm(self.geometry[1][neighbor] - self.geometry[1][atom]) * 1.09 + self.geometry[1][atom])

        new_molecule = Molecule()
        new_molecule.geometry = ([self.geometry[0][i] for i in selected_n] + ["H"] * len(new_at),
                                 numpy.append(self.geometry[1][selected_n], numpy.array(new_at), axis = 0))

        return new_molecule