"""Module to handle molecule operations using Open Babel."""

import networkx
import numpy
import periodictable
from openbabel import openbabel, pybel  # type: ignore

from ocellar.io.driver import Driver


class DOpenbabel(Driver):
    """Class for a driver for interfacing with the openbabel library.

    Attributes
    ----------
    backend : str
        The name of the backend library used, set to "openbabel".

    """

    backend = "openbabel"

    @classmethod
    def _build_geometry(cls, input_geometry: str) -> tuple[list, numpy.ndarray]:
        """Build the geometry from the input file using openbabel.

        Parameters
        ----------
        input_geometry : str
            Path to the input geometry file.

        Returns
        -------
        tuple
            A tuple containing:
            - list: A list of element symbols.
            - numpy.ndarray: An array of atomic coordinates.

        """
        mol = next(pybel.readfile("xyz", input_geometry))
        elements = [periodictable.elements[atom.atomicnum].symbol for atom in mol.atoms]
        coordinates = numpy.array([atom.coords for atom in mol.atoms])
        return elements, coordinates


    @classmethod
    def _build_bonds(cls, geometry: tuple[list, numpy.ndarray]) -> networkx.Graph:
        """Build a graph representation of molecular bonds using openbabel.

        Parameters
        ----------
        geometry : tuple
            A tuple containing:
            - list: A list of element symbols.
            - numpy.ndarray: An array of atomic coordinates.

        Returns
        -------
        networkx.Graph
            A graph representation of the molecular structure with bonds as edges.

        """
        obmol = openbabel.OBMol()

        for i, element in enumerate(geometry[0]):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(
                periodictable.elements.symbol(element).number
            )
            x, y, z = geometry[1][i]
            atom.SetVector(x, y, z)

        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()

        molecule_graph = networkx.Graph()
        for bond in openbabel.OBMolBondIter(obmol):
            molecule_graph.add_edge(bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, order=bond.GetBondOrder())

        return molecule_graph
    

    @classmethod
    def _save_pdb(cls, file_name: str, geometry: tuple[list, numpy.ndarray]) -> None:
        """Save the geometry in PDB format using openbabel.

        Parameters
        ----------
        file_name : str
            The name of the file to save the PDB data.
        geometry : tuple
            A tuple containing:
            - list: A list of element symbols.
            - numpy.ndarray: An array of atomic coordinates.

        Returns
        -------
        None

        """
        obmol = openbabel.OBMol()

        for i, element in enumerate(geometry[0]):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(
                periodictable.elements.symbol(element).number
            )
            x, y, z = geometry[1][i]
            atom.SetVector(x, y, z)
        
        mol = pybel.Molecule(obmol)
        mol.write('pdb', file_name, overwrite=True)