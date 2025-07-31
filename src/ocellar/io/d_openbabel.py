"""Module to handle molecule operations using Open Babel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx
import numpy
import periodictable
from openbabel import openbabel, pybel

from ocellar.io.driver import Driver

if TYPE_CHECKING:
    from ocellar.molecule import Molecule


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
    def _build_bonds(
        cls,
        mol: Molecule,
    ) -> networkx.Graph:
        """Build a graph representation of molecular bonds using openbabel.

        Parameters
        ----------
        mol : ocellar.Molecule
            An object of class Molecule with built geometry.

        Returns
        -------
        networkx.Graph
            A graph representation of the molecular structure with bonds as edges.

        """
        obmol = openbabel.OBMol()

        for i, element in enumerate(mol.geometry[0]):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(periodictable.elements.symbol(element).number)
            x, y, z = mol.geometry[1][i]
            atom.SetVector(x, y, z)

        obmol_new = openbabel.OBMol()
        obmol_new += obmol

        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()

        molecule_graph = networkx.Graph()
        for bond in openbabel.OBMolBondIter(obmol):
            molecule_graph.add_edge(
                bond.GetBeginAtomIdx() - 1,
                bond.GetEndAtomIdx() - 1,
                order=bond.GetBondOrder(),
            )

        if mol.cell_bounds is not None:
            obmol_new.SetPeriodicMol()
            obcell = openbabel.OBUnitCell()
            obcell.SetData(
                mol.cell_bounds[0],
                mol.cell_bounds[1],
                mol.cell_bounds[2],
                mol.cell_bounds[3],
                mol.cell_bounds[4],
                mol.cell_bounds[5],
            )
            obvec = openbabel.vector3()
            obvec.Set(mol.cell_center[0], mol.cell_center[1], mol.cell_center[2])
            obcell.SetOffset(obvec)
            obmol_new.CloneData(obcell)

        obmol_new.ConnectTheDots()
        obmol_new.PerceiveBondOrders()

        molecule_graph_pbc = networkx.Graph()
        for bond in openbabel.OBMolBondIter(obmol_new):
            molecule_graph_pbc.add_edge(
                bond.GetBeginAtomIdx() - 1,
                bond.GetEndAtomIdx() - 1,
                order=bond.GetBondOrder(),
            )

        lonely_atoms = list(set(molecule_graph_pbc.nodes) - set(molecule_graph.nodes))

        for atom in lonely_atoms:
            molecule_graph.add_node(atom)

        return molecule_graph, molecule_graph_pbc

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
            atom.SetAtomicNum(periodictable.elements.symbol(element).number)
            x, y, z = geometry[1][i]
            atom.SetVector(x, y, z)

        mol = pybel.Molecule(obmol)
        mol.write("pdb", file_name, overwrite=True)
