"""Module to handle molecule operations using Open Babel."""

import numpy
import periodictable
from openbabel import openbabel, pybel  # type: ignore

from ocellar.io.driver import Driver


class DOpenbabel(Driver):
    """Class for a driver for interfacing with the openbabel library."""

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
        cls, geometry: tuple[list, numpy.ndarray]
) -> tuple[numpy.ndarray, numpy.ndarray]:
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

        return obmol ### Тут я остановился...
