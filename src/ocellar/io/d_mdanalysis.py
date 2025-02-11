"""Module to handle molecule operations using MDAnalysis."""

import MDAnalysis
import numpy

from ocellar.io.driver import Driver


class DMDAnalysis(Driver):
    """Class for a driver for interfacing with the MDAnalysis library.

    Attributes
    ----------
    backend : str
        The name of the backend library used, set to "MDAnalysis".

    """

    backend = "MDAnalysis"

    @classmethod
    def _build_geometry(cls, input_geometry: str) -> tuple[list, numpy.ndarray]:
        """Build the geometry from a LAMMPS dump file using MDAnalysis.

        Parameters
        ----------
        input_geometry : str
            Path to the input LAMMPS dump file.

        Returns
        -------
        tuple
            A tuple containing:
            - list: A list of element symbols.
            - numpy.ndarray: An array of atomic coordinates.

        """
        u = MDAnalysis.Universe(input_geometry, format="LAMMPSDUMP")

        coordinates = u.atoms.positions.astype(float)
        types = u.atoms.types.astype(int)
        atom_type_symbols = numpy.genfromtxt(input_geometry + "_types", dtype="str")

        elements = [atom_type_symbols[i - 1] for i in types]

        return elements, coordinates
