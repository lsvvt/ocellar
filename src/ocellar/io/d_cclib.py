"""Module to handle molecule operations using cclib."""

import cclib
import numpy
import periodictable

from ocellar.io.driver import Driver


class DCclib(Driver):
    """Class for a driver for interfacing with the cclib library."""

    backend = "cclib"

    @classmethod
    def _build_geometry(cls, input_geometry: str) -> tuple[list, numpy.ndarray]:
        """Build the geometry from the input file using cclib.

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
        parsed_data = cclib.io.ccread(input_geometry)
        elements = [
            periodictable.elements[atom_number].symbol for atom_number in parsed_data.atomnos
        ]
        coordinates = parsed_data.atomcoords[-1]
        return elements, coordinates
