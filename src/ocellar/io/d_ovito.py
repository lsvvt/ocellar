"""Module to handle molecule operations using ovito."""

import numpy as np

from ocellar.io.driver import Driver


class DOvito(Driver):
    """Class for a driver for interfacing with the ovito library.

    Attributes
    ----------
    backend : str
        The name of the backend library used, set to "ovito".

    """

    backend = "ovito"

    @classmethod
    def _build_geometry(
        cls, input_geometry: str, element_types: list[str]
    ) -> tuple[list, np.ndarray]:
        """Build the geometry from a LAMMPS dump file using MDAnalysis.

        Parameters
        ----------
        input_geometry : str
            Path to the input LAMMPS dump file.
        element_types : list[str]
            List of N element symbols corresponding to N atom types in cfg file.

        Returns
        -------
        tuple
            A tuple containing:
            - list: A list of element symbols.
            - numpy.ndarray: An array of atomic coordinates.

        """
        from ovito.io import import_file

        p = import_file(input_geometry)
        data = p.compute(0)
        coordinates = np.asarray(data.particles_.positions_, dtype=float)
        types = np.asarray(data.particles_.particle_types_, dtype=int)
        elements = [element_types[i - 1] for i in types]
        cell = np.asarray(data.cell, dtype=float)

        return elements, coordinates, cell
