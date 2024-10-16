"""Module to handle molecule operations using MDAnalysis."""

import MDAnalysis 
import numpy
import periodictable
import openmm.app

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
        u = MDAnalysis.Universe(input_geometry, format='LAMMPSDUMP')

        coordinates = u.atoms.positions.astype(float)

        with open(input_geometry, "r") as f:
            lines = f.readlines()
            elements = [openmm.app.Element.getByMass(float(line.split()[2])).symbol for line in lines 
                if len(line.split()) > 5 and line.split()[2].replace('.', '', 1).isdigit()]
            # my code
            # coordinates = [list(map(float, line.split()[3:6])) for line in lines 
            #     if len(line.split()) > 5 and line.split()[2].replace('.', '', 1).isdigit()]
        # coordinates = numpy.array(coordinates)

        return elements, coordinates
