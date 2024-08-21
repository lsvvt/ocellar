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


    @classmethod
    def _save_dump(cls, file_name: str, input_geometry: str, idxs: list[int]) -> None:
        """Save the geometry in LAMMPS dump format using cclib.

        Parameters
        ----------
        file_name : str
            The name of the file to save the dump data.
        input_geometry : str or None
            Path to the input geometry file.
        idxs : list[int]
            Indices of atoms.

        Returns
        -------
        None

        """
        s_idxs = list(map(lambda x: str(x + 1), idxs))
        
        with open(input_geometry, "r") as f:
            out_lines = [line for line in f 
                if not (len(line.split()) > 5 and line.split()[0] not in s_idxs and line.split()[2].replace('.', '', 1).isdigit())]
            out_lines[out_lines.index("ITEM: NUMBER OF ATOMS\n") + 1] = str(len(idxs)) + "\n"
            with open(file_name, "w") as f_out:
                f_out.writelines(out_lines)