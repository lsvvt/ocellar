"""Module to handle molecule operations using cclib."""

import cclib
import numpy
import periodictable

from ocellar.io.driver import Driver


class DCclib(Driver):
    """Class for a driver for interfacing with the cclib library.

    Attributes
    ----------
    backend : str
        The name of the backend library used, set to "cclib".

    """

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
            periodictable.elements[atom_number].symbol
            for atom_number in parsed_data.atomnos
        ]
        coordinates = parsed_data.atomcoords[-1]
        return elements, coordinates

    @classmethod
    def _save_xyz(cls, file_name: str, geometry: tuple[list, numpy.ndarray]) -> None:
        """Save the geometry in XYZ format using cclib.

        Parameters
        ----------
        file_name : str
            The name of the file to save the XYZ data.
        geometry : tuple
            A tuple containing:
            - list: A list of element symbols.
            - numpy.ndarray: An array of atomic coordinates.

        Returns
        -------
        None

        """
        attributes = {
            "natom": len(geometry[0]),
            "atomnos": [
                periodictable.elements.symbol(atom).number for atom in geometry[0]
            ],
            "atomcoords": [geometry[1]],
            "charge": 0,
            "mult": 1,
            "metadata": "",
        }

        data = cclib.parser.data.ccData(attributes)
        xyz = cclib.io.xyzwriter.XYZ(data)

        with open(file_name, "w") as f:
            f.writelines(xyz.generate_repr())

        # My code
        # with open(file_name, 'w') as f:
        #     f.write(f"{attributes['natom']}\n")
        #     f.write(f"{attributes['metadata']}\n")
        #     for atom, coord in zip(geometry[0], geometry[1]):
        #         f.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
