"""Module to handle molecule operations using MDAnalysis."""

import numpy as np

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
        import MDAnalysis

        u = MDAnalysis.Universe(input_geometry, format="LAMMPSDUMP")

        coordinates = u.atoms.positions.astype(float)
        types = u.atoms.types.astype(int)
        elements = [element_types[i - 1] for i in types]

        # restore positions to original because MDAnalysis shifts origin to (0,0,0)
        with open(input_geometry) as f:
            lines = f.readlines()
            item_atoms_idx = 0
            for i, line in enumerate(lines):
                if "ITEM: ATOMS" in line:
                    item_atoms_idx = i
                    break
            item_atoms_header = lines[item_atoms_idx].split(" ")
            x_idx = item_atoms_header.index("x") - 2
            first_atom_coords = list(
                map(
                    float,
                    (lines[item_atoms_idx + 1].split(" "))[x_idx : (x_idx + 3)],
                )
            )
            first_atom_coords = np.asarray(first_atom_coords)

        dif = first_atom_coords - coordinates[0]
        coordinates += dif

        return elements, coordinates
