"""Module to handle molecule operations."""

import numpy

from ocellar.io.driver import Driver


class DInternal(Driver):
    """Class for a io driver.

    Attributes
    ----------
    backend : str
        The name of the backend library used.

    """

    backend = "internal"

    @classmethod
    def _build_geometry(
        cls, input_geometry: str, element_types: list[str]
    ) -> tuple[list, numpy.ndarray]:
        """Build the geometry from a cfg file.

        Parameters
        ----------
        input_geometry : str
            Path to the input cfg file.
        element_types : list[str]
            List of N element symbols corresponding to N atom types in cfg file.

        Returns
        -------
        tuple
            A tuple containing:
            - list: A list of element symbols.
            - numpy.ndarray: An array of atomic coordinates.

        """
        with open(input_geometry) as f:
            lines = f.readlines()
            elements = []
            coordinates = []

            for line in lines:
                if len(line.split()) > 7 and line.split()[1].isdigit():
                    elements.append(element_types[int(line.split()[1])])
                    coordinates.append(list(map(float, line.split()[2:5])))
                elif "Energy" in line:
                    break

        return elements, numpy.array(coordinates)

    @classmethod
    def _save_cfg(cls, file_name: str, input_geometry: str, idxs: list[int]) -> None:
        """Save the geometry in cfg format.

        Parameters
        ----------
        file_name : str
            The name of the file to save the cfg data.
        input_geometry : str or None
            Path to the input geometry file.
        idxs : list[int]
            Indices of atoms.

        Returns
        -------
        None

        """
        s_idxs = list(map(lambda x: str(x + 1), idxs))
        out_lines = []
        read_atoms = False

        with open(input_geometry) as f:
            lines = f.readlines()
            num_atoms = int(lines[2])
            for i, line in enumerate(lines):
                if "ITEM: ATOMS" in line:
                    read_atoms = True
                if read_atoms:
                    start_idx = i
                    break
            for i in range(start_idx, start_idx + num_atoms):
                if lines[i].split()[0] in s_idxs:
                    out_lines.append(lines[i])
            out_lines.extend(lines[(start_idx + num_atoms) :])
            out_lines[2] = str(len(idxs)) + "\n"  # change number of atoms in .cfg file
        with open(file_name, "w") as f_out:
            f_out.writelines(out_lines)

    @classmethod
    def _save_dump(cls, file_name: str, input_geometry: str, idxs: list[int]) -> None:
        """Save the geometry in LAMMPS dump format.

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
        out_lines = []
        read_atoms = False

        with open(input_geometry) as f:
            for line in f:
                if read_atoms:
                    if line.split()[0] in s_idxs:
                        out_lines.append(line)
                else:
                    out_lines.append(line)
                if "ITEM: ATOMS" in line:
                    read_atoms = True
            out_lines[out_lines.index("ITEM: NUMBER OF ATOMS\n") + 1] = (
                str(len(idxs)) + "\n"
            )
        with open(file_name, "w") as f_out:
            f_out.writelines(out_lines)
