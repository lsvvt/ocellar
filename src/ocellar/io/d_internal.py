"""Module to handle molecule operations."""

import numpy

from ocellar.io.driver import Driver


class Dinternal(Driver):
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

        with open(input_geometry) as f:
            lines = f.readlines()
            lines = lines[: lines.index("END_CFG\n")]
            out_lines = [
                line
                for line in lines
                if not (
                    len(line.split()) > 7
                    and line.split()[0] not in s_idxs
                    and line.split()[1].isdigit()
                )
            ]
            out_lines[2] = str(len(idxs)) + "\n"
            with open(file_name, "w") as f_out:
                f_out.writelines(out_lines)
                f_out.write("END_CFG\n")

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

        with open(input_geometry) as f:
            out_lines = [
                line
                for line in f
                if not (
                    len(line.split()) > 5
                    and line.split()[0] not in s_idxs
                    and line.split()[2].replace(".", "", 1).isdigit()
                )
            ]
            out_lines[out_lines.index("ITEM: NUMBER OF ATOMS\n") + 1] = (
                str(len(idxs)) + "\n"
            )
            with open(file_name, "w") as f_out:
                f_out.writelines(out_lines)
