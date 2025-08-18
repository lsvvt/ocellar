"""Module to handle molecule operations."""

import numpy as np

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
    ) -> tuple[list, np.ndarray]:
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
            - np.ndarray: An array of atomic coordinates.

        """
        elements = []
        coordinates = []

        with open(input_geometry) as f:
            lines = f.readlines()

        for line in lines:
            if "Size" in line:
                n_atoms = int(next(lines).strip())
            elif "AtomData" in line:
                header = line.split(":", 1)[1].split()
                type_idx, x_idx, y_idx, z_idx = (
                    header.index(label)
                    for label in ("type", "cartes_x", "cartes_y", "cartes_z")
                )
                if n_atoms is None:
                    raise ValueError("Number of atoms record not found before header")
                for _ in range(n_atoms):
                    parts = next(lines).split()
                    atom_type = int(parts[type_idx])
                    elements.append(element_types[atom_type])
                    coordinates.append(
                        [
                            float(parts[x_idx]),
                            float(parts[y_idx]),
                            float(parts[z_idx]),
                        ]
                    )
                break

        return elements, np.array(coordinates)

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

    @classmethod
    def _save_mlipff(
        cls,
        file_name: str,
        coordinates: np.ndarray,
        idxs: list[int],
    ) -> None:
        """Save atom indices and coordinates in a plain-text MLIP-FF format.

        Each output line corresponds to one atom and contains four whitespace-separated
        fields in this exact order:
        `atom_id` `x` `y` `z`

        Parameters
        ----------
        file_name : str
            The name of the file to save the mlipff data.

            `.mlipff` extension is necessary for MLIP-FF.
        coordinates : np.ndarray
            Array of shape (N, 3) with Cartesian coordinates for the selected atoms.
        idxs : list[int]
            List of length N with the atom indices (0-based) that map row-by-row
            to `coordinates`.

        Raises
        ------
        TypeError
            If `coordinates` is not a np.ndarray.
        ValueError
            If `coordinates` is not of shape (N, 3) or if len(`indexes`) != N.

        Returns
        -------
        None

        """
        if not isinstance(coordinates, np.ndarray):
            raise TypeError("coordinates must be a np.ndarray")
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape (N, 3)")
        if len(idxs) != coordinates.shape[0]:
            raise ValueError(
                "len(idxs) must equal the number of rows in coordinates (N)"
            )

        with open(file_name, "w") as f_out:
            for idx, (x, y, z) in zip(idxs, coordinates, strict=True):
                f_out.write(f"{idx} {x:.10f} {y:.10f} {z:.10f}\n")
