from ocellar import io


class Molecule:
    """A class to represent a molecule and its properties.

    Attributes
    ----------
    mol : pybel.Molecule
        The molecule object read from an XYZ file.

    """

    def __init__(self, **kwargs) -> None:
        """Initialize the Molecule object.

        Reading an XYZ file and perceiving bond orders.

        Parameters
        ----------
        xyz_file_name : str
            The name of the XYZ file containing the molecular structure.

        """
        for key, val in kwargs.items():
            setattr(self, key, val)

        # self.mol = next(pybel.readfile("xyz", xyz_file_name))
        # self.mol.OBMol.PerceiveBondOrders()


    def build_geometry(self, backend = "cclib"):
        if self.input_geometry:
            driver = io.Driver(backend)
            self.geometry = driver._build_geometry(self.input_geometry)


    def build_bonds(self, backend = "openbabel"):
        if self.geometry:
            driver = io.Driver(backend)
            self.bonds = driver._build_bonds(self.geometry)


    def get_bond_order(self, bond_index: int) -> str | None:
        """Retrieve the bond order of a specified bond.

        Parameters
        ----------
        bond_index : int
            The index of the bond whose order is to be retrieved.

        Returns
        -------
        str | None
            A string describing the bond order between the two atoms, or
            None if the bond index is invalid.

        """
        bond = self.mol.OBMol.GetBond(bond_index)
        if bond is None:
            return None
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        bond_order = bond.GetBondOrder()
        return (
            f"Bond between atom {atom1} and atom {atom2} has bond order {bond_order}"
        )
