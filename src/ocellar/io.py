from openbabel import openbabel
from openbabel import pybel

class Molecule:
    def __init__(self, xyz_file_name) -> None:
        self.mol = next(pybel.readfile("xyz", xyz_file_name))
        self.mol.OBMol.PerceiveBondOrders()

    def get_bond_order(self, n):
        bond = self.mol.OBMol.GetBond(n)
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        bond_order = bond.GetBondOrder()
        return f"Bond between atom {atom1} and atom {atom2} has bond order {bond_order}"