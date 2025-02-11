from ocellar import molecule

mol = molecule.Molecule()
mol.input_geometry = "dump_el.dump"
mol.build_geometry(
    backend="MDAnalysis"
)  # Build geometry from input file with MDAnalysis backend
mol.save_xyz("tmp.xyz")
mol.build_graph()
mol.build_structure(cut_molecule=False)  # without cutting single bonds

new_mol, idxs = mol.select(
    mol.select_r([1.09220004e0, 8.92530024e-01, 1.03007996e00], 0.1)
)  # 1. Select atom idxs with sphere center and radius 2. Build new Molecule (with hydrogenes) and new idxs from idxs

mol.save_dump(
    "out.dump", mol.input_geometry, idxs
)  # Save dump file using original dump and idxs
