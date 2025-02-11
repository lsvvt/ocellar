from ocellar import molecule

mol = molecule.Molecule()
mol.input_geometry = "butane_cut.cfg"
mol.build_geometry(backend="internal")

mol.build_graph()
mol.build_structure(cut_molecule=False)
new_mol, idxs = mol.select(mol.select_r([44.1066, 57.6431, 52.0], 10))

mol.save_cfg(
    "out.cfg", mol.input_geometry, idxs
)  # Save dump file using original dump and idxs
new_mol.save_xyz("out.xyz")
new_mol.save_pdb("out.pdb")
