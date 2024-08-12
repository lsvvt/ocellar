# ocellar
tools: pdm, ruff, wemake-python-styleguide, pflake8

## installation
```bash
pdm install
```

## usage
```python
from ocellar import molecule


# from xyz
mol = molecule.Molecule()
mol.input_geometry = "tests/data/traj.xyz"
mol.build_geometry()
mol.save_xyz("tmp.xyz")
mol.build_graph()
mol.build_structure()


# from dump
mol = molecule.Molecule()
mol.input_geometry = "tests/data/dump_el.dump"
mol.build_geometry(backend="MDAnalysis")
mol.save_xyz("tmp.xyz")
mol.build_graph()
mol.build_structure()

new_mol, idxs = mol.select(mol.select_r([1.09220004e+0,  8.92530024e-01,  1.03007996e+00], 0.1)) # r

mol.save_dump("out.dump", mol.input_geometry, idxs)

new_mol.build_graph()
new_mol.build_structure()

new_mol.save_xyz("out.xyz")
new_mol.save_pdb("out.pdb")
```