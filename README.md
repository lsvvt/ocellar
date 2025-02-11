# ocellar
tools: pdm, ruff, wemake-python-styleguide, pflake8

## installation
1) Install pdm https://pdm-project.org/en/latest
2) 
```bash
pdm install
. .venv/bin/activate
```

## usage
```python
from ocellar import molecule


# from xyz
mol = molecule.Molecule() # Create Molecule object mol
mol.input_geometry = "tests/data/traj.xyz" # Set input geometry file
mol.build_geometry() # Build geometry from input file
mol.save_xyz("tmp.xyz") # Save xyz
mol.build_graph() # Build graph from geometry
mol.build_structure() # Build structure (connected subgraphs) from graph


# from dump
mol = molecule.Molecule()
mol.input_geometry = "tests/data/dump_el.dump" # dump_el.dump_types must be present
mol.build_geometry(backend="MDAnalysis") # Build geometry from input file with MDAnalysis backend
mol.save_xyz("tmp.xyz")
mol.build_graph()
mol.build_structure(cut_molecule = False) # without cutting single bonds

new_mol, idxs = mol.select(mol.select_r([1.09220004e+0,  8.92530024e-01,  1.03007996e+00], 0.1)) # 1. Select atom idxs with sphere center and radius 2. Build new Molecule (with hydrogenes) and new idxs from idxs

mol.save_dump("out.dump", mol.input_geometry, idxs) # Save dump file using original dump and idxs

# Build and save new mol
new_mol.build_graph()
new_mol.build_structure()

new_mol.save_xyz("out.xyz")
new_mol.save_pdb("out.pdb")

#from cfg
mol = molecule.Molecule()
mol.input_geometry = "butane_cut.cfg" # butane_cut.cfg_types must be present
mol.build_geometry(backend="internal")

mol.build_graph()
mol.build_structure(cut_molecule = False)
new_mol, idxs = mol.select(mol.select_r([44.1066, 57.6431, 52.0], 10))

mol.save_cfg("out.cfg", mol.input_geometry, idxs) # Save dump file using original dump and idxs
new_mol.save_xyz("out.xyz")
new_mol.save_pdb("out.pdb")

```
