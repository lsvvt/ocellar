# ocellar
tools: pdm, ruff, wemake-python-styleguide, pflake8

## installation
use pdm

## usage
```python
from ocellar import molecule

mol = molecule.Molecule()
mol.input_geometry = "tests/data/traj.xyz"
mol.build_geometry()
mol.build_graph()
mol.build_structure()


new_mol = mol.select(mol.select_r([0, 0, 0], 10))
new_mol.build_graph()
new_mol.build_structure()

new_mol.save_xyz("out.xyz")
```