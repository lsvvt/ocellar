# Ocellar

A Python library for extracting spherical molecular environments with respect to topology from different formats: .xyz, LAMMPS .dump, MLIP .cfg.

## Installation

The project uses PDM for dependency management. Follow these steps to set up your development environment:

1. Install PDM from the official documentation at https://pdm-project.org/en/latest
2. Set up the project environment:
```bash
pdm install
. .venv/bin/activate
```

## Usage Examples

### Working with .xyz Files

Load molecular geometry from standard XYZ format files and perform basic operations:

```python
from ocellar import molecule

# Initialize molecule and load geometry
mol = molecule.Molecule("tests/data/traj.xyz")
mol.build_geometry()

# Save processed geometry
mol.save_xyz("output.xyz")

# Build molecular graph and structure
mol.build_graph()
mol.build_structure()
```

### Working with LAMMPS .dump files

Handle molecular dynamics simulation output with element type mapping:

```python
mol = molecule.Molecule("tests/data/dump_el.dump", ["C", "H"])

# Use ovito backend for dump file processing
mol.build_geometry(backend="ovito")
mol.save_xyz("initial_geometry.xyz")

# Build molecular connectivity
mol.build_graph()
mol.build_structure(cut_molecule=False)  # Preserve molecules with parts located inside the sphere

# Choose center point and radius for selection
center = [1.0922, 0.89253, 1.03008]
radius = 2

# Extract molecular fragment
new_mol, atom_indices = mol.select(mol.select_r(center, radius))
# Or extract molecular fragment with respect to functional groups and infer charge
new_mol, atom_indices = mol.select_after_expand(mol.expand_selection(mol.select_r(center, radius)))

# Save extracted fragment
mol.save_dump("fragment.dump", mol.input_geometry, atom_indices)
new_mol.save_xyz("fragment.xyz")
new_mol.save_pdb("fragment.pdb")
```

### Working with MLIP .cfg Files

Process MLIP configuration files with custom element type definitions:

```python
mol = molecule.Molecule("tests/data/butane_cut.cfg", ["C", "C", "H"])

# Use internal parser for CFG files
mol.build_geometry(backend="internal")
mol.build_graph()
mol.build_structure(cut_molecule=True) # passivate loose ends with hydrogens

# Choose center point and radius for selection
center = [44.1066, 57.6431, 52.0]
radius = 5.0
# Extract molecular fragment
new_mol, atom_indices = mol.select(mol.select_r(center, radius))
# Or extract molecular fragment with respect to functional groups and infer charge
new_mol, atom_indices = mol.select_after_expand(mol.expand_selection(mol.select_r(center, radius)))

# Export results
mol.save_cfg("output.cfg", mol.input_geometry, atom_indices)
new_mol.save_xyz("extracted.xyz")
new_mol.save_pdb("extracted.pdb")
```

### Working with .dump or .cfg files with respect to PBC

```python
mol = molecule.Molecule("tests/data/butane_cut.cfg", ["C", "C", "H"])

# Use internal parser for CFG files
mol.build_geometry(backend="internal")
# provide cell in ovito format if NOT present in file
mol.cell = np.asarray([[60, 0, 0, 0], [0, 60, 0, 0], [0, 0, 60, 0]], dtype=float)
mol.replicate_geometry() # make 3x3x3 replication of provided structure
mol.build_graph()
mol.build_structure(cut_molecule=False)

# Choose center point and radius for selection
center = [44.1066, 57.6431, 52.0]
radius = 5.0
# Extract molecular fragment
new_mol, atom_indices = mol.select(mol.select_r(center, radius))
# Or extract molecular fragment with respect to functional groups and infer charge
new_mol, atom_indices = mol.select_after_expand(mol.expand_selection(mol.select_r(center, radius)))

# Export results
mol.save_cfg("output.cfg", mol.input_geometry, atom_indices)
new_mol.save_xyz("extracted.xyz")
new_mol.save_pdb("extracted.pdb")
```

## Backend Options

The library supports multiple backends for different operations:

- **cclib**: Default backend for standard quantum chemistry file formats (XYZ)
- **openbabel**: Chemical structure manipulation and PDB file handling
- **MDAnalysis**: Molecular dynamics trajectory processing
- **ovito**: Molecular dynamics trajectory processing (advised over MDAnalysis)
- **internal**: Custom parsers for specialized formats (CFG, LAMMPS dump)

## File Format Support

Ocellar handles various molecular file formats commonly used in computational chemistry and materials science:
- XYZ: Standard molecular geometry format
- PDB: Protein Data Bank format
- CFG: MLIP configuration files
- LAMMPS dump: Molecular dynamics simulation output

## License

This project is distributed under the MIT License.