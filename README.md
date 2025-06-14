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

## Dependencies

The library integrates with several computational chemistry and data analysis packages:
- **scipy**: Scientific computing functions
- **openbabel-wheel**: Chemical file format handling and molecular graph construction
- **cclib**: Quantum chemistry output file parsing
- **networkx**: Graph-based molecular structure representation
- **MDAnalysis**: Molecular dynamics trajectory analysis

## Usage Examples

### Working with XYZ Files

Load molecular geometry from standard XYZ format files and perform basic operations:

```python
from ocellar import molecule

# Initialize molecule and load geometry
mol = molecule.Molecule()
mol.input_geometry = "tests/data/traj.xyz"
mol.build_geometry()

# Save processed geometry
mol.save_xyz("output.xyz")

# Build molecular graph and structure
mol.build_graph()
mol.build_structure()
```

### Processing LAMMPS Dump Files

Handle molecular dynamics simulation output with element type mapping:

```python
mol = molecule.Molecule()
mol.input_geometry = "tests/data/dump_el.dump"
mol.element_types = ["C", "H"]  # Map atom types to element symbols

# Use MDAnalysis backend for dump file processing
mol.build_geometry(backend="MDAnalysis")
mol.save_xyz("converted.xyz")

# Analyze molecular connectivity
mol.build_graph()
mol.build_structure(cut_molecule=False)  # Preserve molecules with parts located inside the sphere

# Select atoms within spatial region
center_point = [1.0922, 0.89253, 1.03008]
radius = 2
selected_indices = mol.select_r(center_point, radius)

# Extract molecular fragment with hydrogen capping
new_mol, atom_indices = mol.select(selected_indices)

# Save extracted fragment
mol.save_dump("fragment.dump", mol.input_geometry, atom_indices)
new_mol.build_graph()
new_mol.build_structure(cut_molecule=True)
new_mol.save_xyz("fragment.xyz")
new_mol.save_pdb("fragment.pdb")
```

### Working with CFG Files

Process MLIP configuration files with custom element type definitions:

```python
mol = molecule.Molecule()
mol.input_geometry = "tests/data/butane_cut.cfg"
mol.element_types = ["C", "C", "H"]  # Define element types for CFG format

# Use internal parser for CFG files
mol.build_geometry(backend="internal")
mol.build_graph()
mol.build_structure(cut_molecule=False)

# Perform spatial selection
selection_center = [44.1066, 57.6431, 52.0]
selection_radius = 10.0
selected_atoms = mol.select_r(selection_center, selection_radius)
new_mol, indices = mol.select(selected_atoms)

# Export results
mol.save_cfg("output.cfg", mol.input_geometry, indices)
new_mol.save_xyz("extracted.xyz")
new_mol.save_pdb("extracted.pdb")
```

## Backend Options

The library supports multiple backends for different operations:

- **cclib**: Default backend for standard quantum chemistry file formats (XYZ, Gaussian, etc.)
- **openbabel**: Chemical structure manipulation and PDB file handling
- **MDAnalysis**: Molecular dynamics trajectory processing
- **internal**: Custom parsers for specialized formats (CFG, LAMMPS dump)

## Development Tools

The project employs modern Python development practices:
- **pdm**: Project and dependency management
- **ruff**: Code formatting and linting
- **wemake-python-styleguide**: Code quality enforcement
- **pflake8**: Additional style checking

## File Format Support

Ocellar handles various molecular file formats commonly used in computational chemistry and materials science:
- XYZ: Standard molecular geometry format
- PDB: Protein Data Bank format
- CFG: MLIP configuration files
- LAMMPS dump: Molecular dynamics simulation output
- Various quantum chemistry output formats (via cclib)

## License

This project is distributed under the MIT License.