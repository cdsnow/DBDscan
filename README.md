# DBDscan

Systematic scanning of DNA-binding domain (DBD) guest protein placements within scaffold crystal lattices. Given a guest protein and a scaffold crystal structure, DBDscan enumerates every possible sliding-window register along the scaffold DNA, superimposes the guest via C1' atoms, and classifies each placement by steric compatibility with the crystal environment.

## Overview

Protein crystals built from DNA-binding scaffold proteins contain continuous dsDNA helices that thread through the lattice. A "guest" DBD protein can potentially be placed at any register along this DNA. DBDscan answers the question: **which registers are sterically feasible, and how much room does the guest have?**

The pipeline:

1. **Enumerates** all sliding-window registers (forward and reverse orientations)
2. **Builds** the local crystal environment (ASU + symmetry mates with coaxially stacking DNA)
3. **Categorizes** each placement by clash type (scaffold protein, symmetry mate DNA, guest symmetry copies)
4. **Ranks** non-clashing registers by nearest-neighbor distance
5. **Generates** SVG diagrams and PyMOL visualization scripts

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| A | `A_process_guest_models.py` | Extract relevant chains from guest PDB files into `processed_guest_models/` |
| B | `B_find_registers.py` | Superimpose guest onto every DNA sliding window via C1' atoms; write aligned PDBs |
| C | `C_find_symmetry_mates.py` | Find symmetry mates whose DNA coaxially stacks with the scaffold DNA |
| D | `D_categorize_registers.py` | Classify each register (clash with scaffold protein, symmetry mate DNA, guest copies, or clear) |
| E | `E_rank_registers.py` | Score and rank registers; produce ranked CSV, PyMOL scripts, and AlphaFold 3 input JSONs |
| F | `F_register_diagram.py` | Generate SVG register diagram colored by nearest-neighbor distance |

## Quick start

```bash
# Install dependencies
pip install numpy scipy networkx colorama

# Run the full pipeline for all guests in all scaffolds
bash run_all.sh

# Or run a single guest/scaffold combination
bash all_steps_1jgg_in_9YZJ.sh
```

## Input data

### Scaffold models (`scaffold_models/`)

Each scaffold has a PDB file and a JSON metadata file:

| Scaffold | Space group | DNA length | Description |
|----------|-------------|------------|-------------|
| 9YZJ | C 1 2 1 | 31 bp | RepE54 scaffold, monoclinic |
| 9YZK | I 1 2 1 | 42 bp | RepE54 scaffold, expanded lattice |

### Guest proteins (`guest_models/`)

| Guest | Protein | DBD type | DNA window |
|-------|---------|----------|------------|
| 1jgg | Engrailed homeodomain | HTH | 7 bp |
| 1ysa | EVE homeodomain | HTH | 7 bp |
| 1b8i | EVE homeodomain (alt) | HTH | 7 bp |
| 3hdd | Engrailed homeodomain (alt) | HTH | 7 bp |
| 7dta | CREB bZIP | bZIP | 10 bp |
| 4xid | GCN4 bZIP | bZIP | 10 bp |

## Outputs

All outputs are written to `output/{guest}.{scaffold}/` and are fully regenerable from source.

| File | Description |
|------|-------------|
| `*_ranked.csv` | Ranked register table with distances, classifications, and PyMOL commands |
| `*_ranked.pml` | PyMOL script for inspecting ranked placements in the crystal environment |
| `*_register_diagram.svg` | SVG diagram showing non-clashing registers colored by min_dist (cividis) |
| `*_categories.json` | Classification metadata and scaffold protein footprint |
| `*_ranked_sequences.seqs` | DNA sequences for each register placement |
| `af3_input_jsons/` | AlphaFold 3 input JSON files for non-clashing registers |
| `*.pdb` | Aligned guest models and crystal environment PDB files |

### Register diagram

The SVG diagram shows each non-clashing register as a colored bar positioned along the scaffold DNA sequence. Colors use the cividis colormap: blue = largest min_dist (most room), yellow = smallest. Bars overlapping the scaffold protein's own DNA-binding footprint are highlighted with a magenta border.

## Classification system

Each register placement is classified by what it clashes with:

| Category | Label | Meaning |
|----------|-------|---------|
| 1 | `CLASH_SCAFF_PROT` | Guest protein clashes with scaffold protein |
| 2 | `OVERLAP_CORE` | Guest DNA footprint overlaps scaffold protein's binding site |
| 3 | `CLASH_SYM` | Guest clashes with symmetry mate DNA or guest symmetry copy |
| 4 | `NEAR` | No clash, but neighbors within 8 A |
| 5 | `INDEPENDENT` | All neighbors > 8 A away |

## Project structure

```
DBDscan/
  A_process_guest_models.py    # Step A: prepare guest PDBs
  B_find_registers.py          # Step B: enumerate registers
  C_find_symmetry_mates.py     # Step C: find coaxial symmetry mates
  D_categorize_registers.py    # Step D: classify registers
  E_rank_registers.py          # Step E: rank and score
  F_register_diagram.py        # Step F: SVG diagram
  run_all.sh                   # Run all 12 guest/scaffold combinations
  all_steps_*_in_*.sh          # Individual pipeline scripts
  guest_models/                # Input guest PDB files
  scaffold_models/             # Scaffold PDBs + JSON metadata
  pyscaffoldscan/              # Core library modules
    pdbtools.py                #   PDB file parsing and manipulation
    superimpy.py               #   Coordinate superposition
    xtal.py                    #   Crystallographic operations
    sgdata.py                  #   Space group symmetry data
    pyquat.py                  #   Quaternion math
    dna_matching.py            #   DNA sequence/structure matching
    find_minima.py             #   RMSD landscape analysis
```

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- NetworkX
- Colorama

## License

MIT License
