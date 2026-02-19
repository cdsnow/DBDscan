#!/usr/bin/env python3
"""D_categorize_registers.py

Phase 2: Categorize each guest register placement into one of five categories
based on steric clashes with scaffold protein, neighboring DNA, and
symmetry copies of the guest itself.

  Cat 1: Guest protein clashes with scaffold protein (any symmetry copy)
  Cat 2: No protein clash, but guest DNA footprint overlaps scaffold core binding site
  Cat 3: Guest protein clashes with neighboring DNA or guest symmetry copy
  Cat 4: No clash, but guest is within interaction distance (< 8 A)
  Cat 5: Independent (all neighbors >= 8 A)

Contact types tracked per register:
  SPR = nearest scaffold protein atom (from ASU or any symmetry copy)
  SYM = nearest DNA atom from non-coaxial symmetry copies
        (ASU DNA and coaxial/flanking stacked DNA are excluded)
  GSY = nearest guest symmetry copy protein atom

Each register PDB is augmented with:
  - Chain S: nearest guest symmetry copy (protein only)
  - Chain X, element D: dummy contact-point atoms for visualization

Usage:
    python D_categorize_registers.py [guest_code]
    python D_categorize_registers.py 1jgg
"""
import json, sys, os, itertools, time
import numpy as np
from numpy import matrix, array
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pyscaffoldscan'))
from pdbtools import PDB
from sgdata import sgqt

# ── Tunable parameters ────────────────────────────────────────────────────
CLASH_CUTOFF    = 2.0    # A: below this = steric clash
INTERACT_CUTOFF = 8.0    # A: below this = interacting neighbor
ENV_CUTOFF      = 50.0   # A: max atom-to-atom distance to include a symmetry copy

# Core binding sequence of the scaffold protein (on the top strand)
CORE_SEQ = 'TGTGACAAATTGCCCTCAG'

CAT_NAMES = {
    1: 'clash_scaff_prot',
    2: 'footprint_overlap',
    3: 'clash_sym_mate',
    4: 'near_neighbors',
    5: 'independent',
}
CAT_LABELS = {
    1: 'CLASH-PROT',
    2: 'FOOTPRINT',
    3: 'CLASH-SYM',
    4: 'neighbors',
    5: 'indep',
}

# ── Helper functions ──────────────────────────────────────────────────────
def identify_dna_chains(P):
    """Identify which chains in a PDB are DNA based on residue names."""
    dna_resnames = {'DA', 'DT', 'DC', 'DG'}
    dna_chains = []
    for chain in sorted(P.resndict.keys()):
        resnames = {rn.strip() for rn in P.resndict[chain].values()}
        if resnames <= dna_resnames:
            dna_chains.append(chain)
    return dna_chains

def identify_protein_chains(P):
    """Identify protein chains (everything that is not DNA)."""
    dna = set(identify_dna_chains(P))
    return [ch for ch in sorted(P.resndict.keys()) if ch not in dna]

def select_chains(P, chains):
    """Select atoms from specified chains."""
    expr = ' or '.join(f'chain {ch}' for ch in chains)
    return P[expr]

def env_obj_name(sym_idx, cell):
    """Create a clean PyMOL object name for a symmetry copy."""
    a, b, c = cell
    def fmt(v):
        return f"m{abs(v)}" if v < 0 else str(v)
    return f"env_s{sym_idx}_{fmt(a)}_{fmt(b)}_{fmt(c)}"

def format_dummy_atom(serial, atom_name, res_name, chain, res_seq,
                      x, y, z, bfactor):
    """Format a dummy HETATM record with element D."""
    return (f"HETATM{serial:5d} {atom_name:>4s} {res_name:>3s} {chain}"
            f"{res_seq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfactor:6.2f}           D  ")

# ── Generate crystal environment ─────────────────────────────────────────
def generate_crystal_environment(scaffold, prot_coords_asu, dna_coords_asu,
                                 coaxial_keys):
    """Generate all symmetry copies within ENV_CUTOFF of the ASU.

    Stores per-copy: prot_coords, dna_coords, and pre-built KD-trees.
    Coaxial mates are tagged so their DNA is excluded from SYM checks.
    """
    scaffold.xtal_qtlist = sgqt[scaffold.xtal_sg]

    asu_coords = scaffold.GetCoords()
    asu_center = asu_coords.mean(axis=0)
    asu_radius = np.linalg.norm(asu_coords - asu_center, axis=1).max()
    cog_cutoff = ENV_CUTOFF + 2 * asu_radius

    asu_tree = cKDTree(asu_coords)

    print(f"  ASU: {len(asu_coords)} atoms, center=[{asu_center[0]:.1f}, "
          f"{asu_center[1]:.1f}, {asu_center[2]:.1f}], radius={asu_radius:.1f} A")
    print(f"  Protein: {len(prot_coords_asu)} atoms, "
          f"DNA: {len(dna_coords_asu)} atoms")
    print(f"  {len(scaffold.xtal_qtlist)} symmetry operations, "
          f"searching 27 unit cells each ...")
    print(f"  Coaxial mates: {coaxial_keys}")

    env_copies = []
    for sym_idx, m in enumerate(scaffold.xtal_qtlist):
        for a, b, c in itertools.product([-1, 0, 1], repeat=3):
            mat4 = matrix([[m[0],m[1],m[2], m[9]+a],
                           [m[3],m[4],m[5], m[10]+b],
                           [m[6],m[7],m[8], m[11]+c],
                           [0,    0,   0,   1      ]])
            realmat = scaffold.xtal_basis * mat4 * scaffold.xtal_basis.I
            R = np.array(realmat[:3, :3])
            T = np.array(realmat[:3, 3]).flatten()

            copy_center = R @ asu_center + T
            d_cog = np.linalg.norm(copy_center - asu_center)

            if d_cog < 0.1:
                continue
            if d_cog > cog_cutoff:
                continue

            copy_coords = (asu_coords @ R.T) + T
            d_to_asu, _ = asu_tree.query(copy_coords, k=1)
            min_dist = float(d_to_asu.min())
            if min_dist > ENV_CUTOFF:
                continue

            copy_prot = (prot_coords_asu @ R.T) + T
            copy_dna  = (dna_coords_asu @ R.T) + T

            key = (sym_idx, (a, b, c))
            is_coaxial = key in coaxial_keys

            obj = env_obj_name(sym_idx, (a, b, c))
            env_copies.append({
                'all_coords':  copy_coords,
                'prot_coords': copy_prot,
                'dna_coords':  copy_dna,
                'R':           R,
                'T':           T,
                'sym_idx':     sym_idx,
                'cell':        (a, b, c),
                'label':       f"sym{sym_idx}_cell({a},{b},{c})",
                'obj_name':    obj,
                'is_coaxial':  is_coaxial,
                'd_cog':       round(float(d_cog), 1),
                'd_min':       round(min_dist, 1),
            })

    env_copies.sort(key=lambda x: x['d_min'])

    # Pre-build KD-trees (reused for every register)
    for copy in env_copies:
        copy['prot_tree'] = (cKDTree(copy['prot_coords'])
                             if len(copy['prot_coords']) > 0 else None)
        if not copy['is_coaxial'] and len(copy['dna_coords']) > 0:
            copy['dna_tree'] = cKDTree(copy['dna_coords'])
        else:
            copy['dna_tree'] = None

    print(f"  Found {len(env_copies)} symmetry copies within {ENV_CUTOFF} A")
    for i, ec in enumerate(env_copies):
        tag = " [coaxial]" if ec['is_coaxial'] else ""
        print(f"    [{i:>2}] {ec['label']:<25s}  d_cog={ec['d_cog']:>6.1f}  "
              f"d_min={ec['d_min']:>5.1f} A  ({ec['obj_name']}){tag}")

    return env_copies

# ── Classify one register ────────────────────────────────────────────────
def classify_register(reg, scaff_prot_tree, scaff_prot_coords, env_copies,
                      core_start, core_end):
    """Classify a single register placement.

    Computes three independent contact distances:
      SPR = nearest scaffold protein (ASU + all env copies)
      SYM = nearest non-coaxial DNA
      GSY = nearest guest symmetry copy protein
    """
    guest = PDB(reg['pdb_file'])
    guest_prot = identify_protein_chains(guest)
    guest_prot_coords = select_chains(guest, guest_prot).GetCoords()

    # ── SPR: nearest scaffold protein from any copy ──
    min_dist_prot = 999.0
    spr_guest_pos = None
    spr_env_pos = None
    spr_source = None

    # Check ASU protein
    d_asu, idx_asu = scaff_prot_tree.query(guest_prot_coords, k=1)
    d_min_asu = float(d_asu.min())
    if d_min_asu < min_dist_prot:
        min_dist_prot = d_min_asu
        i_g = int(np.argmin(d_asu))
        i_e = int(idx_asu[i_g])
        spr_guest_pos = guest_prot_coords[i_g].round(2).tolist()
        spr_env_pos   = scaff_prot_coords[i_e].round(2).tolist()
        spr_source = 'ASU'

    # Check each env copy's protein
    for ci, copy in enumerate(env_copies):
        pt = copy['prot_tree']
        if pt is None:
            continue
        d, idx = pt.query(guest_prot_coords, k=1)
        d_min = float(d.min())
        if d_min < min_dist_prot:
            min_dist_prot = d_min
            i_g = int(np.argmin(d))
            i_e = int(idx[i_g])
            spr_guest_pos = guest_prot_coords[i_g].round(2).tolist()
            spr_env_pos   = copy['prot_coords'][i_e].round(2).tolist()
            spr_source = copy['obj_name']

    contact_spr = {
        'dist': round(min_dist_prot, 2),
        'guest_pos': spr_guest_pos,
        'env_pos':   spr_env_pos,
        'source':    spr_source,
    }

    # ── SYM: nearest DNA from non-coaxial symmetry copies ──
    min_dist_dna = 999.0
    sym_guest_pos = None
    sym_env_pos = None
    sym_source = None

    for ci, copy in enumerate(env_copies):
        dt = copy['dna_tree']
        if dt is None:
            continue  # skip coaxial (and any with no DNA)
        d, idx = dt.query(guest_prot_coords, k=1)
        d_min = float(d.min())
        if d_min < min_dist_dna:
            min_dist_dna = d_min
            i_g = int(np.argmin(d))
            i_e = int(idx[i_g])
            sym_guest_pos = guest_prot_coords[i_g].round(2).tolist()
            sym_env_pos   = copy['dna_coords'][i_e].round(2).tolist()
            sym_source = copy['obj_name']

    contact_sym = {
        'dist': round(min_dist_dna, 2),
        'guest_pos': sym_guest_pos,
        'env_pos':   sym_env_pos,
        'source':    sym_source,
    }

    # ── GSY: nearest guest symmetry copy protein ──
    min_dist_gsym = 999.0
    closest_gsym_idx = -1
    gsym_guest_pos = None
    gsym_env_pos = None
    guest_prot_tree = cKDTree(guest_prot_coords)
    for ci, copy in enumerate(env_copies):
        R, T_vec = copy['R'], copy['T']
        gsym_coords = (guest_prot_coords @ R.T) + T_vec
        d, idx = guest_prot_tree.query(gsym_coords, k=1)
        d_min = float(d.min())
        if d_min < min_dist_gsym:
            min_dist_gsym = d_min
            closest_gsym_idx = ci
            i_sym = int(np.argmin(d))
            i_self = int(idx[i_sym])
            gsym_guest_pos = guest_prot_coords[i_self].round(2).tolist()
            gsym_env_pos   = gsym_coords[i_sym].round(2).tolist()

    contact_gsym = {
        'dist':     round(min_dist_gsym, 2),
        'guest_pos': gsym_guest_pos,
        'env_pos':   gsym_env_pos,
        'source':    (env_copies[closest_gsym_idx]['obj_name']
                      if closest_gsym_idx >= 0 else None),
        'copy_idx':  closest_gsym_idx,
    }

    # ── Combined distances ──
    min_dist_sym_all = min(min_dist_dna, min_dist_gsym)
    min_dist_any = min(min_dist_prot, min_dist_sym_all)

    # ── Footprint overlap ──
    s, e = reg['start_pos'], reg['end_pos']
    footprint_overlaps = (s <= core_end and e >= core_start)

    # ── Category assignment ──
    if min_dist_prot < CLASH_CUTOFF:
        category = 1
    elif footprint_overlaps:
        category = 2
    elif min_dist_sym_all < CLASH_CUTOFF:
        category = 3
    elif min_dist_any < INTERACT_CUTOFF:
        category = 4
    else:
        category = 5

    return {
        'register':          reg['register'],
        'label':             reg['label'],
        'obj_name':          reg['obj_name'],
        'orientation':       reg['orientation'],
        'start_pos':         reg['start_pos'],
        'end_pos':           reg['end_pos'],
        'region':            reg['region'],
        'rmsd':              reg['rmsd'],
        'pdb_file':          reg['pdb_file'],
        'category':          category,
        'category_name':     CAT_NAMES[category],
        'footprint_overlaps': footprint_overlaps,
        'min_dist_prot':     round(min_dist_prot, 2),
        'min_dist_sym_dna':  round(min_dist_dna, 2),
        'min_dist_guest_sym': round(min_dist_gsym, 2),
        'min_dist_any':      round(min_dist_any, 2),
        'closest_prot':      spr_source,
        'guest_prot_chains': guest_prot,
        'contacts': {
            'asu_prot':  contact_spr,
            'sym_copy':  contact_sym,
            'guest_sym': contact_gsym,
        },
    }

# ── PDB augmentation helpers ─────────────────────────────────────────────
AUGMENT_MARKER = 'REMARK 999 D_CATEGORIZE AUGMENTATION BELOW'

def strip_augmentation(pdb_path):
    """Remove any previous D_categorize augmentation from a PDB file."""
    lines = []
    with open(pdb_path) as f:
        for line in f:
            if AUGMENT_MARKER in line:
                break
            lines.append(line)
    with open(pdb_path, 'w') as f:
        f.writelines(lines)

def augment_register_pdb(result, env_copies):
    """Append nearest guest sym copy (chain S) and dummy atoms (elem D)."""
    pdb_path = result['pdb_file']
    contacts = result['contacts']
    guest_prot_chains = set(result['guest_prot_chains'])

    # Read original content (strip any previous augmentation)
    original = []
    with open(pdb_path) as f:
        for line in f:
            if AUGMENT_MARKER in line:
                break
            original.append(line.rstrip('\n'))

    while original and original[-1].strip() in ('END', 'TER', ''):
        original.pop()

    # Collect protein ATOM lines (for sym copy generation)
    prot_atom_lines = [l for l in original
                       if l.startswith('ATOM') and l[21] in guest_prot_chains]

    out = list(original)
    out.append(AUGMENT_MARKER)

    # ── Nearest guest symmetry copy (protein, chain S+) ──
    gsym = contacts['guest_sym']
    if gsym['copy_idx'] >= 0:
        copy = env_copies[gsym['copy_idx']]
        R, T_vec = copy['R'], copy['T']

        chain_map = {}
        next_ch = ord('S')
        for ch in sorted(guest_prot_chains):
            chain_map[ch] = chr(next_ch)
            next_ch += 1

        out.append('TER')
        for pline in prot_atom_lines:
            old_ch = pline[21]
            new_ch = chain_map[old_ch]
            x = float(pline[30:38])
            y = float(pline[38:46])
            z = float(pline[46:54])
            nc = np.array([x, y, z]) @ R.T + T_vec
            out.append(f"{pline[:21]}{new_ch}{pline[22:30]}"
                       f"{nc[0]:8.3f}{nc[1]:8.3f}{nc[2]:8.3f}"
                       f"{pline[54:]}")

    # ── Dummy contact atoms (element D, chain X) ──
    out.append('TER')
    serial = 9990
    for resseq, resn, contact in [
        (1, 'SPR', contacts['asu_prot']),
        (2, 'SYM', contacts['sym_copy']),
        (3, 'GSY', contacts['guest_sym']),
    ]:
        if contact.get('guest_pos') is None:
            continue
        gx, gy, gz = contact['guest_pos']
        ex, ey, ez = contact['env_pos']
        dist = contact['dist']
        serial += 1
        out.append(format_dummy_atom(serial, 'G', resn, 'X', resseq,
                                     gx, gy, gz, dist))
        serial += 1
        out.append(format_dummy_atom(serial, 'E', resn, 'X', resseq,
                                     ex, ey, ez, dist))

    out.append('END')
    with open(pdb_path, 'w') as f:
        f.write('\n'.join(out) + '\n')

# ── Main ──────────────────────────────────────────────────────────────────
guest_code = sys.argv[1] if len(sys.argv) > 1 else '1jgg'
scaffold_code = sys.argv[2] if len(sys.argv) > 2 else '9YZJ'

# Load scaffold
scaffold_json = f'scaffold_models/{scaffold_code}.json'
with open(scaffold_json) as f:
    scaff = json.load(f)

scaffold = PDB(f"scaffold_models/{scaff['pdb_id']}.pdb")
sch1, sch2 = scaff['dna_chains']

# Identify scaffold protein chain(s)
scaff_prot_chains = [ch for ch, info in scaff['chains'].items()
                     if info['type'] == 'protein']
print(f"Scaffold {scaff['pdb_id']}: protein chain(s) {scaff_prot_chains}, "
      f"DNA chains {[sch1, sch2]}")

# Extract scaffold protein + DNA coords
scaff_prot_coords = select_chains(scaffold, scaff_prot_chains).GetCoords()
scaff_dna_coords  = select_chains(scaffold, [sch1, sch2]).GetCoords()
scaff_prot_tree = cKDTree(scaff_prot_coords)
print(f"  Scaffold protein: {len(scaff_prot_coords)} atoms, "
      f"DNA: {len(scaff_dna_coords)} atoms")

# Load register data
outdir = f'output/{guest_code}.{scaffold_code}'
reg_json = f'{outdir}/{guest_code}_registers.json'
with open(reg_json) as f:
    reg_data = json.load(f)

top_chain = reg_data.get('top_chain', sch1)

# Find core binding footprint on the top chain
top_seq = scaff['chains'][top_chain]['sequence']
core_idx = top_seq.find(CORE_SEQ)
assert core_idx >= 0, f"Core sequence {CORE_SEQ} not found in top chain {top_chain}"
core_start = core_idx + 1
core_end = core_start + len(CORE_SEQ) - 1
print(f"  Core binding footprint: {CORE_SEQ}")
print(f"    Top chain {top_chain} positions {core_start}-{core_end}")

# Load symmetry mates metadata (coaxial stacking partners)
mates_json = f'symmetry_mates/{scaffold_code}_mates.json'
have_mates = os.path.isfile(mates_json)
coaxial_keys = set()
if have_mates:
    with open(mates_json) as f:
        mates = json.load(f)
    for side in ('top', 'bot'):
        m = mates['mates'][side]
        coaxial_keys.add((m['sym_op'], tuple(m['unit_cell'])))

# Generate crystal environment
print(f"\nGenerating crystal environment (cutoff={ENV_CUTOFF} A) ...")
env_copies = generate_crystal_environment(scaffold, scaff_prot_coords,
                                          scaff_dna_coords, coaxial_keys)

num_regs = len(reg_data['registers'])

# Strip any previous augmentation (chain S, elem D) so classify sees clean PDBs
print(f"\nStripping previous augmentation from {num_regs} PDB files ...")
for reg in reg_data['registers']:
    strip_augmentation(reg['pdb_file'])

print(f"Classifying {num_regs} registers for {guest_code} "
      f"(clash={CLASH_CUTOFF} A, interact={INTERACT_CUTOFF} A) ...\n")

# Classify each register
results = []
t0 = time.time()
for i, reg in enumerate(reg_data['registers']):
    result = classify_register(reg, scaff_prot_tree, scaff_prot_coords,
                               env_copies, core_start, core_end)
    results.append(result)
    cat = result['category']
    src = result['closest_prot']
    print(f"  [{i+1:>3}/{num_regs}]  {reg['obj_name']:<20s}  "
          f"d_prot={result['min_dist_prot']:6.1f} ({src:<16s})  "
          f"d_dna={result['min_dist_sym_dna']:6.1f}  "
          f"d_gsym={result['min_dist_guest_sym']:6.1f}  "
          f"Cat {cat} ({CAT_LABELS[cat]})")

elapsed = time.time() - t0
print(f"\n  Classified {num_regs} registers in {elapsed:.1f}s")

# ── Summary ───────────────────────────────────────────────────────────────
cat_counts = {c: 0 for c in CAT_NAMES}
for r in results:
    cat_counts[r['category']] += 1

print(f"\n{'='*75}")
print(f"  Summary for {guest_code} on {scaff['pdb_id']}")
print(f"{'='*75}")
for c in sorted(CAT_NAMES):
    print(f"  Cat {c} ({CAT_NAMES[c]:<20s}):  {cat_counts[c]:>3}")
print(f"  {'':>28s}Total: {num_regs:>3}")

# ── Augment register PDB files ──────────────────────────────────────────
print(f"\nAugmenting register PDB files ...")
for r in results:
    augment_register_pdb(r, env_copies)
print(f"  Augmented {num_regs} PDB files (chain S = nearest guest sym copy, "
      f"elem D = contact points)")

# ── Write symmetry environment PDBs ─────────────────────────────────────
env_pdb_paths = {}
for ci, copy in enumerate(env_copies):
    pdb_path = f'{outdir}/{copy["obj_name"]}.pdb'
    cp = scaffold.Clone()
    cp.Rotate(copy['R'].T)
    cp.Translate(copy['T'])
    cp.WritePDB(pdb_path)
    env_pdb_paths[ci] = pdb_path
print(f"  Wrote {len(env_copies)} symmetry copy PDBs to {outdir}/")

# ── Save JSON ─────────────────────────────────────────────────────────────
json_path = f'{outdir}/{guest_code}_categories.json'

json_results = []
for r in results:
    jr = {k: v for k, v in r.items() if k != 'guest_prot_chains'}
    json_results.append(jr)

output_json = {
    'guest': guest_code,
    'scaffold': scaff['pdb_id'],
    'top_chain': top_chain,
    'core_footprint': {
        'sequence': CORE_SEQ,
        'start': core_start,
        'end': core_end,
    },
    'parameters': {
        'clash_cutoff': CLASH_CUTOFF,
        'interact_cutoff': INTERACT_CUTOFF,
        'env_cutoff': ENV_CUTOFF,
    },
    'num_env_copies': len(env_copies),
    'env_copies': [
        {'obj_name': c['obj_name'], 'label': c['label'],
         'is_coaxial': c['is_coaxial'], 'pdb_file': env_pdb_paths[i]}
        for i, c in enumerate(env_copies)
    ],
    'summary': {CAT_NAMES[c]: cat_counts[c] for c in sorted(CAT_NAMES)},
    'registers': json_results,
}
with open(json_path, 'w') as f:
    json.dump(output_json, f, indent=2)
print(f"  JSON -> {json_path}")

# ── Write PyMOL script ───────────────────────────────────────────────────
scaffold_pdb_path = f"scaffold_models/{scaff['pdb_id']}.pdb"
pml_path = f'{outdir}/{guest_code}_categories.pml'

CAT_COLORS = {
    1: 'red',
    2: 'salmon',
    3: 'yellow',
    4: 'lightteal',
    5: 'palegreen',
}

with open(pml_path, 'w') as pml:
    pml.write(f"# Category visualization for {guest_code} on {scaff['pdb_id']}\n")
    pml.write(f"# Generated by D_categorize_registers.py\n")
    pml.write(f"# Register numbering: positions along top chain ({top_chain})\n")
    pml.write(f"# Core footprint: {CORE_SEQ} (pos {core_start}-{core_end})\n")
    pml.write(f"#\n")
    pml.write(f"# Dummy contact atoms (elem D) in each register PDB:\n")
    pml.write(f"#   resn SPR = nearest scaffold protein (any copy)\n")
    pml.write(f"#   resn SYM = nearest non-coaxial DNA\n")
    pml.write(f"#   resn GSY = nearest guest symmetry copy protein\n")
    pml.write(f"#   atom G = guest-side, atom E = environment-side\n")
    pml.write(f"#   B-factor = contact distance (A)\n\n")

    # Scaffold + coaxial mates
    pml.write("# ── Scaffold ──────────────────────────────────────────\n")
    pml.write(f"load {scaffold_pdb_path}, scaffold\n")
    if have_mates:
        pml.write(f"load {mates['mates']['bot']['pdb_file']}, mate_bot\n")
        pml.write(f"load {mates['mates']['top']['pdb_file']}, mate_top\n")
    pml.write("\n")

    # Symmetry environment PDBs
    pml.write("# ── Crystal environment ───────────────────────────────\n")
    for ci, copy in enumerate(env_copies):
        pml.write(f"load {env_pdb_paths[ci]}, {copy['obj_name']}\n")
    pml.write("\n")

    # Aligned guests
    pml.write("# ── Aligned guests ────────────────────────────────────\n")
    for r in results:
        pml.write(f"load {r['pdb_file']}, {r['obj_name']}\n")
    pml.write("\n")

    # Groups
    pml.write("# ── Groups ────────────────────────────────────────────\n")
    scaff_members = "scaffold"
    if have_mates:
        scaff_members += " mate_bot mate_top"
    pml.write(f"group scaffold_grp, {scaff_members}\n")

    env_objs = ' '.join(c['obj_name'] for c in env_copies)
    pml.write(f"group sym_environment, {env_objs}\n")

    for cat in sorted(CAT_NAMES):
        objs = [r['obj_name'] for r in results if r['category'] == cat]
        if objs:
            pml.write(f"group {CAT_NAMES[cat]}, {' '.join(objs)}\n")
    pml.write("\n")

    # Appearance
    pml.write("# ── Appearance ────────────────────────────────────────\n")
    pml.write("set orthoscopic, 1\n")
    pml.write("hide everything\nshow cartoon\n\n")

    pml.write(f"select scaff_dna, scaffold and chain {sch1} "
              f"or scaffold and chain {sch2}\n")
    pml.write("select scaff_prot, scaffold and not scaff_dna\n")
    pml.write("show sticks, scaff_dna\n")
    pml.write("color gray80, scaff_prot\n")
    pml.write("color palecyan, scaff_dna\n")

    if have_mates:
        pml.write(f"\nselect mate_dna, (mate_bot or mate_top) and chain {sch1}"
                  f" or (mate_bot or mate_top) and chain {sch2}\n")
        pml.write("select mate_prot, (mate_bot or mate_top) and not mate_dna\n")
        pml.write("show sticks, mate_dna\n")
        pml.write("color gray60, mate_prot\n")
        pml.write("color lightblue, mate_dna\n")
    pml.write("\n")

    # Environment copies
    for ci, copy in enumerate(env_copies):
        obj = copy['obj_name']
        pml.write(f"select _{obj}_dna, {obj} and chain {sch1} "
                  f"or {obj} and chain {sch2}\n")
        pml.write(f"select _{obj}_prot, {obj} and not _{obj}_dna\n")
        pml.write(f"show sticks, _{obj}_dna\n")
        pml.write(f"color gray60, _{obj}_prot\n")
        pml.write(f"color lightblue, _{obj}_dna\n")
        pml.write(f"delete _{obj}_dna\ndelete _{obj}_prot\n")
    pml.write("\n")

    # Color guests by category; guest sym copies in cyan; dummy atoms magenta
    pml.write("# Color by category\n")
    for cat in sorted(CAT_NAMES):
        if cat_counts[cat] > 0:
            pml.write(f"color {CAT_COLORS[cat]}, {CAT_NAMES[cat]}\n")
    pml.write("color cyan, chain S\n")
    pml.write("color magenta, elem D\n\n")

    # Disable cluttered groups by default
    pml.write("disable clash_scaff_prot\n")
    pml.write("disable footprint_overlap\n")
    pml.write("disable clash_sym_mate\n")
    pml.write("disable sym_environment\n\n")

    # Clean up selections
    pml.write("delete scaff_dna\ndelete scaff_prot\n")
    if have_mates:
        pml.write("delete mate_dna\ndelete mate_prot\n")
    pml.write("\norient scaffold\nzoom\n")
    pml.write("\nenable sym_environment\n")

print(f"  PyMOL -> {pml_path}")
print(f"\nTo visualize:  pymol {pml_path}")
print(f"  Then:  show spheres, elem D   -- to see contact points")
print(f"         show cartoon, chain S   -- to see nearest guest sym copy")
