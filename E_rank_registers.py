#!/usr/bin/env python3
"""E_rank_registers.py

Rank all guest register placements by how well they fit in the crystal
lattice, producing a ranked CSV and an inspection PML.

Scoring philosophy:
  - The ASU scaffold protein (RepE54) IS included — placing the guest
    where RepE54 sits is a real clash.  The only exemption is for
    DNA footprint overlap (category 2): it is not penalized because
    the guest naturally binds a similar DNA region.
  - ASU DNA and coaxial-stacking DNA are exempt (guest binds there).
  - Neighbors: the ASU scaffold protein plus each symmetry copy.
    For each symmetry copy we measure the closest approach of:
      (a) its scaffold protein  → guest protein   (SPR)
      (b) its non-coaxial DNA   → guest protein   (DNA)
      (c) the guest's own sym-copy protein  → guest protein  (GSY)
  - Candidates with ANY neighbor distance < CLASH_CUTOFF are "clashed"
    and always rank below all non-clashed candidates.
  - Among non-clashed candidates, those farthest from all neighbors rank
    highest; the score accounts for both distance and number of close
    neighbors.

Outputs:
  {guest_code}_ranked.csv   — one row per register, ranked
  {guest_code}_ranked.pml   — PyMOL script for ranked inspection

Usage:
    python E_rank_registers.py [guest_code]
    python E_rank_registers.py 1jgg
"""
import json, sys, os, csv, itertools
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pyscaffoldscan'))
from pdbtools import PDB
from sgdata import sgqt

# ── Parameters ────────────────────────────────────────────────────────
CLASH_CUTOFF    = 2.0    # A: steric clash
INTERACT_CUTOFF = 8.0    # A: interacting neighbor
ENV_CUTOFF      = 50.0   # A: max distance to include a symmetry copy
N_CLOSEST       = 8      # number of closest-neighbor columns in CSV

WC = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

def wc_complement(seq):
    return ''.join(WC[b] for b in seq)

# Core atoms used for clash assessment (backbone + CB, plus Pro ring)
CORE_ATOM_NAMES = {'N', 'CA', 'C', 'O', 'CB'}
PRO_EXTRA_ATOMS = {'CG', 'CD'}

def is_core_atom(atom_name, res_name):
    """True if atom is a backbone/core atom for clash purposes."""
    name = atom_name.strip()
    if name in CORE_ATOM_NAMES:
        return True
    if res_name.strip() == 'PRO' and name in PRO_EXTRA_ATOMS:
        return True
    return False

CAT_NAMES = {
    1: 'clash_scaff_prot',
    2: 'footprint_overlap',
    3: 'clash_sym_mate',
    4: 'near_neighbors',
    5: 'independent',
}

# ── Helpers (shared with D_categorize) ────────────────────────────────
def identify_dna_chains(P):
    dna_resnames = {'DA', 'DT', 'DC', 'DG'}
    dna_chains = []
    for chain in sorted(P.resndict.keys()):
        resnames = {rn.strip() for rn in P.resndict[chain].values()}
        if resnames <= dna_resnames:
            dna_chains.append(chain)
    return dna_chains

def identify_protein_chains(P):
    dna = set(identify_dna_chains(P))
    return [ch for ch in sorted(P.resndict.keys()) if ch not in dna]

def select_chains(P, chains):
    expr = ' or '.join(f'chain {ch}' for ch in chains)
    return P[expr]

def extract_scaffold_core_prot_coords(pdb_path, prot_chains):
    """Extract core-atom coords for scaffold protein chains from a PDB file."""
    coords = []
    prot_set = set(prot_chains)
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] in prot_set:
                if is_core_atom(line[12:16], line[17:20]):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    return np.array(coords) if coords else np.empty((0, 3))

def get_dna_sequence(P, chain):
    """Get single-letter DNA sequence for a chain, ordered by residue ID."""
    resids = sorted(P.resndict[chain].keys())
    seq = ''.join(P.resndict[chain][r].strip()[-1] for r in resids)
    return seq

def env_obj_name(sym_idx, cell):
    a, b, c = cell
    def fmt(v):
        return f"m{abs(v)}" if v < 0 else str(v)
    return f"env_s{sym_idx}_{fmt(a)}_{fmt(b)}_{fmt(c)}"

AUGMENT_MARKER = 'REMARK 999 D_CATEGORIZE AUGMENTATION BELOW'

def extract_coords_and_info(pdb_path, chains):
    """Extract ATOM coords and identity for specified chains.
    Returns (coords_array, info_list) where each info element is
    (chain, resid, atom_name)."""
    chain_set = set(chains)
    coords, info = [], []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] in chain_set:
                coords.append([float(line[30:38]), float(line[38:46]),
                                float(line[46:54])])
                info.append((line[21], int(line[22:26]), line[12:16].strip()))
    return (np.array(coords) if coords else np.empty((0, 3)), info)

def fmt_atom(info_tuple):
    """Format atom identity as chain.resid.name."""
    ch, ri, nm = info_tuple
    return f"{ch}.{ri}.{nm}"

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'MSE':'M','SEC':'U',
}

def extract_protein_sequence(pdb_path, chain_id):
    """Extract protein sequence from PDB.  Tries SEQRES first, ATOM fallback."""
    seqres = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('SEQRES') and line[11] == chain_id:
                for tok in line[19:].split():
                    if tok in THREE_TO_ONE:
                        seqres.append(THREE_TO_ONE[tok])
    if seqres:
        return ''.join(seqres)
    seq = []
    prev = None
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] == chain_id:
                rn = line[17:20].strip()
                ri = int(line[22:26])
                if ri != prev and rn in THREE_TO_ONE:
                    seq.append((ri, THREE_TO_ONE[rn]))
                    prev = ri
    seq.sort()
    return ''.join(r for _, r in seq)

def load_guest_protein_coords(pdb_path):
    """Load guest protein atom coords, stripping any D_categorize augmentation.

    Returns (all_coords, core_coords, atom_info) where core_coords contains
    only backbone atoms (N, CA, C, O, CB) plus CG/CD for prolines, and
    atom_info is a list of (chain, resid, atom_name) parallel to all_coords.
    """
    lines = []
    with open(pdb_path) as f:
        for line in f:
            if AUGMENT_MARKER in line:
                break
            lines.append(line)
    # Identify protein chains from clean lines
    dna_resnames = {'DA', 'DT', 'DC', 'DG'}
    chains_resnames = {}
    for line in lines:
        if not line.startswith('ATOM'):
            continue
        ch = line[21]
        rn = line[17:20].strip()
        chains_resnames.setdefault(ch, set()).add(rn)
    dna_chs = {ch for ch, rns in chains_resnames.items() if rns <= dna_resnames}
    prot_chs = {ch for ch in chains_resnames if ch not in dna_chs}
    all_coords = []
    all_info = []
    core_coords = []
    for line in lines:
        if line.startswith('ATOM') and line[21] in prot_chs:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            all_coords.append([x, y, z])
            atom_name = line[12:16]
            res_name  = line[17:20]
            all_info.append((line[21], int(line[22:26]), atom_name.strip()))
            if is_core_atom(atom_name, res_name):
                core_coords.append([x, y, z])
    empty = np.empty((0, 3))
    return (np.array(all_coords) if all_coords else empty,
            np.array(core_coords) if core_coords else empty,
            all_info)

# ── Generate crystal environment ──────────────────────────────────────
def generate_crystal_environment(scaffold, prot_coords_asu, dna_coords_asu,
                                 coaxial_keys, core_prot_coords_asu=None):
    """Same logic as D_categorize — returns env copies with R, T, KD-trees."""
    from numpy import matrix
    scaffold.xtal_qtlist = sgqt[scaffold.xtal_sg]

    asu_coords = scaffold.GetCoords()
    asu_center = asu_coords.mean(axis=0)
    asu_radius = np.linalg.norm(asu_coords - asu_center, axis=1).max()
    cog_cutoff = ENV_CUTOFF + 2 * asu_radius
    asu_tree = cKDTree(asu_coords)

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
            if d_cog < 0.1 or d_cog > cog_cutoff:
                continue

            copy_coords = (asu_coords @ R.T) + T
            d_to_asu, _ = asu_tree.query(copy_coords, k=1)
            min_dist = float(d_to_asu.min())
            if min_dist > ENV_CUTOFF:
                continue

            copy_prot = (prot_coords_asu @ R.T) + T
            copy_dna  = (dna_coords_asu @ R.T) + T
            copy_core_prot = ((core_prot_coords_asu @ R.T) + T
                              if core_prot_coords_asu is not None
                              else np.empty((0, 3)))

            key = (sym_idx, (a, b, c))
            is_coaxial = key in coaxial_keys

            env_copies.append({
                'prot_coords':      copy_prot,
                'core_prot_coords': copy_core_prot,
                'dna_coords':       copy_dna,
                'R':           R,
                'T':           T,
                'sym_idx':     sym_idx,
                'cell':        (a, b, c),
                'obj_name':    env_obj_name(sym_idx, (a, b, c)),
                'is_coaxial':  is_coaxial,
                'd_cog':       round(float(d_cog), 1),
                'd_min':       round(min_dist, 1),
            })

    env_copies.sort(key=lambda x: x['d_min'])

    # Pre-build KD-trees
    for copy in env_copies:
        if not copy['is_coaxial'] and len(copy['dna_coords']) > 0:
            copy['dna_tree'] = cKDTree(copy['dna_coords'])
        else:
            copy['dna_tree'] = None
        copy['prot_tree'] = (cKDTree(copy['prot_coords'])
                             if len(copy['prot_coords']) > 0 else None)
        copy['core_prot_tree'] = (cKDTree(copy['core_prot_coords'])
                                  if len(copy['core_prot_coords']) > 0 else None)

    print(f"  {len(env_copies)} symmetry copies within {ENV_CUTOFF} A "
          f"({sum(1 for c in env_copies if c['is_coaxial'])} coaxial)")
    return env_copies

# ── Per-register neighbor analysis ────────────────────────────────────
def analyze_neighbors(guest_prot_coords, guest_core_coords,
                      env_copies, scaff_prot_tree, scaff_core_prot_tree,
                      guest_info=None, scaff_prot_info=None,
                      scaff_dna_info=None):
    """Compute per-neighbor distances for one register.

    Two distance sets per neighbor:
      dist      — all-atom (used for scoring/ranking)
      core_dist — core-atoms only: N,CA,C,O,CB + Pro CG,CD (used for clash)

    When atom info arrays are provided, each neighbor also gets:
      guest_atom / env_atom — (chain, resid, atom_name) for the closest pair.

    Returns sorted list of neighbor dicts (sorted by all-atom dist).
    """
    guest_tree = cKDTree(guest_prot_coords)
    guest_core_tree = cKDTree(guest_core_coords)
    track = guest_info is not None
    neighbors = []

    # ASU scaffold protein — the RepE54 in the home building block
    d_asu, idx_asu = scaff_prot_tree.query(guest_prot_coords, k=1)
    d_asu_min = float(d_asu.min())
    d_asu_core, _ = scaff_core_prot_tree.query(guest_core_coords, k=1)
    d_asu_core_min = float(d_asu_core.min())
    nb_asu = {
        'dist':      round(d_asu_min, 2),
        'core_dist': round(d_asu_core_min, 2),
        'obj_name':  'ASU',
        'type':      'SPR',
        'prot_dist': round(d_asu_min, 2),
        'gsym_dist': None,
        'dna_dist':  None,
    }
    if track:
        ig = int(np.argmin(d_asu))
        nb_asu['guest_atom'] = guest_info[ig]
        nb_asu['env_atom'] = scaff_prot_info[int(idx_asu[ig])]
    neighbors.append(nb_asu)

    for copy in env_copies:
        dists_this_copy = []       # (type, all-atom dist, guest_idx, env_info)
        core_dists_this_copy = []  # (type, core-atom dist)

        # Scaffold protein from this symmetry copy
        pt = copy['prot_tree']
        cpt = copy['core_prot_tree']
        if pt is not None:
            d, idx = pt.query(guest_prot_coords, k=1)
            d_prot = float(d.min())
            ig = int(np.argmin(d))
            dists_this_copy.append(('SPR', d_prot, ig,
                                    scaff_prot_info[int(idx[ig])] if track else None))
        else:
            d_prot = None
        if cpt is not None:
            d, _ = cpt.query(guest_core_coords, k=1)
            core_dists_this_copy.append(('SPR', float(d.min())))

        # Guest-sym protein distance
        R, T_vec = copy['R'], copy['T']
        gsym_coords = (guest_prot_coords @ R.T) + T_vec
        d, idx = guest_tree.query(gsym_coords, k=1)
        d_gsym = float(d.min())
        i_sym = int(np.argmin(d))
        i_self = int(idx[i_sym])
        dists_this_copy.append(('GSY', d_gsym, i_self,
                                guest_info[i_sym] if track else None))
        # Core version
        gsym_core_coords = (guest_core_coords @ R.T) + T_vec
        d, _ = guest_core_tree.query(gsym_core_coords, k=1)
        core_dists_this_copy.append(('GSY', float(d.min())))

        # Non-coaxial DNA distance
        dt = copy['dna_tree']
        if dt is not None:
            d, idx = dt.query(guest_prot_coords, k=1)
            d_dna = float(d.min())
            ig = int(np.argmin(d))
            dists_this_copy.append(('DNA', d_dna, ig,
                                    scaff_dna_info[int(idx[ig])] if track else None))
            d_core, _ = dt.query(guest_core_coords, k=1)
            core_dists_this_copy.append(('DNA', float(d_core.min())))
        else:
            d_dna = None

        # Pick the closest interaction with this copy
        best = min(dists_this_copy, key=lambda x: x[1])
        best_type, best_dist, best_ig, best_env_info = best
        _, best_core = min(core_dists_this_copy, key=lambda x: x[1])
        nb = {
            'dist':      round(best_dist, 2),
            'core_dist': round(best_core, 2),
            'obj_name':  copy['obj_name'],
            'type':      best_type,
            'prot_dist': round(d_prot, 2) if d_prot is not None else None,
            'gsym_dist': round(d_gsym, 2),
            'dna_dist':  round(d_dna, 2) if d_dna is not None else None,
        }
        if track:
            nb['guest_atom'] = guest_info[best_ig]
            nb['env_atom'] = best_env_info
        neighbors.append(nb)

    neighbors.sort(key=lambda x: x['dist'])
    return neighbors

def compute_score(neighbors, n_closest=N_CLOSEST):
    """Compute a ranking score from neighbor distances.

    Score = sum of the closest N all-atom neighbor distances (higher = better).
    Clash is assessed using core atoms only (N, CA, C, O, CB + Pro CG/CD).
    """
    closest = neighbors[:n_closest]
    dists = [n['dist'] for n in closest]
    while len(dists) < n_closest:
        dists.append(999.0)
    total = sum(dists)
    has_clash = any(n['core_dist'] < CLASH_CUTOFF for n in neighbors)
    n_clash = sum(1 for n in neighbors if n['core_dist'] < CLASH_CUTOFF)
    n_interact = sum(1 for n in neighbors if n['dist'] < INTERACT_CUTOFF)
    min_core = min(n['core_dist'] for n in neighbors)
    return {
        'total_score':  round(total, 2),
        'has_clash':    has_clash,
        'n_clash':      n_clash,
        'n_interact':   n_interact,
        'min_dist':     dists[0],
        'min_core_dist': min_core,
        'closest_dists': dists[:n_closest],
    }

# ── Classification label ──────────────────────────────────────────────
def classify_for_ranking(has_clash, n_interact, min_dist):
    """Simple classification for the CSV."""
    if has_clash:
        return 'CLASH'
    elif min_dist >= INTERACT_CUTOFF:
        return 'independent'
    elif n_interact >= 3:
        return 'multi-connected'
    else:
        return 'near-neighbor'

# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
guest_code = sys.argv[1] if len(sys.argv) > 1 else '1jgg'
scaffold_code = sys.argv[2] if len(sys.argv) > 2 else '9YZJ'

# Load scaffold
scaffold_json = f'scaffold_models/{scaffold_code}.json'
with open(scaffold_json) as f:
    scaff = json.load(f)

scaffold = PDB(f"scaffold_models/{scaff['pdb_id']}.pdb")
sch1, sch2 = scaff['dna_chains']

scaff_prot_chains = [ch for ch, info in scaff['chains'].items()
                     if info['type'] == 'protein']
print(f"Scaffold {scaff['pdb_id']}: protein {scaff_prot_chains}, "
      f"DNA [{sch1}, {sch2}]")

scaff_pdb_path    = f"scaffold_models/{scaff['pdb_id']}.pdb"
scaff_prot_coords, scaff_prot_info = extract_coords_and_info(
    scaff_pdb_path, scaff_prot_chains)
scaff_prot_tree   = cKDTree(scaff_prot_coords)
scaff_core_prot_coords = extract_scaffold_core_prot_coords(
    scaff_pdb_path, scaff_prot_chains)
scaff_core_prot_tree = cKDTree(scaff_core_prot_coords)
scaff_dna_coords, scaff_dna_info = extract_coords_and_info(
    scaff_pdb_path, [sch1, sch2])
print(f"  Scaffold protein: {len(scaff_prot_coords)} atoms "
      f"({len(scaff_core_prot_coords)} core)")

# Load mates
mates_json = f'symmetry_mates/{scaffold_code}_mates.json'
have_mates = os.path.isfile(mates_json)
coaxial_keys = set()
if have_mates:
    with open(mates_json) as f:
        mates = json.load(f)
    for side in ('top', 'bot'):
        m = mates['mates'][side]
        coaxial_keys.add((m['sym_op'], tuple(m['unit_cell'])))

outdir = f'output/{guest_code}.{scaffold_code}'

# Load categories JSON (for register metadata + original categories)
cat_json = f'{outdir}/{guest_code}_categories.json'
with open(cat_json) as f:
    cat_data = json.load(f)

top_chain = cat_data['top_chain']
registers = cat_data['registers']

# Load asu_bp for junction deduplication
reg_json = f'{outdir}/{guest_code}_registers.json'
with open(reg_json) as f:
    reg_meta = json.load(f)
asu_bp = reg_meta['asu_bp']
window_size = reg_meta['window_size']

# Load DNA sequences
bot_chain = sch2 if top_chain == sch1 else sch1
scaff_top_seq = scaff['chains'][top_chain]['sequence']
scaff_bot_seq = scaff['chains'][bot_chain]['sequence']
# Bottom strand displayed 3'→5' = WC complement of top strand
scaff_bot_display = wc_complement(scaff_top_seq)

# Guest DNA sequences
guest_pdb = PDB(f'processed_guest_models/{guest_code}.pdb')
guest_dna_chains = identify_dna_chains(guest_pdb)
gch1, gch2 = guest_dna_chains
guest_seq1 = get_dna_sequence(guest_pdb, gch1)
guest_seq2 = get_dna_sequence(guest_pdb, gch2)

print(f"  {len(registers)} registers loaded from {cat_json} (asu_bp={asu_bp})")
print(f"  Scaffold top strand ({top_chain}): 5'-{scaff_top_seq}-3'")
print(f"  Guest DNA: chain {gch1}={guest_seq1}, chain {gch2}={guest_seq2}")

# Generate crystal environment
print(f"\nGenerating crystal environment ...")
env_copies = generate_crystal_environment(scaffold, scaff_prot_coords,
                                          scaff_dna_coords, coaxial_keys,
                                          scaff_core_prot_coords)

# ── Analyze each register ─────────────────────────────────────────────
print(f"\nAnalyzing {len(registers)} registers ...\n")

ranked = []
for i, reg in enumerate(registers):
    guest_prot_coords, guest_core_coords, guest_atom_info = \
        load_guest_protein_coords(reg['pdb_file'])
    if len(guest_prot_coords) == 0:
        print(f"  WARNING: no protein atoms in {reg['pdb_file']}, skipping")
        continue

    neighbors = analyze_neighbors(guest_prot_coords, guest_core_coords,
                                  env_copies, scaff_prot_tree,
                                  scaff_core_prot_tree,
                                  guest_info=guest_atom_info,
                                  scaff_prot_info=scaff_prot_info,
                                  scaff_dna_info=scaff_dna_info)
    score = compute_score(neighbors)
    classification = classify_for_ranking(score['has_clash'],
                                          score['n_interact'],
                                          score['min_dist'])

    nb0 = neighbors[0]
    ranked.append({
        'obj_name':       reg['obj_name'],
        'label':          reg['label'],
        'orientation':    reg['orientation'],
        'start_pos':      reg['start_pos'],
        'end_pos':        reg['end_pos'],
        'region':         reg['region'],
        'rmsd':           reg['rmsd'],
        'pdb_file':       reg['pdb_file'],
        'orig_category':  reg['category_name'],
        'classification': classification,
        'total_score':    score['total_score'],
        'has_clash':      score['has_clash'],
        'n_clash':        score['n_clash'],
        'n_interact':     score['n_interact'],
        'min_dist':       score['min_dist'],
        'min_core_dist':  score['min_core_dist'],
        'closest_dists':  score['closest_dists'],
        'neighbors':      neighbors[:N_CLOSEST],
        'min_dist_guest_atom': nb0.get('guest_atom'),
        'min_dist_env_atom':   nb0.get('env_atom'),
        'min_dist_env_obj':    nb0['obj_name'],
        'min_dist_type':       nb0['type'],
    })

    tag = 'CLASH' if score['has_clash'] else 'ok'
    print(f"  [{i+1:>3}/{len(registers)}]  {reg['obj_name']:<20s}  "
          f"score={score['total_score']:7.1f}  min={score['min_dist']:5.1f}  "
          f"core={score['min_core_dist']:5.1f}  "
          f"interact={score['n_interact']}  clash={score['n_clash']}  [{tag}]")

# ── Deduplicate junction equivalents ──────────────────────────────────
# Registers spanning junc_bot and junc_top that differ by exactly asu_bp
# in start_pos (same orientation) are the same physical placement via
# crystal periodicity.  Keep the one with the lower RMSD.
if have_mates:
    by_key = {}
    for r in ranked:
        # Canonical key: orientation + the smaller of the two equivalent start_pos
        canon_start = r['start_pos'] % asu_bp
        key = (r['orientation'], canon_start)
        by_key.setdefault(key, []).append(r)

    deduped = []
    n_dropped = 0
    for key, group in by_key.items():
        if len(group) == 1:
            deduped.append(group[0])
        else:
            # Keep the one with better RMSD (lower = better fit)
            group.sort(key=lambda r: r['rmsd'])
            kept = group[0]
            deduped.append(kept)
            for dropped in group[1:]:
                n_dropped += 1
                print(f"  Dedup: {dropped['obj_name']} is equivalent to "
                      f"{kept['obj_name']} (delta={asu_bp} bp), dropping")
    ranked = deduped
    if n_dropped:
        print(f"  Removed {n_dropped} redundant junction-equivalent registers")

# ── Deduplicate wrapped registers ─────────────────────────────────────
# Registers that wrap around the scaffold (e.g. pos=-1 and pos=30 on 31bp)
# are the same physical placement. Keep the one with larger min_dist.
N_bp = len(scaff_top_seq)
seen = {}
deduped = []
for r in ranked:
    key = (r['start_pos'] % N_bp, r['orientation'])
    if key in seen:
        prev = seen[key]
        if r['min_dist'] > prev['min_dist']:
            deduped.remove(prev)
            seen[key] = r
            deduped.append(r)
    else:
        seen[key] = r
        deduped.append(r)
ranked = deduped

# ── Rank ──────────────────────────────────────────────────────────────
# Sort: non-clashers first, then by min_dist DESC (farthest from closest
# neighbor = best), with total_score DESC as tiebreaker.
ranked.sort(key=lambda r: (r['has_clash'], -r['min_dist'], -r['total_score']))

# Assign ranks
for rank_idx, r in enumerate(ranked):
    r['rank'] = rank_idx + 1

# ── Summary ───────────────────────────────────────────────────────────
n_clash = sum(1 for r in ranked if r['has_clash'])
n_ok = len(ranked) - n_clash
print(f"\n{'='*70}")
print(f"  Ranking summary for {guest_code} on {scaff['pdb_id']}")
print(f"{'='*70}")
print(f"  Non-clashing:  {n_ok:>3}")
print(f"  Clashing:      {n_clash:>3}")
print(f"  Total:         {len(ranked):>3}")

if n_ok > 0:
    print(f"\n  Top 10 non-clashing candidates:")
    for r in ranked[:min(10, n_ok)]:
        print(f"    rank {r['rank']:>2}  {r['obj_name']:<20s}  "
              f"score={r['total_score']:7.1f}  min={r['min_dist']:5.1f}  "
              f"interact={r['n_interact']}  [{r['classification']}]")

# ── Write CSV ─────────────────────────────────────────────────────────
csv_path = f'{outdir}/{guest_code}_ranked.csv'

header = ['rank', 'min_dist', 'min_core_dist', 'min_dist_atoms',
          'obj_name', 'start_pos', 'end_pos', 'orientation', 'classification',
          'n_clash', 'n_interact', 'rmsd']
for k in range(1, N_CLOSEST + 1):
    header.append(f'neighbor_{k}_dist')
    header.append(f'neighbor_{k}_name')
header.append('pymol_dist_cmd')

with open(csv_path, 'w', newline='') as f:
    w = f.write
    w(f"# Ranked register analysis: guest {guest_code} on scaffold "
      f"{scaff['pdb_id']}\n")
    w(f"# Positions are 1-indexed on the top strand (chain {top_chain}). "
      f"Negative positions extend into the bottom coaxial stacking mate; "
      f"positions > {asu_bp} extend into the top mate.\n")
    w(f"# fwd/rev: guest DBD tried in both directions along the duplex. "
      f"In fwd chain {gch1} ({guest_seq1}) maps to top; "
      f"in rev chain {gch2} ({guest_seq2}) maps to top.\n")
    w(f"# min_dist = closest all-atom distance to any lattice neighbor (A). "
      f"min_core_dist = closest core-atom (N CA C O CB + Pro CG CD) "
      f"distance (A).\n")
    w(f"# Ranking: non-clashing first sorted by min_dist descending then "
      f"by sum-of-{N_CLOSEST}-closest-distances descending.\n")
    w(f"# Classification: CLASH = core-atom < {CLASH_CUTOFF} A | "
      f"independent = no neighbor within {INTERACT_CUTOFF} A | "
      f"multi-connected = >= 3 neighbors within {INTERACT_CUTOFF} A | "
      f"near-neighbor = 1-2 neighbors within {INTERACT_CUTOFF} A\n")
    w(f"# min_dist_atoms = chain.resid.name for guest / environment atoms "
      f"at the closest contact. pymol_dist_cmd = PyMOL command to "
      f"display that distance.\n")
    w(f"# Neighbors: ASU = scaffold protein in the home building block | "
      f"env_sN_A_B_C = symmetry copy N at unit cell offset "
      f"(A B C) where m = minus. "
      f"Distance is the closest of scaffold protein / "
      f"guest sym-copy protein / non-coaxial DNA from that copy.\n")

    writer = csv.writer(f)
    writer.writerow(header)
    for r in ranked:
        # Format min_dist atom identities
        ga = r['min_dist_guest_atom']
        ea = r['min_dist_env_atom']
        if ga and ea:
            min_dist_atoms = f"{fmt_atom(ga)} / {fmt_atom(ea)}"
        else:
            min_dist_atoms = ''

        # PyMOL distance command
        if ga and ea:
            g_ch, g_ri, g_nm = ga
            e_ch, e_ri, e_nm = ea
            if not r['has_clash']:
                guest_obj = f"rank{r['rank']}.{r['obj_name']}"
            else:
                guest_obj = r['obj_name']
            env_obj_nm = r['min_dist_env_obj']
            if env_obj_nm == 'ASU':
                env_obj = 'scaffold'
            elif r['min_dist_type'] == 'GSY':
                env_obj = guest_obj
                e_ch = 'S'
            else:
                env_obj = env_obj_nm
            label = f"min_rank{r['rank']}"
            g_sele = (f"{guest_obj} and chain {g_ch} and resi {g_ri} "
                      f"and name {g_nm}")
            e_sele = (f"{env_obj} and chain {e_ch} and resi {e_ri} "
                      f"and name {e_nm}")
            pymol_cmd = f"distance {label}, ({g_sele}), ({e_sele})"
        else:
            pymol_cmd = ''

        row = [
            r['rank'], r['min_dist'], r['min_core_dist'], min_dist_atoms,
            r['obj_name'], r['start_pos'], r['end_pos'],
            r['orientation'], r['classification'],
            r['n_clash'], r['n_interact'], round(r['rmsd'], 4),
        ]
        for k in range(N_CLOSEST):
            if k < len(r['neighbors']):
                nb = r['neighbors'][k]
                row.append(nb['dist'])
                row.append(nb['obj_name'])
            else:
                row.append(999.0)
                row.append('')
        row.append(pymol_cmd)
        writer.writerow(row)

print(f"\n  CSV -> {csv_path}")

# ── Write PML ─────────────────────────────────────────────────────────
scaffold_pdb_path = f"scaffold_models/{scaff['pdb_id']}.pdb"
pml_path = f'{outdir}/{guest_code}_ranked.pml'

with open(pml_path, 'w') as pml:
    pml.write(f"# Ranked register visualization for {guest_code} "
              f"on {scaff['pdb_id']}\n")
    pml.write(f"# Generated by E_rank_registers.py\n")
    pml.write(f"# Ranking: non-clashing first (farthest=best), "
              f"clashing last\n")
    pml.write(f"# ASU scaffold protein included as neighbor in scoring\n\n")

    # ── Scaffold + mates ──
    pml.write("# ── Scaffold ──────────────────────────────────────────\n")
    pml.write(f"load {scaffold_pdb_path}, scaffold\n")
    if have_mates:
        pml.write(f"load {mates['mates']['bot']['pdb_file']}, mate_bot\n")
        pml.write(f"load {mates['mates']['top']['pdb_file']}, mate_top\n")
    pml.write("\n")

    # ── Crystal environment ──
    pml.write("# ── Crystal environment ───────────────────────────────\n")
    for copy in env_copies:
        pdb_path = f'{outdir}/{copy["obj_name"]}.pdb'
        pml.write(f"load {pdb_path}, {copy['obj_name']}\n")
    pml.write("\n")

    # ── Load ranked guests ──
    pml.write("# ── Ranked guests ─────────────────────────────────────\n")

    # Non-clashing: top-level objects with rank prefix
    non_clash = [r for r in ranked if not r['has_clash']]
    clashed   = [r for r in ranked if r['has_clash']]

    for r in non_clash:
        rank_name = f"rank{r['rank']}.{r['obj_name']}"
        pml.write(f"load {r['pdb_file']}, {rank_name}\n")

    # Clashing: load with original names, will be grouped
    for r in clashed:
        pml.write(f"load {r['pdb_file']}, {r['obj_name']}\n")
    pml.write("\n")

    # ── Groups ──
    pml.write("# ── Groups ────────────────────────────────────────────\n")
    scaff_members = "scaffold"
    if have_mates:
        scaff_members += " mate_bot mate_top"
    pml.write(f"group scaffold_grp, {scaff_members}\n")

    env_objs = ' '.join(c['obj_name'] for c in env_copies)
    pml.write(f"group sym_environment, {env_objs}\n")

    if clashed:
        clash_objs = ' '.join(r['obj_name'] for r in clashed)
        pml.write(f"group clashed, {clash_objs}\n")
    pml.write("\n")

    # ── Appearance ──
    pml.write("# ── Appearance ────────────────────────────────────────\n")
    pml.write("set orthoscopic, 1\n")
    pml.write("hide everything\nshow cartoon\n\n")

    # Scaffold DNA/protein
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
    for copy in env_copies:
        obj = copy['obj_name']
        pml.write(f"select _{obj}_dna, {obj} and chain {sch1} "
                  f"or {obj} and chain {sch2}\n")
        pml.write(f"select _{obj}_prot, {obj} and not _{obj}_dna\n")
        pml.write(f"show sticks, _{obj}_dna\n")
        pml.write(f"color gray60, _{obj}_prot\n")
        pml.write(f"color lightblue, _{obj}_dna\n")
        pml.write(f"delete _{obj}_dna\ndelete _{obj}_prot\n")
    pml.write("\n")

    # ── Color guests ──
    pml.write("# ── Color ranked guests ───────────────────────────────\n")

    # Non-clashing: color by rank (green gradient — best=bright, worst=pale)
    if non_clash:
        n = len(non_clash)
        for i, r in enumerate(non_clash):
            rank_name = f"rank{r['rank']}.{r['obj_name']}"
            # Gradient from bright green (rank 1) to pale yellow-green
            frac = i / max(n - 1, 1)
            red   = frac * 0.8
            green = 1.0 - frac * 0.3
            blue  = frac * 0.3
            pml.write(f"set_color rank{r['rank']}_color, "
                      f"[{red:.2f}, {green:.2f}, {blue:.2f}]\n")
            pml.write(f"color rank{r['rank']}_color, {rank_name}\n")

    # Clashing: red
    if clashed:
        pml.write("color red, clashed\n")
    pml.write("color cyan, chain S\n\n")

    # ── Disable everything except rank 1 and environment ──
    pml.write("# ── Initial view: rank 1 + environment ────────────────\n")

    # Disable all guests first
    for r in non_clash:
        rank_name = f"rank{r['rank']}.{r['obj_name']}"
        pml.write(f"disable {rank_name}\n")
    if clashed:
        pml.write("disable clashed\n")
    pml.write("disable sym_environment\n\n")

    # Enable rank 1 and environment
    if non_clash:
        rank1_name = f"rank{non_clash[0]['rank']}.{non_clash[0]['obj_name']}"
        pml.write(f"enable {rank1_name}\n")
    pml.write("enable sym_environment\n\n")

    # Clean up temp selections
    pml.write("delete scaff_dna\ndelete scaff_prot\n")
    if have_mates:
        pml.write("delete mate_dna\ndelete mate_prot\n")
    pml.write("\norient scaffold\nzoom\n")

print(f"  PML -> {pml_path}")

# ── Precompute composite sequences for each register ─────────────────
# In B_find_registers: fwd = vstack([gch1, gch2]) → gch1 maps to top strand
# rev = vstack([gch2, gch1]) → gch2 maps to top strand
fwd_top_seq = guest_seq1   # gch1 maps to top strand in fwd
rev_top_seq = guest_seq2   # gch2 maps to top strand in rev

for r in ranked:
    sp = r['start_pos']
    overwrite_seq = fwd_top_seq if r['orientation'] == 'fwd' else rev_top_seq
    top = list(scaff_top_seq)
    n = len(top)
    for k in range(len(overwrite_seq)):
        idx = (sp + k - 1) % n  # wrap around
        top[idx] = overwrite_seq[k].lower()
    r['composite_top'] = ''.join(top)
    r['composite_bot'] = ''.join(
        WC[b.upper()].lower() if b.islower() else WC[b] for b in top)

# ── Write composite sequence file ────────────────────────────────────
seq_path = f'{outdir}/{guest_code}_ranked_sequences.seqs'

with open(seq_path, 'w') as sf:
    sf.write(f">scaffold_dna_{scaff['pdb_id']}_chain{top_chain}+{bot_chain}\n")
    sf.write(f"  top  5'-{scaff_top_seq}-3'\n")
    sf.write(f"  bot  3'-{scaff_bot_display}-5'\n")
    sf.write(f">guest_dbd_{guest_code}_chain{gch1}+{gch2}\n")
    bot_guest_display = wc_complement(guest_seq1)
    sf.write(f"  top  5'-{guest_seq1}-3'  (chain {gch1}, used as top in fwd)\n")
    sf.write(f"  bot  3'-{bot_guest_display}-5'  (chain {gch2}, used as top in rev)\n")
    for r in ranked:
        rank_label = f"rank{r['rank']}.{r['obj_name']}"
        tag = r['classification']
        sf.write(f">{rank_label}  pos={r['start_pos']}..{r['end_pos']}  "
                 f"{r['orientation']}  min={r['min_dist']}A  [{tag}]\n")
        sf.write(f"  top  5'-{r['composite_top']}-3'\n")
        sf.write(f"  bot  3'-{r['composite_bot']}-5'\n")

print(f"  SEQ -> {seq_path}")

# ── Generate AF3 input JSONs for non-clashing registers ──────────────
af3_dir = f'{outdir}/af3_input_jsons'
os.makedirs(af3_dir, exist_ok=True)

# Extract protein sequences
scaff_prot_ch = scaff_prot_chains[0]
scaff_prot_seq = scaff['chains'][scaff_prot_ch].get('sequence')
if not scaff_prot_seq:
    scaff_prot_seq = extract_protein_sequence(scaff_pdb_path, scaff_prot_ch)

# Guest protein: try SEQRES from original PDB, fall back to processed
guest_orig_pdb = f'guest_models/{guest_code}.pdb'
guest_prot_chains = identify_protein_chains(guest_pdb)
guest_prot_ch = guest_prot_chains[0]
if os.path.isfile(guest_orig_pdb):
    guest_prot_seq = extract_protein_sequence(guest_orig_pdb, guest_prot_ch)
else:
    guest_prot_seq = extract_protein_sequence(
        f'processed_guest_models/{guest_code}.pdb', guest_prot_ch)

print(f"\n  AF3 protein sequences:")
print(f"    Scaffold ({scaff_prot_ch}): {len(scaff_prot_seq)} aa")
print(f"    Guest    ({guest_prot_ch}): {len(guest_prot_seq)} aa")

n_af3 = 0
for r in non_clash:
    dna_top = r['composite_top'].upper()
    dna_bot = wc_complement(dna_top)

    af3_name = f"{guest_code}_rank{r['rank']:02d}_{r['obj_name']}"
    af3_data = [{
        "name": af3_name,
        "modelSeeds": ["571004816"],
        "sequences": [
            {"proteinChain": {
                "sequence": scaff_prot_seq,
                "count": 1,
                "useStructureTemplate": True,
            }},
            {"proteinChain": {
                "sequence": guest_prot_seq,
                "count": 1,
                "useStructureTemplate": True,
            }},
            {"dnaSequence": {"sequence": dna_top, "count": 1}},
            {"dnaSequence": {"sequence": dna_bot, "count": 1}},
        ],
        "dialect": "alphafoldserver",
        "version": 1,
    }]

    af3_path = os.path.join(af3_dir, f"{af3_name}.json")
    with open(af3_path, 'w') as jf:
        json.dump(af3_data, jf, indent=1)
    n_af3 += 1

print(f"  AF3 -> {af3_dir}/  ({n_af3} JSON files for non-clashing registers)")

print(f"\nTo visualize: start pymol in this folder and then @{pml_path}")
if non_clash:
    r1 = non_clash[0]
    print(f"  Rank 1: {r1['obj_name']}  (score={r1['total_score']}, "
          f"min_dist={r1['min_dist']}, {r1['classification']})")
