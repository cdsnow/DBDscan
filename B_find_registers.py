#!/usr/bin/env python3
"""B_find_registers.py

Enumerate all possible sliding-window registers for aligning a guest protein's
dsDNA onto the scaffold crystal's dsDNA, using C1' atom superposition.

If symmetry mates exist in symmetry_mates/, the scan extends across the
ASU-symmetry mate junctions so that windows can span the boundary.

Outputs:
  - Aligned guest PDB files in output/{guest_code}/
  - A PyMOL script (output/{guest_code}_view.pml) to visualise everything
  - A JSON summary (output/{guest_code}_registers.json)

Usage:
    python B_find_registers.py [guest_code]
    python B_find_registers.py 1jgg
"""
import json, sys, os
import numpy as np
from pyscaffoldscan.pdbtools import PDB
from pyscaffoldscan.superimpy import superpose_rot_trans

WC = {'A':'T', 'T':'A', 'G':'C', 'C':'G'}

def get_dna_sequence(P, chain):
    """Get single-letter DNA sequence for a chain, ordered by residue ID."""
    resids = sorted(P.resndict[chain].keys())
    seq = ''.join(P.resndict[chain][r].strip()[-1] for r in resids)
    return resids, seq

def get_c1prime_coords(P, chain):
    """Extract C1' coordinates for a DNA chain, ordered by residue ID."""
    sel = P[f"chain {chain} and name C1'"]
    data = [(d['resi'], d['x'], d['y'], d['z']) for d in sel.listdict]
    data.sort()
    resids = [d[0] for d in data]
    coords = np.array([[d[1], d[2], d[3]] for d in data])
    return resids, coords

def identify_dna_chains(P):
    """Identify which chains in a PDB are DNA based on residue names."""
    dna_resnames = {'DA', 'DT', 'DC', 'DG'}
    dna_chains = []
    for chain in sorted(P.resndict.keys()):
        resnames = {rn.strip() for rn in P.resndict[chain].values()}
        if resnames <= dna_resnames:
            dna_chains.append(chain)
    return dna_chains

def find_base_pairing(resids1, seq1, resids2, seq2):
    """Find the best antiparallel Watson-Crick alignment between two DNA strands."""
    L1, L2 = len(seq1), len(seq2)
    seq2_rev = seq2[::-1]
    best_score, best_shift = -1, 0

    for d in range(-(L2 - 1), L1):
        lo = max(0, d)
        hi = min(L1, L2 + d)
        if hi - lo <= 0:
            continue
        matches = sum(1 for i in range(lo, hi)
                      if seq1[i] == WC.get(seq2_rev[i - d], ''))
        if matches > best_score:
            best_score = matches
            best_shift = d

    lo = max(0, best_shift)
    hi = min(L1, L2 + best_shift)
    pairs = []
    for i in range(lo, hi):
        j = L2 - 1 - (i - best_shift)
        is_wc = (seq1[i] == WC.get(seq2[j], ''))
        pairs.append((resids1[i], resids2[j], is_wc))

    return best_shift, best_score, hi - lo, pairs

def superimpose_and_rmsd(mobile, target):
    """Superimpose mobile onto target, return (rmsd, R, T)."""
    R, T = superpose_rot_trans(mobile, target)
    aligned = np.dot(mobile, R) + T
    rmsd = np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1)))
    return rmsd, R, T

def format_pos(p):
    """Format position for PyMOL-safe names: negative uses 'm' prefix, zero-padded."""
    return f"m{abs(p):02d}" if p < 0 else f"{p:02d}"

def reg_obj_name(start, end, orient):
    """PyMOL object name, e.g. R01to07_fwd or Rm05to01_fwd."""
    return f"R{format_pos(start)}to{format_pos(end)}_{orient}"

# ── CLI arguments ─────────────────────────────────────────────────────────
guest_code = sys.argv[1] if len(sys.argv) > 1 else '1jgg'
scaffold_code = sys.argv[2] if len(sys.argv) > 2 else '9YZJ'

# ── Load scaffold ──────────────────────────────────────────────────────────
scaffold_json = f'scaffold_models/{scaffold_code}.json'
with open(scaffold_json) as f:
    scaff = json.load(f)

scaffold_pdb_path = f"scaffold_models/{scaff['pdb_id']}.pdb"
scaffold = PDB(scaffold_pdb_path)
sch1, sch2 = scaff['dna_chains']

scaff_resids = {}
scaff_coords = {}
scaff_seq = {}
for ch in [sch1, sch2]:
    scaff_resids[ch], scaff_seq[ch] = get_dna_sequence(scaffold, ch)
    _, scaff_coords[ch] = get_c1prime_coords(scaffold, ch)

N = len(scaff_resids[sch1])  # nt per strand in the ASU

# ── Determine scaffold base pairing from sequence ─────────────────────────
print(f"Scaffold {scaff['pdb_id']} DNA:")
print(f"  Chain {sch1}: {N} nt, resids {scaff_resids[sch1][0]}-{scaff_resids[sch1][-1]}")
print(f"  Chain {sch2}: {N} nt, resids {scaff_resids[sch2][0]}-{scaff_resids[sch2][-1]}")

shift, wc_count, overlap_bp, bp_pairs = find_base_pairing(
    scaff_resids[sch1], scaff_seq[sch1],
    scaff_resids[sch2], scaff_seq[sch2])

mismatches = [p for p in bp_pairs if not p[2]]
print(f"  Antiparallel alignment: shift={shift}, {wc_count}/{overlap_bp} Watson-Crick pairs")
if mismatches:
    print(f"  Non-WC positions: {[(r1,r2) for r1,r2,_ in mismatches]}")

# ── Identify "top" chain (contains AAATT in scaffold binding sequence) ────
top_chain = None
for ch in [sch1, sch2]:
    if 'AAATT' in scaff_seq[ch]:
        top_chain = ch
        break
assert top_chain is not None, "Could not identify top chain (no AAATT found)"
print(f"  Top chain: {top_chain} (contains AAATT)")

# Ensure sch1 = top chain so forward orientation numbers along the top strand
if top_chain != sch1:
    sch1, sch2 = sch2, sch1
    print(f"  (Swapped strand order: sch1={sch1}, sch2={sch2})")

# ── Build extended DNA using symmetry mates (if available) ────────────────
mates_json = f'symmetry_mates/{scaffold_code}_mates.json'
have_mates = os.path.isfile(mates_json)
bot_pdb_path = top_pdb_path = None

if have_mates:
    with open(mates_json) as f:
        mates = json.load(f)
    bot_pdb_path = mates['mates']['bot']['pdb_file']
    top_pdb_path = mates['mates']['top']['pdb_file']
    bot_pdb = PDB(bot_pdb_path)
    top_pdb = PDB(top_pdb_path)
    print(f"\n  Symmetry mates loaded")
    print(f"    Bottom: {bot_pdb_path} (gap={mates['mates']['bot']['gap']} A)")
    print(f"    Top:    {top_pdb_path} (gap={mates['mates']['top']['gap']} A)")

    bot_coords, top_coords = {}, {}
    bot_seq, top_seq = {}, {}
    for ch in [sch1, sch2]:
        _, bot_coords[ch] = get_c1prime_coords(bot_pdb, ch)
        _, top_coords[ch] = get_c1prime_coords(top_pdb, ch)
        _, bot_seq[ch] = get_dna_sequence(bot_pdb, ch)
        _, top_seq[ch] = get_dna_sequence(top_pdb, ch)

    ext_s1_coords = np.vstack([bot_coords[sch1], scaff_coords[sch1], top_coords[sch1]])
    ext_s2_coords = np.vstack([bot_coords[sch2][::-1], scaff_coords[sch2][::-1], top_coords[sch2][::-1]])

    ext_s1_seq = bot_seq[sch1] + scaff_seq[sch1] + top_seq[sch1]
    ext_s2_seq = bot_seq[sch2][::-1] + scaff_seq[sch2][::-1] + top_seq[sch2][::-1]

    N_ext = len(ext_s1_seq)
    asu_start = N
    asu_end = 2 * N

    wc_ok = sum(1 for i in range(N_ext) if ext_s1_seq[i] == WC.get(ext_s2_seq[i], ''))
    print(f"  Extended DNA: {N_ext} bp ({N} bot + {N} ASU + {N} top), WC check: {wc_ok}/{N_ext}")

else:
    print(f"\n  No symmetry mates found — scanning ASU only.")
    ext_s1_coords = scaff_coords[sch1]
    ext_s2_coords = scaff_coords[sch2][::-1]
    ext_s1_seq = scaff_seq[sch1]
    ext_s2_seq = scaff_seq[sch2][::-1]
    N_ext = N
    asu_start = 0
    asu_end = N

# ── Load guest ─────────────────────────────────────────────────────────────
guest = PDB(f'processed_guest_models/{guest_code}.pdb')

guest_dna = identify_dna_chains(guest)
assert len(guest_dna) == 2, f"Expected 2 DNA chains in guest, found {guest_dna}"
gch1, gch2 = guest_dna

guest_resids = {}
guest_coords = {}
for ch in [gch1, gch2]:
    guest_resids[ch], guest_coords[ch] = get_c1prime_coords(guest, ch)

W = len(guest_resids[gch1])
assert W == len(guest_resids[gch2]), \
    f"Guest DNA strands differ in length: {gch1} vs {gch2}"

mobile = np.vstack([guest_coords[gch1], guest_coords[gch2]])
mobile_rev = np.vstack([guest_coords[gch2], guest_coords[gch1]])

print(f"\nGuest {guest_code}: {W}-bp window ({2*W} C1' atoms)")

# ── Prepare output directory ──────────────────────────────────────────────
outdir = f'output/{guest_code}.{scaffold_code}'
os.makedirs(outdir, exist_ok=True)

# ── Enumerate registers ───────────────────────────────────────────────────
j_lo = max(0, asu_start - W + 1)
j_hi = min(N_ext - W, asu_end - 1)
total_regs = 2 * (j_hi - j_lo + 1)

print(f"\nScanning {total_regs} registers ...")

# Clean previous register PDBs from output dir
for old_f in os.listdir(outdir):
    if old_f.endswith('.pdb'):
        os.remove(os.path.join(outdir, old_f))

results = []
reg = 0

for orient in ['fwd', 'rev']:
    mob = mobile if orient == 'fwd' else mobile_rev

    for j in range(j_lo, j_hi + 1):
        reg_num = reg + 1

        # Register position along the top chain
        start_pos = j - asu_start + 1
        end_pos = start_pos + W - 1
        label = f"R:{start_pos}:{end_pos}"
        obj_name = reg_obj_name(start_pos, end_pos, orient)

        print(f"  [{reg_num:>3}/{total_regs}]  {label:<11s} {orient}", end="", flush=True)

        tgt_s1 = ext_s1_coords[j:j + W]
        tgt_s2 = ext_s2_coords[j:j + W][::-1]
        target = np.vstack([tgt_s1, tgt_s2])

        rmsd, R, T = superimpose_and_rmsd(mob, target)

        # Classify region
        if j >= asu_start and j + W <= asu_end:
            region = 'asu'
        elif have_mates:
            region = 'junc_bot' if j < asu_start else 'junc_top'
        else:
            region = 'asu'

        # Write aligned guest PDB
        pdb_path = f"{outdir}/{obj_name}.pdb"
        aligned_guest = guest.Clone()
        aligned_guest.Rotate(np.array(R))
        aligned_guest.Translate(np.array(T).flatten())
        aligned_guest.WritePDB(pdb_path)

        print(f"  {region:<9s}  RMSD={rmsd:.4f}  -> {obj_name}.pdb")

        results.append({
            'register': reg,
            'label': label,
            'obj_name': obj_name,
            'orientation': orient,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'ext_offset': j,
            'region': region,
            'rmsd': round(float(rmsd), 4),
            'pdb_file': pdb_path,
        })
        reg += 1

# ── Summary ───────────────────────────────────────────────────────────────
num_reg = len(results)
n_asu = sum(1 for r in results if r['region'] == 'asu')
n_junc = sum(1 for r in results if 'junc' in r['region'])
rmsds = [r['rmsd'] for r in results]

print(f"\nDone. {num_reg} registers ({n_asu} ASU-internal, {n_junc} junction-spanning)")
print(f"  RMSD range: {min(rmsds):.4f} - {max(rmsds):.4f} A")

# ── Save JSON ─────────────────────────────────────────────────────────────
json_path = f'{outdir}/{guest_code}_registers.json'
output_json = {
    'guest': guest_code,
    'scaffold': scaff['pdb_id'],
    'top_chain': top_chain,
    'window_size': W,
    'asu_bp': N,
    'extended_bp': N_ext,
    'asu_range': [asu_start, asu_end],
    'num_registers': num_reg,
    'registers': results,
}
with open(json_path, 'w') as f:
    json.dump(output_json, f, indent=2)
print(f"  JSON  -> {json_path}")

# ── Write PyMOL script ────────────────────────────────────────────────────
pml_path = f'{outdir}/{guest_code}_view.pml'
with open(pml_path, 'w') as pml:
    pml.write(f"# PyMOL visualization for {guest_code} on {scaff['pdb_id']}\n")
    pml.write(f"# Generated by B_find_registers.py\n")
    pml.write(f"# Register numbering: positions along top chain ({top_chain})\n\n")

    # -- Scaffold structures --
    pml.write("# ── Scaffold ──────────────────────────────────────────\n")
    pml.write(f"load {scaffold_pdb_path}, scaffold\n")
    if have_mates:
        pml.write(f"load {bot_pdb_path}, mate_bot\n")
        pml.write(f"load {top_pdb_path}, mate_top\n")
    pml.write("\n")

    # -- Aligned guests --
    pml.write("# ── Aligned guests ────────────────────────────────────\n")
    for r in results:
        pml.write(f"load {r['pdb_file']}, {r['obj_name']}\n")
    pml.write("\n")

    pml.write("# ── Groups ────────────────────────────────────────────\n")
    pml.write("group scaffold_grp, scaffold")
    if have_mates:
        pml.write(" mate_bot mate_top")
    pml.write("\n")

    for orient in ['fwd', 'rev']:
        objs = [r['obj_name'] for r in results if r['orientation'] == orient]
        pml.write(f"group {orient}, {' '.join(objs)}\n")
    pml.write("\n")

    # -- Appearance --
    pml.write("# ── Appearance ────────────────────────────────────────\n")
    pml.write("set orthoscopic, 1\n")
    pml.write("hide everything\nshow cartoon\n\n")

    pml.write(f"select scaff_dna, scaffold and chain {sch1}+{sch2}\n")
    pml.write("select scaff_prot, scaffold and not scaff_dna\n")
    pml.write("show sticks, scaff_dna\n")
    pml.write("color gray80, scaff_prot\n")
    pml.write("color palecyan, scaff_dna\n")
    if have_mates:
        pml.write(f"\nselect mate_dna, (mate_bot or mate_top) and chain {sch1}+{sch2}\n")
        pml.write("select mate_prot, (mate_bot or mate_top) and not mate_dna\n")
        pml.write("show sticks, mate_dna\n")
        pml.write("color gray60, mate_prot\n")
        pml.write("color lightblue, mate_dna\n")
    pml.write("\n")

    pml.write("# Guests: forward = warm, reverse = cool\n")
    pml.write("color lightorange, fwd\n")
    pml.write("color lightpink, rev\n\n")

    pml.write("disable rev\n\n")

    pml.write("delete scaff_dna\ndelete scaff_prot\n")
    if have_mates:
        pml.write("delete mate_dna\ndelete mate_prot\n")
    pml.write("\norient scaffold\nzoom\n")

print(f"  PyMOL -> {pml_path}")
print(f"\nTo visualize: start pymol in this folder and then @{pml_path}")
