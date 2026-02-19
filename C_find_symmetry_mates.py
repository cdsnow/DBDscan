#!/usr/bin/env python3
"""C_find_symmetry_mates.py

Find the two symmetry mates whose dsDNA coaxially stacks with the scaffold
DNA (one on each end) and save them as PDB files in symmetry_mates/.

Approach: for every symmetry operation + unit cell translation, transform the
terminal base-pair C1' midpoints and check whether the transformed terminal
lands close to (< cutoff) the original's opposite terminal.  Coaxiality is
verified by checking that the DNA axis dot product is ~1.

Usage:
    python C_find_symmetry_mates.py
"""
import json, os
import numpy as np
from numpy import matrix, array
import itertools
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pyscaffoldscan'))
from pdbtools import PDB
from sgdata import sgqt

# ── Load scaffold ──────────────────────────────────────────────────────────
scaffold_code = sys.argv[1] if len(sys.argv) > 1 else '9YZJ'
scaffold_json = f'scaffold_models/{scaffold_code}.json'
with open(scaffold_json) as f:
    scaff = json.load(f)

P = PDB(f"scaffold_models/{scaff['pdb_id']}.pdb")
P.xtal_qtlist = sgqt[P.xtal_sg]

sch1, sch2 = scaff['dna_chains']  # A, B

print(f"Scaffold {scaff['pdb_id']}, space group {P.xtal_sg}")
print(f"  {len(P.xtal_qtlist)} symmetry operations")
print(f"  Unit cell: {P.xtal_edges}, angles {P.xtal_angles}")

# ── Terminal base-pair midpoints ───────────────────────────────────────────
def get_c1p(pdb, chain, resid):
    sel = pdb[f"chain {chain} and resi {resid} and name C1'"]
    d = sel.listdict[0]
    return np.array([d['x'], d['y'], d['z']])

# From scaffold JSON and base pair analysis (B_find_registers.py):
# A[1] <-> B[43]  (low end)
# A[31] <-> B[13] (high end)
resids_lo = {sch1: scaff['chains'][sch1]['resid_range'][0],
             sch2: scaff['chains'][sch2]['resid_range'][1]}
resids_hi = {sch1: scaff['chains'][sch1]['resid_range'][1],
             sch2: scaff['chains'][sch2]['resid_range'][0]}

end_lo = 0.5 * (get_c1p(P, sch1, resids_lo[sch1]) +
                get_c1p(P, sch2, resids_lo[sch2]))
end_hi = 0.5 * (get_c1p(P, sch1, resids_hi[sch1]) +
                get_c1p(P, sch2, resids_hi[sch2]))

dna_vec = end_hi - end_lo
dna_len = np.linalg.norm(dna_vec)
dna_axis = dna_vec / dna_len

print(f"\n  DNA end_lo ({sch1}:{resids_lo[sch1]}/{sch2}:{resids_lo[sch2]} midpoint):"
      f" [{end_lo[0]:.1f}, {end_lo[1]:.1f}, {end_lo[2]:.1f}]")
print(f"  DNA end_hi ({sch1}:{resids_hi[sch1]}/{sch2}:{resids_hi[sch2]} midpoint):"
      f" [{end_hi[0]:.1f}, {end_hi[1]:.1f}, {end_hi[2]:.1f}]")
print(f"  DNA axis:   [{dna_axis[0]:.3f}, {dna_axis[1]:.3f}, {dna_axis[2]:.3f}]")
print(f"  DNA length: {dna_len:.1f} A")

# ── Search for stacking mates ─────────────────────────────────────────────
CUTOFF = 10.0       # max distance between stacking terminal midpoints (A)
AXIS_DOT_MIN = 0.95 # min dot product for coaxiality

print(f"\nSearching symmetry copies (cutoff={CUTOFF} A, axis_dot>{AXIS_DOT_MIN})...")

top_hit = None  # mate stacking on the high end
bot_hit = None  # mate stacking on the low end

for sym_idx, m in enumerate(P.xtal_qtlist):
    for a, b, c in itertools.product([-1, 0, 1], repeat=3):
        mat4 = matrix([[m[0],m[1],m[2], m[9]+a],
                       [m[3],m[4],m[5], m[10]+b],
                       [m[6],m[7],m[8], m[11]+c],
                       [0,    0,   0,   1      ]])
        realmat = P.xtal_basis * mat4 * P.xtal_basis.I
        R = np.array(realmat[:3,:3])

        # Transform terminal midpoints (homogeneous column vector convention)
        t_lo = np.array(realmat * matrix(list(end_lo) + [1.0]).T)[:3].flatten()
        t_hi = np.array(realmat * matrix(list(end_hi) + [1.0]).T)[:3].flatten()

        # Skip identity
        if np.linalg.norm(t_lo - end_lo) < 0.1:
            continue

        # Coaxiality check
        mate_vec = t_hi - t_lo
        mate_axis = mate_vec / np.linalg.norm(mate_vec)
        axis_dot = abs(np.dot(dna_axis, mate_axis))
        if axis_dot < AXIS_DOT_MIN:
            continue

        is_pure_trans = np.allclose(R, np.eye(3), atol=0.01)

        # Top stacking: mate's low end near our high end
        d_top = np.linalg.norm(t_lo - end_hi)
        if d_top < CUTOFF:
            if top_hit is None or d_top < top_hit['dist']:
                top_hit = dict(sym_idx=sym_idx, cell=(a,b,c), dist=d_top,
                               pure_trans=is_pure_trans, axis_dot=axis_dot,
                               realmat=realmat, R=R)

        # Bottom stacking: mate's high end near our low end
        d_bot = np.linalg.norm(t_hi - end_lo)
        if d_bot < CUTOFF:
            if bot_hit is None or d_bot < bot_hit['dist']:
                bot_hit = dict(sym_idx=sym_idx, cell=(a,b,c), dist=d_bot,
                               pure_trans=is_pure_trans, axis_dot=axis_dot,
                               realmat=realmat, R=R)

# ── Report results ─────────────────────────────────────────────────────────
for label, hit in [('TOP', top_hit), ('BOT', bot_hit)]:
    if hit is None:
        print(f"\n  {label}: NOT FOUND")
        continue
    sym, cell = hit['sym_idx'], hit['cell']
    print(f"\n  {label}: sym_op={sym}, cell={cell}, gap={hit['dist']:.2f} A,"
          f" pure_translation={hit['pure_trans']}, axis_dot={hit['axis_dot']:.4f}")

assert top_hit and bot_hit, "Could not find both stacking mates"

# ── Generate and save PDB files ───────────────────────────────────────────
os.makedirs('symmetry_mates', exist_ok=True)

mate_info = {}
for label, hit in [('top', top_hit), ('bot', bot_hit)]:
    R = hit['R']
    T = np.array(hit['realmat'][:3, 3]).flatten()
    M = P.Clone()
    M.Rotate(R.T)
    M.Translate(T)

    pdb_file = f"symmetry_mates/{scaff['pdb_id']}_{label}.pdb"
    M.WritePDB(pdb_file)
    print(f"\n  Wrote {pdb_file}")

    # Verify: check terminal distances
    m_lo = 0.5 * (get_c1p(M, sch1, resids_lo[sch1]) +
                  get_c1p(M, sch2, resids_lo[sch2]))
    m_hi = 0.5 * (get_c1p(M, sch1, resids_hi[sch1]) +
                  get_c1p(M, sch2, resids_hi[sch2]))

    if label == 'top':
        gap = np.linalg.norm(m_lo - end_hi)
        print(f"    Verification: mate_lo to orig_hi = {gap:.2f} A")
    else:
        gap = np.linalg.norm(m_hi - end_lo)
        print(f"    Verification: mate_hi to orig_lo = {gap:.2f} A")

    mate_info[label] = {
        'pdb_file': pdb_file,
        'sym_op': hit['sym_idx'],
        'unit_cell': list(hit['cell']),
        'gap': round(float(hit['dist']), 2),
        'pure_translation': bool(hit['pure_trans']),
    }

# ── Save metadata ──────────────────────────────────────────────────────────
meta = {
    'scaffold': scaff['pdb_id'],
    'space_group': P.xtal_sg,
    'dna_chains': scaff['dna_chains'],
    'dna_axis': [round(float(x), 4) for x in dna_axis],
    'dna_length': round(float(dna_len), 1),
    'mates': mate_info,
}
meta_path = f'symmetry_mates/{scaffold_code}_mates.json'
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)
print(f"\n  Metadata saved to {meta_path}")
