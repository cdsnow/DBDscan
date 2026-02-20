"""pipeline.py — Bundled GuestScan pipeline for browser execution via PyScript/Pyodide.

Combines pyscaffoldscan internals (pdbtools, superimpy, sgdata) with pipeline
steps A–F adapted for in-memory operation (no file I/O, string-based PDB data).
"""
import re
import json
import itertools
import numpy as np
from numpy import (zeros, array, eye, vstack, linalg, equal, c_, r_, dot,
                   isnan, around, matrix)
from scipy.spatial import cKDTree

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: sgdata — Space group symmetry operations
# ══════════════════════════════════════════════════════════════════════════════
A = 0.5 * (2 ** 0.5)
B = 0.5 * (3 ** 0.5)
C = 0.5
D = 0.25
E = 0.75
F = 1.0 / 3
G = 2.0 / 3
H = 1.0 / 6
I_sg = 5.0 / 6  # renamed to avoid shadowing builtin

sgqt = {
    'P 1': [(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)],
    'P 2': [(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
            (-1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0)],
    'P 21': [(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
             (-1, 0, 0, 0, 1, 0, 0, 0, -1, 0, C, 0)],
    'C 2': [(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
            (-1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0),
            (1, 0, 0, 0, 1, 0, 0, 0, 1, C, C, 0),
            (-1, 0, 0, 0, 1, 0, 0, 0, -1, C, C, 0)],
    'I 2': [(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
            (-1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0),
            (1, 0, 0, 0, 1, 0, 0, 0, 1, C, C, C),
            (-1, 0, 0, 0, 1, 0, 0, 0, -1, C, C, C)],
    'P 2 2 2': [(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
                (-1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0),
                (1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0),
                (-1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0)],
    'P 21 21 21': [(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
                   (-1, 0, 0, 0, -1, 0, 0, 0, 1, C, 0, C),
                   (1, 0, 0, 0, -1, 0, 0, 0, -1, C, C, 0),
                   (-1, 0, 0, 0, 1, 0, 0, 0, -1, 0, C, C)],
    'C 2 2 21': [(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
                 (-1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, C),
                 (1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0),
                 (-1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, C),
                 (1, 0, 0, 0, 1, 0, 0, 0, 1, C, C, 0),
                 (-1, 0, 0, 0, -1, 0, 0, 0, 1, C, C, C),
                 (1, 0, 0, 0, -1, 0, 0, 0, -1, C, C, 0),
                 (-1, 0, 0, 0, 1, 0, 0, 0, -1, C, C, C)],
    'P 41 21 2': [(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
                  (0, -1, 0, 1, 0, 0, 0, 0, 1, C, C, D),
                  (-1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, C),
                  (0, 1, 0, -1, 0, 0, 0, 0, 1, C, C, E),
                  (1, 0, 0, 0, -1, 0, 0, 0, -1, C, C, D),
                  (0, 1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0),
                  (-1, 0, 0, 0, 1, 0, 0, 0, -1, C, C, E),
                  (0, -1, 0, -1, 0, 0, 0, 0, -1, 0, 0, C)],
    'P 43 21 2': [(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
                  (0, -1, 0, 1, 0, 0, 0, 0, 1, C, C, E),
                  (-1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, C),
                  (0, 1, 0, -1, 0, 0, 0, 0, 1, C, C, D),
                  (1, 0, 0, 0, -1, 0, 0, 0, -1, C, C, E),
                  (0, 1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0),
                  (-1, 0, 0, 0, 1, 0, 0, 0, -1, C, C, D),
                  (0, -1, 0, -1, 0, 0, 0, 0, -1, 0, 0, C)],
}

# Aliases
sgqt['C 1 2 1'] = sgqt['C 2']
sgqt['I 1 2 1'] = sgqt['I 2']
sgqt['P 1 21 1'] = sgqt['P 21']

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: superimpy — SVD-based coordinate superposition
# ══════════════════════════════════════════════════════════════════════════════

def _calc_a_P(Xc, Yc):
    """Setup variables for superposition eigenproblem."""
    A_vec = zeros(3)
    P_mat = zeros([3, 3])
    for k in range(len(Yc)):
        a = Yc[k]
        b = Xc[k]
        a0b0 = a[0] * b[0]
        a1b1 = a[1] * b[1]
        a2b2 = a[2] * b[2]
        a1b2 = a[1] * b[2]
        a2b1 = a[2] * b[1]
        a2b0 = a[2] * b[0]
        a0b2 = a[0] * b[2]
        a0b1 = a[0] * b[1]
        a1b0 = a[1] * b[0]
        a0b1a1b0 = a0b1 + a1b0
        a0b2a2b0 = a0b2 + a2b0
        a1b2a2b1 = a1b2 + a2b1
        A_vec += array([a1b2 - a2b1, a2b0 - a0b2, a0b1 - a1b0])
        P_mat += array([[-2 * a1b1 - 2 * a2b2, a0b1a1b0, a0b2a2b0],
                        [a0b1a1b0, -2 * a0b0 - 2 * a2b2, a1b2a2b1],
                        [a0b2a2b0, a1b2a2b1, -2 * a0b0 - 2 * a1b1]])
    return A_vec, P_mat


def superpose_rot_trans(X, Y):
    """Superimpose X onto Y. Returns (R, T) where aligned = X @ R + T."""
    EYE = eye(3)
    if equal(X, Y).all():
        return [EYE, array([0., 0., 0.])]

    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    a_vec, P_mat = _calc_a_P(Xc, Yc)

    gamma = vstack((c_[P_mat, a_vec], r_[a_vec, zeros(1)]))
    D_eig, V = linalg.eig(gamma)

    if abs(V[-1]).min() < 0.00001:
        return [EYE, Y.mean(0) - X.mean(0)]

    Vnorm = V / V[-1, :]
    bestSSD = 999999
    bestS = None
    for k in range(4):
        S = array([[0, -Vnorm[2, k], Vnorm[1, k]],
                    [Vnorm[2, k], 0, -Vnorm[0, k]],
                    [-Vnorm[1, k], Vnorm[0, k], 0]])
        if linalg.det(EYE - S) != 0:
            Q = linalg.inv(dot(EYE + S, linalg.inv(EYE - S)))
            SSD = ((dot(Q, Xc.T).T - Yc) ** 2).sum()
            if isnan(SSD):
                return [EYE, array([0., 0., 0.])]
            if SSD < bestSSD:
                bestSSD = SSD
                bestS = S.copy()

    if bestS is None:
        return [EYE, Y.mean(0) - X.mean(0)]

    rotation = linalg.inv(dot(EYE + bestS, linalg.inv(EYE - bestS)))
    translation = Y.mean(0) - dot(rotation, X.mean(0))
    R = around(rotation.T, decimals=12)
    T = around(translation, decimals=12)
    return R, T


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: pdbtools — PDB class adapted for string I/O
# ══════════════════════════════════════════════════════════════════════════════

def DummyAtom(x, y, z, aname='D', rname='DUM', rnum=1, anum=1, elem='D',
              atype='ATOM  ', altloc=' ', chain=' ', occupancy=1.00, temp=0):
    """Format an ATOM/HETATM record."""
    values = (atype, anum, aname, altloc, rname, chain, rnum, x, y, z,
              occupancy, temp, elem)
    if aname[0].isdigit():
        return '%6s%5d %-4s%1s%3s %1s%4d    %8.3f%8.3f%8.3f %5.2f %5.2f          %2s' % values
    else:
        return '%6s%5d  %-3s%1s%3s %1s%4d    %8.3f%8.3f%8.3f %5.2f %5.2f          %2s' % values


def first_model_string(pdb_string):
    """Extract first MODEL from an NMR ensemble PDB string.
    Returns the PDB string with only the first model."""
    lines = pdb_string.split('\n')
    has_model = False
    for line in lines:
        if line.startswith('ATOM'):
            if not has_model:
                return pdb_string
            break
        if line.startswith('MODEL'):
            has_model = True
            break
    if not has_model:
        return pdb_string

    result = []
    for line in lines:
        if line.startswith('ENDMDL'):
            break
        if not line.startswith('MODEL'):
            result.append(line)
    return '\n'.join(result)


class PDB:
    """PDB object that stores structure data parsed from a PDB string.

    Usage:
        P = PDB(pdb_string)
        P = PDB(pdb_string, hetatm=True)
    """

    def __init__(self, arg=None, hetatm=False, verbose=False):
        if arg is None:
            pass  # Cloning creates empty PDB
        elif isinstance(arg, str):
            self.path = None
            self.orig = arg
            self.Parse(hetatm=hetatm, verbose=verbose)
            if hasattr(self, 'orig'):
                del self.orig
        else:
            raise ValueError("PDB() requires a PDB-format string")

    def __repr__(self):
        if not hasattr(self, 'listdict') or not self.listdict:
            return '<PDB object (empty)>'
        n = self.num_segments
        if n == 1:
            if min(self.resids) == max(self.resids):
                return '<PDB: residue %d: %s>' % (max(self.resids), self.seq)
            else:
                return '<PDB: residues %d-%d: %s>' % (min(self.resids), max(self.resids), self.seq)
        else:
            return '<PDB: %d segments, %d atoms>' % (n, self.num_atoms)

    def __str__(self):
        return self.current

    def __len__(self):
        return self.num_alpha_carbons

    def __add__(self, other):
        return PDB(str(self) + str(other))

    def __getitem__(self, selection):
        """PyMOL-like selection syntax:
        P['chain A'], P['chain A and resi 1-10'], P['name CA']
        """
        if isinstance(selection, int):
            seg2chainresids = self._get_segment2chainresids()
            assert selection in seg2chainresids
            chain, lo, hi = seg2chainresids[selection]
            selection = 'chain %s and resi > %d and resi < %d' % (chain, lo - 1, hi + 1)
        elif isinstance(selection, slice):
            return self.JustResidues(self.resids[selection])

        selection = re.sub(r'\bresi\s+(\d+)\b(?!\s*-\s*\d)', r'resi == \1', selection)
        selection = re.sub(r'\bresi\s+(\d+)-(\d+)\b',
                           lambda m: f'resi > {int(m.group(1)) - 1} and resi < {int(m.group(2)) + 1}',
                           selection)
        selection = re.sub(r'([()<>])', r' \1 ', selection)
        selection = selection.replace('==', ' == ')

        properties = {'num', 'x', 'y', 'z', 'bfact', 'q', 'resi', 'line'}
        props = {'name', 'resn', 'chain', 'elem', 'altloc'}
        boolops = {'and', 'AND', 'or', 'OR', 'not', '(', ')', '==', '<', '>'}
        words = selection.split()
        replaced = []
        lastword = None
        for word in words:
            if "'" in word:
                replaced.append('"%s"' % word)
            elif word in properties:
                replaced.append("atom['%s']" % word)
            elif word in props:
                replaced.append("atom['%s'] ==" % word)
            elif word.isalpha() and word not in boolops:
                replaced.append("'%s'" % word)
            elif word.isalnum() and not word.isdigit() and word not in boolops:
                replaced.append("'%s'" % word)
            elif word not in boolops and lastword == 'chain':
                replaced.append("'%s'" % word)
            else:
                replaced.append(word)
            lastword = word
        criteria = ' '.join(replaced)
        matchlines = [atom['line'] for atom in self.listdict if eval(criteria)]
        assert matchlines != [], 'No matches for ' + selection
        matchstr = '\n'.join(matchlines)
        hetatm = self.hetatmflag
        return PDB(matchstr, hetatm=hetatm)

    def _get_segment2chainresids(self):
        self._update_chainresi2seg()
        segments = sorted(set(self.chainresi2seg.values()))
        segment2chainresids = dict((seg, (' ', 999999, -999999)) for seg in segments)
        for (chain, resi), segment in list(self.chainresi2seg.items()):
            if resi > segment2chainresids[segment][2] and resi < segment2chainresids[segment][1]:
                segment2chainresids[segment] = (chain, resi, resi)
            elif resi > segment2chainresids[segment][2]:
                segment2chainresids[segment] = (chain, segment2chainresids[segment][1], resi)
            elif resi < segment2chainresids[segment][1]:
                segment2chainresids[segment] = (chain, resi, segment2chainresids[segment][2])
        return segment2chainresids

    def _update_current_str_from_listdict(self):
        self.current = ''
        self.num_atoms = 0
        for d in self.listdict:
            if (d['chain'], d['resi'], d['name']) in self.removed_atoms:
                continue
            self.current += d['line'] + '\n'
            self.num_atoms += 1

    def _update_pdbstrings_from_listdict(self, verbose=False):
        lastresid = self.listdict[0]['resi']
        lastchain = self.listdict[0]['chain']
        pdbstrings = []
        pdbstring = ''
        for atom in self.listdict:
            resi, chain, line = atom['resi'], atom['chain'], atom['line']
            if resi not in [lastresid, lastresid + 1] or chain != lastchain:
                pdbstrings.append(pdbstring)
                pdbstring = ''
            pdbstring += line + '\n'
            lastresid, lastchain = resi, chain
        pdbstrings.append(pdbstring)
        self.pdbstrings = pdbstrings

    def SetChain(self, chain):
        """Set chain ID for all atoms."""
        if isinstance(chain, str) and len(chain) == 1:
            for d in self.listdict:
                d['chain'] = chain
                d['line'] = d['line'][:21] + chain + d['line'][22:]
            self._update_current_str_from_listdict()
            self.chainresi2seg = {(chain, resi): seg for (x, resi), seg in self.chainresi2seg.items()}
            self.seg2chain = {seg: chain for seg, x in self.seg2chain.items()}
            self.resndict = {chain: rdict for x, rdict in self.resndict.items()}
            self.namedict = {chain: rdict for x, rdict in self.namedict.items()}
            self.chain = chain
        elif isinstance(chain, dict):
            for d in self.listdict:
                newchain = chain[d['chain']]
                d['chain'] = newchain
                d['line'] = d['line'][:21] + newchain + d['line'][22:]
            self._update_current_str_from_listdict()
            self.chainresi2seg = dict(((chain[x], resi), seg) for (x, resi), seg in list(self.chainresi2seg.items()))
            self.seg2chain = dict((seg, chain[x]) for seg, x in list(self.seg2chain.items()))
            if len(set(self.seg2chain.values())) == 1:
                self.chain = set(self.seg2chain.values()).pop()
            self.resndict = dict((chain[x], rdict) for x, rdict in list(self.resndict.items()))
            self.namedict = dict((chain[x], rdict) for x, rdict in list(self.namedict.items()))

    def Rotate(self, R):
        """Apply rotation matrix to all coordinates."""
        if isinstance(R, list):
            R = np.array(R)
        if R.shape == (1, 4):
            a, b, c, d = R
            R = np.zeros([3, 3])
            R[0][0] = a * a + b * b - c * c - d * d
            R[0][1] = 2 * (b * c - a * d)
            R[0][2] = 2 * (b * d + a * c)
            R[1][0] = 2 * (b * c + a * d)
            R[1][1] = a * a - b * b + c * c - d * d
            R[1][2] = 2 * (c * d - a * b)
            R[2][0] = 2 * (b * d - a * c)
            R[2][1] = 2 * (c * d + a * b)
            R[2][2] = a * a - b * b - c * c + d * d
        for d in self.listdict:
            oldline, x, y, z = d['line'], d['x'], d['y'], d['z']
            X = R[0][0] * x + R[1][0] * y + R[2][0] * z
            Y = R[0][1] * x + R[1][1] * y + R[2][1] * z
            Z = R[0][2] * x + R[1][2] * y + R[2][2] * z
            d['line'] = oldline[:30] + '%8.3f%8.3f%8.3f' % (X, Y, Z) + oldline[54:]
            d['x'], d['y'], d['z'] = X, Y, Z
        x, y, z = self.cog
        X = R[0][0] * x + R[1][0] * y + R[2][0] * z
        Y = R[0][1] * x + R[1][1] * y + R[2][1] * z
        Z = R[0][2] * x + R[1][2] * y + R[2][2] * z
        self.cog[0], self.cog[1], self.cog[2] = X, Y, Z
        self._update_current_str_from_listdict()

    def Translate(self, T):
        """Apply translation vector to all coordinates."""
        for d in self.listdict:
            oldline, x, y, z = d['line'], d['x'], d['y'], d['z']
            X, Y, Z = T[0] + x, T[1] + y, T[2] + z
            d['line'] = oldline[:30] + '%8.3f%8.3f%8.3f' % (X, Y, Z) + oldline[54:]
            d['x'], d['y'], d['z'] = X, Y, Z
        self.cog = self.GetCoords().mean(0)
        self._update_current_str_from_listdict()

    def Clone(self):
        """Create a deep copy of this PDB object."""
        p = PDB()
        props_list = ['removed_atoms', 'pdbstrings', 'segmentseqs', 'resids',
                      'resIDs', 'first_resids', 'partials']
        for prop in props_list:
            if hasattr(self, prop):
                setattr(p, prop, getattr(self, prop)[:])

        props_scalar = ['num_atoms', 'current', 'chain', 'seq', 'radius',
                        'num_alpha_carbons', 'mw', 'num_segments', 'hetatmflag']
        for prop in props_scalar:
            if hasattr(self, prop):
                setattr(p, prop, getattr(self, prop))

        props_dict = ['three2one', 'res_heavies', 'res_mass',
                      'namedict', 'resndict', 'chainresi2seg', 'seg2chain']
        for prop in props_dict:
            if hasattr(self, prop):
                setattr(p, prop, getattr(self, prop).copy())

        if hasattr(self, 'cog'):
            p.cog = self.cog.copy()

        # Crystal data
        for attr in ['xtal_sg', 'xtal_edges', 'xtal_angles', 'xtal_basis',
                      'xtal_va', 'xtal_vb', 'xtal_vc', 'xtal_num_in_unit_cell']:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if hasattr(val, 'copy'):
                    setattr(p, attr, val.copy())
                else:
                    setattr(p, attr, val)

        p.listdict = [x.copy() for x in self.listdict]
        return p

    def GetCoords(self):
        """Return coordinates as Nx3 numpy array."""
        return np.array([(d['x'], d['y'], d['z']) for d in self.listdict])

    def _snap_to_zero(self, v, epsilon=1e-9):
        indices = [i for i, x in enumerate(v) if abs(x) < epsilon]
        v.put(indices, [0.0])

    def _parse_xtal(self, line):
        """Parse CRYST1 header line for unit cell info."""
        l = line.split()
        self.xtal_edges = [float(l[1]), float(l[2]), float(l[3])]
        self.xtal_angles = [float(l[4]), float(l[5]), float(l[6])]
        self.xtal_sg = line[55:67].strip()
        self.xtal_num_in_unit_cell = int(l[-1])
        rad = [np.radians(x) for x in self.xtal_angles]
        B_mat = np.identity(4)
        B_mat[0][1] = np.cos(rad[2])
        B_mat[1][1] = np.sin(rad[2])
        B_mat[0][2] = np.cos(rad[1])
        B_mat[1][2] = (np.cos(rad[0]) - B_mat[0][1] * B_mat[0][2]) / B_mat[1][1]
        B_mat[2][2] = np.sqrt(1 - B_mat[0][2] ** 2 - B_mat[1][2] ** 2)
        self.xtal_basis = matrix(B_mat * (self.xtal_edges + [1.0]))
        self.xtal_va = np.array(self.xtal_basis[:3, :3].T)[0]
        self.xtal_vb = np.array(self.xtal_basis[:3, :3].T)[1]
        self.xtal_vc = np.array(self.xtal_basis[:3, :3].T)[2]
        self._snap_to_zero(self.xtal_va)
        self._snap_to_zero(self.xtal_vb)
        self._snap_to_zero(self.xtal_vc)

    def _update_chainresi2seg(self):
        chainresi2seg = {}
        segmentseqs = {}
        seg = 0
        lastchain = self.listdict[0]['chain']
        lastresi = self.listdict[0]['resi']
        chainresi2seg[(lastchain, lastresi)] = seg
        for x in self.listdict:
            if x['chain'] != lastchain or x['resi'] < lastresi or x['resi'] > lastresi + 1:
                seg += 1
            if x['name'] == 'CA' and x['altloc'] in [' ', 'A'] and x['resn'] in self.three2one:
                try:
                    segmentseqs[seg] += self.three2one[x['resn']]
                except KeyError:
                    segmentseqs[seg] = self.three2one[x['resn']]
            chainresi2seg[(x['chain'], x['resi'])] = seg
            lastchain, lastresi = x['chain'], x['resi']
        self.chainresi2seg = chainresi2seg
        self.seg2chain = dict((seg, chain) for (chain, resi), seg in list(chainresi2seg.items()))
        self.num_segments = len(self.seg2chain)
        self.segmentseqs = [x[1] for x in sorted(segmentseqs.items())]

    def Parse(self, hetatm=False, verbose=False):
        """Parse the PDB string."""
        self.three2one = {
            'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'ASN': 'N', 'GLN': 'Q',
            'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F',
            'ALA': 'A', 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R',
            'TRP': 'W', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
            'MSE': 'M'}

        self.removed_atoms = []
        self.hetatms = ''
        self.hetatmflag = hetatm
        self.res_heavies = {
            'ALA': ['N', 'CA', 'C', 'O', 'CB'],
            'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
            'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
            'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
            'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'GLY': ['N', 'CA', 'C', 'O'],
            'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
            'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CD1', 'CG2'],
            'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
            'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
            'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
            'MSE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SE', 'CE'],
            'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
            'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
            'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
            'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
            'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
            'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
            'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
            'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
            'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH']
        }
        self.res_mass = {
            'ALA': 71.09, 'CYS': 103.15, 'ASP': 115.09, 'GLU': 129.12,
            'PHE': 147.18, 'GLY': 57.05, 'HIS': 137.14, 'ILE': 113.16,
            'LYS': 128.17, 'LEU': 113.16, 'MET': 131.19, 'ASN': 114.11,
            'PRO': 97.12, 'GLN': 128.14, 'ARG': 156.19, 'SER': 87.08,
            'THR': 101.11, 'VAL': 99.14, 'TRP': 186.21, 'TYR': 163.18,
            'MSE': 131.19}

        segment, lastresnum, lastchain = 0, -999, 'Z'
        pdbstrings, segmentseqs, listdict = [], [], []
        first_resids, resids, resIDs = [], [], []
        resndict, namedict, chainresi2seg = {}, {}, {}
        seq, pdbstring = '', ''
        self.num_atoms = 0

        lines = self.orig.split('\n')
        for line in lines:
            if len(line) < 4:
                continue
            if line[:6] == 'CRYST1':
                self._parse_xtal(line)
            if not hetatm and line[:4] != 'ATOM':
                continue
            if line[:6] == 'HETATM':
                self.hetatms += line + '\n'
            if hetatm and line[:6] != 'HETATM' and line[:4] != 'ATOM':
                continue

            num = int(line[6:11])
            name = line[12:16].strip()
            altloc = line[16]
            resn = line[17:20]
            chain = line[21]
            resi = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            try:
                occup = float(line[54:60])
            except:
                occup = None
            try:
                bfact = float(line[60:66])
            except:
                bfact = None
            try:
                elem = line[76:78].strip()
            except:
                elem = None
            try:
                q = float(line[78:80])
            except:
                q = None

            if chain not in resndict:
                resndict[chain] = {}
            if chain not in namedict:
                namedict[chain] = {}

            if resi != lastresnum or chain != lastchain:
                resndict[chain][resi] = resn
                namedict[chain][resi] = []
                resids.append(resi)
                resIDs.append((resi, chain))
                if resi != lastresnum + 1 or chain != lastchain:
                    first_resids.append(resi)
                    if lastresnum != -999:
                        pdbstrings.append(pdbstring)
                        pdbstring = ''
                        if seq != '':
                            segmentseqs.append(seq)
                        seq = ''
                        segment += 1
                chainresi2seg[(chain, resi)] = segment

            namedict[chain][resi].append(name)
            d = {'num': num, 'name': name, 'altloc': altloc, 'resn': resn,
                 'chain': chain, 'resi': resi, 'x': x, 'y': y, 'z': z,
                 'occup': occup, 'bfact': bfact, 'elem': elem, 'q': q,
                 'line': line}
            listdict.append(d)
            if name in ['OXT', '1H', '2H', '3H', 'H1', 'H2', 'H3']:
                self.removed_atoms.append((chain, resi, name))
            else:
                pdbstring += line + '\n'
                self.num_atoms += 1
            if name == 'CA' and altloc in [' ', 'A']:
                if resn in self.three2one:
                    seq += self.three2one[resn]
                else:
                    seq += '?'
            elif name == "C1'" and altloc in [' ', 'A']:
                seq += resn[-1]
            lastresnum, lastchain = resi, chain

        pdbstrings.append(pdbstring)
        if seq != '':
            segmentseqs.append(seq)

        assert listdict, 'No coordinates parsed'

        self.pdbstrings = pdbstrings
        self.current = ''.join(pdbstrings)
        self.segmentseqs = segmentseqs
        self.seq = ''.join(segmentseqs)
        self.resids = resids
        self.resIDs = resIDs
        self.first_resids = first_resids
        self.listdict = listdict
        self.cog = self.GetCoords().mean(0)
        self.radius = (((self.GetCoords() - self.cog) ** 2).sum(1) ** 0.5).max()

        alpha_listdict = [atom for atom in self.listdict
                          if atom['name'] == 'CA' and atom['altloc'] in [' ', 'A']]
        self.num_alpha_carbons = len(alpha_listdict)
        try:
            self.mw = sum(self.res_mass[atom['resn']] for atom in alpha_listdict)
        except KeyError:
            self.mw = None
        self.namedict = namedict
        self.resndict = resndict
        self.chainresi2seg = chainresi2seg
        self.seg2chain = dict((seg, chain) for (chain, resi), seg in list(chainresi2seg.items()))
        self.num_segments = len(self.segmentseqs)

        if len(set(self.seg2chain.values())) == 1:
            self.chain = set(self.seg2chain.values()).pop()

        self.partials = []
        for chain in self.resndict:
            for resi, resn in list(self.resndict[chain].items()):
                names = self.namedict[chain][resi]
                try:
                    wanted = self.res_heavies[resn]
                    segment = chainresi2seg[(chain, resi)]
                    if not set(wanted).issubset(names):
                        self.partials.append((segment, resi))
                except:
                    pass


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: Shared helper functions
# ══════════════════════════════════════════════════════════════════════════════
WC = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}


def wc_complement(seq):
    return ''.join(WC[b] for b in seq)


def identify_dna_chains(P):
    """Identify DNA chains based on residue names."""
    dna_resnames = {'DA', 'DT', 'DC', 'DG'}
    dna_chains = []
    for chain in sorted(P.resndict.keys()):
        resnames = {rn.strip() for rn in P.resndict[chain].values()}
        if resnames <= dna_resnames:
            dna_chains.append(chain)
    return dna_chains


def identify_protein_chains(P):
    """Identify protein chains (everything not DNA)."""
    dna = set(identify_dna_chains(P))
    return [ch for ch in sorted(P.resndict.keys()) if ch not in dna]


def select_chains(P, chains):
    """Select atoms from specified chains."""
    expr = ' or '.join(f'chain {ch}' for ch in chains)
    return P[expr]


def get_dna_sequence(P, chain):
    """Get DNA sequence for a chain, ordered by residue ID."""
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
    return f"m{abs(p):02d}" if p < 0 else f"{p:02d}"


def reg_obj_name(start, end, orient):
    return f"R{format_pos(start)}to{format_pos(end)}_{orient}"


# Core atoms for clash assessment
CORE_ATOM_NAMES = {'N', 'CA', 'C', 'O', 'CB'}
PRO_EXTRA_ATOMS = {'CG', 'CD'}


def is_core_atom(atom_name, res_name):
    name = atom_name.strip()
    if name in CORE_ATOM_NAMES:
        return True
    if res_name.strip() == 'PRO' and name in PRO_EXTRA_ATOMS:
        return True
    return False


# ── Parameters ──
CLASH_CUTOFF = 2.0
INTERACT_CUTOFF = 8.0
ENV_CUTOFF = 50.0
N_CLOSEST = 8
CORE_SEQ = 'TGTGACAAATTGCCCTCAG'

CAT_NAMES = {
    1: 'clash_scaff_prot', 2: 'footprint_overlap', 3: 'clash_sym_mate',
    4: 'near_neighbors', 5: 'independent',
}


def env_obj_name_str(sym_idx, cell):
    a, b, c = cell
    def fmt(v):
        return f"m{abs(v)}" if v < 0 else str(v)
    return f"env_s{sym_idx}_{fmt(a)}_{fmt(b)}_{fmt(c)}"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: Pipeline step functions
# ══════════════════════════════════════════════════════════════════════════════

def validate_guest_selection(pdb_string, protein_chains, dna1_chain, dna1_lo,
                             dna1_hi, dna2_chain, dna2_lo, dna2_hi):
    """Validate the guest selection. Returns (ok, message, info_dict)."""
    try:
        P = PDB(pdb_string)
    except Exception as e:
        return False, f"Failed to parse PDB: {e}", {}

    all_chains = sorted(P.resndict.keys())
    info = {'chains': all_chains, 'chain_info': {}}
    for ch in all_chains:
        resids = sorted(P.resndict[ch].keys())
        resnames = {rn.strip() for rn in P.resndict[ch].values()}
        dna_resnames = {'DA', 'DT', 'DC', 'DG'}
        is_dna = resnames <= dna_resnames
        info['chain_info'][ch] = {
            'type': 'DNA' if is_dna else 'protein',
            'resid_range': [min(resids), max(resids)] if resids else [0, 0],
            'num_residues': len(resids),
        }

    # Check chains exist
    for ch in protein_chains:
        if ch not in all_chains:
            return False, f"Protein chain '{ch}' not found. Available: {all_chains}", info

    for label, ch, lo, hi in [('DNA1', dna1_chain, dna1_lo, dna1_hi),
                               ('DNA2', dna2_chain, dna2_lo, dna2_hi)]:
        if ch not in all_chains:
            return False, f"{label} chain '{ch}' not found. Available: {all_chains}", info
        resids = sorted(P.resndict[ch].keys())
        if lo < min(resids) or hi > max(resids):
            return False, f"{label} range {lo}-{hi} out of bounds (chain {ch}: {min(resids)}-{max(resids)})", info
        n_bp = hi - lo + 1
        if n_bp < 3:
            return False, f"{label} needs at least 3 base pairs, got {n_bp}", info

    n1 = dna1_hi - dna1_lo + 1
    n2 = dna2_hi - dna2_lo + 1
    if n1 != n2:
        return False, f"DNA strand lengths differ: {n1} vs {n2}", info

    info['window_size'] = n1
    return True, f"Valid: {len(protein_chains)} protein chain(s), {n1}-bp DNA window", info


def step_A_process_guest(pdb_string, protein_chains, dna1_chain, dna1_lo,
                         dna1_hi, dna2_chain, dna2_lo, dna2_hi, hetatm=False):
    """Process guest: extract selected chains/residues. Returns PDB string."""
    P = PDB(first_model_string(pdb_string), hetatm=hetatm)

    sele_parts = [f'chain {ch}' for ch in protein_chains]
    sele_parts.append(f'(chain {dna1_chain} and resi {dna1_lo}-{dna1_hi})')
    sele_parts.append(f'(chain {dna2_chain} and resi {dna2_lo}-{dna2_hi})')
    sele = ' or '.join(sele_parts)

    graft = P[sele]
    return str(graft)


def step_B_find_registers(guest_pdb_str, scaffold_pdb_str, scaffold_json,
                          mates_json=None, bot_pdb_str=None, top_pdb_str=None,
                          progress_cb=None):
    """Enumerate all sliding-window registers. Returns dict with results."""
    scaff = scaffold_json if isinstance(scaffold_json, dict) else json.loads(scaffold_json)
    scaffold = PDB(scaffold_pdb_str)
    sch1, sch2 = scaff['dna_chains']

    scaff_resids, scaff_coords, scaff_seq = {}, {}, {}
    for ch in [sch1, sch2]:
        scaff_resids[ch], scaff_seq[ch] = get_dna_sequence(scaffold, ch)
        _, scaff_coords[ch] = get_c1prime_coords(scaffold, ch)

    N = len(scaff_resids[sch1])

    shift, wc_count, overlap_bp, bp_pairs = find_base_pairing(
        scaff_resids[sch1], scaff_seq[sch1],
        scaff_resids[sch2], scaff_seq[sch2])

    # Identify top chain
    top_chain = None
    for ch in [sch1, sch2]:
        if 'AAATT' in scaff_seq[ch]:
            top_chain = ch
            break
    if top_chain is None:
        top_chain = sch1

    if top_chain != sch1:
        sch1, sch2 = sch2, sch1

    # Build extended DNA
    have_mates = mates_json is not None
    if have_mates:
        mates = mates_json if isinstance(mates_json, dict) else json.loads(mates_json)
        bot_pdb = PDB(bot_pdb_str)
        top_pdb = PDB(top_pdb_str)

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
    else:
        ext_s1_coords = scaff_coords[sch1]
        ext_s2_coords = scaff_coords[sch2][::-1]
        ext_s1_seq = scaff_seq[sch1]
        ext_s2_seq = scaff_seq[sch2][::-1]
        N_ext = N
        asu_start = 0
        asu_end = N

    # Load guest
    guest = PDB(guest_pdb_str)
    guest_dna = identify_dna_chains(guest)
    assert len(guest_dna) == 2, f"Expected 2 DNA chains, found {guest_dna}"
    gch1, gch2 = guest_dna

    guest_resids_dict, guest_coords_dict = {}, {}
    for ch in [gch1, gch2]:
        guest_resids_dict[ch], guest_coords_dict[ch] = get_c1prime_coords(guest, ch)

    W = len(guest_resids_dict[gch1])
    mobile = np.vstack([guest_coords_dict[gch1], guest_coords_dict[gch2]])
    mobile_rev = np.vstack([guest_coords_dict[gch2], guest_coords_dict[gch1]])

    # Enumerate registers
    j_lo = max(0, asu_start - W + 1)
    j_hi = min(N_ext - W, asu_end - 1)
    total_regs = 2 * (j_hi - j_lo + 1)

    results = []
    register_pdbs = {}
    reg = 0

    for orient in ['fwd', 'rev']:
        mob = mobile if orient == 'fwd' else mobile_rev
        for j in range(j_lo, j_hi + 1):
            start_pos = j - asu_start + 1
            end_pos = start_pos + W - 1
            obj_name = reg_obj_name(start_pos, end_pos, orient)

            if progress_cb:
                progress_cb('Finding registers', reg + 1, total_regs)

            tgt_s1 = ext_s1_coords[j:j + W]
            tgt_s2 = ext_s2_coords[j:j + W][::-1]
            target = np.vstack([tgt_s1, tgt_s2])

            rmsd, R, T = superimpose_and_rmsd(mob, target)

            if j >= asu_start and j + W <= asu_end:
                region = 'asu'
            elif have_mates:
                region = 'junc_bot' if j < asu_start else 'junc_top'
            else:
                region = 'asu'

            aligned_guest = guest.Clone()
            aligned_guest.Rotate(np.array(R))
            aligned_guest.Translate(np.array(T).flatten())
            pdb_str = str(aligned_guest)
            register_pdbs[obj_name] = pdb_str

            results.append({
                'register': reg,
                'label': f"R:{start_pos}:{end_pos}",
                'obj_name': obj_name,
                'orientation': orient,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'ext_offset': j,
                'region': region,
                'rmsd': round(float(rmsd), 4),
                'pdb_string': pdb_str,
            })
            reg += 1

    return {
        'top_chain': top_chain,
        'window_size': W,
        'asu_bp': N,
        'extended_bp': N_ext,
        'asu_range': [asu_start, asu_end],
        'num_registers': len(results),
        'registers': results,
        'register_pdbs': register_pdbs,
    }


def _generate_crystal_environment(scaffold, prot_coords_asu, dna_coords_asu,
                                  coaxial_keys, core_prot_coords_asu=None):
    """Generate symmetry copies within ENV_CUTOFF of the ASU."""
    scaffold.xtal_qtlist = sgqt[scaffold.xtal_sg]

    asu_coords = scaffold.GetCoords()
    asu_center = asu_coords.mean(axis=0)
    asu_radius = np.linalg.norm(asu_coords - asu_center, axis=1).max()
    cog_cutoff = ENV_CUTOFF + 2 * asu_radius
    asu_tree = cKDTree(asu_coords)

    env_copies = []
    for sym_idx, m in enumerate(scaffold.xtal_qtlist):
        for a, b, c in itertools.product([-1, 0, 1], repeat=3):
            mat4 = matrix([[m[0], m[1], m[2], m[9] + a],
                           [m[3], m[4], m[5], m[10] + b],
                           [m[6], m[7], m[8], m[11] + c],
                           [0, 0, 0, 1]])
            realmat = scaffold.xtal_basis * mat4 * scaffold.xtal_basis.I
            R = np.array(realmat[:3, :3])
            T_vec = np.array(realmat[:3, 3]).flatten()

            copy_center = R @ asu_center + T_vec
            d_cog = np.linalg.norm(copy_center - asu_center)
            if d_cog < 0.1 or d_cog > cog_cutoff:
                continue

            copy_coords = (asu_coords @ R.T) + T_vec
            d_to_asu, _ = asu_tree.query(copy_coords, k=1)
            min_dist = float(d_to_asu.min())
            if min_dist > ENV_CUTOFF:
                continue

            copy_prot = (prot_coords_asu @ R.T) + T_vec
            copy_dna = (dna_coords_asu @ R.T) + T_vec
            copy_core_prot = ((core_prot_coords_asu @ R.T) + T_vec
                              if core_prot_coords_asu is not None
                              else np.empty((0, 3)))

            key = (sym_idx, (a, b, c))
            is_coaxial = key in coaxial_keys

            env_copies.append({
                'prot_coords': copy_prot,
                'core_prot_coords': copy_core_prot,
                'dna_coords': copy_dna,
                'R': R, 'T': T_vec,
                'sym_idx': sym_idx, 'cell': (a, b, c),
                'obj_name': env_obj_name_str(sym_idx, (a, b, c)),
                'is_coaxial': is_coaxial,
                'd_cog': round(float(d_cog), 1),
                'd_min': round(min_dist, 1),
            })

    env_copies.sort(key=lambda x: x['d_min'])

    for copy in env_copies:
        if not copy['is_coaxial'] and len(copy['dna_coords']) > 0:
            copy['dna_tree'] = cKDTree(copy['dna_coords'])
        else:
            copy['dna_tree'] = None
        copy['prot_tree'] = (cKDTree(copy['prot_coords'])
                             if len(copy['prot_coords']) > 0 else None)
        copy['core_prot_tree'] = (cKDTree(copy['core_prot_coords'])
                                  if len(copy['core_prot_coords']) > 0 else None)

    return env_copies


def _classify_register(reg, scaff_prot_tree, scaff_prot_coords, env_copies,
                       core_start, core_end):
    """Classify a single register placement."""
    guest = PDB(reg['pdb_string'])
    guest_prot = identify_protein_chains(guest)
    guest_prot_coords = select_chains(guest, guest_prot).GetCoords()

    # SPR: nearest scaffold protein
    min_dist_prot = 999.0
    spr_source = None
    d_asu, idx_asu = scaff_prot_tree.query(guest_prot_coords, k=1)
    d_min_asu = float(d_asu.min())
    if d_min_asu < min_dist_prot:
        min_dist_prot = d_min_asu
        spr_source = 'ASU'

    for ci, copy in enumerate(env_copies):
        pt = copy['prot_tree']
        if pt is None:
            continue
        d, idx = pt.query(guest_prot_coords, k=1)
        d_min = float(d.min())
        if d_min < min_dist_prot:
            min_dist_prot = d_min
            spr_source = copy['obj_name']

    # SYM: nearest non-coaxial DNA
    min_dist_dna = 999.0
    for ci, copy in enumerate(env_copies):
        dt = copy['dna_tree']
        if dt is None:
            continue
        d, idx = dt.query(guest_prot_coords, k=1)
        d_min = float(d.min())
        if d_min < min_dist_dna:
            min_dist_dna = d_min

    # GSY: nearest guest symmetry copy protein
    min_dist_gsym = 999.0
    guest_prot_tree = cKDTree(guest_prot_coords)
    for ci, copy in enumerate(env_copies):
        R, T_vec = copy['R'], copy['T']
        gsym_coords = (guest_prot_coords @ R.T) + T_vec
        d, idx = guest_prot_tree.query(gsym_coords, k=1)
        d_min = float(d.min())
        if d_min < min_dist_gsym:
            min_dist_gsym = d_min

    min_dist_sym_all = min(min_dist_dna, min_dist_gsym)
    min_dist_any = min(min_dist_prot, min_dist_sym_all)

    s, e = reg['start_pos'], reg['end_pos']
    footprint_overlaps = (s <= core_end and e >= core_start)

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
        'register': reg['register'],
        'label': reg['label'],
        'obj_name': reg['obj_name'],
        'orientation': reg['orientation'],
        'start_pos': reg['start_pos'],
        'end_pos': reg['end_pos'],
        'region': reg['region'],
        'rmsd': reg['rmsd'],
        'pdb_string': reg['pdb_string'],
        'category': category,
        'category_name': CAT_NAMES[category],
        'footprint_overlaps': footprint_overlaps,
        'min_dist_prot': round(min_dist_prot, 2),
        'min_dist_sym_dna': round(min_dist_dna, 2),
        'min_dist_guest_sym': round(min_dist_gsym, 2),
        'min_dist_any': round(min_dist_any, 2),
        'closest_prot': spr_source,
    }


def step_D_categorize(registers_data, scaffold_pdb_str, scaffold_json,
                      mates_json=None, progress_cb=None):
    """Classify registers. Returns categorized register list."""
    scaff = scaffold_json if isinstance(scaffold_json, dict) else json.loads(scaffold_json)
    scaffold = PDB(scaffold_pdb_str)
    sch1, sch2 = scaff['dna_chains']

    scaff_prot_chains = [ch for ch, info in scaff['chains'].items()
                         if info['type'] == 'protein']
    scaff_prot_coords = select_chains(scaffold, scaff_prot_chains).GetCoords()
    scaff_dna_coords = select_chains(scaffold, [sch1, sch2]).GetCoords()
    scaff_prot_tree = cKDTree(scaff_prot_coords)
    scaff_core_prot_coords = _extract_scaffold_core_prot_coords(
        scaffold_pdb_str, scaff_prot_chains)

    # Determine top chain and core footprint
    top_chain = registers_data.get('top_chain', sch1)
    top_seq = scaff['chains'][top_chain]['sequence']
    core_idx = top_seq.find(CORE_SEQ)
    if core_idx >= 0:
        core_start = core_idx + 1
        core_end = core_start + len(CORE_SEQ) - 1
    else:
        core_start = 0
        core_end = 0

    # Load coaxial keys
    coaxial_keys = set()
    if mates_json is not None:
        mates = mates_json if isinstance(mates_json, dict) else json.loads(mates_json)
        for side in ('top', 'bot'):
            m = mates['mates'][side]
            coaxial_keys.add((m['sym_op'], tuple(m['unit_cell'])))

    if progress_cb:
        progress_cb('Generating crystal environment', 0, 1)

    env_copies = _generate_crystal_environment(
        scaffold, scaff_prot_coords, scaff_dna_coords, coaxial_keys,
        scaff_core_prot_coords)

    regs = registers_data['registers']
    results = []
    for i, reg in enumerate(regs):
        if progress_cb:
            progress_cb('Classifying registers', i + 1, len(regs))
        result = _classify_register(reg, scaff_prot_tree, scaff_prot_coords,
                                    env_copies, core_start, core_end)
        results.append(result)

    return {
        'top_chain': top_chain,
        'core_footprint': {'sequence': CORE_SEQ, 'start': core_start, 'end': core_end},
        'num_env_copies': len(env_copies),
        'registers': results,
        'env_copies': env_copies,
    }


def _extract_scaffold_core_prot_coords(pdb_str, prot_chains):
    """Extract core-atom coords for scaffold protein chains."""
    coords = []
    prot_set = set(prot_chains)
    for line in pdb_str.split('\n'):
        if line.startswith('ATOM') and len(line) > 54 and line[21] in prot_set:
            if is_core_atom(line[12:16], line[17:20]):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords) if coords else np.empty((0, 3))


def _extract_coords_and_info(pdb_str, chains):
    """Extract ATOM coords and identity for specified chains from string."""
    chain_set = set(chains)
    coords, info = [], []
    for line in pdb_str.split('\n'):
        if line.startswith('ATOM') and len(line) > 54 and line[21] in chain_set:
            coords.append([float(line[30:38]), float(line[38:46]),
                           float(line[46:54])])
            info.append((line[21], int(line[22:26]), line[12:16].strip()))
    return (np.array(coords) if coords else np.empty((0, 3)), info)


def _load_guest_protein_coords(pdb_str):
    """Load guest protein atom coords from string.
    Returns (all_coords, core_coords, atom_info)."""
    dna_resnames = {'DA', 'DT', 'DC', 'DG'}
    chains_resnames = {}
    lines = pdb_str.split('\n')
    for line in lines:
        if not line.startswith('ATOM') or len(line) < 54:
            continue
        ch = line[21]
        rn = line[17:20].strip()
        chains_resnames.setdefault(ch, set()).add(rn)
    dna_chs = {ch for ch, rns in chains_resnames.items() if rns <= dna_resnames}
    prot_chs = {ch for ch in chains_resnames if ch not in dna_chs}

    all_coords, all_info, core_coords = [], [], []
    for line in lines:
        if line.startswith('ATOM') and len(line) > 54 and line[21] in prot_chs:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            all_coords.append([x, y, z])
            atom_name = line[12:16]
            res_name = line[17:20]
            all_info.append((line[21], int(line[22:26]), atom_name.strip()))
            if is_core_atom(atom_name, res_name):
                core_coords.append([x, y, z])

    empty = np.empty((0, 3))
    return (np.array(all_coords) if all_coords else empty,
            np.array(core_coords) if core_coords else empty,
            all_info)


def _analyze_neighbors(guest_prot_coords, guest_core_coords, env_copies,
                       scaff_prot_tree, scaff_core_prot_tree,
                       guest_info=None, scaff_prot_info=None,
                       scaff_dna_info=None):
    """Compute per-neighbor distances for one register."""
    guest_tree = cKDTree(guest_prot_coords)
    guest_core_tree = cKDTree(guest_core_coords)
    track = guest_info is not None
    neighbors = []

    # ASU scaffold protein
    d_asu, idx_asu = scaff_prot_tree.query(guest_prot_coords, k=1)
    d_asu_min = float(d_asu.min())
    d_asu_core, _ = scaff_core_prot_tree.query(guest_core_coords, k=1)
    d_asu_core_min = float(d_asu_core.min())
    nb_asu = {
        'dist': round(d_asu_min, 2),
        'core_dist': round(d_asu_core_min, 2),
        'obj_name': 'ASU',
        'type': 'SPR',
    }
    if track:
        ig = int(np.argmin(d_asu))
        nb_asu['guest_atom'] = guest_info[ig]
        nb_asu['env_atom'] = scaff_prot_info[int(idx_asu[ig])]
    neighbors.append(nb_asu)

    for copy in env_copies:
        dists_this_copy = []
        core_dists_this_copy = []

        pt = copy['prot_tree']
        cpt = copy['core_prot_tree']
        if pt is not None:
            d, idx = pt.query(guest_prot_coords, k=1)
            d_prot = float(d.min())
            ig = int(np.argmin(d))
            dists_this_copy.append(('SPR', d_prot, ig,
                                    scaff_prot_info[int(idx[ig])] if track else None))
        if cpt is not None:
            d, _ = cpt.query(guest_core_coords, k=1)
            core_dists_this_copy.append(('SPR', float(d.min())))

        R, T_vec = copy['R'], copy['T']
        gsym_coords = (guest_prot_coords @ R.T) + T_vec
        d, idx = guest_tree.query(gsym_coords, k=1)
        d_gsym = float(d.min())
        i_sym = int(np.argmin(d))
        i_self = int(idx[i_sym])
        dists_this_copy.append(('GSY', d_gsym, i_self,
                                guest_info[i_sym] if track else None))
        gsym_core_coords = (guest_core_coords @ R.T) + T_vec
        d, _ = guest_core_tree.query(gsym_core_coords, k=1)
        core_dists_this_copy.append(('GSY', float(d.min())))

        dt = copy['dna_tree']
        if dt is not None:
            d, idx = dt.query(guest_prot_coords, k=1)
            d_dna = float(d.min())
            ig = int(np.argmin(d))
            dists_this_copy.append(('DNA', d_dna, ig,
                                    scaff_dna_info[int(idx[ig])] if track else None))
            d_core, _ = dt.query(guest_core_coords, k=1)
            core_dists_this_copy.append(('DNA', float(d_core.min())))

        if not dists_this_copy:
            continue

        best = min(dists_this_copy, key=lambda x: x[1])
        best_type, best_dist, best_ig, best_env_info = best
        _, best_core = min(core_dists_this_copy, key=lambda x: x[1])
        nb = {
            'dist': round(best_dist, 2),
            'core_dist': round(best_core, 2),
            'obj_name': copy['obj_name'],
            'type': best_type,
        }
        if track:
            nb['guest_atom'] = guest_info[best_ig]
            nb['env_atom'] = best_env_info
        neighbors.append(nb)

    neighbors.sort(key=lambda x: x['dist'])
    return neighbors


def _compute_score(neighbors, n_closest=N_CLOSEST):
    """Compute ranking score from neighbor distances."""
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
        'total_score': round(total, 2),
        'has_clash': has_clash,
        'n_clash': n_clash,
        'n_interact': n_interact,
        'min_dist': dists[0],
        'min_core_dist': min_core,
        'closest_dists': dists[:n_closest],
    }


def step_E_rank(categorized_data, scaffold_pdb_str, scaffold_json,
                guest_pdb_str, mates_json=None, progress_cb=None):
    """Rank registers. Returns ranked list."""
    scaff = scaffold_json if isinstance(scaffold_json, dict) else json.loads(scaffold_json)
    scaffold = PDB(scaffold_pdb_str)
    sch1, sch2 = scaff['dna_chains']

    scaff_prot_chains = [ch for ch, info in scaff['chains'].items()
                         if info['type'] == 'protein']
    scaff_pdb_str_local = scaffold_pdb_str

    scaff_prot_coords, scaff_prot_info = _extract_coords_and_info(
        scaff_pdb_str_local, scaff_prot_chains)
    scaff_prot_tree = cKDTree(scaff_prot_coords)
    scaff_core_prot_coords = _extract_scaffold_core_prot_coords(
        scaff_pdb_str_local, scaff_prot_chains)
    scaff_core_prot_tree = cKDTree(scaff_core_prot_coords)
    scaff_dna_coords, scaff_dna_info = _extract_coords_and_info(
        scaff_pdb_str_local, [sch1, sch2])

    coaxial_keys = set()
    if mates_json is not None:
        mates = mates_json if isinstance(mates_json, dict) else json.loads(mates_json)
        for side in ('top', 'bot'):
            m = mates['mates'][side]
            coaxial_keys.add((m['sym_op'], tuple(m['unit_cell'])))

    if progress_cb:
        progress_cb('Generating crystal environment', 0, 1)

    env_copies = _generate_crystal_environment(
        scaffold, scaff_prot_coords, scaff_dna_coords, coaxial_keys,
        scaff_core_prot_coords)

    registers = categorized_data['registers']
    top_chain = categorized_data.get('top_chain', sch1)
    asu_bp = None

    # Get DNA sequences for composite
    guest_pdb = PDB(guest_pdb_str)
    guest_dna_chains = identify_dna_chains(guest_pdb)
    gch1, gch2 = guest_dna_chains
    _, guest_seq1 = get_dna_sequence(guest_pdb, gch1)
    _, guest_seq2 = get_dna_sequence(guest_pdb, gch2)
    scaff_top_seq = scaff['chains'][top_chain]['sequence']

    ranked = []
    for i, reg in enumerate(registers):
        if progress_cb:
            progress_cb('Ranking registers', i + 1, len(registers))

        guest_prot_coords, guest_core_coords, guest_atom_info = \
            _load_guest_protein_coords(reg['pdb_string'])
        if len(guest_prot_coords) == 0:
            continue

        neighbors = _analyze_neighbors(
            guest_prot_coords, guest_core_coords, env_copies,
            scaff_prot_tree, scaff_core_prot_tree,
            guest_info=guest_atom_info,
            scaff_prot_info=scaff_prot_info,
            scaff_dna_info=scaff_dna_info)
        score = _compute_score(neighbors)

        has_clash = score['has_clash']
        n_interact = score['n_interact']
        min_dist = score['min_dist']
        if has_clash:
            classification = 'CLASH'
        elif min_dist >= INTERACT_CUTOFF:
            classification = 'independent'
        elif n_interact >= 3:
            classification = 'multi-connected'
        else:
            classification = 'near-neighbor'

        # Composite sequence
        sp = reg['start_pos']
        overwrite_seq = guest_seq1 if reg['orientation'] == 'fwd' else guest_seq2
        top = list(scaff_top_seq)
        for k in range(len(overwrite_seq)):
            idx = sp + k - 1
            if 0 <= idx < len(top):
                top[idx] = overwrite_seq[k].lower()
        composite_top = ''.join(top)
        composite_bot = ''.join(
            WC[b.upper()].lower() if b.islower() else WC[b] for b in top)

        nb0 = neighbors[0]
        ranked.append({
            'obj_name': reg['obj_name'],
            'label': reg['label'],
            'orientation': reg['orientation'],
            'start_pos': reg['start_pos'],
            'end_pos': reg['end_pos'],
            'region': reg['region'],
            'rmsd': reg['rmsd'],
            'pdb_string': reg['pdb_string'],
            'category': reg.get('category', 0),
            'category_name': reg.get('category_name', ''),
            'classification': classification,
            'total_score': score['total_score'],
            'has_clash': has_clash,
            'n_clash': score['n_clash'],
            'n_interact': n_interact,
            'min_dist': score['min_dist'],
            'min_core_dist': score['min_core_dist'],
            'closest_dists': score['closest_dists'],
            'neighbors': neighbors[:N_CLOSEST],
            'composite_top': composite_top,
            'composite_bot': composite_bot,
        })

    # Sort: non-clashers first, then by min_dist DESC
    ranked.sort(key=lambda r: (r['has_clash'], -r['min_dist'], -r['total_score']))

    for rank_idx, r in enumerate(ranked):
        r['rank'] = rank_idx + 1

    return {
        'ranked': ranked,
        'top_chain': top_chain,
        'scaff_top_seq': scaff_top_seq,
        'guest_dna_chains': [gch1, gch2],
        'guest_seq1': guest_seq1,
        'guest_seq2': guest_seq2,
    }


def step_F_diagram_svg(ranked_data, scaffold_json, categories_data):
    """Generate side-by-side SVG with fwd (left) and rev (right) panels. Returns SVG string."""
    scaff = scaffold_json if isinstance(scaffold_json, dict) else json.loads(scaffold_json)

    top_chain = ranked_data.get('top_chain') or categories_data.get('top_chain')
    dna_chains = scaff['dna_chains']
    bot_chain = [c for c in dna_chains if c != top_chain][0]
    top_seq = scaff['chains'][top_chain]['sequence']
    bot_display = ''.join(WC[b] for b in top_seq)
    seq_len = len(top_seq)

    foot = categories_data.get('core_footprint', {})
    foot_start = foot.get('start')
    foot_end = foot.get('end')

    ranked = ranked_data['ranked']
    registers = [r for r in ranked if r['classification'] != 'CLASH']

    fwd = sorted([r for r in registers if r['orientation'] == 'fwd'],
                 key=lambda r: r['start_pos'])
    rev = sorted([r for r in registers if r['orientation'] == 'rev'],
                 key=lambda r: r['start_pos'])

    if not registers:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="30"><text x="5" y="20" fill="#999">No non-clashing registers</text></svg>'

    # Color mapping
    CIVIDIS = [
        (0.00, (0.000, 0.135, 0.305)),
        (0.10, (0.065, 0.176, 0.353)),
        (0.20, (0.135, 0.219, 0.396)),
        (0.30, (0.210, 0.264, 0.425)),
        (0.40, (0.290, 0.312, 0.438)),
        (0.50, (0.373, 0.364, 0.435)),
        (0.60, (0.461, 0.418, 0.416)),
        (0.70, (0.558, 0.477, 0.384)),
        (0.80, (0.666, 0.541, 0.336)),
        (0.90, (0.787, 0.614, 0.269)),
        (1.00, (0.993, 0.906, 0.144)),
    ]

    def cividis_rgb(t):
        t = max(0.0, min(1.0, t))
        for i in range(len(CIVIDIS) - 1):
            t0, c0 = CIVIDIS[i]
            t1, c1 = CIVIDIS[i + 1]
            if t <= t1:
                f = (t - t0) / (t1 - t0)
                return tuple(c0[j] + f * (c1[j] - c0[j]) for j in range(3))
        return CIVIDIS[-1][1]

    def cividis_hex(t):
        r, g, b = cividis_rgb(t)
        return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'

    def text_color(t):
        r, g, b = cividis_rgb(t)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        return '#ffffff' if lum < 0.45 else '#000000'

    all_dists = [r['min_dist'] for r in registers]
    d_lo, d_hi = min(all_dists), max(all_dists)
    d_range = d_hi - d_lo if d_hi > d_lo else 1.0

    def dist_to_t(d):
        return (d_hi - d) / d_range

    # SVG layout constants (1.5x scale)
    BP_W = 2.55
    BAR_H = 5.4
    BAR_GAP = 0.9
    MARGIN_L = 33.0
    MARGIN_R = 12.0
    SEQ_FONT = 3.9
    NUM_FONT = 3.9
    LABEL_FONT = 3.0
    BRACE_FONT = 4.68
    SCALE_H = 4.5
    SCALE_GAP = 9.0
    PANEL_GAP = 18.0
    SUPER_H = 6.0
    SUPER_GAP = 3.0

    BRACE_END_X = 14.25
    BRACE_TIP_X = 10.5
    BRACE_LBL_X = 6.75

    def bp_x(pos):
        return MARGIN_L + (pos - 1) * BP_W + BP_W / 2

    def bp_left(pos):
        return MARGIN_L + (pos - 1) * BP_W

    # Panel dimensions
    panel_w = MARGIN_L + seq_len * BP_W + MARGIN_R
    n_fwd = len(fwd)
    n_rev = len(rev)
    n_max = max(n_fwd, n_rev, 1)

    y_super = 6.0
    y_num = y_super + SUPER_H + SUPER_GAP + 3.0
    y_top = y_num + 6.75
    y_bot = y_top + 5.25
    y_bars0 = y_bot + 7.5
    y_bars_end = y_bars0 + n_max * (BAR_H + BAR_GAP)
    y_scale = y_bars_end + SCALE_GAP
    total_h = y_scale + SCALE_H + 12
    total_w = 2 * panel_w + PANEL_GAP

    s = []
    a = s.append

    a(f'<svg xmlns="http://www.w3.org/2000/svg" '
      f'width="{total_w:.1f}mm" height="{total_h:.1f}mm" '
      f'viewBox="0 0 {total_w:.1f} {total_h:.1f}">')

    a('<style>')
    a(f'  .seq {{ font-family: Courier, monospace; font-size: {SEQ_FONT}px; '
      f'text-anchor: middle; fill: #e0e0e0; }}')
    a(f'  .num {{ font-family: Arial, sans-serif; font-size: {NUM_FONT}px; '
      f'text-anchor: middle; fill: #aaa; }}')
    a(f'  .barlbl {{ font-family: Arial, sans-serif; font-size: {LABEL_FONT}px; '
      f'text-anchor: middle; dominant-baseline: central; pointer-events: none; }}')
    a(f'  .bracelbl {{ font-family: Arial, sans-serif; font-size: {BRACE_FONT}px; '
      f'fill: #aaa; text-anchor: middle; }}')
    a(f'  .scalelbl {{ font-family: Arial, sans-serif; font-size: {NUM_FONT}px; '
      f'fill: #aaa; }}')
    a(f'  .superlbl {{ font-family: Arial, sans-serif; font-size: {LABEL_FONT + 0.3}px; '
      f'text-anchor: middle; dominant-baseline: central; fill: #85c1e9; }}')
    a('  .register-bar:hover { opacity: 0.8; }')
    a('  .register-bar { cursor: pointer; }')
    a('  .register-bar.active { stroke: #00ff88; stroke-width: 0.6; }')
    a('</style>')

    # ── Helper functions ──

    def draw_brace(y_start, n_bars, label):
        if n_bars == 0:
            return
        yt = y_start
        yb = y_start + n_bars * (BAR_H + BAR_GAP) - BAR_GAP
        ym = (yt + yb) / 2
        ex = BRACE_END_X
        tx = BRACE_TIP_X
        dx = ex - tx
        tip_off = 1.5
        tip_h = 2.25
        half = (yb - yt) / 2
        if dx + tip_h > half:
            sc = half / (dx + tip_h)
            dx *= sc
            tip_h *= sc
        arm_top = ym - tip_h - (yt + dx)
        arm_bot = (yb - dx) - (ym + tip_h)
        path = (
            f'M {ex},{yt:.2f} '
            f'C {ex - 0.818 * dx:.4f},{yt - 0.052 * dx:.4f} '
            f'{tx},{yt + 0.448 * dx:.4f} '
            f'{tx},{yt + dx:.4f} '
            f'v {arm_top:.4f} '
            f'c {-0.182 * tip_off:.4f},{0.535 * tip_h:.4f} '
            f'{-0.515 * tip_off:.4f},{0.869 * tip_h:.4f} '
            f'{-tip_off:.4f},{tip_h:.4f} '
            f'{0.561 * tip_off:.4f},{0.131 * tip_h:.4f} '
            f'{0.894 * tip_off:.4f},{0.465 * tip_h:.4f} '
            f'{tip_off:.4f},{tip_h:.4f} '
            f'v {arm_bot:.4f} '
            f'c 0,{0.552 * dx:.4f} '
            f'{0.321 * dx:.4f},{dx + 0.077 * dx:.4f} '
            f'{dx:.4f},{dx:.4f}'
        )
        a(f'<path d="{path}" fill="none" stroke="#888" stroke-width="0.4"/>')
        a(f'<text class="bracelbl" '
          f'transform="translate({BRACE_LBL_X:.1f},{ym:.2f}) rotate(-90)">'
          f'{label}</text>')

    def overlaps_footprint(start, end):
        if foot_start is None or foot_end is None:
            return False
        return start <= foot_end and end >= foot_start

    def draw_bars(regs, y_start):
        for i, r in enumerate(regs):
            y = y_start + i * (BAR_H + BAR_GAP)
            x = bp_left(r['start_pos'])
            w = (r['end_pos'] - r['start_pos'] + 1) * BP_W
            t = dist_to_t(r['min_dist'])
            fill = cividis_hex(t)
            tc = text_color(t)
            overlap = overlaps_footprint(r['start_pos'], r['end_pos'])
            stroke = ' stroke="#c71585" stroke-width="0.3"' if overlap else ''
            obj = r['obj_name']
            rank = r['rank']
            a(f'<rect class="register-bar" data-obj-name="{obj}" '
              f'data-rank="{rank}" '
              f'x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
              f'height="{BAR_H}" rx="0.8" fill="{fill}"{stroke} '
              f'onclick="onRegisterBarClick(\'{obj}\')"/>')
            txt = f"{r['start_pos']}\u2013{r['end_pos']}"
            cx = x + w / 2
            cy = y + BAR_H / 2
            a(f'<text class="barlbl" x="{cx:.2f}" y="{cy:.2f}" fill="{tc}" '
              f'onclick="onRegisterBarClick(\'{obj}\')">{txt}</text>')

    def draw_panel(regs, brace_label):
        """Draw scaffold ruler, protein footprint, column markers, and bars."""
        n_bars = len(regs)

        # Scaffold protein super-register
        if foot_start is not None and foot_end is not None:
            sx = bp_left(foot_start)
            sw = (foot_end - foot_start + 1) * BP_W
            a(f'<rect x="{sx:.2f}" y="{y_super:.2f}" width="{sw:.2f}" '
              f'height="{SUPER_H}" rx="1.2" fill="#1a4a6e" stroke="#5dade2" '
              f'stroke-width="0.5"/>')
            scx = sx + sw / 2
            scy = y_super + SUPER_H / 2
            a(f'<text class="superlbl" x="{scx:.2f}" y="{scy:.2f}">'
              f'Scaffold Protein: {foot_start}\u2013{foot_end}</text>')

        # Background column markers (every 5 bp)
        col_top = y_num - NUM_FONT
        col_h = y_bars_end + 1 - col_top
        for p in range(1, seq_len + 1):
            if (p - 1) % 5 == 0:
                x = bp_x(p) - BP_W * 0.4
                color = '#2a2a2a' if (p - 1) % 10 == 0 else '#252525'
                a(f'<rect x="{x:.2f}" y="{col_top:.1f}" '
                  f'width="{BP_W * 0.8:.2f}" height="{col_h:.1f}" fill="{color}"/>')

        # Position numbers
        for p in range(1, seq_len + 1):
            if (p - 1) % 5 == 0:
                a(f'<text class="num" x="{bp_x(p):.2f}" y="{y_num}">{p}</text>')

        # DNA top strand 5'->3'
        a(f'<text class="seq" x="{bp_x(1) - BP_W:.2f}" y="{y_top}" '
          f'style="font-size:{SEQ_FONT * 0.7:.1f}px;fill:#666">5\'</text>')
        for i, base in enumerate(top_seq):
            a(f'<text class="seq" x="{bp_x(i + 1):.2f}" y="{y_top}">{base}</text>')
        a(f'<text class="seq" x="{bp_x(seq_len) + BP_W:.2f}" y="{y_top}" '
          f'style="font-size:{SEQ_FONT * 0.7:.1f}px;fill:#666">3\'</text>')

        # DNA bottom strand 3'->5'
        a(f'<text class="seq" x="{bp_x(1) - BP_W:.2f}" y="{y_bot}" '
          f'style="font-size:{SEQ_FONT * 0.7:.1f}px;fill:#666">3\'</text>')
        for i, base in enumerate(bot_display):
            a(f'<text class="seq" x="{bp_x(i + 1):.2f}" y="{y_bot}">{base}</text>')
        a(f'<text class="seq" x="{bp_x(seq_len) + BP_W:.2f}" y="{y_bot}" '
          f'style="font-size:{SEQ_FONT * 0.7:.1f}px;fill:#666">5\'</text>')

        # Brace + bars
        draw_brace(y_bars0, n_bars, brace_label)
        draw_bars(regs, y_bars0)

    # ── Left panel (forward) ──
    a(f'<g transform="translate(0,0)">')
    draw_panel(fwd, 'Forward alignment')
    a('</g>')

    # ── Right panel (reverse) ──
    right_x = panel_w + PANEL_GAP
    a(f'<g transform="translate({right_x:.1f},0)">')
    draw_panel(rev, 'Reverse alignment')
    a('</g>')

    # ── Color scale bar (centered, spanning both panels) ──
    scale_w = seq_len * BP_W
    scale_x = (total_w - scale_w) / 2
    n_steps = 64
    step_w = scale_w / n_steps

    a('<!-- color scale bar -->')
    a(f'<text class="scalelbl" x="{scale_x:.1f}" y="{y_scale - 1:.1f}">'
      f'min_dist (\u00c5)</text>')

    for k in range(n_steps):
        t = k / (n_steps - 1)
        sx = scale_x + k * step_w
        fill = cividis_hex(t)
        a(f'<rect x="{sx:.2f}" y="{y_scale:.1f}" '
          f'width="{step_w + 0.1:.2f}" height="{SCALE_H}" fill="{fill}"/>')

    n_ticks = 5
    for k in range(n_ticks + 1):
        frac = k / n_ticks
        val = d_hi - frac * d_range
        tx = scale_x + frac * scale_w
        ty = y_scale + SCALE_H + NUM_FONT + 0.5
        anchor = 'start' if k == 0 else ('end' if k == n_ticks else 'middle')
        a(f'<text class="scalelbl" x="{tx:.2f}" y="{ty:.1f}" '
          f'text-anchor="{anchor}">{val:.1f}</text>')

    a('</svg>')
    return '\n'.join(s)
