import os, tempfile, numpy, re

def WriteNewChains(list_of_PDBs, keeper_chains, outpdb, append=False):
    chunks = [x['chain %s'%y] for x in list_of_PDBs for y in keeper_chains]
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if os.path.isfile(outpdb) and not append: os.remove(outpdb)
    for chunk,chain in zip(chunks,3*alphabet):
        chunk.SetChain(chain)
        chunk.WritePDB(outpdb,append=True)

def Fetch(pdb_id):
    """Download from the PDB by id code"""
    if pdb_id.endswith('.pdb'): pdb_id = pdb_id[:-4]
    from urllib import request
    request.urlretrieve('http://files.rcsb.org/download/%s.pdb'%pdb_id, '%s.pdb'%(pdb_id.lower()))

def DummyAtom(x,y,z,aname='D',rname='DUM',rnum=1,anum=1,elem='D',atype='ATOM  ',altloc=' ',chain=' ',occupancy=1.00,temp=0):
    """Handles formatting of ATOM records for PDB output"""
    values=(atype,anum,aname,altloc,rname,chain,rnum,x,y,z,occupancy,temp,elem)
    if aname[0].isdigit():
        return '%6s%5d %-4s%1s%3s %1s%4d    %8.3f%8.3f%8.3f %5.2f %5.2f          %2s' % values
    else:
        return '%6s%5d  %-3s%1s%3s %1s%4d    %8.3f%8.3f%8.3f %5.2f %5.2f          %2s' % values

def first_model(pdb_path):
    """Return path to just the first MODEL of an NMR ensemble PDB.
    For non-ensemble files (no MODEL record before first ATOM), returns
    the original path unchanged."""
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM'):
                return pdb_path
            if line.startswith('MODEL'):
                break
        else:
            return pdb_path
    lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ENDMDL'):
                break
            if not line.startswith('MODEL'):
                lines.append(line)
    tmp = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w')
    tmp.writelines(lines)
    tmp.close()
    return tmp.name

class PDB:
    """A PDB object stores the Protein Data Bank representation of a structure and parsed metadata.

    Arguments: (str)pdbpath OR (str)pdbstring
    
    Usage:
    P = PDB('2ij2.pdb')
    Q = PDB(open('2ij2.pdb').read())

    Note:
    The PDB objects exist in order to provide an intermediate between the PDB files on disk and System objects.
    """ 
    def __init__(self, arg=None,hetatm=False,verbose=False,checkforfile=True,warn=True):
        if arg == None: pass  ## Cloning involves creating an empty PDB
        elif isinstance(arg,str) and os.path.isfile(arg): ## If user provides a path to a file on disk 
            self.path = arg
            self.orig = open(arg).read()
            self.Parse(hetatm=hetatm,verbose=verbose)
        else:
            self.path = None 
            self.orig = arg 
            if checkforfile and len(arg)<200 and len(arg)>3 and arg[:4] != 'ATOM':
                if arg.endswith('.pdb'):
                    assert False, 'Could not find the the pdb: '+arg 
            self.Parse(hetatm=hetatm,verbose=verbose)
        if hasattr(self,'orig'): del(self.orig)

    def __repr__(self):
        n = self.num_segments
        if n == 1:
            if min(self.resids)==max(self.resids):
                reprstr = '<PDB object at %s. Residue %d: %s>' % (hex(id(self)), max(self.resids), self.seq)
            else:
                reprstr = '<PDB object at %s. Residues %d-%d: %s>' % (hex(id(self)), min(self.resids), max(self.resids), self.seq)
        elif n > 1:
            reprstr = '<PDB object at %s.' % (hex(id(self))) 
            seg2chainresids = self._get_segment2chainresids()
            reprstr += ', '.join( ' Seg %d: %s %d-%d' % (seg, x[0],x[1],x[2]) for seg, x in list(seg2chainresids.items()))
            reprstr += '>' 
        else:
            reprstr = self.current
        return reprstr 

    def __str__(self): return self.current
    def __len__(self): return self.num_alpha_carbons 
    def __add__(self, other):
        both = str(self) + str(other)
        return PDB(both)

    def _get_segment2chainresids(self):
        self._update_chainresi2seg()
        segments = sorted(set(self.chainresi2seg.values()))
        segment2chainresids = dict( (seg,(' ',999999,-999999)) for seg in segments)
        for (chain, resi), segment in list(self.chainresi2seg.items()):
            if resi > segment2chainresids[segment][2] and resi < segment2chainresids[segment][1]:
                segment2chainresids[segment] = (chain,resi,resi)
            elif resi > segment2chainresids[segment][2]:
                segment2chainresids[segment] = (chain,segment2chainresids[segment][1],resi)
            elif resi < segment2chainresids[segment][1]:
                segment2chainresids[segment] = (chain,resi,segment2chainresids[segment][2])
        return segment2chainresids

    def __getitem__(self, selection, hetatm=None, warn=True):
        """Returns a PDB object with just the specified portion of the current PDB object.

        Usage:
        If you provide an integer you get the corresponding "segment"
        Pfirstseg = P[0]  ## Grab just the first contiguous chunk of protein
        If you provide a slice, the code attempts to return the corresponding residues by ID
        peptide = P[:10]  ## Grab the first ten residues
        More complicated queries mimic the selection syntax provided in PyMOL
        A, B = P['chain A'], P['chain B']
        CAtrace = P['name CA']
        active_site = P['chain A and resi > 19 and resi < 31']
        """
        if isinstance(selection, int): ## User providing a segment ID
            assert selection in self.seg2chain
            seg2chainresids = self._get_segment2chainresids()
            chain, lo, hi = seg2chainresids[selection]
            selection = 'chain %s and resi > %d and resi < %d' % (chain, lo - 1, hi + 1) 
        elif isinstance(selection,slice): ## User slicing by residue
            return self.JustResidues(self.resids[selection])

        # Preprocessing to handle special 'resi' syntax
        selection = re.sub(r'\bresi\s+(\d+)\b(?!\s*-\s*\d)', r'resi == \1', selection)
        selection = re.sub(r'\bresi\s+(\d+)-(\d+)\b', lambda m: f'resi > {int(m.group(1)) - 1} and resi < {int(m.group(2)) + 1}', selection)

        selection = re.sub(r'([()<>])', r' \1 ', selection)
        selection = selection.replace('==',' == ')

        properties = {'num','x','y','z','bfact','q','resi','line'}
        props = {'name','resn','chain','elem','altloc'}  ## Properties that do not need == 
        boolops = {'and','AND','or','OR','not','(',')','==','<','>'}
        words = selection.split()
        replaced = []
        lastword = None
        for word in words:
            if "'" in word: replaced.append( '\"%s\"' % word )
            elif word in properties: replaced.append( "atom['%s']"%word )
            elif word in props: replaced.append( "atom['%s'] =="%word )
            elif word.isalpha() and word not in boolops: replaced.append( "'%s'"%word )
            elif word.isalnum() and not word.isdigit() and word not in boolops: replaced.append( "'%s'"%word )
            elif word not in boolops and lastword == 'chain': replaced.append( "'%s'"%word ) 
            else: replaced.append( word )
            lastword = word
        criteria = ' '.join(replaced)
        matchlines = [atom['line'] for atom in self.listdict if eval(criteria)]
        assert matchlines != [], 'No matches for '+selection
        matchstr = '\n'.join(matchlines)
        if hetatm is None:  ## Unless specified, inherit the same hetatm settings
            hetatm = self.hetatmflag
        return PDB(matchstr,hetatm=hetatm,warn=warn)

    def _update_current_str_from_listdict(self):
        self.current = ''
        self.num_atoms = 0
        for d in self.listdict:
            if (d['chain'],d['resi'],d['name']) in self.removed_atoms: continue
            self.current += d['line'] + '\n'
            self.num_atoms += 1

    def _update_pdbstrings_from_listdict(self, verbose=False):
        lastresid, lastchain = self.listdict[0]['resi'], self.listdict[0]['chain']
        pdbstrings = []
        pdbstring = ''
        for atom in self.listdict:
            resi, chain, line = atom['resi'], atom['chain'], atom['line']
            ## Detect chain breaks
            if resi not in [lastresid, lastresid+1] or chain != lastchain:
                if verbose: print('Chain break', resi, chain)
                pdbstrings.append(pdbstring)
                pdbstring = '' 
            pdbstring += line + '\n'
            lastresid, lastchain = resi, chain
        pdbstrings.append(pdbstring)
        self.pdbstrings = pdbstrings              

    def Info(self):
        n = self.num_segments
        if n == 1:
            print('This PDB consists of %d residues in 1 contiguous segment' % len(self))
        if n > 1:
            print('This PDB consists of %d contiguous segment(s)' % n)
            seg2chainresids = self._get_segment2chainresids()
            for seg, (chain,minres,maxres) in list(seg2chainresids.items()):
                print('Segment %d  (chain %s)' % (seg, chain))
                segresids = list(range(minres, maxres+1))
                for ri in range(minres, maxres, 50):
                    end = min(ri + 50, maxres)
                    try:
                        read = self.segmentseqs[seg][segresids.index(ri):segresids.index(end)+1]
                        print('%4d %s %4d' % (ri, read, end))
                    except IndexError:
                        print('<No sequence>. Probably heteroatoms.')
                        break
    def SetChain(self,chain):
        """Edit the current string representation of the PDB to enforce the specified chain

        Arguments: (str)chain or (dict)chain
        Returns: None

        Usage: 
        P = PDB('1pgb.pdb')
        P = PDB['chain B']
        P.SetChain('A')
        or if P has two chains 'A', and 'B' you might:
        chain_update = {'A':'C', 'B':'D'}
        P.SetChain(chain_update)
        """
        if isinstance(chain,str) and len(chain)==1:
            for d in self.listdict: 
                d['chain'] = chain 
                d['line'] = d['line'][:21] + chain + d['line'][22:] 
            self._update_current_str_from_listdict()
            self.chainresi2seg = {(chain, resi): seg for (x, resi), seg in self.chainresi2seg.items()}
            self.seg2chain = {seg:chain for seg, x in self.seg2chain.items()}
            self.resndict = {chain:rdict for x, rdict in self.resndict.items()}
            self.namedict = {chain:rdict for x, rdict in self.namedict.items()}
            self.chain = chain
        elif isinstance(chain, dict):
            for d in self.listdict: 
                newchain = chain[d['chain']]
                assert isinstance(newchain,str) and len(newchain)==1
                d['chain'] = newchain 
                d['line'] = d['line'][:21] + newchain + d['line'][22:] 
            self._update_current_str_from_listdict()
            self.chainresi2seg = dict(((chain[x],resi),seg) for (x,resi),seg in list(self.chainresi2seg.items()))
            self.seg2chain = dict((seg,chain[x]) for seg, x in list(self.seg2chain.items()))
            if len(set(self.seg2chain.values())) == 1: self.chain = set(self.seg2chain.values()).pop()
            self.resndict = dict( (chain[x],rdict) for x, rdict in list(self.resndict.items()))
            self.namedict = dict( (chain[x],rdict) for x, rdict in list(self.namedict.items()))

    def Renumber(self,resid_map={}):
        """Edit the current PDB renumbering residues contiguously, or optionally according to supplied dict 

        Arguments: Optional (dict)resid_map 
        Returns: None

        Usage: 
        P = PDB('1pgb.pdb')
        P.Renumber()

        resid_update_dict = dict((resid, resid+1) for resid in P.resids)
        P.Renumber(resid_update_dict)

        P = PDB('1bfm.pdb')
        chain = 'B'
        resid_update_dict = dict(((chain,resid), resid+100) for resid in P.resids)
        P.Renumber(resid_update_dict)
        """
        if isinstance(resid_map, str) or resid_map=={}:
            if resid_map == 'contiguous' or resid_map=={}:
                chains = sorted(self.namedict.keys())
                last_resid = 0
                resid_map = {}
                for chain in chains:
                    current_resids = list(self.namedict[chain].keys())
                    target_first_resid = last_resid + 1
                    current_first_resid = sorted(current_resids)[0]
                    renumber_by = target_first_resid - current_first_resid
                    resid_map.update( dict(((chain, resid), resid+renumber_by) for resid in current_resids) )
                    last_resid = sorted(resid_map.values())[-1] 

        examplekey = list(resid_map.keys())[0]
        if isinstance(examplekey, int): ### Simple renumber: resid key
            assert set(resid_map.keys()) == set(self.resids)
            for d in self.listdict: 
                new_resi = resid_map[d['resi']] 
                d['resi'] = new_resi
                d['line'] = d['line'][:22] + '%4s'%new_resi  + d['line'][26:] 
        elif isinstance(examplekey,tuple) and isinstance(examplekey[0],str):
            for d in self.listdict: 
                key = (d['chain'], d['resi'])
                if key not in resid_map: continue
                new_resi = resid_map[key] 
                d['resi'] = new_resi
                d['line'] = d['line'][:22] + '%4s'%new_resi  + d['line'][26:] 
        self._update_current_str_from_listdict()
        self.resids = sorted(set(d['resi'] for d in self.listdict))

        new_resndict = {}
        for chain, olddict in list(self.resndict.items()):
            if chain not in new_resndict: new_resndict[chain] = {}
            for resi,aa in list(olddict.items()): 
                if resi in resid_map: new_resndict[chain][resid_map[resi]] = aa
                elif (chain,resi) in resid_map: new_resndict[chain][resid_map[(chain, resi)]] = aa
                else: new_resndict[chain][resi] = aa
        self.resndict = new_resndict

        for chain, olddict in list(self.namedict.items()):
            for resi,atomnames in list(olddict.items()): 
                if resi in resid_map: self.namedict[chain][resid_map[resi]] = atomnames
                elif (chain,resi) in resid_map: self.namedict[chain][resid_map[(chain, resi)]] = atomnames
        newchainresi2seg = {}
        for (chain,resi), seg in list(self.chainresi2seg.items()):
            if resi in resid_map:
                newchainresi2seg[(chain, resid_map[resi])] = seg 
            elif (chain,resi) in resid_map:
                newchainresi2seg[(chain, resid_map[(chain,resi)])] = seg
            else: newchainresi2seg[(chain,resi)] = seg
        self.chainresi2seg = newchainresi2seg 
        seg2firstresid = dict( (seg,9999999) for seg in list(self.seg2chain.keys())) 
        for (chain,resi), seg in list(self.chainresi2seg.items()):
            if resi < seg2firstresid[seg]: seg2firstresid[seg] = resi
        self.first_resids = [seg2firstresid[seg] for seg in list(self.seg2chain.keys())]
        
        updatedpartials = []
        for segment, resid in self.partials:
            if resid in resid_map:  updatedpartials.append( (segment, resid_map[resid]) )
            else: updatedpartials.append( (segment, resid) )
        self.partials = updatedpartials
        self._update_pdbstrings_from_listdict()

    def PickAltLoc(self,altloc):
        """Recreate the current string representation, excluding alternate locations other than the specified

        Arguments: (str)altloc
        Returns: None

        Usage:
        P.PickAltLoc('A')
        """
        self.listdict = [x for x in self.listdict if x['altloc'] in [' ',altloc]]
        for d in self.listdict: 
            d['altloc'] = ' '
            d['line'] = d['line'][:16] + ' ' + d['line'][17:] 
        self._update_current_str_from_listdict()

    def Rotate(self,R):
        """Given a rotation matrix or unit quaternion, update the coordinates"""         
        if isinstance(R, list):
            R = numpy.array(R)
        if R.shape == (1,4): ## Convert unit quaternion to rotation matrix 
            a, b, c, d = R 
            R = numpy.zeros([3,3])
            R[0][0] = a*a + b*b - c*c - d*d
            R[0][1] = 2*(b*c - a*d)
            R[0][2] = 2*(b*d + a*c)
            R[1][0] = 2*(b*c + a*d)
            R[1][1] = a*a - b*b + c*c - d*d
            R[1][2] = 2*(c*d - a*b)
            R[2][0] = 2*(b*d - a*c)
            R[2][1] = 2*(c*d + a*b)
            R[2][2] = a*a - b*b - c*c + d*d
        for d in self.listdict:
            oldline, x, y, z = d['line'], d['x'], d['y'], d['z'] 
            X = R[0][0]*x + R[1][0]*y + R[2][0]*z
            Y = R[0][1]*x + R[1][1]*y + R[2][1]*z
            Z = R[0][2]*x + R[1][2]*y + R[2][2]*z
            d['line']=oldline[:30] + '%8.3f%8.3f%8.3f'%(X,Y,Z) + oldline[54:] 
            d['x'], d['y'], d['z'] = X, Y, Z
        x,y,z = self.cog
        X = R[0][0]*x + R[1][0]*y + R[2][0]*z
        Y = R[0][1]*x + R[1][1]*y + R[2][1]*z
        Z = R[0][2]*x + R[1][2]*y + R[2][2]*z
        self.cog[0], self.cog[1], self.cog[2] = X, Y, Z
        self._update_current_str_from_listdict()

    def Translate(self,T):
        """Given a translation vector, update the coordinates.
     
        Usage:
        P.Translate([20.0, 0.0, 0.0])
        xyz = numpy.array([0.0, 10.0, -5.0])
        P.Translate(xyz)
        """         
        for d in self.listdict:
            oldline, x, y, z = d['line'], d['x'], d['y'], d['z']
            X, Y, Z = T[0] + x, T[1] + y, T[2] + z
            d['line']=oldline[:30] + '%8.3f%8.3f%8.3f'%(X,Y,Z) + oldline[54:] 
            d['x'], d['y'], d['z'] = X, Y, Z
        self.cog = self.GetCoords().mean(0)
        
        self._update_current_str_from_listdict()

    def MoveToOrigin(self):
        """Move the center-of-geometry (cog) to the origin"""
        cog = self.GetCoords().mean(0)
        self.Translate(-cog)

    def Clone(self):
        """Create a new PDB object that matches the current object.
           TODO: Try using copy.deepcopy()
        """
        p = PDB()

        props = ['removed_atoms','pdbstrings','segmentseqs','resids','resIDs',
                 'first_resids', 'partials']
        for prop in props:
            try: exec('p.%s = self.%s[:]'%(prop,prop))
            except: pass

        props = ['num_atoms','current','chain',
                 'seq','radius','num_alpha_carbons','mw','num_segments','orig','hetatmflag']
        for prop in props:
            try: exec('p.%s = self.%s'%(prop,prop))
            except: pass

        props = ['three2one','res_heavies','res_mass','xyz',
                     'namedict','resndict','chainresi2seg','seg2chain','cog']
        for prop in props:
            try: exec('p.%s = self.%s.copy()'%(prop,prop))
            except: pass

        p.listdict = [x.copy() for x in self.listdict]

        return p

    def WritePDB(self, pdbfilename, append=False):
        """Output PDB to specified path (or return the string).

        Arguments: (str)pdbfilename
        Returns: None

        Usage: 
        P.WritePDB('output.pdb')
        P.WritePDB('output.pdb',append=True)
        """
        open(pdbfilename,'a' if append else 'w').write(str(self))

    def GetCoords(self):
        """Return the coordinates as an array.

        Arguments: None
        Returns: numpy array

        Usage:
        coords = P.GetCoords()

        """
        return numpy.array([(d['x'], d['y'], d['z']) for d in self.listdict])
         
    def _parse_hetatm(self, line):
        """Parse a HETATM line"""
        self.hetatms += line + '\n'

    def _snap_to_zero(self, v, epsilon=1e-9):
        """Get rid of tiny terms in an array"""
        indices = [i for i,x in enumerate(v) if abs(x) < epsilon]
        v.put(indices, [0.0])

    def _parse_xtal(self, line):
        """Parse a CRYST1 Header line"""
        l = line.split()
        self.xtal_edges = [float(l[1]), float(l[2]), float(l[3])]
        self.xtal_angles = [float(l[4]), float(l[5]), float(l[6])]
        self.xtal_sg = line[55:67].strip() 
        self.xtal_num_in_unit_cell = int(l[-1])
        rad = [numpy.radians(x) for x in self.xtal_angles]
        B = numpy.identity(4)
        B[0][1], B[1][1], = numpy.cos(rad[2]), numpy.sin(rad[2])
        B[0][2] = numpy.cos(rad[1])
        B[1][2] = (numpy.cos(rad[0]) - B[0][1]*B[0][2]) / B[1][1]
        B[2][2] = numpy.sqrt(1 - B[0][2]**2 - B[1][2]**2)
        self.xtal_basis = numpy.matrix(B * (self.xtal_edges + [1.0]) )
        self.xtal_va,self.xtal_vb,self.xtal_vc = numpy.array(self.xtal_basis[:3,:3].T)
        self._snap_to_zero(self.xtal_va)
        self._snap_to_zero(self.xtal_vb)
        self._snap_to_zero(self.xtal_vc)

    def _update_chainresi2seg(self):
        chainresi2seg = {}
        segmentseqs = {}
        seg = 0
        lastchain, lastresi = self.listdict[0]['chain'], self.listdict[0]['resi']
        chainresi2seg[(lastchain, lastresi)] = seg
        for x in self.listdict:
            if x['chain'] != lastchain or x['resi'] < lastresi or x['resi'] > lastresi + 1: seg += 1
            if x['name']=='CA' and x['altloc'] in [' ','A'] and x['resn'] in self.three2one:
                try: segmentseqs[seg] += self.three2one[x['resn']]
                except KeyError: segmentseqs[seg] = self.three2one[x['resn']]
            chainresi2seg[(x['chain'], x['resi'])] = seg
            lastchain, lastresi = x['chain'], x['resi']
        self.chainresi2seg = chainresi2seg
        self.seg2chain = dict( (seg,chain) for (chain,resi),seg in list(chainresi2seg.items()))  ## To which chain belongs each contiguous block of residues?
        self.num_segments = len(self.seg2chain) ## How many contiguous blocks of residues are there?
        self.segmentseqs = [x[1] for x in sorted(segmentseqs.items())]

    def Parse(self, hetatm=False, verbose=False):
        """Parse the original pdb string, line by line"""
        self.three2one = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'ASN': 'N', 'GLN': 'Q', 
                     'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 
                     'ALA': 'A', 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 
                     'TRP': 'W', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
                     'MSE': 'M'}

        #### Atoms that CHOMP cannot parse will be removed, but a list will be retained for remediation purposes
        self.removed_atoms = []
        self.hetatms = ''
        self.hetatmflag = hetatm

        #### Note: these are in order to allow simple Chi value calculation with successive windows of 4 atoms
        self.res_heavies = { \
            'ALA': ['N','CA','C','O','CB'],
            'CYS': ['N','CA','C','O','CB','SG'],
            'ASP': ['N','CA','C','O','CB','CG','OD1','OD2'],
            'GLU': ['N','CA','C','O','CB','CG','CD','OE1','OE2'],
            'PHE': ['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ'],
            'GLY': ['N','CA','C','O'],
            'HIS': ['N','CA','C','O','CB','CG','ND1','CD2','CE1','NE2'],
            'ILE': ['N','CA','C','O','CB','CG1','CD1','CG2'],
            'LYS': ['N','CA','C','O','CB','CG','CD','CE','NZ'],
            'LEU': ['N','CA','C','O','CB','CG','CD1','CD2'],
            'MET': ['N','CA','C','O','CB','CG','SD','CE'],
            'MSE': ['N','CA','C','O','CB','CG','SE','CE'],			# Selenium
            'ASN': ['N','CA','C','O','CB','CG','OD1','ND2'],
            'PRO': ['N','CA','C','O','CB','CG','CD'],
            'GLN': ['N','CA','C','O','CB','CG','CD','OE1','NE2'],
            'ARG': ['N','CA','C','O','CB','CG','CD','NE','CZ','NH1','NH2'],
            'SER': ['N','CA','C','O','CB','OG'],
            'THR': ['N','CA','C','O','CB','OG1','CG2'],
            'VAL': ['N','CA','C','O','CB','CG1','CG2'],
            'TRP': ['N','CA','C','O','CB','CG','CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2'],
            'TYR': ['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ','OH']
        }

        self.res_mass = { 'ALA':71.09, 'CYS':103.15, 'ASP':115.09, 'GLU':129.12, 'PHE':147.18, 
                          'GLY':57.05, 'HIS':137.14, 'ILE':113.16, 'LYS':128.17, 'LEU':113.16, 
                          'MET':131.19, 'ASN':114.11, 'PRO':97.12, 'GLN':128.14, 'ARG':156.19, 
                          'SER':87.08, 'THR':101.11, 'VAL':99.14, 'TRP':186.21, 'TYR':163.18,
                          'MSE':131.19}


        segment, lastresnum, lastchain = 0, -999, 'Z'
        pdbstrings, segmentseqs, listdict, first_resids, resids, resIDs = [], [], [], [], [], []
        resndict, namedict, chainresi2seg, xyz = {}, {}, {}, {}
        seq, pdbstring = '', ''
        self.num_atoms = 0

        lines = self.orig.split('\n')
        for line in lines:
            if len(line)<4: continue
            if line[:6]=='CRYST1': self._parse_xtal(line)
            ## Normal mode, parse only ATOM records
            if not hetatm and line[:4]!='ATOM': continue
            if line[:6]=='HETATM': self._parse_hetatm(line)
            ## In hetatm mode, we parse ATOM records or HETATM records
            if hetatm and line[:6] != 'HETATM' and line[:4] != 'ATOM': continue

            ## Parse each ATOM line
            num = int(line[6:11])
            if verbose and num%100==0: print('Parsing pdb: %d\r' % num, end=' ')
            name = line[12:16].strip()
            altloc = line[16]
            resn = line[17:20]
            chain = line[21]
            resi = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            ## Items past the coordinates optional 
            try: occup = float(line[54:60])
            except: occup = None 
            try: bfact = float(line[60:66])
            except: bfact = None
            try: elem = line[76:78].strip()
            except: elem = None
            try: q = float(line[78:80])
            except: q = None

            if chain not in resndict: resndict[chain] = {}
            if chain not in namedict: namedict[chain] = {}

            ## When we have reached a new residue
            if resi != lastresnum or chain != lastchain:
                resndict[chain][resi] = resn
                namedict[chain][resi] = [] 
                resids.append(resi)
                resIDs.append((resi,chain))
                if resi != lastresnum + 1 or chain != lastchain:
                    first_resids.append(resi)
                    if lastresnum != -999:
                        pdbstrings.append(pdbstring)
                        pdbstring = ''
                        if seq != '': segmentseqs.append(seq)
                        seq = ''
                        segment += 1
                chainresi2seg[(chain,resi)] = segment

            xyz[(chain,resi,name,altloc)] = numpy.array([x,y,z])
            namedict[chain][resi].append(name)
            d = {}
            for propty in ['num','name','altloc','resn','chain','resi',
                             'x','y','z','occup','bfact','elem','q','line']:
                try: d[propty] = eval(propty)
                except NameError: pass
            listdict.append(d)
            if name in ['OXT','1H','2H','3H','H1','H2','H3']: self.removed_atoms.append( (chain, resi, name) )
            else: 
                pdbstring += line + '\n'
                self.num_atoms += 1
            if name=='CA' and altloc in [' ','A']:
                if resn in self.three2one:
                    seq += self.three2one[resn]
                else:
                    seq += '?'
            elif name=="C1'" and altloc in [' ','A']:  ## Nucleic acid sequence
                seq += resn[-1]
            lastresnum, lastchain = resi, chain

        pdbstrings.append(pdbstring)
        if seq != '': segmentseqs.append(seq)

        assert xyz != {}, 'No coordinates parsed'

        self.pdbstrings = pdbstrings
        self.current = ''.join(pdbstrings)
        self.segmentseqs = segmentseqs
        self.seq = ''.join(segmentseqs)
        self.resids = resids
        self.resIDs = resIDs
        self.first_resids = first_resids
        self.listdict = listdict
        self.cog = self.GetCoords().mean(0)
        self.radius = (((self.GetCoords() - self.cog)**2).sum(1)**0.5).max()

        alpha_listdict = [atom for atom in self.listdict if atom['name'] == 'CA' and atom['altloc'] in [' ','A']]
        self.num_alpha_carbons = len(alpha_listdict)
        try: self.mw = sum(self.res_mass[atom['resn']] for atom in alpha_listdict)
        except KeyError: self.mw = None
        self.namedict = namedict
        self.resndict = resndict
        self.chainresi2seg = chainresi2seg  ## To which segment (contiguous block of residues) belongs each residue
        self.seg2chain = dict( (seg,chain) for (chain,resi),seg in list(chainresi2seg.items()))  ## To which chain belongs each contiguous block of residues?
        self.num_segments = len(self.segmentseqs) ## How many contiguous blocks of residues are there?
        
        if len(set(self.seg2chain.values())) == 1: self.chain = set(self.seg2chain.values()).pop()

        self.partials = []  #Checking for missing heavy atoms
        for chain in self.resndict: 
            for resi, resn in list(self.resndict[chain].items()):  ## For every residue...
                names = self.namedict[chain][resi] ## What atom names are there for this residue?
                try:
                    wanted = self.res_heavies[resn]  ## What atom names are expected for this residue?
                    segment = chainresi2seg[(chain,resi)]  ## Which block of continguous residues (segment) is this residue in?
                    if not set(wanted).issubset(names): self.partials.append( (segment,resi) )  ## Take note if there are missing atoms
                except: pass
