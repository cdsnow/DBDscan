from pdbtools import DummyAtom
from numpy import array, mean, square, sqrt, matrix, floor, arccos, dot
import itertools
from sgdata import sgqt
from collections import defaultdict


def unitcell_axes_to_edgeangles(va,vb,vc):
    """Calculate the unit cell vector lengths (a, b, c) and angles (alpha, beta, gamma), given the three basis vectors.

    Usage:
    a, b, c, A, B, G = unitcell_axes_to_edgeangles(va, vb, vc)
    """
    a, b, c = ((va**2).sum())**0.5, ((vb**2).sum())**0.5, ((vc**2).sum())**0.5
    A = arccos(dot(vb,vc) / (b*c))
    B = arccos(dot(va,vc) / (a*c))
    G = arccos(dot(va,vb) / (a*b))
    return (a,b,c,A,B,G)

def LookupSymmetryOperations(P):
    """Given the space group from a PDB object (P.xtal_sg), set the list of quaternion-translations (P.xtal_qtlist).

    Usage:
    LookupSymmetryOperations(P)
    """
    P.xtal_qtlist = sgqt[P.xtal_sg]

def CalculateSymmetryTransforms(P):
    """Calculate the rotations and translations needed to fill the unit cell. Store as P.RTlist.

    Usage:
    CalculateSymmetryTransforms(P)
    print P.RTlist
    """
    P.RTlist = []
    for m in P.xtal_qtlist:
        mat = matrix([[m[0],m[1],m[2],m[9]],[m[3],m[4],m[5],m[10]],[m[6],m[7],m[8],m[11]],[0,0,0,1]])
        realmat = P.xtal_basis * mat * P.xtal_basis.I
        R, T = realmat[:3,:3], realmat[:3,3]
        P.RTlist.append( (R,T) )

def CalculateSymmetryMatrices(P):
    """Calculate the 4x4 matrices useful for generating symmetry copies.

    Usage:
    CalculateSymmetryMatrices(P)
    print P.xtal_matrices
    """
    try: qtlist = P.xtal_qtlist
    except AttributeError: LookupSymmetryOperations(P)
    P.xtal_matrices = []
    for m in P.xtal_qtlist:
        P.xtal_matrices.append(matrix([[m[0],m[1],m[2],m[9]],[m[3],m[4],m[5],m[10]],[m[6],m[7],m[8],m[11]],[0,0,0,1]]))

def FillUnitCell(P,a=0,b=0,c=0):
    """Fill the specified unit cell with PDB objects using the crystallographic symmetry specified within P.

    Usage:
    UC = FillUnitCell(P)
    UC2 = FillUnitCell(P, a=0, b=0, c=1)
    """
    try: qtlist = P.xtal_qtlist
    except AttributeError: LookupSymmetryOperations(P)
    symcopies = []
    chainnames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,m in enumerate(P.xtal_qtlist):
        mat = matrix([[m[0],m[1],m[2],m[9]+a],[m[3],m[4],m[5],m[10]+b],[m[6],m[7],m[8],m[11]+c],[0,0,0,1]])
        realmat = P.xtal_basis * mat * P.xtal_basis.I
        R, T = realmat[:3,:3], realmat[:3,3]
        M = P.Clone()
        #M.Rotate(array(R.T))
        M.Rotate(array(R))
        M.Translate(T)
        M.SetChain(chainnames[i % len(chainnames)])
        symcopies.append(M)
    return symcopies

def Supercell(P,a=1,b=1,c=1,repeatChains=True):
    """Mimic the symmetry filling of the PyMOL tool: SuperSym

    Usage:
    UC = Supercell(P)
    UC2 = Supercell(P, a=0, b=0, c=1, repeatChains=False)
    """
    try: qtlist = P.xtal_qtlist
    except AttributeError: LookupSymmetryOperations(P)
    unitcells = list(itertools.product(list(range(a)),list(range(b)),list(range(c))))

    #print('BASIS',P.xtal_basis)
    symcopies = []
    #chainnames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    chainnames = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'*100)
    tmpxyz = P.GetCoords()
    #extent = tmpxyz.max(0) - tmpxyz.min(0)
    #print('extent:',extent)
    #center = matrix(list(P.cog) + [1.0]).T
    center = 0.5*tmpxyz.max(0) + 0.5*tmpxyz.min(0)
    center = matrix(center.tolist() + [1.0]).T

    #print('center', center)
    #print('basis.I', P.xtal_basis.I)
    center_cell = P.xtal_basis.I * center
    #print('center_cell:',center_cell)
    for UCabc in unitcells:
        for i,m in enumerate(P.xtal_qtlist):
            mat = matrix([[m[0],m[1],m[2],m[9]],[m[3],m[4],m[5],m[10]],[m[6],m[7],m[8],m[11]],[0,0,0,1]])
            shift = floor(mat * center_cell)
            #print('\n\n',mat,shift)
            mat[0:3,3] = mat[0:3,3] - shift[0:3,0]
            mat[0:3,3] = mat[0:3,3] + [[float(x)] for x in UCabc] 
            #print(UCabc,i,mat)
            realmat = P.xtal_basis * mat * P.xtal_basis.I
            R, T = realmat[:3,:3], realmat[:3,3]
            #print('R',R,'T',T)
            M = P.Clone()
            #M.Rotate(array(R.T))
            M.Rotate(array(R).T)
            M.Translate(T)
            #if repeatChains: chainname = chainnames[i % len(chainnames)]
            #else: chainname = chainnames[len(symcopies) % len(chainnames)]
            #M.SetChain(chainname)
            myorigchains = sorted(M.resndict.keys())
            numchains = len(list(M.resndict.keys()))
            mychains = [chainnames.pop(0) for i in range(numchains)]
            rechain = dict(list(zip(myorigchains, mychains)))
            M.SetChain(rechain)
            symcopies.append(M)
    return symcopies

def SetCOGFromCentroids(P):
    """Given a PDB, System, or SystemVector, compute a c.o.g. using residue centroids.

    Usage:
    SetCOGFromCentroids(P)
    print P.cog
    """
    centroids = P.ResidueCentroids()
    P.cog = centroids.mean(0)

def SetRadiusFromCentroids(P):
    """Given a PDB, System, or SystemVector, compute a radius using residue centroids.

    Usage:
    SetRadiusFromCentroids(P)
    print P.radius
    """
    centroids = P.ResidueCentroids()
    cog = centroids.mean(0)
    P.radius = max(((centroids - cog)**2).sum(1))**0.5

def SymmetryCopies(P, cutoff=20, verbose=False, debugging=False):
    """Given a PDB, System, or SystemVector with the right meta information, generate nearby symmetry copies.

    Usage:
    neighbors = SymmetryCopies(P)
    """
    try: qtlist = P.xtal_qtlist
    except AttributeError: LookupSymmetryOperations(P)
    try: cog = P.cog
    except AttributeError: SetCOGFromCentroids(P)
    try: radius = P.radius
    except AttributeError: SetRadiusFromCentroids(P)

    cutoff2 = cutoff**2
    cogcutoff2 = (cutoff + 2*P.radius)**2
    unitcells = list(itertools.product([-1,0,1],[-1,0,1],[-1,0,1]))

    xyz = P.ResidueCentroids()
    if debugging:
        outp = open('debugging.master.pdb','w')
        for i2, (x,y,z) in enumerate(xyz):
            print(DummyAtom(x,y,z,rnum=i2,anum=i2), file=outp)
    ns = NeighborSearch(xyz)
    copies = []
    chainnames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    index = 1
    for qt in P.xtal_qtlist:
        for a,b,c in unitcells:
            if a==0 and b==0 and c==0 and qt == (1,0,0,0,1,0,0,0,1,0,0,0): continue ## Identity
            m = matrix([[qt[0],qt[1],qt[2],qt[9]+a],
                        [qt[3],qt[4],qt[5],qt[10]+b],
                        [qt[6],qt[7],qt[8],qt[11]+c],
                        [0,0,0,1]])
            realmat = P.xtal_basis * m * P.xtal_basis.I
            R, T = realmat[:3,:3], array(realmat[:3,3].T)[0]
            ### Quickly check if the monomer might be close enough
            test = matrix(P.cog) * R.T + T
            d2 = (array(test - P.cog)**2).sum()
            if d2 > cogcutoff2: continue  ## Too far

            ### Full distance check
            sym = xyz * R.T + T
            #nearest, d2 = ns.nn(array(sym), num = 1)
            #if min(d2)[0] > cutoff2: continue ## Too far away
            #mind2 = min(ns.kdt.query(sym,1)[0])
            #if mind2 > cutoff2: continue ## Too far away
            mind = min(ns.kdt.query(sym,1)[0])
            if mind > cutoff: continue ## Too far away
            if verbose: print('distance =', mind)

            ### Create output
            copy = Clone(P)
            copy.Rotate(array(R.T))
            copy.Translate(T)

            chain = chainnames[index % len(chainnames)]
            if isinstance(copy, PDB): copy.SetChain(chain)
            if debugging:
                print(chain) 
                outp = open('debugging.%s.pdb'%chain,'w')
                for i2, (x,y,z) in enumerate(array(sym)):
                    print(DummyAtom(x,y,z,rnum=index,anum=i2), file=outp)
                copy.WritePDB('grr.%s.pdb'%chain)
                assert abs(sym - copy.ResidueCentroids() ).all() < 0.001

            index += 1
            copies.append( copy )
    return copies 

def DesignNeighborMap(S):
    """Long range neighbor map suitable for protein design.

    Usage:
    n = DesignNeighborMap(S)
    """
    return NeighborMap(S, array([[14]*20]*20))

def invariant_large_rotamer_generator(neighbormap, dunbrack_lib):
    """Returns a RotamerGenerator that gives all positions the extra rotamers given by default to buried positions.

    rg = invariant_large_rotamer_generator(n, dunbrack_lib)
    """
    rotgen = RotamerGenerator(neighbormap, dunbrack_lib)
    rgp = rotgen.GetParameters()
    rgp.baseBuriedCutoff = 0
    rgp.buriedCutoff = 0
    rgp.maxPercentSurface = rgp.maxPercentBuried
    rgp.maxRotamersSurface = rgp.maxRotamersBuried
    rotgen.SetParameters(rgp)
    return rotgen

    
def SymmetrySystemVectors(SV,cutoff=20,verbose=False):
    """Given a SystemVector return a list of neighboring symmetry SystemVectors"""
    assert 'xtal_basis' in SV.__dict__, 'SymmetrySystemVectors requires a SystemVector input with .xtal_basis defined' 
    assert 'xtal_sg' in SV.__dict__, 'SymmetrySystemVectors requires a SystemVector input with .xtal_sg defined' 
    basis, sg = SV.xtal_basis, SV.xtal_sg
    if verbose: print(sg)
    try: matrices = SV.xtal_matrices
    except AttributeError:
        CalculateSymmetryMatrices(SV)
        matrices = SV.xtal_matrices 

#def CreateSymSystems(P,cutoff=20,verbose=False):
#    """Given a PDB return a System and a list of neighboring symmetry Systems"""
#    S = NewSystem(P)
#    S.xtal_basis, S.xtal_sg = P.xtal_basis, P.xtal_sg
#    symneighbors = SymmetryCopies(S, verbose=verbose)
#    return S, symneighbors

#def SymmetryCopies(S,cutoff=20,verbose=False):
#    """Given a System or SystemVector return neighboring symmetry copies"""
#    assert 'xtal_basis' in S.__dict__, 'SymmetryNeighborSystems requires .xtal_basis to be set'
#    assert 'xtal_sg' in S.__dict__, 'SymmetryNeighborSystems requires .xtal_sg to be set'
#    basis, sg = S.xtal_basis, S.xtal_sg
#    if verbose: print sg
#    try: matrices = S.xtal_matrices
#    except AttributeError:
#        CalculateSymmetryMatrices(S)
#        matrices = S.xtal_matrices 
#    xyz = S.ResidueCentroids()
#    ns = NeighborSearch(xyz,method='hash')
#
#    unitcells = list(itertools.product([-1,0,1],[-1,0,1],[-1,0,1]))
#    bbox_center = 0.5*(xyz.min(0) + xyz.max(0))
#    center_cell = basis.I * matrix(list(bbox_center) + [1.0]).T
#    systems = []
#    for mat in matrices:
#        if verbose: print ''
#        shift = floor(mat * center_cell)[:3,0]
#        #mat[:3,3] -= shift
#        for a,b,c in unitcells:
#            m = mat.copy()
#            if verbose: print 'From source matrix:\n', m
#            m[:3,3] += [[a],[b],[c]]
#            realmat = basis * m * basis.I  ## Get real translation numbers
#            R, T = realmat[:3,:3], realmat[:3,3]
#            if verbose: 
#                print 'The real rotation matrix:'
#                print R
#                print 'Which is'
#                print Quat(array(R))
#            Quat(array(R)).AsThetaAxis() 
#            t =  array([snap2zero(x) for x in T.flatten().tolist()[0]])
#            sym = xyz * R.T + t
#            if verbose: print 'Finish with translation:',t
#            if rmsd(sym, xyz) < 1: continue  ## Duplicate of original protein
#            #nearatom_indices = ns.set_within_radius(array(sym), radius = 20, estimate = 20)
#            nearest, d2 = ns.nn(array(sym), num = 1)
#            if min(d2)[0] > cutoff**2: continue ## Too far away
#            copy = S.Clone
#            copy.Rotate(R.T)
#            #print R.T
#            copy.Translate(t)
#            #print t
#            systems.append(copy)
#    return S, systems

def FillUnitCells(S,basis,sg,a,b,c):
    """Given a System, a basis, a spacegroup, and the number of unit cells along a, b, and c return a list of Systems.

    Usage:
    symsystems = FillUnitCells(S, basis, 'P 21 21 21', a, b, c)
    """
    matrices = [matrix(x) for x in sg_sym_to_mat_list(sg)]
    xyz = S.GetSystemCoords()
    bbox_center = 0.5*(xyz.min(0) + xyz.max(0))
    center_cell = basis.I * matrix(list(bbox_center) + [1.0]).T
    systems = []
    rangea = [int(x - a/2) for x in range(a)]
    rangeb = [int(x - b/2) for x in range(b)]
    rangec = [int(x - c/2) for x in range(c)]
    unitcells = list(itertools.product(rangea, rangeb, rangec))
    for mat in matrices:
        shift = floor(mat * center_cell)[:3,0]
        mat[:3,3] -= shift
        for a,b,c in unitcells:
            m = mat.copy()
            m[:3,3] += [[a],[b],[c]]
            realmat = basis * m * basis.I  ## Get real translation numbers
            R, T = realmat[:3,:3], realmat[:3,3]
            t =  array([snap2zero(x) for x in T.flatten().tolist()[0]])
            sym = xyz * R.T + t
            #if rmsd(sym, xyz) < 1: continue  ## Duplicate of original protein
            copy = S.Clone
            copy.Rotate(R.T)
            copy.Translate(t)
            systems.append(copy)

def MapToCells(xyz,basis,sg,a,b,c):
    """Given coordinates, a basis, a spacegroup, and the number of unit cells along a, b, and c return a list of coordinates"""
    qtlist = sgqt[sg]
    matrices = []
    for m in qtlist:
        matrices.append( matrix([[m[0],m[1],m[2],m[9]+a],[m[3],m[4],m[5],m[10]+b],[m[6],m[7],m[8],m[11]+c],[0,0,0,1]]) )

    if xyz.shape == (3,):
        bbox_center = xyz
    else:
        bbox_center = 0.5*(xyz.min(0) + xyz.max(0))
    center_cell = basis.I * matrix(list(bbox_center) + [1.0]).T
    systems = []
    rangea = [int(x - a/2) for x in range(a)]
    rangeb = [int(x - b/2) for x in range(b)]
    rangec = [int(x - c/2) for x in range(c)]
    unitcells = list(itertools.product(rangea, rangeb, rangec))
    mates = []
    for mat in matrices:
        shift = floor(mat * center_cell)[:3,0]
        mat[:3,3] -= shift
        for a,b,c in unitcells:
            m = mat.copy()
            m[:3,3] += [[a],[b],[c]]
            realmat = basis * m * basis.I  ## Get real translation numbers
            R, T = realmat[:3,:3], realmat[:3,3]
            t =  array([snap2zero(x) for x in T.flatten().tolist()[0]])
            sym = xyz * R.T + t
            mates.append( sym )
    return mates 

def SymmetryResidueSystems(M0,Mlist,verbose=False):
    """Given a System and a list of neighboring Systems, cut 1-Residue Systems from the latter."""
    assert isinstance(M0,System) and isinstance(Mlist[0],System)
    
    n = NeighborMap([M0] + Mlist, array([[14]*20]*20)) ## Interaction of any a.a. pair
    composite = [M0]
    for Mid, M in enumerate(Mlist):
        for i, ri in enumerate(M):
            for r in M0:
                if not n.AreNeighbors(r,ri): continue
                if ri.ID < r.ID: continue ## avoiding double counting
                composite.append( M[i:i+1] ) ## 1-Residue System slice
                if verbose: print('Monomer %d, res %d, neighbor of %d, check = %d' % (Mid,ri.ID,r.ID,composite[-1][0].ID))
                break
    return composite

def SymmetryResidueSysVec(M0,Mlist,verbose=False):
    """Given a SystemVector and a list of neighboring SystemVectors, cut 1-Residue Systems from the latter."""
    assert isinstance(M0,SystemVector) and isinstance(Mlist[0],SystemVector)
    ensemble = M0.Clone()
    for M in Mlist: ensemble.extend(M)
    n = NeighborMap(ensemble, array([[14]*20]*20)) ## Interaction of any a.a. pair

    ### HERE

    composite = []
    for Mid, M in enumerate(Mlist):
      for segid, Mseg in enumerate(M):
        for i, ri in enumerate(Mseg):
          for M0seg in M0:
            for r in M0seg:
                if not n.AreNeighbors(r,ri): continue
                if ri.ID < r.ID: continue ## avoiding double counting
                composite.append( M[segid][i:i+1] ) ## 1-Residue System slice
                if verbose: print('Monomer %d, segment %d, res %d, neighbor of %d, check = %d' % (Mid,segid,ri.ID,r.ID,composite[-1][0].ID))
                break
    return composite

def SymmetryRotamerLibrary(composite,rg,core=[0], activeres=[]):
    """Build a RotamerLibrary using the supplied RotamerGenerator. Ensure corresponding residues get the same rotamers.

    Arguments: (list of Systems)composite OR (SystemVector)composite AND (RotamerGenerator)rg
    Optional: (list)core, (list)activeres
    Returns: RotamerLibrary

    Usage:
    rl = SymmetryRotamerLibrary(composite, rg)
    rl = SymmetryRotamerLibrary(composite, core=range(5))

    Note:
    Rotamers are added first added to the Systems listed, by index, in core. 
    If activeres is specified, rotamers are only added to residues with IDs in activeres.
    For non-core portions of the input, rotamers are added so as to match the rotamers in the core.
    """

    rl = RotamerLibrary(composite)
    ## Add rotamers to the core portions of the input
    for seg in core:
        if activeres==[]:
            rl.GenerateRotamers(composite[seg], rg)
        else:
            for r in composite[seg]:
                if r.ID not in activeres: continue
                rl.GenerateRotamers(r, rg)

    ## Record the core (M0) rotamers
    rl.M0_rotamerIDs = []
    rl.M0_dict = defaultdict(list) 
    for seg in core:
        for r in composite[seg]:
            rl.M0_rotamerIDs.extend(rl.GetRotamerIDRange(r))
            rl.M0_dict[r.ID].extend(rl.GetRotamerIDRange(r))

    ## Delete all other rotamers
    nonM0_rotids = sorted(set(rl.GetRotamerIDs()).difference(rl.M0_rotamerIDs))
    rl.DeleteRotamers(nonM0_rotids)

    ## Add back matching rotamers for other residues
    rl.symmetry_matches = {}
    for Mi, M in enumerate(composite):
        if Mi in core: continue  ## Leave the core segments alone
        assert len(M)==1, 'SymmetryRotamerLibrary expects to find only 1-Residue Systems'
        r = M[0]
        resid, aa = r.ID, r.ResName
     
        ## Identify the correct template residue from the core
        template = None
        for seg in core:
            for segres in composite[seg]:
                if (segres.ID, segres.ResName) == (resid, aa): 
                    template = segres
        assert template, 'Could not find an appropriate template for symmetry residue %s%d' % (aa,resid)

        ## Add rotamers to match the template residue
        templatechis = rl.GetRotamerChis(template)
        rl.AddRotamers(r, templatechis)
        assert rl.GetRotamerChis(r) == rl.GetRotamerChis(template)
        rl.symmetry_matches.update( dict(list(zip(rl.GetRotamerIDRange(r),rl.GetRotamerIDRange(template)))) )
    return rl

def NullSymmetryRotamerLibrary(composite, core=[0]):
    """Build a Null RotamerLibrary using the supplied RotamerGenerator. 
       Ensure corresponding residues get the same rotamers"""
    rl = RotamerLibrary(composite)
    rl.M0_rotamerIDs = []
    rl.M0_dict = defaultdict(list) 
    for seg in core:
        for r in composite[seg]:
            rl.M0_rotamerIDs.extend(rl.GetRotamerIDRange(r))
            rl.M0_dict[r.ID].extend(rl.GetRotamerIDRange(r))
    nonM0_rotids = sorted(set(rl.GetRotamerIDs()).difference(rl.M0_rotamerIDs))
    rl.DeleteRotamers(nonM0_rotids)
    rl.symmetry_matches = {}
    for Mid, M in enumerate(composite):
        if Mid in core: continue
        for r in M:
            resid, aa = r.ID, r.ResName

            template = None
            for seg in core:
                for segres in composite[seg]:
                    if (segres.ID, segres.ResName) == (resid, aa):
                        template = segres
            assert template, 'Could not find an appropriate template for symmetry residue %s%d' % (aa,resid)
 
            templatechis = rl.GetRotamerChis(template)
            rl.AddRotamers(r, templatechis)
            assert rl.GetRotamerChis(r) == rl.GetRotamerChis(template)
            rl.symmetry_matches.update( dict(list(zip(rl.GetRotamerIDRange(r),rl.GetRotamerIDRange(template)))) )
    return rl

def MergeEGnodeOld(eg,donor,acceptor):
    """Copy edge information from one EnergyGraph node into another"""
    edges = eg.GetVertexEdges(donor)
    if edges == []: return
    for partner, E in edges:
        if partner < acceptor: continue ## avoid double counting
        elif partner == acceptor:  ## e.g. Rotamer pairwise interaction with self
            oldE = eg.GetVertexEnergy(acceptor)
            eg.SetVertexEnergy(acceptor, oldE + E)
        try:
            oldE = eg.GetEdgeEnergy(partner, acceptor)
            eg.SetEdgeEnergy(partner, acceptor, oldE + E)
        except RuntimeError: ## No current edge
            eg.AddEdge(partner, acceptor, E)

def MergeEGnode(eg,donor,acceptor):
    """Copy edge information from one EnergyGraph node into another"""
    edges = eg.GetVertexEdges(donor)
    if edges == []: return
    for partner, E in edges:
        if partner == acceptor:  ## e.g. Rotamer pairwise interaction with self
            oldE = eg.GetVertexEnergy(acceptor)
            eg.SetVertexEnergy(acceptor, oldE + 0.5 * E)
            continue
        try:
            oldE = eg.GetEdgeEnergy(partner, acceptor)
            eg.SetEdgeEnergy(partner, acceptor, oldE + 0.5 * E)
        except RuntimeError: ## No current edge
            eg.AddEdge(partner, acceptor, 0.5 * E)

def SymmetryEnergyGraph(composite, rl, n, energy_function, core=[0], precision=0.01, doM0=True, doSYM=True, verbose=False):
    """Build an EnergyGraph assuming all Systems after the first can only see the first System"""
    ### Build the initial EnergyGraph including edges to symmetry residues
    eg = EnergyGraph()
    eg.SetEnergyThreshold(precision)
    ## We'll use the RotamerIterationMap to avoid duplicate calculations
    ri = RotamerIterationMap(rl)

    ## Avoid double counting by not considering terms that only involve symmetry copies
    ri.DisableAll()
    coresegs = [composite[seg] for seg in core]
    symsegs = [composite[seg] for seg in sorted(set(range(len(composite))).difference(core))]

    if doM0: 
        ## Turn on interactions within M0 segments
        for coreseg in coresegs:
            if verbose: print('Turning on interactions within M0 segment',coreseg.GetResidueIDs())
            ri.SetSystemSingleWeight(coreseg, 1.0)  ## Calculate 1-body terms within M0 segment
            ri.SetSystemPairWeight(coreseg, coreseg, 1.0) ## Calculate 2-body terms within M0 segment 

        ## Turn on interactions between M0 segments, unless they are mutually exlusive design choices
        for coresegA, coresegB in itertools.combinations(coresegs, 2):
            if len(coresegA) > 1 or len(coresegB) > 1:
                if verbose: print('Turning on interactions between M0 segments',coresegA.GetResidueIDs(), coresegB.GetResidueIDs())
                ri.SetSystemPairWeight(coresegA, coresegB, 1.0) ## Calculate 2-body terms between M0 segments 

            ## If both of these core segments are 1-Residue Systems, they might be mutually exclusive
            rc, rs = coresegA[0], coresegB[0]
            if rc.ID == rs.ID and rc.ResName != rs.ResName: continue  ## Cannot coexist
            #print rc.ID, rs.ID, rc.ResName, rs.ResName, 'between M0 segments'
            if verbose: print('Turning on interactions between M0 segments',coresegA.GetResidueIDs(), coresegB.GetResidueIDs())
            ri.SetSystemPairWeight(coresegA, coresegB, 1.0) ## Calculate 2-body terms from M0 segment to symmetry image

    if doSYM:
        ## Turn on interactions between M0 segments and symmetry segments
        for coreseg, symseg in itertools.product(coresegs, symsegs):
            if len(coreseg) > 1: 
                if verbose: print('Turning on symmetry interaction between',coreseg.GetResidueIDs(), symseg.GetResidueIDs()) 
                ri.SetSystemPairWeight(coreseg, symseg, 1.0) ## Calculate 2-body terms from M0 segment to symmetry image
                continue

            ## If the coreseg is a 1-Residue System, it might be a design position that cannot coexist with the symmetry 1-Residue System
            rc, rs = coreseg[0], symseg[0]
            if rc.ID == rs.ID and rc.ResName != rs.ResName: continue  ## Cannot coexist
            #print rc.ID, rs.ID, rc.ResName, rs.ResName, 'with symmetry'
            if verbose: print('Turning on symmetry interaction between',coreseg.GetResidueIDs(), symseg.GetResidueIDs()) 
            ri.SetSystemPairWeight(coreseg, symseg, 1.0) ## Calculate 2-body terms from M0 segment to symmetry image
        
    ## Now the only non-productive terms in the energy graph should be rotamer-rotamer interactions when/if the same residue
    ## interacts with a symmetry mirror (since both monomers will have to adopt the same rotamer ID eventually)

    if verbose: print('Filling the EnergyGraph')
    energy_function.FillEnergyGraph(rl, n, ri, eg)
    if verbose: print('Done filling the EnergyGraph')

    if verbose: print('Adding the symmetry residue interactions')
    for symnode,node in list(rl.symmetry_matches.items()):
        MergeEGnode(eg,symnode,node)

    ### Cut out just the EnergyGraph associated with the M0 rotamers
    subeg = eg.Subgraph(rl.M0_rotamerIDs)

    return subeg


def check_scanpair_result(rl, n, resi, resj, energy_function):
    eg = EnergyGraph()
    ri = RotamerIterationMap(rl)
    energy_function.FillEnergyGraph(rl, n, ri, eg)
    minE,mini,minj = 999999,0,0
    for roti in rl.GetRotamerIDRange(resi):
      for rotj in rl.GetRotamerIDRange(resj):
         if eg[roti,rotj] < minE:
           minE = eg[roti,rotj]
           mini = roti
           minj = rotj
    return minE

def exposed_pair_generator(composite):
    exposed_resids = pickle.load(open('exposed.pkl'))
    pairs_of_systems = itertools.combinations(composite,2)

def calc_aa_vs_aa_bestcaseE_novo(resi,resj, energy_function):
    print('Working on',resi.ID,resj.ID)
    wtaai, wtaaj = resi.ResName, resj.ResName
    aaseti = set([wtaai] + chargedaa)
    aasetj = set([wtaaj] + chargedaa)
    results = {}
    for aai, aaj in itertools.product(aaseti, aasetj):
        resi.Mutate(amino_dir[aai])
        resj.Mutate(amino_dir[aaj])
        ri = NewSystem()
        ri.Append(resi)
        ri.Parse(str(ri))  ## Convert the residue to a 1-residue System
        rj = NewSystem()
        rj.Append(resj)
        rj.Parse(str(rj))  ## Convert the residue to a 1-residue System
        ij = [ri,rj]
        rl = RotamerLibrary(ij)
        tmpn = NewNeighborMap(ij)
        newrg = invariant_large_rotamer_generator(tmpn,dunbrack_lib)
        rl.GenerateRotamers(newrg)
        bestE = energy_function.ScanPairs(rl, tmpn, ij[0][0], ij[1][0], True, True)
        results[(resi.ID,aai,resj.ID,aaj)] = bestE
    return results

def get_obe_for_rotamers(rl, n, resi, energy_function):
    eg = EnergyGraph()
    ri = RotamerIterationMap(rl)
    energy_function.FillEnergyGraph(rl, n, ri, eg)
    return[ eg.GetVertexEnergy(v) for v in rl.GetRotamerIDRange(resi)]


def calc_aa_vs_aa_bestcaseE(resi,resj, energy_function):
    print('Working on',resi.ID,resj.ID)
    wtaai, wtaaj = resi.ResName, resj.ResName
    aaseti = set([wtaai] + chargedaa)
    aasetj = set([wtaaj] + chargedaa)
    results = {}
    for aai, aaj in itertools.product(aaseti, aasetj):
        resi.Mutate(amino_dir[aai])
        resj.Mutate(amino_dir[aaj])
        rl = RotamerLibrary(composite)
        rl.GenerateRotamers(resi, rg, discardOld = True)
        rl.GenerateRotamers(resj, rg, discardOld = True)
        bestE = energy_function.ScanPairs(rl, n, resi, resj, False, False)
        results[(resi.ID,aai,resj,ID,aaj)] = bestE
    return results


def write_display_pml(pdb, surfacelist, symresidueset):
    pml = open('display_active_res.pml','w')
    reslist =' or resi '.join('%d'%x for x in surfacelist)
    print("""
load %s.pdb
hide everything, solvent
select M0surf, %s and (resi %s)
show sticks, M0surf
globload('monomer.*.pdb')
""" % (pdb, pdb, reslist), file=pml)

    for mid,resi in symresidueset:
        print('show sticks, monomer.%d and resi %d' % (mid,resi), file=pml)
    print("""
hide lines
show ribbon
hide_bb_sticks
orient %s 
remove elem H
""" % (pdb), file=pml)
    pml.close()


def snap2zero(x, epsilon=1e-10):
    if abs(x) < epsilon: return 0.0
    else: return x


def TwoBodyEnergiesDict(tbe):
    dictdict = {}
    for i in range(len(tbe)):
      for j in range(i+1,len(tbe)):
          d = {}
          e = tbe[i,j]
          if max(e) == 0 and min(e) == 0: continue
          d['atr'], d['rep'], d['solv'] = e[0], e[1], e[2]
          d['hb_sc_sc'], d['hb_sc_bb'], d['hb_bb_bb_sr'], d['hb_bb_bb_lr'] = e[3], e[4], e[5], e[6]
          d['rotpair'] = e[7]
          dictdict[(i,j)] = d
    return dictdict

def rmsd(A,B): return sqrt(mean(square(A - B).sum(1)))


def prepend_CRYST1_record(pdbfile, P):
    """Use crystallography meta data in P, to write a CRYST1 record to pdbfile"""
    edges, angles = P.xtal_edges, P.xtal_angles, 
    sg, num = P.xtal_sg, P.xtal_num_in_unit_cell
    orig = open(pdbfile).read()
    template = 'CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f %-10s %4d' 
    values = (edges[0],edges[1],edges[2],angles[0],angles[1],angles[2],sg,num)
    out = open(pdbfile,'w') 
    print(template % values, file=out)
    print(orig, file=out)

def TransferSymmetry(A,B):
    """Transfer crystallography meta data from A to B"""
    try: B.xtal_edges = A.xtal_edges
    except AttributeError: pass
    try: B.xtal_angles = A.xtal_angles
    except AttributeError: pass
    try: B.xtal_sg = A.xtal_sg
    except AttributeError: pass
    try: B.xtal_basis = A.xtal_basis
    except AttributeError: pass
    try: B.xtal_num_in_unit_cell = A.xtal_num_in_unit_cell
    except AttributeError: pass

def WriteSystemVectorEnsemble(ensemble,outpdb):
    """Write a list of SystemVectors to a pdb file"""
    WritePDB(ensemble[0], outpdb=outpdb, chain='A')
    for i, c in enumerate('BCDEFGHIJKLMNOPQRSTUVWXYZ'):
        if i >= len(ensemble) - 1: continue
        WritePDB(ensemble[i+1], outpdb=outpdb, chain=c, append=True)

def IdentifySymmetryRotamers(rl, pl=None, verbose=False, epsilon=0.1):
    """Go through a RotamerLibrary and identify rotamers that only differ by chain identifier. Jan 2015

    Arguments: (RotamerLibrary)rl
    Optional: (PoseLibrary)pl, (bool)verbose, (float)epsilon
    Returns: PoseLibrary

    Usage:
    pl = IdentifySymmetryRotamers(rl)

    Note: in the case of a symmetry repacking, the input optional PoseLibrary is None
    the function just deletes the redundant rotamers and returns a PoseLibrary with just the remaining ids
    In the case of a design problem, the input PoseLibrary must be pruned more carefully.

    The optional argument epsilon determines how dissimilar a chi value must be to count as unique.
    """
    redundant = {}
    rev_red = {}
    rld = list(rl.View().items())
    for rotI, (resiI, resnI, chainI, chitupleI) in rld:
        for rotJ, (resiJ, resnJ, chainJ, chitupleJ) in rld:
            if rotJ <= rotI: continue
            if rotJ in redundant: continue
            if chainJ <= chainI: continue
            if resiI != resiJ: continue
            if resnI != resnJ: continue

            chidiffs = [abs(float(chitupleI[i]) - float(chitupleJ[i])) for i in range(4)]
            periodicdiffs = []
            for chidiff in chidiffs:
                if chidiff > 180: periodicdiffs.append(abs(chidiff - 360))
                else: periodicdiffs.append(chidiff)
            if max(periodicdiffs) > epsilon: continue
    
            #fuzzyI = tuple(x[:-1] for x in chitupleI)
            #fuzzyJ = tuple(x[:-1] for x in chitupleJ)
            #if fuzzyI != fuzzyJ: continue

            #if chitupleI != chitupleJ: continue
            #if resiI == 53: print chitupleI, chitupleJ, periodicdiffs
            if verbose: print('rotamer %6d (%4d %s) matches rotamer %6d (%4d %s)' % (rotJ, resiJ, chainJ, rotI, resiI, chainI))
            redundant[rotJ] = rotI
            if rotI not in rev_red: rev_red[rotI] = []
            rev_red[rotI].append(rotJ)
    #fold = len(rl)/len(redundant)
    #numred = len(set(redundant.keys()))
    numuniq = len(set(redundant.values()))
    fold = len(rl)/numuniq
    if len(rl) % numuniq == 0:
    #test = set(len(x) for x in rev_red.values())
    #if len(test)==1:
    #    fold = test.pop() + 1
        print('Perfect %d-fold symmetry' % fold)
        test = set(len(x) for x in list(rev_red.values()))
        if len(test)>1:
            assert False, 'A problem with the rotamer matching. Try sorted([(x[0],len(x[1])) for x in rev_red.items()], key=lambda x: x[1])'
        testfold = test.pop() + 1
        assert testfold == fold 
    else:
        print('Identified %d symmetry rotamers (from %d original rotamers)' % (len(redundant), len(rl)))
        assert False, 'The remaining code is only tested with perfect symmetry'
    if pl is None:  ## Repacking case, no existing PoseLibrary
        rl2 = rl.Clone()
        rl2.DeleteRotamers(redundant)
        pl = NewPoseLibrary(dict((k,v) for k,v in list(rl2.Dict().items()) if v))
    elif hasattr(pl, 'Dict'):  ## Design case
        pldict = {}
        for k,v in list(pl.Dict().items()):
            if len(v)==1 and v[0] in redundant: continue
            uniq = sorted(set(v).difference(redundant))
            if uniq == []: continue
            pldict[k] = uniq
        pl = NewPoseLibrary(pldict)
    pl.redundant = redundant
    pl.rev_red = rev_red
    return pl

def BuildSymmetryEnergyGraph(eg, pl, verbose=False):
    """Build a new EnergyGraph from an old one, merging nodes specified as redundant in a dictionary (pl.redundant). Jan 2015

    Arguments: (EnergyGraph)eg, (PoseLibrary)pl
    Optional: (bool)verbose
    Returns: EnergyGraph 

    Usage:
    pl = IdentifySymmetryRotamers(rl, eg.pl, epsilon=0.01, verbose=True)
    symEG = BuildSymmetryEnergyGraph(eg, pl)
    """
    ## We'll merge all nodes for two entire monomers -- then multiply all energies by 0.5
    ## First, we populate a new EnergyGraph with the non-redundant vertices from the parent graph
    newEG = EnergyGraph()
    for vid in eg.GetVertexIDs():
        if vid in pl.redundant: continue 
        newEG.AddVertex(vid, eg.GetVertexEnergy(vid))
        #if vid == 18: print 'Created node %d with initial OBE of %.3f' % (vid, newEG.GetVertexEnergy(vid))
    ## Next, redundant vertices are merged into their representative master vertex
    ## The one body energy of the representative is incremented accordingly
    ## If there is a self-self interaction edge, that also goes into the one-body energy
    for vid in eg.GetVertexIDs():
        if vid in pl.redundant:
            rep = pl.redundant[vid]
            repE = newEG.GetVertexEnergy(rep)
            #if verbose and rep==100: print 'Merging',vid,'into',rep, 'pre vertex E =', newEG.GetVertexEnergy(rep)
            newEG.SetVertexEnergy(rep, repE + eg.GetVertexEnergy(vid))  ## OBE SUM
            #if rep == 18: print 'Merged OBE (%.3f) for %d into %d' % (eg.GetVertexEnergy(vid), vid, rep)
            #if verbose and rep==100: print 'Merging',vid,'into',rep, 'new vertex E =', newEG.GetVertexEnergy(rep)
            try:
                edgeE = eg.GetEdgeEnergy(vid, rep)
                newEG.SetVertexEnergy(rep, newEG.GetVertexEnergy(rep) + edgeE)
                #print 'Added simple self-self edge E for',vid,rep
                #if rep == 18: print 'Added simple self-self edge E (%.3f) for'%edgeE,vid,rep,'to',rep
            except RuntimeError: pass

    ## What about pairs of redundant rotamers that interact with each other...
   

    # To get a correctly behaving EnergyGraph it is important to not have spurious edges
    # Thus we must not add edges between mutually exlusive possible choices
    rotidsets = [set(x) for x in list(pl.Dict().values())]
    #taboo = {}
    #for k,v in pl.Dict().items():
    #    if len(v)==1: continue
    #    vset = set(v)
    #    for v0 in v:
    #        taboo[v0] = vset.difference([v0])


    for (i, j), E in eg.EdgeGenerator():
        if i in pl.redundant: vi = pl.redundant[i]
        else: vi = i
        if j in pl.redundant: vj = pl.redundant[j]
        else: vj = j

        ## possible to have a pair of redundant rotamers that interact with each other
        if i in pl.redundant and j in pl.redundant and vi == vj:
            newEG.SetVertexEnergy(vi, newEG.GetVertexEnergy(vi) + E)
            #if vi == 18: print 'Added self-self edge E (%.3f) for'%E,i,j,'to',vi
            #except RuntimeError: pass

        ## Don't add edges for mutually exclusive rotamers
        same_residue = False
        for rotidset in rotidsets:
            if vi in rotidset and vj in rotidset: 
                same_residue = True
                break
        if same_residue: continue
        #if vi in taboo and vj in taboo[vi]: continue

        try:
            oldE = newEG.GetEdgeEnergy(vi, vj)
            newEG.SetEdgeEnergy(vi, vj, oldE + E)
        except RuntimeError:
            newEG.AddEdge(vi, vj, E)
    return newEG

def TranslateSymmetryCombo(combo, pl):
    """Convert the SymmetryEnergyGraph combination into a combination appropriate for the original EnergyGraph.

    Arguments: (iterable)combo, (PoseLibrary)pl
    Returns: list

    Usage:
    newEG = BuildSymmetryEnergyGraph(eg, pl)
    combo, E = sa.Pack(pl, newEG)
    oligo_combo = TranslateSymmetryCombo(combo, pl)
    """
    assert hasattr(pl, 'rev_red')
    assert len(set(len(x) for x in list(pl.rev_red.values()))) == 1
    fold = len(list(pl.rev_red.values())[0]) + 1
    combo = list(combo)
    #assert pl.NumDimensions() % len(combo) == 0
    #fold = pl.NumDimensions()/len(combo)
    print('TranslateSymmetryCombo expanding combination %d-fold' % fold)
    dupe_combo = combo * fold
    oligo_combo = []
    for i in range(fold):
        for j in range(len(combo)):
            rep = combo[j]
            if i==0: actual = rep
            else: actual = pl.rev_red[rep][i-1]
            oligo_combo.append(actual)
    #return oligo_combo
    
    #return combo + [pl.rev_red[x] for x in dupe_combo]

    #oligo_combo = []
    ##for rotid, choices in zip(dupe_combo, rl.Dict().values()):
    ##for rotid, choices in zip(dupe_combo, [x[1] for x in sorted(rl.Dict().items())]):
    #for dim, rotid in enumerate(dupe_combo):
    #    choices = rl.DimensionIDRange(dim)
    #    picked = False
    #    for choice in choices:
    #        try:
    #            translated = pl.redundant[choice]
    #        except:
    #            translated = choice
    #        if translated == rotid:
    #            oligo_combo.append(choice)
    #            picked = True
    #    assert picked
    test = []
    for rotid in oligo_combo:
        if rotid in pl.redundant:
            test.append(pl.redundant[rotid])
        else:
            test.append(rotid)
    assert test == dupe_combo
    print('TranslateSymmetryCombo seems to check out')
    return oligo_combo


def GenerateSymmetryDesignRotamers(composite, dp, rg, rl):
    """Generate rotamers for the supplied composite, rl, rg, and design palette dp, but only for chain A.
       Then ensure all amino acids with the same residue ID and name have exactly the same rotamers.

    Arguments: (list of Systems)composite, (DesignPalette)dp, (RotamerGenerator)rg, (RotamerLibrary)rl
    Returns: rl 

    Usage:
    rl = GenerateSymmetryDesignRotamers(composite, dp, rg, rl)
    """
    chis = {}
    for r in ResidueGenerator(composite):
        if r.Chain != 'A': continue
        if (r.ID,r.Chain) not in dp.frozen: 
            rl.GenerateRotamers(rg, r)
        chis[(r.ID,r.ResName)] = rl.GetRotamerChis(r)

    for r in ResidueGenerator(composite):
        #if (r.ID,r.Chain) in dp.frozen: continue
        #if r.Chain == 'A': continue
        rl.AddRotamers(r, chis[(r.ID,r.ResName)], discardOld=True)  ## HMM
    return rl

def GenerateSymmetryRepackRotamers(S, rg, rl, activeresids=None):
    """Generate rotamers for the supplied list of Systems, rl, rg, and design palette dp, but only for chain A.
       Then ensure all amino acids with the same residue ID and name have exactly the same rotamers.

    Arguments: (list of Systems)S, (RotamerGenerator)rg, (RotamerLibrary)rl
    Optional: (set)activeresids
    Returns: rl 

    Usage:
    rl = GenerateSymmetryRepackRotamers(S, rg, rl)

    Note: Use the activeresids argument to limit rotamer generation to a subset of the residues by ID
    """
    chis = {}
    for r in ResidueGenerator(S):
        if r.Chain != 'A': continue
        if not activeresids or r.ID in activeresids: 
            rl.GenerateRotamers(rg, r)
        chis[(r.ID,r.ResName)] = rl.GetRotamerChis(r)

    for r in ResidueGenerator(S):
        #if r.Chain == 'A': continue
        #if activeresids and r.ID not in activeresids: continue
        rl.AddRotamers(r, chis[(r.ID,r.ResName)], discardOld=True)
    return rl
