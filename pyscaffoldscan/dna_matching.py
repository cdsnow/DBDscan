import numpy as np
from scipy.spatial.distance import cdist
from math import ceil, floor
from colorama import Fore
import networkx as nx

def vector_direction(A):
    """Compute the normalized vector from the first to the last point of A."""
    return (A[-1] - A[0]) / np.linalg.norm(A[-1] - A[0])

def is_same_direction(A, B):
    """Check if vectors A and B are in the same general direction."""
    dot_product = np.dot(A, B)
    return dot_product > 0

def find_matching_positions(Axyz, Bxyz, cutoff=2.0):
    """Find positions in A that have a matching position in B within the cutoff distance."""
    matches = []
    for a in Axyz:
        distances = np.linalg.norm(Bxyz - a, axis=1)
        if np.any(distances <= cutoff):
            matches.append(a)
    return matches

def generate_match_cgo(Axyz, Bxyz, matches, color=(1,1,0), radius=0.3):
    """Generate PyMOL CGO commands to visualize matches between C1' atoms using cylinders
    
    Parameters:
    -----------
    Axyz : np.ndarray
        Coordinates of C1' atoms in first structure
    Bxyz : np.ndarray
        Coordinates of C1' atoms in second structure
    matches : list
        List of (i,j,dist) tuples indicating matched positions
    color : tuple
        RGB color values for the cylinders (default: yellow)
    radius : float
        Radius of the cylinders (default: 0.3 Angstroms)
        
    Returns:
    --------
    str
        PyMOL CGO script commands
    """
    cgo_commands = []
    cgo_commands.append("from pymol.cgo import *")
    cgo_commands.append("cmd.delete('dna_matches')")
    
    # Start the CGO object
    cgo_commands.append("obj = [")
    
    # Add cylinders for each match
    for i, j, _ in matches:
        # Get coordinates
        ax, ay, az = Axyz[i]
        bx, by, bz = Bxyz[j]
        
        # Add cylinder
        cgo_commands.append("    CYLINDER,")
        cgo_commands.append(f"    {ax}, {ay}, {az},")  # start point
        cgo_commands.append(f"    {bx}, {by}, {bz},")  # end point
        cgo_commands.append(f"    {radius},")  # radius
        cgo_commands.append(f"    {color[0]}, {color[1]}, {color[2]},")  # color start
        cgo_commands.append(f"    {color[0]}, {color[1]}, {color[2]},")  # color end
    
    # Close the CGO object
    cgo_commands.append("]")
    
    # Create the object in PyMOL
    cgo_commands.append("cmd.load_cgo(obj, 'dna_matches')")
    
    return "\n".join(cgo_commands)

def generate_match_lines(Axyz, Bxyz, matches, color=(1,1,0)):
    """Generate PyMOL CGO commands to visualize matches between C1' atoms
    
    Parameters:
    -----------
    Axyz : np.ndarray
        Coordinates of C1' atoms in first structure
    Bxyz : np.ndarray
        Coordinates of C1' atoms in second structure
    matches : list
        List of (i,j,dist) tuples indicating matched positions
    color : tuple
        RGB color values for the connection lines (default: yellow)
        
    Returns:
    --------
    str
        PyMOL CGO script commands
    """
    cgo_commands = []
    cgo_commands.append("from pymol.cgo import *")
    cgo_commands.append("cmd.delete('dna_matches')")
    
    # Start the CGO object
    cgo_commands.append("obj = [")
    cgo_commands.append("    BEGIN, LINES,")
    cgo_commands.append(f"    COLOR, {color[0]}, {color[1]}, {color[2]},")
    
    # Add lines for each match
    for i, j, _ in matches:
        # Get coordinates
        ax, ay, az = Axyz[i]
        bx, by, bz = Bxyz[j]
        
        # Add line vertices
        cgo_commands.append(f"    VERTEX, {ax}, {ay}, {az},")
        cgo_commands.append(f"    VERTEX, {bx}, {by}, {bz},")
    
    # Close the CGO object
    cgo_commands.append("    END")
    cgo_commands.append("]")
    
    # Create the object in PyMOL
    cgo_commands.append("cmd.load_cgo(obj, 'dna_matches')")
    
    return "\n".join(cgo_commands)

def calculate_rmsd(A, B):
    """Calculate the RMSD between two sets of matched positions."""
    differences = A - B
    squared_differences = np.square(differences)
    mean_squared_differences = np.mean(squared_differences)
    rmsd = np.sqrt(mean_squared_differences)
    return rmsd

def AssessDNAOverlap(Axyz, Bxyz):
    """Given two sets of coordinates that should be C1' traces, assess the fusability"""
    Axyz, Bxyz = np.array(Axyz), np.array(Bxyz)
    # Calculate mindist
    tmpdists = cdist(Axyz, Bxyz)
    
    # Verify directionality
    Adir = vector_direction(Axyz)
    Bdir = vector_direction(Bxyz)
    if not is_same_direction(Adir, Bdir):
        assert False, "AssessDNAOverlap: Strands are not oriented in the same direction"
    
    # Match positions within cutoff
    matched_positions = find_matching_positions(Axyz, Bxyz)
    if not matched_positions:
        return 99999, 0, tmpdists.min()
    
    # Convert matched positions to numpy array for calculation
    matched_positions = np.array(matched_positions)
    # For simplicity, we'll just select the closest match in B for each match in A to calculate RMSD.
    # In a real scenario, you might need a more sophisticated method to ensure the matches correspond correctly.
    closest_matches = [Bxyz[np.argmin(np.linalg.norm(Bxyz - a, axis=1))] for a in matched_positions]
    closest_matches = np.array(closest_matches)
    
    # Calculate RMSD over matching positions
    rmsd = calculate_rmsd(matched_positions, closest_matches)

    return rmsd, len(matched_positions), tmpdists.min()

def FuseDNA(M0,M1,twochains,numoverlap):
    assert len(twochains)==2, 'expecting exactly two chain identifiers (e.g. BC)'

    share1 = numoverlap // 2
    share2 = numoverlap - share1

    x = twochains[0] ## C below
    y = twochains[1] ## B below

    M1C = M1[f'chain {x}']
    newB = M1C['resi < %d' % M1C.resids[-share1]]
    newB.SetChain(f'{y}')
    newB.Renumber()

    M0B = M0[f'chain {y}']
    newB2 = M0B['resi > %d' % M0B.resids[share2-1]]
    newB2.Renumber()
    renum_dict = dict((i+1,i+len(newB.seq)+1) for i in range(len(newB2.seq)))
    newB2.Renumber(renum_dict)

    testB = newB + newB2

    M0C = M0[f'chain {x}']
    newC2 = M0C['resi < %d' % M0C.resids[-share2]]
    newC2.Renumber()
    M1B = M1[f'chain {y}']
    newC = M1B['resi > %d' % M1B.resids[share1-1]]
    newC.SetChain(f'{x}')
    newC.Renumber()
    renum_dict = dict((i+1,i+len(newC2.seq)+1) for i in range(len(newC.seq)))
    newC.Renumber(renum_dict)
    testC = newC2 + newC
    return testB + testC

def find_contiguous_overlaps(Axyz, Bxyz, cutoff=2.0):
    """Find the longest stretch of contiguous overlapping positions"""
    distances = cdist(Axyz, Bxyz)
    all_matches = []
    used_A = set()
    used_B = set()
    
    # First find all possible matches within cutoff
    for i in range(len(Axyz)):
        for j in range(len(Bxyz)):
            if distances[i,j] <= cutoff:
                all_matches.append((i, j, distances[i,j]))
    
    # Sort by position in first chain
    all_matches.sort()
    
    # Find all possible contiguous segments
    segments = []
    current_segment = []
    
    for match in all_matches:
        if not current_segment:
            current_segment.append(match)
            continue
            
        prev_match = current_segment[-1]
        # Check if this match continues the segment
        if (match[0] == prev_match[0] + 1 and  # Adjacent in A
            match[1] == prev_match[1] + 1):     # Adjacent in B
            current_segment.append(match)
        else:
            if len(current_segment) > 1:
                segments.append(current_segment)
            current_segment = [match]
    
    if len(current_segment) > 1:
        segments.append(current_segment)
    
    # Find longest segment
    if not segments:
        return [], 99999
        
    best_segment = max(segments, key=len)
    
    # Calculate RMSD for best segment
    matched_A = np.array([Axyz[m[0]] for m in best_segment])
    matched_B = np.array([Bxyz[m[1]] for m in best_segment])
    rmsd = np.sqrt(np.mean(np.sum((matched_A - matched_B)**2, axis=1)))
    
    return best_segment, rmsd

def find_contiguous_overlaps_nx(Axyz, Bxyz, cutoff=2.0, min_length=6, verbose=0):
    """Find the longest stretch of contiguous overlapping positions using NetworkX
    
    Parameters:
    -----------
    Axyz : np.ndarray
        Coordinates for first structure
    Bxyz : np.ndarray
        Coordinates for second structure
    cutoff : float
        Maximum distance (in same units as coordinates) for positions to be considered matching
    min_length : int
        Minimum number of consecutive positions required to consider it a valid alignment
        
    Returns:
    --------
    matches : list
        List of (i, j, distance) tuples representing matched positions
    rmsd : float
        Root mean square deviation of the matched positions
    """
    V = verbose
    # Calculate all pairwise distances
    distances = cdist(Axyz, Bxyz)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes for all matches within cutoff
    nodes = []
    for i in range(len(Axyz)):
        for j in range(len(Bxyz)):
            if distances[i,j] <= cutoff:
                G.add_node((i,j), distance=distances[i,j])
                nodes.append((i,j))
                
    if V>0: print(f"Created {len(nodes)} nodes for matches within cutoff")
    
    # Add edges between compatible consecutive matches
    for i,j in nodes:
        for k,l in nodes:
            if k == i+1 and l == j+1:
                G.add_edge((i,j), (k,l))
                
    if V>0: print(f"Graph has {G.number_of_edges()} edges between consecutive matches")
    
    # Find all simple paths between any pair of nodes
    longest_path = []
    max_length = min_length - 1  # Initialize to less than min_length
    
    # Only start from nodes that have no predecessors
    start_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    if V>0: print(f"Searching for paths from {len(start_nodes)} possible start points")
    
    # Only consider paths that could potentially be long enough
    for start in start_nodes:
        # Get nodes that are reachable from this start node
        reachable = nx.descendants(G, start)
        reachable.add(start)
        
        # If there aren't enough reachable nodes, skip this start point
        if len(reachable) < min_length:
            continue
            
        end_nodes = [n for n in reachable if G.out_degree(n) == 0]
        for end in end_nodes:
            try:
                paths = nx.all_simple_paths(G, start, end)
                for path in paths:
                    if len(path) >= min_length and len(path) > max_length:
                        max_length = len(path)
                        longest_path = path
            except nx.NetworkXNoPath:
                continue
                
    if not longest_path:
        if V>0: print(f"No paths of minimum length {min_length} found")
        return [], 99999
        
    # Convert path back to matches format with distances
    matches = [(i, j, distances[i,j]) for i,j in longest_path]
    
    # Calculate RMSD for best segment
    matched_A = np.array([Axyz[m[0]] for m in matches])
    matched_B = np.array([Bxyz[m[1]] for m in matches])
    rmsd = np.sqrt(np.mean(np.sum((matched_A - matched_B)**2, axis=1)))
    
    if V>0: 
        print(f"\nFound longest path of length {len(matches)}")
        print(f"A positions: {[m[0] for m in matches]}")
        print(f"B positions: {[m[1] for m in matches]}")
        print(f"RMSD: {rmsd:.3f}")
    
    return matches, rmsd


def old_find_contiguous_overlaps_nx(Axyz, Bxyz, cutoff=2.0, verbose=0):
    """Find the longest stretch of contiguous overlapping positions using NetworkX"""
    V = verbose
    # Calculate all pairwise distances
    distances = cdist(Axyz, Bxyz)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes for all matches within cutoff
    # Node will be (i,j) representing matching position i in A to j in B
    nodes = []
    for i in range(len(Axyz)):
        for j in range(len(Bxyz)):
            if distances[i,j] <= cutoff:
                G.add_node((i,j), distance=distances[i,j])
                nodes.append((i,j))
                
    if V>0: print(f"Created {len(nodes)} nodes for matches within cutoff")
    
    # Add edges between compatible consecutive matches
    # (i,j) -> (k,l) is valid if k=i+1 and l=j+1
    for i,j in nodes:
        for k,l in nodes:
            if k == i+1 and l == j+1:
                G.add_edge((i,j), (k,l))
                
    if V>0: print(f"Graph has {G.number_of_edges()} edges between consecutive matches")
    
    # Find all simple paths between any pair of nodes
    # We'll keep track of the longest one
    longest_path = []
    max_length = 0
    
    # For efficiency, we'll only start from nodes that have no predecessors
    start_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    if V>0: print(f"Searching for paths from {len(start_nodes)} possible start points")
    
    for start in start_nodes:
        # We only need to check paths to nodes with no successors
        end_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
        for end in end_nodes:
            try:
                # Use all_simple_paths to get all possible paths
                paths = nx.all_simple_paths(G, start, end)
                for path in paths:
                    if len(path) > max_length:
                        max_length = len(path)
                        longest_path = path
            except nx.NetworkXNoPath:
                continue
                
    if not longest_path:
        return [], 99999
        
    # Convert path back to matches format with distances
    matches = [(i, j, distances[i,j]) for i,j in longest_path]
    
    # Calculate RMSD for best segment
    matched_A = np.array([Axyz[m[0]] for m in matches])
    matched_B = np.array([Bxyz[m[1]] for m in matches])
    rmsd = np.sqrt(np.mean(np.sum((matched_A - matched_B)**2, axis=1)))
    
    if V>0:
        print(f"\nFound longest path of length {len(matches)}")
        print(f"A positions: {[m[0] for m in matches]}")
        print(f"B positions: {[m[1] for m in matches]}")
        print(f"RMSD: {rmsd:.3f}")
    
    return matches, rmsd

def AssessDNAOverlapContiguous(Axyz, Bxyz, cutoff=2.0, debug_cgo_py=None):
    """Modified version of AssessDNAOverlap requiring contiguous matches
    
    Parameters:
    -----------
    Axyz : array-like
        Coordinates of C1' atoms in first structure
    Bxyz : array-like
        Coordinates of C1' atoms in second structure
    cutoff : float
        Distance cutoff for considering positions matched (default: 2.0)
    debug_cgo_py : str or None
        If provided, path to output PyMOL CGO script for visualization
        
    Returns:
    --------
    tuple
        (rmsd, number_of_matches, minimum_distance)
    """
    Axyz, Bxyz = np.array(Axyz), np.array(Bxyz)
    
    # Calculate mindist
    tmpdists = cdist(Axyz, Bxyz)
    mindist = tmpdists.min()
    
    # Verify directionality
    Adir = vector_direction(Axyz)
    Bdir = vector_direction(Bxyz)
    if not is_same_direction(Adir, Bdir):
        return 99999, 0, mindist
    
    # Find best contiguous overlap
    matches, rmsd = find_contiguous_overlaps_nx(Axyz, Bxyz, cutoff=cutoff)
    
    # Generate CGO script if requested
    if debug_cgo_py and matches:
        cgo_script = generate_match_cgo(Axyz, Bxyz, matches)
        with open(debug_cgo_py, 'w') as f:
            f.write(cgo_script)
    
    return rmsd, len(matches), mindist

def CheckProteinClashes(M0, M1, protchains='AHDE', clashdist=3, clashtolerance=0):
    protM0 = M0[' or '.join([f'chain {cc}' for cc in protchains])]
    protM1 = M1[' or '.join([f'chain {cc}' for cc in protchains])]
    X = protM0.GetCoords()
    Y = protM1.GetCoords()
    alldists = cdist(X, Y)
    clashcount = (alldists < clashdist).sum()
    if clashcount > clashtolerance:
        print(Fore.RED + 'Protein clashcount %d' % clashcount + Fore.RESET)
        return True
    return False

def Manage_DNA_Overlap(M0, contig1chains, contig2chains, overlap, overhang, verbose=0):
    assert overhang < overlap, f"Requested an overhang ({overhang}) longer than the overlapping section length ({overlap})"
    if overlap % 2 == 1:
        print(Fore.RED + 'WARNING: Manage_DNA_Overlap is not optimized for odd number sized overlaps currently.')
    if overhang % 2 == 1:
        print(Fore.RED + 'WARNING: Manage_DNA_Overlap can only make even number sized overhangs currently.')
        print(f'Instead of {overhang}, will generate {2*ceil(overhang/2)}',Fore.RESET)
    V = verbose
    activechains = contig1chains + contig2chains
    remainderM0 = M0[' and '.join([f'not chain {cc}' for cc in activechains])]
    c1 = contig1chains[0] ## should be the 5'->3' chain towards overlap in M0
    c2 = contig1chains[1] ## should be the 5'->3' chain away from overlap in M1
    c3 = contig2chains[0] ## should be the 5'->3' chain towards overlap in M1
    c4 = contig2chains[1] ## should be the 5'->3' chain away from overlap in M0
    s1 = M0[f'chain {c1}']
    s2 = M0[f'chain {c2}']
    s3 = M0[f'chain {c3}']
    s4 = M0[f'chain {c4}']

    L = len(s1.seq)
    s1_5trim = ceil(overlap/2) #+ floor(overhang/2)
    s1_3trim = ceil(overlap/2) #+ ceil(overhang/2)
    firstresid_s1 = s1.resids[s1_5trim]
    lastresid_s1 = s1.resids[-s1_3trim]

    s2_5trim = ceil(overlap/2) #- floor(overhang/2)
    s2_3trim = ceil(overlap/2) #+ ceil(overhang/2)
    firstresid_s2 = s2.resids[s2_5trim]
    lastresid_s2 = s2.resids[-s2_3trim]

    s3_5trim = ceil(overlap/2) #- floor(overhang/2)
    s3_3trim = ceil(overlap/2) #+ ceil(overhang/2)
    firstresid_s3 = s3.resids[s3_5trim]
    lastresid_s3 = s3.resids[-s3_3trim]

    s4_5trim = ceil(overlap/2) #- floor(overhang/2)
    s4_3trim = ceil(overlap/2) #+ ceil(overhang/2)
    firstresid_s4 = s4.resids[s4_5trim]
    lastresid_s4 = s4.resids[-s4_3trim]

    ## Overhang time
    ## with no modification here, we should end up with a blunt end junction
    ## If we extend the 3' end of strand1, we should also increment the firstresid 
    ## this scheme can only make overhangs with even numbers of bp
    firstresid_s1 += ceil(overhang/2)
    lastresid_s1 += ceil(overhang/2)
    firstresid_s2 += ceil(overhang/2)
    lastresid_s2 += ceil(overhang/2)
    firstresid_s3 += ceil(overhang/2)
    lastresid_s3 += ceil(overhang/2)

    if V>0:
        print(f'The provided overlap is {overlap}')
        print(f"strand1 is {L} nt long, so we could trim {s1_5trim} from the 5' and {s1_3trim} from the 3'")
        print(f"s1.resids:",s1.resids)
        print(f"s1.resids[{s1_5trim}]",firstresid_s1) 
        print(f"s1.resids[-{s1_3trim}]",lastresid_s1) 
        print(f"s2.resids:",s2.resids)
        print(f"s2.resids[{s2_5trim}]",firstresid_s2) 
        print(f"s2.resids[-{s2_3trim}]",lastresid_s2) 
    
    #s1_sele = f'resi > {firstresid_s1 - 1} and resi < {lastresid_s1}'
    #s2_sele = f'resi > {firstresid_s2 - 1} and resi < {lastresid_s2}'
    s1_sele = f'resi < {lastresid_s1}'
    s2_sele = f'resi > {firstresid_s2 - 1}'
    s1 = s1[s1_sele]
    s2 = s2[s2_sele]
    #s3 = s3[s1_sele] ## C6 rosette
    #s4 = s4[s2_sele] ## C6 rosette
    #s3 = s3[s2_sele]
    #s4 = s4[s1_sele]
    #s3_sele = f'resi > {firstresid_s3 - 1} and resi < {lastresid_s3}'
    #s4_sele = f'resi > {firstresid_s4 - 1} and resi < {lastresid_s4}'
    s3_sele = f'resi < {lastresid_s3}'
    s4_sele = f'resi > {firstresid_s4 - 1}'
    s3 = s3[s3_sele]
    s4 = s4[s4_sele]
    if V>0:
        print(f"s1_sele:",s1_sele) 
        print(f"s2_sele:",s2_sele) 
        print(f"s3_sele:",s3_sele) 
        print(f"s4_sele:",s4_sele) 
        print('s1 len',len(s1.seq),'overlap',overlap,'target',L - overlap)

    return remainderM0 + s1 + s2 + s3 + s4 

def TrimChain(P, chain, termini, num): 
    resids = P[f'chain {chain}'].resids 
    #print('TrimChain',chain,resids,termini, num)
    if termini==5:
        sele = f'not (chain {chain} and resi < {resids[num]})'
    elif termini==3:
        sele = f'not (chain {chain} and resi > {resids[-(num+1)]})'
    else:
        assert False, 'TrimChain needs either 5 or 3 for termini'
    return P[sele]

