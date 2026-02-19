from pyscaffoldscan.pdbtools import PDB, first_model

indir = 'guest_models'
outdir = 'processed_guest_models'

############################
## EVE-HD  #################
############################
code = '1jgg'
guest_chain = 'B'
DNA1 = 'C'
DNA2 = 'D'
DNA1_lo, DNA1_hi = [201,207] 
DNA2_lo, DNA2_hi = [214,220] 

P = PDB(f'{indir}/{code}.pdb')
sele  = f'chain {guest_chain}'
sele += f' or (chain {DNA1} and resi {DNA1_lo}-{DNA1_hi})'
sele += f' or (chain {DNA2} and resi {DNA2_lo}-{DNA2_hi})'
graft = P[sele]
graft.WritePDB(f'{outdir}/{code}.pdb')

############################
## bZip  ################### 
############################
code = '1ysa'
guest_chain1 = 'C'
guest_chain2 = 'D'
DNA1 = 'A'
DNA2 = 'B'
DNA1_lo, DNA1_hi = [6,14] 
DNA2_lo, DNA2_hi = [28,36] 

P = PDB(f'{indir}/{code}.pdb')
sele  = f'chain {guest_chain1}'
sele += f' or chain {guest_chain2}'
sele += f' or (chain {DNA1} and resi {DNA1_lo}-{DNA1_hi})'
sele += f' or (chain {DNA2} and resi {DNA2_lo}-{DNA2_hi})'
graft = P[sele]
graft.WritePDB(f'{outdir}/{code}.pdb')

############################
## UBX-HD  #################
############################
code = '1b8i'
guest_chain = 'B'
DNA1 = 'C'
DNA2 = 'D'
DNA1_lo, DNA1_hi = [9,14] 
DNA2_lo, DNA2_hi = [19,24] 

P = PDB(f'{indir}/{code}.pdb')
sele  = f'chain {guest_chain}'
sele += f' or (chain {DNA1} and resi {DNA1_lo}-{DNA1_hi})'
sele += f' or (chain {DNA2} and resi {DNA2_lo}-{DNA2_hi})'
graft = P[sele]
graft.WritePDB(f'{outdir}/{code}.pdb')

############################
## C-Clamp  ################
############################
code = '7dta'
guest_chain = 'A'
DNA1 = 'B'
DNA2 = 'C'
DNA1_lo, DNA1_hi = [4,8]
DNA2_lo, DNA2_hi = [15,19]

P = PDB(first_model(f'{indir}/{code}.pdb'), hetatm=True)  #hetatm=True to retain the Zn
sele  = f'chain {guest_chain}'
sele += f' or (chain {DNA1} and resi {DNA1_lo}-{DNA1_hi})'
sele += f' or (chain {DNA2} and resi {DNA2_lo}-{DNA2_hi})'
graft = P[sele]
graft.WritePDB(f'{outdir}/{code}.pdb')

############################
## EnH-HD  #################
############################
code = '3hdd'
guest_chain = 'A'
DNA1 = 'C'
DNA2 = 'D'
DNA1_lo, DNA1_hi = [211,216] 
DNA2_lo, DNA2_hi = [328,333] 

P = PDB(f'{indir}/{code}.pdb')
sele  = f'chain {guest_chain}'
sele += f' or (chain {DNA1} and resi {DNA1_lo}-{DNA1_hi})'
sele += f' or (chain {DNA2} and resi {DNA2_lo}-{DNA2_hi})'
graft = P[sele]
graft.WritePDB(f'{outdir}/{code}.pdb')

############################
## ANTP-HD  ################
############################
code = '4xid'
guest_chain = 'A'
DNA1 = 'B'
DNA2 = 'C'
DNA1_lo, DNA1_hi = [6,11] 
DNA2_lo, DNA2_hi = [28,33] 

P = PDB(f'{indir}/{code}.pdb')
sele  = f'chain {guest_chain}'
sele += f' or (chain {DNA1} and resi {DNA1_lo}-{DNA1_hi})'
sele += f' or (chain {DNA2} and resi {DNA2_lo}-{DNA2_hi})'
graft = P[sele]
graft.WritePDB(f'{outdir}/{code}.pdb')

