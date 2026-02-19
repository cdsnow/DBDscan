The overall goal of this project is to take candidate DNA-binding protein models that have been cleaned up
(and placed in processed_guest_models/) and systematically align these to each dsDNA sliding window within
a scaffold crystal. The models that define the scaffold crystals will be in scaffold_models/. 
A reasonable way to align to each DNA window agnostic to the current sequence will be to use the C1' atom positions.
Preexisting useful code includes: 
pyscaffoldscan/pdbtools.py for Parsing pdb files
The PDB class therein has useful functions like GetCoords()
and a powerful selection syntax.
P["name C1'"]  would create a subset of PDB object P with just the C1' atoms.
Other methods include Translate and Rotate.

pyscaffoldscan/superimpy.py a module for finding the rotation and translation needed
to optimally superimpose a mobile set of coordinates onto a fixed target set of coordinates.

Once we have scripts that output all possible sliding window superpositions to output/ we can proceed to Phase 2.

Phase 2 will focus on categorizing each potential binding site. Category 1. The potential guest position will be infeasible since the guest protein will have steric clashes with the other protein inside the scaffold model.
Category 2. The guest protein will clash with a symmetry mate. We will need to model the local crystal environment.
This can be done with:
pyscaffoldscan/xtal.py, pyquat.py, and sgdata.py
Category 3. The potential guest placement will not clash, but might still interact with symmetry mates or the scaffold protein from the same asymmetric unit.
Category 4. The potential guest placement is quite independent, with no atom within 8 Angstroms of any other biomolecule (including symmetry mates) other than the two DNA chains it is bound to.

Phase 3. Will be outputting a nice diagram, map that illustrates these categories and potential neighbor distances as a function of binding site. Also output a PDB model of the local crystal environment for the feasible placements.

Phase 4. Make a build script that in-lines the various needed Python code into a single PyScript HTML file so I can offer this calculation on my University hosted website.
