#!/bin/bash
# Run the full GuestScan pipeline for guest 1ysa in scaffold 9YZK
# Usage: bash all_steps_1ysa_in_9YZK.sh
set -e

GUEST=1ysa
SCAFFOLD=9YZK

echo "============================================================"
echo "  GuestScan pipeline for $GUEST in $SCAFFOLD"
echo "============================================================"

echo ""
echo "Step A: Process guest models"
echo "------------------------------------------------------------"
python A_process_guest_models.py

echo ""
echo "Step C: Find symmetry mates (scaffold-only, run once)"
echo "------------------------------------------------------------"
if [ ! -f symmetry_mates/${SCAFFOLD}_mates.json ]; then
    python C_find_symmetry_mates.py $SCAFFOLD
else
    echo "  symmetry_mates/${SCAFFOLD}_mates.json already exists, skipping."
fi

echo ""
echo "Step B: Find registers"
echo "------------------------------------------------------------"
python B_find_registers.py $GUEST $SCAFFOLD

echo ""
echo "Step D: Categorize registers"
echo "------------------------------------------------------------"
python D_categorize_registers.py $GUEST $SCAFFOLD

echo ""
echo "Step E: Rank registers"
echo "------------------------------------------------------------"
python E_rank_registers.py $GUEST $SCAFFOLD

echo ""
echo "Step F: Register diagram"
echo "------------------------------------------------------------"
python F2_sidebyside_register_diagram.py $GUEST $SCAFFOLD

echo ""
echo "============================================================"
echo "  Done. Outputs in output/$GUEST.$SCAFFOLD/"
echo "============================================================"
echo ""
ls -lh output/$GUEST.$SCAFFOLD/${GUEST}_ranked.csv \
       output/$GUEST.$SCAFFOLD/${GUEST}_ranked.pml \
       output/$GUEST.$SCAFFOLD/${GUEST}_ranked_sequences.seqs
echo ""
echo "AF3 JSONs: $(ls output/$GUEST.$SCAFFOLD/af3_input_jsons/*.json 2>/dev/null | wc -l | tr -d ' ') files in output/$GUEST.$SCAFFOLD/af3_input_jsons/"
echo ""
echo "To visualize:  start PyMol in this folder and then @output/$GUEST.$SCAFFOLD/${GUEST}_ranked.pml"
