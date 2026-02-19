#!/bin/bash
# Run the full GuestScan pipeline for all guests in all scaffolds.
# Usage: bash run_all.sh
set -e

SCRIPTS=(
    all_steps_1jgg_in_9YZJ.sh
    all_steps_1ysa_in_9YZJ.sh
    all_steps_1b8i_in_9YZJ.sh
    all_steps_3hdd_in_9YZJ.sh
    all_steps_7dta_in_9YZJ.sh
    all_steps_4xid_in_9YZJ.sh
    all_steps_1jgg_in_9YZK.sh
    all_steps_1ysa_in_9YZK.sh
    all_steps_1b8i_in_9YZK.sh
    all_steps_3hdd_in_9YZK.sh
    all_steps_7dta_in_9YZK.sh
    all_steps_4xid_in_9YZK.sh
)

total=${#SCRIPTS[@]}
i=0

for script in "${SCRIPTS[@]}"; do
    i=$((i + 1))
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "  [$i/$total]  $script"
    echo "╚════════════════════════════════════════════════════════════╝"
    bash "$script"
done

echo ""
echo "All $total pipelines complete."
