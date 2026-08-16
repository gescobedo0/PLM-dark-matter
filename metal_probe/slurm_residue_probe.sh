#!/bin/bash
# Merge residue shards + run the residue probe. Auto-submitted by
# slurm_residues.sh, or run on its own once embeddings exist:
#     sbatch slurm_residue_probe.sh
# EDIT --partition once (same as slurm_residues.sh); env is in activate_env.sh.
#SBATCH --job-name=mp_resprobe
#SBATCH --partition=gpu                 # <-- EDIT: a partition you can run on (CPU is fine here)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/resprobe_%j.out
set -euo pipefail
MODEL="${MODEL:-650M}"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source activate_env.sh

python 06b_merge_residues.py --model "$MODEL"
python 11_residue_probe.py --model "$MODEL"
echo "done -> results/residue/residue_probe.md"
