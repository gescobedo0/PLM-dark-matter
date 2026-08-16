#!/bin/bash
# Per-residue embedding, plain sbatch style (like slurm_embed.sh). Just:
#     sbatch slurm_residues.sh
# ---------------------------------------------------------------------------
# EDIT ONCE:
#   1. --partition below  -> your GPU partition (sinfo)
#   2. env setup          -> in activate_env.sh (shared by all jobs)
#   3. keep --array size == NSHARDS below
# When the array finishes it AUTO-SUBMITS merge+probe (slurm_residue_probe.sh);
# to disable that, comment out the marked block at the bottom.
# ---------------------------------------------------------------------------
#SBATCH --job-name=mp_resembed
#SBATCH --partition=gpu                 # <-- EDIT: your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-7                      # <-- keep in sync with NSHARDS
#SBATCH --output=logs/resembed_%A_%a.out
set -euo pipefail
NSHARDS=8
MODEL="${MODEL:-650M}"                    # override at submit: sbatch --export=ALL,MODEL=35M slurm_residues.sh

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source activate_env.sh

python 05b_embed_residues.py --model "$MODEL" --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS"

# --- auto-submit merge+probe once, after the whole array succeeds ------------
if [ "${SLURM_ARRAY_TASK_ID}" = "${SLURM_ARRAY_TASK_MAX}" ]; then
    sbatch --export=ALL,MODEL="$MODEL" --dependency=afterok:"${SLURM_ARRAY_JOB_ID}" slurm_residue_probe.sh
fi
# ----------------------------------------------------------------------------
