#!/bin/bash
# ---------------------------------------------------------------------------
# TWO REQUIRED EDITS before this will run (or just use ../run_pipeline.sh):
#   1. --partition below: set to YOUR cluster's GPU partition (see: sinfo)
#   2. the environment-activation line near the bottom (torch + fair-esm)
# `sbatch` only QUEUES the job and returns immediately; wait for it to finish
# (squeue -u $USER) before running 06/07/08.
# ---------------------------------------------------------------------------
#SBATCH --job-name=metalprobe_embed
#SBATCH --partition=gpu                 # <-- REQUIRED: your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-19                     # 20 shards; must match NSHARDS below
#SBATCH --output=logs/embed_%A_%a.out
#
# ESM-2 embedding extraction as a Slurm array (spec §4: load the model once per
# job, stream a slice of the catalog). One model per submission; pass the model
# key via env MODEL. Merge shards after the array completes.
#
# Submit:
#   mkdir -p logs
#   MODEL=650M sbatch slurm_embed.sh          # report run
#   MODEL=150M sbatch --array=0-9  slurm_embed.sh   # dev / size sweep
#   MODEL=35M  sbatch --array=0-3  slurm_embed.sh
# After the array finishes:
#   python 06_merge_embeddings.py --model $MODEL
#
set -euo pipefail
MODEL="${MODEL:-650M}"
NSHARDS="${NSHARDS:-20}"                  # keep in sync with --array size

module load cuda 2>/dev/null || true
# activate your environment (conda/venv) here, e.g.:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate metalprobe

python 05_embed_esm2.py \
    --model "$MODEL" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" \
    --nshards "$NSHARDS"
