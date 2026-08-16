#!/bin/bash
# Per-residue phase, correctly ordered (embed residues -> merge -> probe), chained
# with a Slurm dependency. Same PARTITION / ENV_SETUP contract as run_pipeline.sh.
#
#   PARTITION=gpu ENV_SETUP="source ~/miniconda3/bin/activate metalprobe" ./run_residue_pipeline.sh
#
#   MODEL=650M     model key (also 150M, 35M for the residue emergence sweep)
#   NSHARDS=8      array size (per-residue extraction is heavier than pooled)
#
# Requires catalog/residue_labels.parquet (committed; produced by 01b locally).
set -euo pipefail
: "${PARTITION:?set PARTITION=<your gpu partition>  (sinfo)}"
MODEL="${MODEL:-650M}"
NSHARDS="${NSHARDS:-8}"
ENV_SETUP="${ENV_SETUP:-true}"
GPU_GRES="${GPU_GRES:-gpu:1}"
cd "$(dirname "$0")"
mkdir -p logs

EMBED_ID=$(sbatch --parsable \
  --job-name=mp_resembed --partition="$PARTITION" --gres="$GPU_GRES" \
  --cpus-per-task=4 --mem=48G --time=03:00:00 \
  --array=0-$((NSHARDS-1)) --output=logs/resembed_%A_%a.out \
  --wrap "$ENV_SETUP; python 05b_embed_residues.py --model $MODEL --shard \$SLURM_ARRAY_TASK_ID --nshards $NSHARDS")
echo "residue embed array : $EMBED_ID  (model=$MODEL, shards=$NSHARDS)"

POST_ID=$(sbatch --parsable --dependency=afterok:"$EMBED_ID" \
  --job-name=mp_resprobe --partition="$PARTITION" --cpus-per-task=8 --mem=32G \
  --time=01:00:00 --output=logs/resprobe_%j.out \
  --wrap "$ENV_SETUP; python 06b_merge_residues.py --model $MODEL && python 11_residue_probe.py --model $MODEL")
echo "merge + residue probe: $POST_ID  (after $EMBED_ID succeeds)"
echo "results -> results/residue/residue_probe.md"
