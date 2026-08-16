#!/bin/bash
# One command to run the whole GPU pipeline in the CORRECT order.
#
# It submits the embedding array job, then a SECOND job that merges + probes +
# plots, chained with a Slurm dependency so it only starts after the embeddings
# finish successfully. No more running 06/07/08 before the embeddings exist.
#
# REQUIRED: tell it your GPU partition (find it with `sinfo`):
#   PARTITION=gpu ./run_pipeline.sh
#
# Common options (env vars):
#   MODEL=650M           model key (also 150M, 35M for the size sweep)
#   NSHARDS=4            array size (2.5k short seqs embed fast; 1-4 is plenty)
#   ENV_SETUP="source ~/miniconda3/bin/activate metalprobe"   # activate your env
#   GPU_GRES=gpu:1       gres request
#
# The ESM-2 weights (~2.5 GB for 650M) auto-download on first run inside the
# embed job, into ~/.cache/torch/hub/checkpoints/ — there is no separate
# download step.
set -euo pipefail
: "${PARTITION:?set PARTITION=<your gpu partition>  (list them with: sinfo)}"
MODEL="${MODEL:-650M}"
NSHARDS="${NSHARDS:-4}"
ENV_SETUP="${ENV_SETUP:-true}"
GPU_GRES="${GPU_GRES:-gpu:1}"
cd "$(dirname "$0")"
mkdir -p logs

EMBED_ID=$(sbatch --parsable \
  --job-name=mp_embed --partition="$PARTITION" --gres="$GPU_GRES" \
  --cpus-per-task=4 --mem=32G --time=02:00:00 \
  --array=0-$((NSHARDS-1)) --output=logs/embed_%A_%a.out \
  --wrap "$ENV_SETUP; python 05_embed_esm2.py --model $MODEL --shard \$SLURM_ARRAY_TASK_ID --nshards $NSHARDS")
echo "embed array : $EMBED_ID   (model=$MODEL, shards=$NSHARDS, partition=$PARTITION)"

POST_ID=$(sbatch --parsable --dependency=afterok:"$EMBED_ID" \
  --job-name=mp_post --partition="$PARTITION" --cpus-per-task=4 --mem=16G \
  --time=00:30:00 --output=logs/post_%j.out \
  --wrap "$ENV_SETUP; python 06_merge_embeddings.py --model $MODEL && python 07_probe.py && python 08_visualize.py --model $MODEL --pooling mean && python 09_distance_enrichment.py --model $MODEL --pooling mean")
echo "post-process: $POST_ID   (runs only after $EMBED_ID succeeds)"
echo
echo "watch:   squeue -u \$USER"
echo "results: embeddings/esm2_${MODEL}.h5, results/probe_results.md, results/figures/"
echo "if embed fails: sacct -j $EMBED_ID --format=JobID,State,ExitCode ; cat logs/embed_${EMBED_ID}_*.out"
