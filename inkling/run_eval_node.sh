#!/usr/bin/env bash
# Launch 8-GPU sharded evaluation on a single 8xB300 node.
# Usage: run_eval_node.sh <mode> <m> <n> <output_dir> <checkpoint_or_none> <shard_offset> <total_shards> <bench1> [bench2 ...]
set -euo pipefail

MODE="$1"
M="$2"
N="$3"
OUT="$4"
CKPT="$5"
OFFSET="$6"
TOTAL="$7"
shift 7
BENCHES=("$@")

mkdir -p "$OUT" /dev/shm/inkling/logs
cd /dev/shm/inkling/code

CKPT_ARG=()
if [[ "$CKPT" != "none" && -n "$CKPT" ]]; then
  CKPT_ARG=(--checkpoint "$CKPT")
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PIDS=()
for g in 0 1 2 3 4 5 6 7; do
  SHARD=$((OFFSET + g))
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. python3 -u -m inkling.eval_inkling \
    --model-path /dev/shm/inkling/model \
    --data /dev/shm/inkling/data/eval_inkling \
    --output "$OUT" \
    --benches "${BENCHES[@]}" \
    --mode "$MODE" --m "$M" --n "$N" \
    "${CKPT_ARG[@]}" \
    --shard-index "$SHARD" --num-shards "$TOTAL" \
    --batch-tokens 98304 --max-batch 96 --chunk-size 2048 \
    ${EXTRA_ARGS:-} \
    > "$OUT/shard_${SHARD}.log" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done
exit $FAIL
