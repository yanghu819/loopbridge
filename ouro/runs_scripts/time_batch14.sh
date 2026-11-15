#!/bin/bash
# time_batch14.sh GPU...: batched serving (batch_requests.py, continuous batching) for Ouro-1.4B, as time_batch.sh for
# Ouro-2.6B: the 100 hash-chosen requests of GSM8K, MATH-500, BBH and MMLU-Pro (prompts and stop rules of the
# O14x-a41-<task> evaluations) for Original n/n, Distilled n/n (X14-E48f-p1d1, X14-D48f-p2d2, merged) and LoopBridge
# 4/n (O14-a-universal, maps folded in), n = 1, 2, at batch sizes $BATCHES (default 32 8).
# Output: runs/e2e/<OUT><B>-14-<name>-<task>.jsonl and .summary.json (env OUT, default batch).
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/btime_q.txt; mkdir -p $R/e2e; cd $R; O=${OUT:-batch}
A=$R/O14-a-universal/checkpoint-48m.pt; E=$R/X14-E48f-p1d1/checkpoint-48m.pt; D=$R/X14-D48f-p2d2/checkpoint-48m.pt
job() {  # COND CKPT NAME B
  for t in gsm math bbh mmlupro; do
    echo "cd /root/big-small && CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 PYTHONPATH=/root/big-small environments/ouro/.venv/bin/python ouro/scripts/batch_requests.py $M/model-1.4b $1 $2 $R/lmeval/out/O14x-a41-$t 100 $4 $R/e2e/$O$4-14-$3-$t.jsonl > $R/e2e/$O$4-14-$3-$t.log 2>&1"
  done
}
{
  for b in ${BATCHES:-32 8}; do
    job native1st - native1st $b; job D1st $E D1st $b; job a4x1pm $A a4x1pm $b
    job native2st - native2st $b; job D2st $D D2st $b; job a4x2pm $A a4x2pm $b
  done
} | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 1 > /dev/null 2>&1 < /dev/null & done
