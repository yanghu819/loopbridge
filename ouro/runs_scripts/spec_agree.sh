#!/bin/bash
# spec_agree.sh GPU...: one-decode-loop drafters for speculative decoding of Original 4/4 (ouro/scripts/spec_agree.py)
# on the 400 timed requests: Original 1/1, Distilled 1/1 (rank 16 and 80), Ours fixed m=4 and LoopBridge at 4/1, each
# teacher-forced on Original 4/4's greedy answers (runs/e2e/clean-native4st-<task>.jsonl). All draft at the same cost
# per token (one loop). Output: runs/e2e/spec-<name>-<task>.jsonl. Two jobs per GPU (grid_worker MAX=2).
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/spec_q.txt; cd $R; P=$R/e2e/prompts
A=$R/X26-a-universal/checkpoint-48m.pt; F=$R/X26-a-fixed4-lora/checkpoint-48m.pt
E=$R/X26-E48f-p1d1/checkpoint-48m.pt; E80=$R/X26-E48fr80-p1d1/checkpoint-48m.pt
PY="CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 PYTHONPATH=/root/big-small environments/ouro/.venv/bin/python"
job() {  # COND CKPT NAME [ENV]
  for t in gsmfull math0shot bbh mmlupro; do
    echo "cd /root/big-small && $4 $PY ouro/scripts/spec_agree.py $M/model-2.6b $1 $2 $P/X26-a41-$t 100 $R/e2e/clean-native4st-$t.jsonl $R/e2e/spec-$3-$t.jsonl > $R/e2e/spec-$3-$t.log 2>&1"
  done
}
{ job native1st - native1st; job D1st $E D1st; job D1st $E80 D1r80st "OURO_LORA_RANK=80"
  job a4x1pm $F f4x1pm; job a4x1pm $A a4x1pm; } | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do env -u OURO_LORA_RANK setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
