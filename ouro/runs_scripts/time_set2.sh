#!/bin/bash
# time_set2.sh GPU...: a second block of timed requests (the next 100 per task in the same hash order, OURO_TIME_START=100)
# for the batch-1 comparisons of Section 5.5, with the protocol of time_math0.sh: two passes, the second in the opposite
# order, MATH-500 zero-shot. Conditions: Original 1/1, 2/2, 4/4; Distilled 1/1, 2/2 (rank 16 and 80); Ours 4/1, 4/2
# (LoopBridge, fixed m=4; folded maps); LoopBridge 4/1 with per-step maps (one pass, for its accuracy).
# Output: runs/e2e/{clnB,clnB3}-<name>-<task>.jsonl. Run on an otherwise idle box, one job per GPU.
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/tset2_q.txt; cd $R; P=$R/e2e/prompts
A=$R/X26-a-universal/checkpoint-48m.pt; F=$R/X26-a-fixed4-lora/checkpoint-48m.pt
E=$R/X26-E48f-p1d1/checkpoint-48m.pt; D=$R/X26-D48f-p2d2/checkpoint-48m.pt
E80=$R/X26-E48fr80-p1d1/checkpoint-48m.pt; D80=$R/X26-D48fr80-p2d2/checkpoint-48m.pt
PY="OURO_TIME_START=100 CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 PYTHONPATH=/root/big-small environments/ouro/.venv/bin/python"
one() {  # PASS COND CKPT NAME [ENV]
  for t in gsmfull math0shot bbh mmlupro; do
    echo "cd /root/big-small && $5 $PY ouro/scripts/time_requests.py $M/model-2.6b $2 $3 $P/X26-a41-$t 100 $R/e2e/$1-$4-$t.jsonl > $R/e2e/$1-$4-$t.log 2>&1"
  done
}
pass() {  # PASS
  for d in 1 2 4; do one $1 native${d}st - native${d}st; done
  one $1 D1st $E D1st; one $1 D2st $D D2st
  one $1 D1st $E80 D1r80st "OURO_LORA_RANK=80"; one $1 D2st $D80 D2r80st "OURO_LORA_RANK=80"
  one $1 a4x1pm $A a4x1pm; one $1 a4x2pm $A a4x2pm; one $1 a4x1pm $F f4x1pm; one $1 a4x2pm $F f4x2pm
}
{ pass clnB; one clnB a4x1sb $A a4x1sb; pass clnB3 | tac; } | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do env -u OURO_LORA_RANK setsid nohup ./grid_worker.sh $g $Q 1 > /dev/null 2>&1 < /dev/null & done
