#!/bin/bash
# batch_set2.sh GPU...: batched serving (batch_requests.py, continuous batching, batch 8 and 32, three passes) of the
# second block of timed requests (the next 100 per task, OURO_TIME_START=100; time_set2.sh times them at batch 1), for
# the conditions of time_batch.sh. Output: runs/e2e/{btB,btBb,btBc}<B>-<name>-<task>.jsonl and .summary.json.
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/bset2_q.txt; cd $R; P=$R/e2e/prompts
A=$R/X26-a-universal/checkpoint-48m.pt; F=$R/X26-a-fixed4-lora/checkpoint-48m.pt
E=$R/X26-E48f-p1d1/checkpoint-48m.pt; D=$R/X26-D48f-p2d2/checkpoint-48m.pt
E80=$R/X26-E48fr80-p1d1/checkpoint-48m.pt; D80=$R/X26-D48fr80-p2d2/checkpoint-48m.pt
PY="OURO_TIME_START=100 CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 PYTHONPATH=/root/big-small environments/ouro/.venv/bin/python"
job() {  # PASS COND CKPT NAME B [ENV]
  for t in gsmfull math0shot bbh mmlupro; do
    echo "cd /root/big-small && $6 $PY ouro/scripts/batch_requests.py $M/model-2.6b $2 $3 $P/X26-a41-$t 100 $5 $R/e2e/$1$5-$4-$t.jsonl > $R/e2e/$1$5-$4-$t.log 2>&1"
  done
}
{ for p in btB btBb btBc; do for b in 32 8; do
    job $p native1st - native1st $b; job $p D1st $E D1st $b; job $p D1st $E80 D1r80st $b "OURO_LORA_RANK=80"
    job $p a4x1pm $A a4x1pm $b; job $p a4x1pm $F f4x1pm $b
    job $p native2st - native2st $b; job $p D2st $D D2st $b; job $p D2st $D80 D2r80st $b "OURO_LORA_RANK=80"
    job $p a4x2pm $A a4x2pm $b; job $p a4x2pm $F f4x2pm $b
  done; done; } | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do env -u OURO_LORA_RANK setsid nohup ./grid_worker.sh $g $Q 1 > /dev/null 2>&1 < /dev/null & done
