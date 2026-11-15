#!/bin/bash
# time_math0.sh GPU...: re-time the MATH-500 requests with the zero-shot prompt that Table 1 uses (the earlier timing
# replayed the 4-shot Minerva prompt). Same 100 problems (chosen by a hash of the item id), same conditions, same
# protocol as time_sweep.sh / time_controls.sh (batch 1: two passes, the second in the opposite order) and
# time_batch.sh (batch 8 and 32: three passes). Prompts: runs/e2e/prompts/X26-a41-math0shot (copy of the lm-eval run).
# Output: runs/e2e/{clean,clean3}-<name>-math0shot.jsonl and {batch,batchb,batchc}<B>-<name>-math0shot.jsonl.
# Run on an otherwise idle box, one job per GPU.
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/tmath0_q.txt; mkdir -p $R/e2e; cd $R; P=$R/e2e/prompts/X26-a41-math0shot
A=$R/X26-a-universal/checkpoint-48m.pt; F=$R/X26-a-fixed4-lora/checkpoint-48m.pt
E=$R/X26-E48f-p1d1/checkpoint-48m.pt; D=$R/X26-D48f-p2d2/checkpoint-48m.pt
E80=$R/X26-E48fr80-p1d1/checkpoint-48m.pt; D80=$R/X26-D48fr80-p2d2/checkpoint-48m.pt
PY="CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 PYTHONPATH=/root/big-small environments/ouro/.venv/bin/python"
one() {  # PASS COND CKPT NAME [ENV]: batch 1
  echo "cd /root/big-small && $5 $PY ouro/scripts/time_requests.py $M/model-2.6b $2 $3 $P 100 $R/e2e/$1-$4-math0shot.jsonl > $R/e2e/$1-$4-math0shot.log 2>&1"
}
bat() {  # PASS COND CKPT NAME B [ENV]: continuous batching
  echo "cd /root/big-small && $6 $PY ouro/scripts/batch_requests.py $M/model-2.6b $2 $3 $P 100 $5 $R/e2e/$1$5-$4-math0shot.jsonl > $R/e2e/$1$5-$4-math0shot.log 2>&1"
}
pass1() {  # PASS
  for d in 1 2 3 4; do one $1 native${d}st - native${d}st; done
  one $1 D1st $E D1st; one $1 D2st $D D2st
  one $1 D1st $E80 D1r80st "OURO_LORA_RANK=80"; one $1 D2st $D80 D2r80st "OURO_LORA_RANK=80"
  for c in a1x1sb a1x2sb a4x1sb a4x2sb a1x1pm a1x2pm a3x1pm a3x2pm a4x1pm a4x2pm; do one $1 $c $A $c; done
  one $1 a4x1pm $F f4x1pm; one $1 a4x2pm $F f4x2pm
}
passb() {  # PASS B
  bat $1 native1st - native1st $2; bat $1 D1st $E D1st $2; bat $1 D1st $E80 D1r80st $2 "OURO_LORA_RANK=80"
  bat $1 a4x1pm $A a4x1pm $2; bat $1 a4x1pm $F f4x1pm $2
  bat $1 native2st - native2st $2; bat $1 D2st $D D2st $2; bat $1 D2st $D80 D2r80st $2 "OURO_LORA_RANK=80"
  bat $1 a4x2pm $A a4x2pm $2; bat $1 a4x2pm $F f4x2pm $2
}
{ pass1 clean; pass1 clean3 | tac
  for p in batch batchb batchc; do for b in 32 8; do passb $p $b; done; done
} | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do env -u OURO_LORA_RANK setsid nohup ./grid_worker.sh $g $Q 1 > /dev/null 2>&1 < /dev/null & done
