#!/bin/bash
# usage: launch_o.sh [NUM_GPUS=4]
# Arm O: single-bank universal loop adapter (S) + self-live shared answer history (one bank for the prompt AND one
# for the answer at any (p, d): KV of a one-loop model). Trains 48M tokens on GPUS (reserved), then queues
# RULER 4K/8K/16K (O42/O22/O43/O33/O44) and MATH-500 + GSM8K + MMLU (O42/O43) on the dispatcher.
# NEED GPUs are claimed one by one as they become idle (each is reserved from the dispatcher the moment it is claimed).
NEED=${1:-4}; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; N=X26-O-universal
until [ -f $M/adapter-distill-200m/data/fineweb_train.bin ]; do sleep 60; done
got=""
while [ $(echo $got | wc -w) -lt $NEED ]; do
  for g in 0 1 2 3 4 5 6 7; do
    [ $(echo $got | wc -w) -ge $NEED ] && break
    echo " $got " | grep -q " $g " && continue
    grep -qw $g /dev/shm/reserved_gpus 2>/dev/null && continue
    if [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; then got="$got $g"; echo " $g" >> /dev/shm/reserved_gpus; fi
  done
  sleep 30
done
GPUS=$(echo $got | tr " " ",")
echo "$(date +%T) $N on $GPUS"
[ -f $R/$N/TRAIN_DONE ] || DATA=$M/mix-data-48m ./run_mix2.sh $N O $GPUS 29821 48000000 --micro 1 || exit 1
for g in ${GPUS//,/ }; do sed -i "s/\b$g\b//" /dev/shm/reserved_gpus; done
CK=$R/$N/checkpoint-48m.pt
{ for c in O42 O22 O43 O33 O44; do
    echo "MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' BATCH=24576 OURO_NO_CUDNN_SDP=1 ./bench3.sh X26-${c}f $c $CK \$GPU 1"; done
  for c in O42 O43; do MA="path=$M/model-2.6b,condition=$c,checkpoint=$CK,max_length=16384"
    echo "./lm1.sh \$GPU o26x-$c-math - \"$MA\" \"--tasks minerva_math500\""
    echo "./lm1.sh \$GPU X26-$c-gsmfull - \"$MA\" \"--tasks gsm8k_cot --num_fewshot 3\""
    echo "./lm1.sh \$GPU X26-$c-mmlufull - \"$MA\" \"--tasks mmlu --num_fewshot 5\""; done; } | flock jobs.txt.lock tee -a jobs.txt > /dev/null
echo "$(date +%T) evals queued"
