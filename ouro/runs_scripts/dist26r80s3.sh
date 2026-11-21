#!/bin/bash
# dist26r80s3.sh GPUS PORT: the third training seed of the parameter-matched baseline on Ouro-2.6B, Distilled 1/1 with
# LoRA rank 80 (as many trainable parameters as LoopBridge), trained by launch_x26_distill.sh (RANK=80 SEED=3,
# first-token target, run X26-E48fr80-p1d1-s3; global batch 24 at any GPU count), then the 11 benchmarks of Table 1 at
# 1/1 with the settings of after_x26_distill.sh (MATH-500 zero-shot), two per GPU on the same GPUs.
# Evaluations o26x-E48fr80s3-<task>, RULER X26-E48fr80s3f (paper_numbers R80_SEEDS, dist_seed_key); every evaluation
# line carries OURO_LORA_RANK=80 and the workers are started without it (a worker that inherits the rank breaks other
# runs' jobs).
GPUS=$1; PORT=$2; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; N=X26-E48fr80-p1d1-s3
[ -f $R/$N/TRAIN_DONE ] || { RANK=80 SEED=3 ./launch_x26_distill.sh E $GPUS $PORT first || exit 1; }
CK=$R/$N/checkpoint-48m.pt; Q=$R/dist26r80s3_q.txt; L=o26x-E48fr80s3; B=X26-E48fr80s3f; P="OURO_LORA_RANK=80 "
MA="path=$M/model-2.6b,condition=E,checkpoint=$CK,max_length=16384"
lm() { ls lmeval/out/$L-$1/*/results_* > /dev/null 2>&1 || echo "$P./lm1.sh \$GPU $L-$1 - \"$MA\" \"$2\""; }
{ lm math0shot "--tasks minerva_math500 --num_fewshot 0"; lm gsmfull "--tasks gsm8k_cot --num_fewshot 3"
  lm mmlufull "--tasks mmlu --num_fewshot 5"; lm bbh "--tasks bbh_cot_fewshot --limit 60"
  lm arc "--tasks arc_challenge --num_fewshot 25"; lm mmlupro "--tasks mmlu_pro --limit 30"
  lm hellaswag "--tasks hellaswag --num_fewshot 10"; lm winogrande "--tasks winogrande --num_fewshot 5"
  lm evalplus "--tasks humanevalplus_chat_local,mbppplus_chat_local"
  ls bench/$B/s*/ruler16k.jsonl > /dev/null 2>&1 || echo "cd /root/big-small && ${P}MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' ./ouro/runs_scripts/bench3.sh $B E $CK \$GPU 1"
} | flock $Q.lock sh -c "cat >> $Q"
for g in ${GPUS//,/ }; do for w in 1 2; do   # grid_worker runs one job at a time: two workers give two jobs per GPU
  env -u OURO_LORA_RANK setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done; done
