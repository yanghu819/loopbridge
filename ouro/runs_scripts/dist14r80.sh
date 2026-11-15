#!/bin/bash
# dist14r80.sh GPUS PORT: the parameter-matched baseline on Ouro-1.4B, Distilled 1/1 with LoRA rank 80 (as many trainable
# parameters as the 1.4B LoopBridge), trained by launch_x26_distill.sh (SIZE=14 RANK=80, first-token target, run
# X14-E48fr80-p1d1), then its Table 9 benchmarks at 1/1 (settings of seed14.sh), two per GPU on the same GPUs.
# Evaluations O14x-E48fr80-<task>, RULER O14-E48fr80f; every evaluation line carries OURO_LORA_RANK=80 and the workers
# are started without it (HANDOFF 11-13: a worker that inherits the rank breaks other runs' jobs).
GPUS=$1; PORT=$2; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; N=X14-E48fr80-p1d1
[ -f $R/$N/checkpoint-48m.pt ] || { SIZE=14 RANK=80 ./launch_x26_distill.sh E $GPUS $PORT first || exit 1; }
CK=$R/$N/checkpoint-48m.pt; Q=$R/dist14r80_q.txt; L=O14x-E48fr80; B=O14-E48fr80f; P="OURO_LORA_RANK=80 "
MA="path=$M/model-1.4b,condition=E,checkpoint=$CK,max_length=16384"
lm() { ls lmeval/out/$L-$1/*/results_* > /dev/null 2>&1 || echo "$P./lm1.sh \$GPU $L-$1 - \"$MA\" \"$2\""; }
{ lm math0shot "--tasks minerva_math500 --num_fewshot 0"; lm gsm "--tasks gsm8k_cot --num_fewshot 3 --limit 400"
  lm mmlu "--tasks mmlu --num_fewshot 5 --limit 20"; lm bbh "--tasks bbh_cot_fewshot --limit 60"
  lm arc "--tasks arc_challenge --num_fewshot 25"; lm mmlupro "--tasks mmlu_pro --limit 30"
  lm hellaswag "--tasks hellaswag --num_fewshot 10"; lm winogrande "--tasks winogrande --num_fewshot 5"
  lm evalplus "--tasks humanevalplus_chat_local,mbppplus_chat_local"
  ls bench/$B/s*/ruler16k.jsonl > /dev/null 2>&1 || echo "cd /root/big-small && ${P}MODEL=$M/model-1.4b DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' ./ouro/runs_scripts/bench3.sh $B E $CK \$GPU 1"
} | flock $Q.lock sh -c "cat >> $Q"
for g in ${GPUS//,/ }; do for w in 1 2; do   # grid_worker runs one job at a time: two workers give two jobs per GPU
  env -u OURO_LORA_RANK setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done; done
