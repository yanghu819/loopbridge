#!/bin/bash
# seed14.sh ARM SEED GPUS PORT: another training seed of an Ouro-1.4B row of Table 9 (four prefill loops, one decode
# loop), trained exactly as the seed-1 run but with --seed SEED (skipped if the checkpoint exists), then its benchmarks
# run two per GPU on the same GPUs.
#   ARM=a  LoopBridge   O14-a-seed<SEED>       cell a41   runs O14xs<SEED>-a41-<task>, RULER O14s<SEED>-a41f
#   ARM=E  Distilled 1/1 X14-E48f-p1d1-s<SEED>  condition E runs O14x-E48fs<SEED>-<task>, RULER O14-E48fs<SEED>f
# Benchmarks and settings are those of the seed-1 runs (O14x-a41-*, O14x-E48f-*), MATH-500 zero-shot.
A=$1; S=$2; GPUS=$3; PORT=$4; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R
if [ $A = a ]; then
  N=O14-a-seed$S; C=a41; L=O14xs$S-a41; B=O14s$S-a41f
  [ -f $R/$N/checkpoint-48m.pt ] || { MODEL=$M/model-1.4b DATA=$M/mix-data-48m ./run_mix2.sh $N a $GPUS $PORT 48000000 --micro 1 --seed $S || exit 1; }
else
  N=X14-E48f-p1d1-s$S; C=E; L=O14x-E48fs$S; B=O14-E48fs${S}f
  [ -f $R/$N/checkpoint-48m.pt ] || { SIZE=14 SEED=$S ./launch_x26_distill.sh E $GPUS $PORT first || exit 1; }
fi
CK=$R/$N/checkpoint-48m.pt; Q=$R/seed14${A}$S\_q.txt
MA="path=$M/model-1.4b,condition=$C,checkpoint=$CK,max_length=16384"
lm() { ls lmeval/out/$L-$1/*/results_* > /dev/null 2>&1 || echo "./lm1.sh \$GPU $L-$1 - \"$MA\" \"$2\""; }
{ lm math0shot "--tasks minerva_math500 --num_fewshot 0"; lm gsm "--tasks gsm8k_cot --num_fewshot 3 --limit 400"
  lm mmlu "--tasks mmlu --num_fewshot 5 --limit 20"; lm bbh "--tasks bbh_cot_fewshot --limit 60"
  lm arc "--tasks arc_challenge --num_fewshot 25"; lm mmlupro "--tasks mmlu_pro --limit 30"
  lm hellaswag "--tasks hellaswag --num_fewshot 10"; lm winogrande "--tasks winogrande --num_fewshot 5"
  lm evalplus "--tasks humanevalplus_chat_local,mbppplus_chat_local"
  ls bench/$B/s*/ruler16k.jsonl > /dev/null 2>&1 || echo "cd /root/big-small && MODEL=$M/model-1.4b DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' ./ouro/runs_scripts/bench3.sh $B $C $CK \$GPU 1"
} | flock $Q.lock sh -c "cat >> $Q"
for g in ${GPUS//,/ }; do env -u OURO_LORA_RANK setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
