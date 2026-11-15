#!/bin/bash
# fixed14.sh SEED GPUS PORT: "Ours, fixed m=4" on Ouro-1.4B, as X26-a-fixed4-lora on Ouro-2.6B (launch_variant.sh fixed4):
# the aligned adapter's LoRA trained only on the cells 4/1..4/4 with the maps frozen at the identity, same data, tokens,
# batch and teacher (Original 4/4) as the 1.4B LoopBridge O14-a-universal. SEED=1 is the default seed (no --seed).
# Then its Ouro-1.4B Table 9 benchmarks at 4/1 (settings of seed14.sh), two per GPU on the same GPUs.
# Runs: O14-a-fixed4-lora[-s<SEED>]; evaluations O14xf4[s<SEED>]-a41-<task>, RULER O14f4[s<SEED>]-a41f.
S=$1; GPUS=$2; PORT=$3; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R
SUF=$([ "$S" = 1 ] || echo "s$S"); N=O14-a-fixed4-lora$([ "$S" = 1 ] || echo "-s$S"); L=O14xf4$SUF-a41; B=O14f4$SUF-a41f
EXTRA=$([ "$S" = 1 ] || echo "--seed $S")
[ -f $R/$N/checkpoint-48m.pt ] || { OURO_CELLS=4x1,4x2,4x3,4x4 MODEL=$M/model-1.4b DATA=$M/mix-data-48m \
  ./run_mix2.sh $N a $GPUS $PORT 48000000 --micro 1 --freeze-maps $EXTRA || exit 1; }
CK=$R/$N/checkpoint-48m.pt; Q=$R/fixed14s${S}_q.txt
MA="path=$M/model-1.4b,condition=a41,checkpoint=$CK,max_length=16384"
lm() { ls lmeval/out/$L-$1/*/results_* > /dev/null 2>&1 || echo "./lm1.sh \$GPU $L-$1 - \"$MA\" \"$2\""; }
{ lm math0shot "--tasks minerva_math500 --num_fewshot 0"; lm gsm "--tasks gsm8k_cot --num_fewshot 3 --limit 400"
  lm mmlu "--tasks mmlu --num_fewshot 5 --limit 20"; lm bbh "--tasks bbh_cot_fewshot --limit 60"
  lm arc "--tasks arc_challenge --num_fewshot 25"; lm mmlupro "--tasks mmlu_pro --limit 30"
  lm hellaswag "--tasks hellaswag --num_fewshot 10"; lm winogrande "--tasks winogrande --num_fewshot 5"
  lm evalplus "--tasks humanevalplus_chat_local,mbppplus_chat_local"
  ls bench/$B/s*/ruler16k.jsonl > /dev/null 2>&1 || echo "cd /root/big-small && MODEL=$M/model-1.4b DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' ./ouro/runs_scripts/bench3.sh $B a41 $CK \$GPU 1"
} | flock $Q.lock sh -c "cat >> $Q"
for g in ${GPUS//,/ }; do env -u OURO_LORA_RANK setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
