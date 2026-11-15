#!/bin/bash
# think_seed.sh SEED GPUS PORT: another training seed of LoopBridge on Ouro-2.6B-Thinking, trained exactly as
# T26-a-universal (launch_aligned_more.sh think: mix-think-48m, 48M tokens, micro 1) but with --seed SEED, then the
# eight benchmarks of Figure 11 at prefill 4 / decode 2 with the settings of eval_think_aligned.sh, two per GPU.
# Runs: T26-a-seed<SEED>; evaluations T26s<SEED>-a42-<task>_s<k>, T26s<SEED>x-a42-<task>, T26s<SEED>ne-a42-helmet.
S=$1; GPUS=$2; PORT=$3; M=/dev/shm/big-small-ouro; R=$M/runs; MP=$M/model-2.6b-think; cd $R
N=T26-a-seed$S; P=T26s$S; Q=$R/thinkseed${S}_q.txt
[ -f $R/$N/checkpoint-48m.pt ] || { MODEL=$MP DATA=$M/mix-think-48m ./run_mix2.sh $N a $GPUS $PORT 48000000 --micro 1 --seed $S || exit 1; }
CK=$R/$N/checkpoint-48m.pt; c=a42
MA="path=$MP,condition=$c,checkpoint=$CK,max_length=16384"
{ for s in 0 1 2 3; do echo "./lm1.sh \$GPU $P-$c-gsm8k_think_s$s - \"$MA,batch_tokens=160000,max_batch=64\" \"--tasks gsm8k_think_s$s --apply_chat_template\""; done
  for s in 0 1 2 3 4; do echo "./lm1.sh \$GPU $P-$c-math500_think_s$s - \"$MA,batch_tokens=160000,max_batch=64\" \"--tasks math500_think_s$s --apply_chat_template\""; done
  echo "./lm1.sh \$GPU ${P}x-$c-mmlupro - \"$MA\" \"--tasks mmlu_pro --limit 30\""
  echo "./lm1.sh \$GPU ${P}x-$c-arc - \"$MA\" \"--tasks arc_challenge --num_fewshot 25\""
  echo "./lm1.sh \$GPU ${P}x-$c-bbh - \"$MA\" \"--tasks bbh_cot_fewshot --limit 60\""
  echo "./lm1.sh \$GPU ${P}x-$c-evalplus - \"$MA\" \"--tasks humanevalplus_chat_local,mbppplus_chat_local\""
  echo "cd /root/big-small && NO_EMPTY=1 MODEL=$MP DATA=$M/bench-mini BENCHES=helmet BATCH=65536 ./ouro/runs_scripts/bench3.sh ${P}ne-$c-helmet $c $CK \$GPU 1"
} | flock $Q.lock sh -c "cat >> $Q"
for g in ${GPUS//,/ }; do env -u OURO_LORA_RANK setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
