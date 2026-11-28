#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/dev/shm/inkling/code
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p /dev/shm/inkling/eval_swebench_verified /dev/shm/inkling/vllm_logs /dev/shm/inkling/logs

M_VL="/dev/shm/inkling/model_varloop"
M_MINI="/dev/shm/inkling/model"
URLS="http://127.0.0.1:8000 http://127.0.0.1:8001 http://127.0.0.1:8002 http://127.0.0.1:8003 http://127.0.0.1:8004 http://127.0.0.1:8005 http://127.0.0.1:8006 http://127.0.0.1:8007"
SWE_EFFORT="${SWE_EFFORT:-medium}"
SWE_MAX_STEPS="${SWE_MAX_STEPS:-60}"

wait_gpus_empty() {
  pkill -9 -f "[v]llm serve" || true
  pkill -9 -f "[v]llm_loopbridge" || true
  sleep 2
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d " "); do
    kill -9 "$p" 2>/dev/null || true
  done
  for _ in $(seq 1 40); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0} $1>m{m=$1} END{print m}')
    if [ "${used:-0}" -lt 500 ]; then return 0; fi
    sleep 2
  done
}

wait_ports() {
  local label="$1"
  local ngpu="${2:-8}"
  for g in $(seq 0 $((ngpu - 1))); do
    for _ in $(seq 1 240); do
      if curl -sf "http://127.0.0.1:$((8000+g))/health" >/dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] swe_${label} GPU $g READY"; break
      fi
      sleep 2
    done
    curl -sf "http://127.0.0.1:$((8000+g))/health" >/dev/null
  done
}

serve_swe() {
  local ngpu="$1" model="$2" mode="$3" m="$4" n="$5" ckpt="$6" pairing="$7" policy="$8" apc="$9" label="${10}"
  wait_gpus_empty
  local extra=()
  [ -n "$ckpt" ] && extra+=(--checkpoint "$ckpt")
  [ -n "$pairing" ] && extra+=(--pass-pairing "$pairing")
  [ -n "$policy" ] && extra+=(--multi-turn-policy "$policy")
  [ "$apc" = "1" ] && extra+=(--enable-prefix-caching)
  for g in $(seq 0 $((ngpu - 1))); do
    CUDA_VISIBLE_DEVICES=$g /dev/shm/vllm-venv/bin/python -m inkling.vllm_loopbridge serve \
      --model "$model" --mode "$mode" --m "$m" --n "$n" "${extra[@]}" \
      --port $((8000+g)) --max-model-len 262144 --gpu-memory-utilization 0.92 \
      > "/dev/shm/inkling/vllm_logs/swe_${label}_gpu${g}.log" 2>&1 &
  done
  wait_ports "$label" "$ngpu"
}

run_swe_arm() {
  local limit="$1" model="$2" mode="$3" m="$4" n="$5" ckpt="$6" pairing="$7" policy="$8" apc="$9" label="${10}"
  local outf="/dev/shm/inkling/eval_swebench_verified/${label}/swebench_verified.jsonl"
  if [ -f "$outf" ] && [ "$(wc -l < "$outf")" -ge "$limit" ]; then
    echo "[$(date +%H:%M:%S)] === SKIP ALREADY DONE (N=$limit): $label ==="
    return 0
  fi
  echo "[$(date +%H:%M:%S)] === SWE-BENCH VERIFIED (N=$limit): $label ($mode $m/$n pairing=$pairing policy=$policy apc=$apc) ==="
  serve_swe 8 "$model" "$mode" "$m" "$n" "$ckpt" "$pairing" "$policy" "$apc" "n${limit}_${label}"
  /dev/shm/vllm-venv/bin/python -u -m inkling.eval_swebench_verified \
    --urls $URLS --model "$model" \
    --output "/dev/shm/inkling/eval_swebench_verified/${label}" \
    --reasoning-effort "$SWE_EFFORT" --max-steps "$SWE_MAX_STEPS" \
    --max-tokens-per-step 8192 --max-context-tokens 196608 \
    --n-samples 1 --concurrency 24 --limit "$limit" \
    --temperature 1.0 --top-p 0.95 --seed 42
  wait_gpus_empty
}

run_suite_at_limit() {
  local lim="$1"
  # 1. Locked 24M n=1 control + Suffix-Fold (Policy APC-S) fast breakthrough checks FIRST!
  [ -f /dev/shm/inkling/runs/vl_distilled_1_1_24m/checkpoint-24m.pt ] && \
    run_swe_arm "$lim" "$M_VL" distilled 1 1 /dev/shm/inkling/runs/vl_distilled_1_1_24m/checkpoint-24m.pt "" reprefill 1 vl_distilled_1_1_apc
  run_swe_arm "$lim" "$M_VL" loopbridge 3 1 /dev/shm/inkling/runs/vl_lb_3_1_24m/checkpoint-24m.pt tail suffix_fold 1 vl_lb_3_1_policy_apcs
  run_swe_arm "$lim" "$M_VL" loopbridge 4 2 /dev/shm/inkling/runs/vl_lb_4_2_kvonly_12m/checkpoint-12m.pt anchored suffix_fold 1 vl_lb_4_2_kv_policy_apcs
  run_swe_arm "$lim" "$M_MINI" loopbridge 3 2 /dev/shm/inkling/runs/lb_3_2_24m/checkpoint-24m.pt tail suffix_fold 1 mini_lb_3_2_policy_apcs

  # 2. Core Baselines & Variable-depth TML-d24-varloop Hero Arms
  run_swe_arm "$lim" "$M_VL"   training_free 4 4 "" "" reprefill 1 vl_orig_4_4_apc
  run_swe_arm "$lim" "$M_VL"   training_free 3 3 "" "" reprefill 1 vl_orig_3_3_apc
  run_swe_arm "$lim" "$M_VL"   training_free 2 2 "" "" reprefill 1 vl_orig_2_2_apc
  [ -f /dev/shm/inkling/runs/vl_distilled_2_2_t4_12m/checkpoint-12m.pt ] && \
    run_swe_arm "$lim" "$M_VL" distilled 2 2 /dev/shm/inkling/runs/vl_distilled_2_2_t4_12m/checkpoint-12m.pt "" reprefill 1 vl_distilled_2_2_apc
  run_swe_arm "$lim" "$M_VL"   loopbridge 3 2 /dev/shm/inkling/runs/vl_lb_universal_kvonly_nosynth_18m/checkpoint-18m.pt anchored d0 1 vl_univ_3_2_policy_d0
  run_swe_arm "$lim" "$M_VL"   loopbridge 4 2 /dev/shm/inkling/runs/vl_lb_4_2_kvonly_12m/checkpoint-12m.pt anchored d0 1 vl_lb_4_2_kv_policy_d0
  run_swe_arm "$lim" "$M_VL"   loopbridge 4 2 /dev/shm/inkling/runs/vl_lb_4_2_kvonly_12m/checkpoint-12m.pt anchored reprefill 0 vl_lb_4_2_kv_policy_ri
  run_swe_arm "$lim" "$M_VL"   loopbridge 4 3 /dev/shm/inkling/runs/vl_lb_4_3_kvonly_12m/checkpoint-12m.pt anchored d0 1 vl_lb_4_3_kv_policy_d0
  run_swe_arm "$lim" "$M_VL"   training_free 1 1 "" "" reprefill 1 vl_orig_1_1_apc
  run_swe_arm "$lim" "$M_VL"   loopbridge 3 1 /dev/shm/inkling/runs/vl_lb_3_1_24m/checkpoint-24m.pt tail d0 1 vl_lb_3_1_policy_d0
  run_swe_arm "$lim" "$M_VL"   loopbridge 3 1 /dev/shm/inkling/runs/vl_lb_3_1_24m/checkpoint-24m.pt tail reprefill 0 vl_lb_3_1_policy_ri

  # 3. Fixed-depth Inkling-2-Mini
  run_swe_arm "$lim" "$M_MINI" training_free 3 3 "" "" reprefill 1 mini_orig_3_3_apc
  [ -f /dev/shm/inkling/runs/distilled_2_2_v3/checkpoint-24m.pt ] && \
    run_swe_arm "$lim" "$M_MINI" distilled 2 2 /dev/shm/inkling/runs/distilled_2_2_v3/checkpoint-24m.pt "" reprefill 1 mini_distilled_2_2_apc
  run_swe_arm "$lim" "$M_MINI" loopbridge 3 2 /dev/shm/inkling/runs/lb_3_2_24m/checkpoint-24m.pt tail reprefill 0 mini_lb_3_2_policy_ri
  run_swe_arm "$lim" "$M_MINI" loopbridge 3 2 /dev/shm/inkling/runs/lb_3_2_24m/checkpoint-24m.pt tail d0 1 mini_lb_3_2_policy_d0
}

# PASS 1: Fast stratified N=50 across all 15 arms (~5-8 min/arm -> rapid full baseline + LoopBridge signal!)
run_suite_at_limit 50

# PASS 2: Auto-resume and expand all 15 arms to ALL 155 gold-verified official SWE-bench Verified tasks!
run_suite_at_limit 155

echo "[$(date +%H:%M:%S)] === ALL SWE-BENCH VERIFIED PASSES (N=50 & N=155) COMPLETE! RESUMING WAVE-3 QUEUE ==="
exec bash /dev/shm/inkling/run_a1_rebuttal_w3.sh
