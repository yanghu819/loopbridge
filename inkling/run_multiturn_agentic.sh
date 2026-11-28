#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/dev/shm/inkling/code
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p /dev/shm/inkling/eval_bfcl_multiturn /dev/shm/inkling/systems_multiturn /dev/shm/inkling/vllm_logs /dev/shm/inkling/logs

M_VL="/dev/shm/inkling/model_varloop"
M_MINI="/dev/shm/inkling/model"
URLS="http://127.0.0.1:8000 http://127.0.0.1:8001 http://127.0.0.1:8002 http://127.0.0.1:8003 http://127.0.0.1:8004 http://127.0.0.1:8005 http://127.0.0.1:8006 http://127.0.0.1:8007"

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
        echo "[$(date +%H:%M:%S)] mt_${label} GPU $g READY"; break
      fi
      sleep 2
    done
    curl -sf "http://127.0.0.1:$((8000+g))/health" >/dev/null
  done
}

serve_mt() {
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
      > "/dev/shm/inkling/vllm_logs/mt_${label}_gpu${g}.log" 2>&1 &
  done
  wait_ports "$label" "$ngpu"
}

run_bfcl_arm() {
  local model="$1" mode="$2" m="$3" n="$4" ckpt="$5" pairing="$6" policy="$7" apc="$8" effort="$9" nsamp="${10}" label="${11}"
  echo "[$(date +%H:%M:%S)] === BFCL-MT ARM: $label ($mode $m/$n pairing=$pairing policy=$policy apc=$apc effort=$effort avg@$nsamp) ==="
  serve_mt 8 "$model" "$mode" "$m" "$n" "$ckpt" "$pairing" "$policy" "$apc" "$label"
  /dev/shm/vllm-venv/bin/python -u -m inkling.eval_bfcl_multiturn \
    --urls $URLS --model "$model" --bfcl-root /dev/shm/inkling/bfcl_v4 \
    --output "/dev/shm/inkling/eval_bfcl_multiturn/${label}" \
    --splits multi_turn_base multi_turn_long_context \
    --reasoning-effort "$effort" --max-tokens-per-step 8192 --max-steps-per-turn 12 \
    --n-samples "$nsamp" --concurrency 32 --temperature 1.0 --top-p 0.95 --seed 42
  wait_gpus_empty
}

run_sys_arm() {
  local mode="$1" m="$2" n="$3" ckpt="$4" pairing="$5" policy="$6" apc="$7" probe="$8" label="$9"
  echo "[$(date +%H:%M:%S)] === MT-SYS 1xB300: $label ==="
  serve_mt 1 "$M_VL" "$mode" "$m" "$n" "$ckpt" "$pairing" "$policy" "$apc" "sys_${label}"
  local pflag=()
  [ "$probe" = "1" ] && pflag+=(--probe-incremental)
  /dev/shm/vllm-venv/bin/python -u -m inkling.bench_multiturn_systems \
    --url http://127.0.0.1:8000 --model "$M_VL" --label "$label" \
    --repeats 2 "${pflag[@]}" \
    --output "/dev/shm/inkling/systems_multiturn/${label}.json"
  wait_gpus_empty
}

# ==============================================================================
# PART 0: FAST SMOKE VERIFICATION (1xB300 Policy D0 APC + BFCL tool execution)
# ==============================================================================
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "[$(date +%H:%M:%S)] === SMOKE TEST: Policy D0 APC on TML-d24-varloop 4/2 KV-Only ==="
  serve_mt 1 "$M_VL" loopbridge 4 2 /dev/shm/inkling/runs/vl_lb_4_2_kvonly_12m/checkpoint-12m.pt anchored d0 1 smoke_d0
  /dev/shm/vllm-venv/bin/python -u -m inkling.eval_bfcl_multiturn \
    --urls http://127.0.0.1:8000 --model "$M_VL" --bfcl-root /dev/shm/inkling/bfcl_v4 \
    --output /dev/shm/inkling/eval_bfcl_multiturn/_smoke_d0 \
    --splits multi_turn_base multi_turn_long_context \
    --reasoning-effort medium --n-samples 1 --limit 4 --concurrency 4 --temperature 1.0 --top-p 0.95 --seed 42
  wait_gpus_empty
fi

# ==============================================================================
# PART 1: FULL BFCL v4 MULTI-TURN AGENTIC EVAL (base 200 + long_context 200, avg@2 = 800 eps/arm!)
# All 8x B300 GPUs at 100% util; effort=medium (realistic agentic tool-use length!)
# ==============================================================================
# --- A. Variable-depth 26.4B TML-d24-varloop ---
run_bfcl_arm "$M_VL" training_free 4 4 "" "" reprefill 1 medium 2 vl_orig_4_4_apc
run_bfcl_arm "$M_VL" training_free 3 3 "" "" reprefill 1 medium 2 vl_orig_3_3_apc
run_bfcl_arm "$M_VL" training_free 2 2 "" "" reprefill 1 medium 2 vl_orig_2_2_apc
[ -f /dev/shm/inkling/runs/vl_distilled_2_2_t4_12m/checkpoint-12m.pt ] && \
  run_bfcl_arm "$M_VL" distilled 2 2 /dev/shm/inkling/runs/vl_distilled_2_2_t4_12m/checkpoint-12m.pt "" reprefill 1 medium 2 vl_distilled_2_2_apc
run_bfcl_arm "$M_VL" training_free 4 2 "" anchored reprefill 0 medium 2 vl_tf_4_2_anch_policy_ri
run_bfcl_arm "$M_VL" loopbridge 4 2 /dev/shm/inkling/runs/vl_lb_4_2_kvonly_12m/checkpoint-12m.pt anchored reprefill 0 medium 2 vl_lb_4_2_kv_policy_ri
run_bfcl_arm "$M_VL" loopbridge 4 2 /dev/shm/inkling/runs/vl_lb_4_2_kvonly_12m/checkpoint-12m.pt anchored d0 1 medium 2 vl_lb_4_2_kv_policy_d0
run_bfcl_arm "$M_VL" loopbridge 4 3 /dev/shm/inkling/runs/vl_lb_4_3_kvonly_12m/checkpoint-12m.pt anchored reprefill 0 medium 2 vl_lb_4_3_kv_policy_ri
run_bfcl_arm "$M_VL" loopbridge 4 3 /dev/shm/inkling/runs/vl_lb_4_3_kvonly_12m/checkpoint-12m.pt anchored d0 1 medium 2 vl_lb_4_3_kv_policy_d0
run_bfcl_arm "$M_VL" loopbridge 3 2 /dev/shm/inkling/runs/vl_lb_universal_kvonly_nosynth_18m/checkpoint-18m.pt anchored d0 1 medium 2 vl_univ_3_2_policy_d0

# --- B. Fixed-depth 26.4B Inkling-2-Mini ---
run_bfcl_arm "$M_MINI" training_free 3 3 "" "" reprefill 1 medium 2 mini_orig_3_3_apc
run_bfcl_arm "$M_MINI" training_free 2 2 "" "" reprefill 1 medium 2 mini_orig_2_2_apc
[ -f /dev/shm/inkling/runs/distilled_2_2_v3/checkpoint-24m.pt ] && \
  run_bfcl_arm "$M_MINI" distilled 2 2 /dev/shm/inkling/runs/distilled_2_2_v3/checkpoint-24m.pt "" reprefill 1 medium 2 mini_distilled_2_2_apc
run_bfcl_arm "$M_MINI" loopbridge 3 2 /dev/shm/inkling/runs/lb_3_2_24m/checkpoint-24m.pt tail reprefill 0 medium 2 mini_lb_3_2_policy_ri
run_bfcl_arm "$M_MINI" loopbridge 3 2 /dev/shm/inkling/runs/lb_3_2_24m/checkpoint-24m.pt tail d0 1 medium 2 mini_lb_3_2_policy_d0

# --- C. Low-Effort Short-Output Stress Test (effort=low -> ultra-short tool outputs!) ---
run_bfcl_arm "$M_VL" training_free 4 4 "" "" reprefill 1 low 2 vl_orig_4_4_apc_low
run_bfcl_arm "$M_VL" training_free 2 2 "" "" reprefill 1 low 2 vl_orig_2_2_apc_low
run_bfcl_arm "$M_VL" loopbridge 4 2 /dev/shm/inkling/runs/vl_lb_4_2_kvonly_12m/checkpoint-12m.pt anchored reprefill 0 low 2 vl_lb_4_2_kv_policy_ri_low
run_bfcl_arm "$M_VL" loopbridge 4 2 /dev/shm/inkling/runs/vl_lb_4_2_kvonly_12m/checkpoint-12m.pt anchored d0 1 low 2 vl_lb_4_2_kv_policy_d0_low

# ==============================================================================
# PART 2: ISOLATED 1xB300 MULTI-TURN SYSTEMS BENCHMARK (APC vs Policy R vs I vs D0)
# (~8 min total uncontended timing across short-output & reasoning multi-turn grids)
# ==============================================================================
[ ! -f /dev/shm/inkling/systems_multiturn/vl_orig_4_4_apc.json ] && \
  run_sys_arm training_free 4 4 "" "" reprefill 1 0 vl_orig_4_4_apc
[ ! -f /dev/shm/inkling/systems_multiturn/vl_orig_4_4_inc_probe.json ] && \
  run_sys_arm training_free 4 4 "" "" reprefill 1 1 vl_orig_4_4_inc_probe
[ ! -f /dev/shm/inkling/systems_multiturn/vl_orig_3_3_apc.json ] && \
  run_sys_arm training_free 3 3 "" "" reprefill 1 0 vl_orig_3_3_apc
[ ! -f /dev/shm/inkling/systems_multiturn/vl_orig_2_2_apc.json ] && \
  run_sys_arm training_free 2 2 "" "" reprefill 1 0 vl_orig_2_2_apc
[ ! -f /dev/shm/inkling/systems_multiturn/vl_lb_4_2_policy_r.json ] && \
  run_sys_arm loopbridge 4 2 /dev/shm/inkling/runs/vl_lb_4_2_kvonly_12m/checkpoint-12m.pt anchored reprefill 0 0 vl_lb_4_2_policy_r
[ ! -f /dev/shm/inkling/systems_multiturn/vl_lb_4_2_policy_d0.json ] && \
  run_sys_arm loopbridge 4 2 /dev/shm/inkling/runs/vl_lb_4_2_kvonly_12m/checkpoint-12m.pt anchored d0 1 0 vl_lb_4_2_policy_d0
[ ! -f /dev/shm/inkling/systems_multiturn/vl_lb_4_3_policy_d0.json ] && \
  run_sys_arm loopbridge 4 3 /dev/shm/inkling/runs/vl_lb_4_3_kvonly_12m/checkpoint-12m.pt anchored d0 1 0 vl_lb_4_3_policy_d0

echo "[$(date +%H:%M:%S)] === ALL MULTI-TURN AGENTIC + SYSTEMS BENCHMARKS COMPLETE! LAUNCHING SWE-BENCH VERIFIED ==="
exec bash /dev/shm/inkling/run_a1_swebench_verified.sh
