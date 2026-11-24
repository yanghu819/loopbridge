# Reproducing the paper

> **Quick jump:** [Main-text map](#main-text-results-and-where-they-come-from) · [Environment & data](#environment-models-data) · [Provenance](#provenance) · [Training](#training-one-b300-node-18-gpus-per-run) · [Evaluation](#evaluation) · [Appendix map](#appendix-elements--source)

The paper's numbers come from these places, all in this commit:

1. **Per-item scores** (`ouro/results/items/`): one file per evaluation run used by the paper (item id → score).
   `python ouro/scripts/paper_numbers.py compute` recomputes every paired comparison, interval and mean in the tables
   and text (paired bootstrap, 5,000 resamples, seed 0) and writes `ouro/results/paper_numbers.json`. It needs numpy and math-verify (Python 3.10+;
   MATH-500 answers are scored with math-verify).
2. **Timing records** (`ouro/results/e2e/`, `ouro/results/throughput-sb4/`): the end-to-end timing table, time ratios
   and decode throughput, also recomputed by `paper_numbers.py compute`.
3. **Decision-model records** (`ouro/results/lev/`): per-depth LevBench development outputs for the Lev appendix.
4. **26.4B sparse-MoE Day-0 vLLM records** (`inkling/results/eval_official_exact.json`, `eval_vllm_*.json`, `vllm_sse_*.json`):
   per-item official-exact scores (`GPQA Diamond` 198, `LiveCodeBench v6` 175, `MRCR v2 8n 256K–512K` 100,
   `GraphWalks BFS 256K` 100, `LongBench v2 1M` 503) and isolated single-GPU Day-0 vLLM Blackwell timing (`512..256K`).
   `uv run --with numpy --with matplotlib python inkling/make_figs.py` recomputes every 26.4B paired bootstrap 95% CI
   (5,000 resamples, seed 0) and redraws Figures 15–16.

`ouro/scripts/paper_figs.py` draws Figures 1–14 and generated Ouro tables from `paper_numbers.json`.

## Main-text results and where they come from

| Paper item | `paper_numbers.py` function | Runs (launcher in `ouro/runs_scripts/`) |
|---|---|---|
| Figure 1 (`fig_tps_teaser`, accuracy vs. end-to-end tokens/s) | `time_breakdown`, `thinking_batched` | `time_controls.sh`, `time_sweep.sh`, Thinking batched serving (`paper_figs.fig_tps_teaser`) |
| Figure 2, Section 5.1 (time split, `fig_time`) | `time_breakdown` | `time_controls.sh`, `time_sweep.sh` (passes `clean`, `clean3`) |
| Figure 3 (method diagram `fig0_method`, pseudocode in Figure 12 / Appendix E) | none | `ouro/adapter_model.py`, `ouro/batched.py` |
| Table 1, Section 5.2 (`tab_main`, LoopBridge vs. Original & Distilled baselines) and its seeds | `main_table`, `main_gains` | `launch_aligned.sh` (LoopBridge), `seed_main.sh`, `launch_x26_distill.sh` (`RANK`, `SEED`) |
| Figure 4, Section 5.3 (`fig_taskmap`: all 16 pairs m, n in 1..4; full table: Table 9) | `aligned_cells` | `launch_aligned.sh` grid evaluations |
| Figure 5, Section 5.4 (`fig_depth_main`; full: Figure 10 / Tables 5–8) | `depth_sweep`, `depth_gains`, `same_weights`, `same_weights_seeds`, `plateau`, `plateau_seeds` | `sweep_depth.sh`, `launch_variant.sh wide`, `seed_main.sh ARM=a CELLS="11 12"` |
| Figure 6, Section 5.5 (`fig_budget`; paired bar chart: Figure 9; batch 8/32: Figure 13 / Table 14) | `time_breakdown`, `budget_pairs`, `time_matched`, `batched_serving` | `time_controls.sh` (folded maps `a<m>x<n>pm`), `time_batch.sh` |
| Figure 7, Section 5.5 (`fig_tbatched`, Thinking GSM8K batch 16/32; Table 15) | `thinking_batched` | Thinking continuous-batching runs |
| Table 2, Section 5.6 (a: `tab_build`, b: `tab_tokens`, c: `tab_lr`) | `main_table`, `ablations`, `lr_check`, `lora80` | `launch_variant.sh lora` (`RANK=80`: parameter-matched control), `ablate_tokens.sh`, `LRS=0.3/3 launch_x26_distill.sh`, `LRS=... launch_variant.sh lr` |
| Figure 8, Section 5.7 (`fig_lev`, fixed m = all loops; all pairs: Table 16) | `lev` | Lev scripts (Appendix M) |
| Appendix Tables 11–13 & Figure 11 (Ouro-1.4B & Thinking models) | `main_table_14`, `equal_aligned`, `same_weights` | `launch_aligned_more.sh`, `eval_think_aligned.sh`, `launch_t26df.sh` |
| Appendix Figure 14 (`fig2_frontier`, one-cache 8K decode microbenchmark) | `think_frontier`, `throughput` | Thinking evaluations, `tput_*.sh` microbenchmarks |
| Appendix Table 6 (Trainable prefill, Appendix B) | `prefill_trainable` | `launch_variant.sh prefill / w8` (`SEED=2,3`) |
| Appendix L (Design variants: unrestricted key maps & fixed-depth `m=4` LoRA) | `main_table`, `CMP` (`Table 6`), `time_breakdown`, `batched_serving` | `launch_variant.sh fixed4`, `launch_v.sh` |
| Appendix H (MATH-500 exemplar check & Table 10 reproduction) | `repro`, `CMP` rows "Aligned MATH" | `math0_q.sh ARM=a` |

Every number quoted in the text has a check in `paper_numbers.py`; `compute` prints `!!` next to any that disagrees.

To regenerate the per-item scores themselves you need GPUs: train the LoopBridge adapters, run the evaluations, then
`python ouro/scripts/paper_numbers.py export RUNS_DIR` collects exactly the items the paper uses.

## Environment, models, data

```bash
cd environments/ouro && uv sync && cd ../..          # core runtime: torch, transformers 4.x (Ouro), datasets, numpy, scipy
```

The pinned `environments/ouro/pyproject.toml` / `uv.lock` covers the core PyTorch/Transformers/RULER/LongBench runtime.
Evaluation and paper scripts additionally use `lm-eval`, `evalplus`, `math-verify`, and `matplotlib` (for CPU-only
verification of every paper number and figure, `uv run --with numpy --with math-verify python ouro/scripts/paper_numbers.py compute`
and `uv run --with numpy --with matplotlib python ouro/scripts/paper_figs.py` require no GPU or `uv sync`).

- **Released LoopBridge Checkpoints (ModelScope, public, SHA256-verified):**
  [`https://modelscope.cn/models/torushy/LoopBridge`](https://modelscope.cn/models/torushy/LoopBridge)
  (8 Ouro adapters: `ouro-2.6b/loopbridge`, `loopbridge-wide`, `loopbridge-onecache`; `ouro-2.6b-thinking/loopbridge`, `loopbridge-onecache`; `ouro-1.4b/loopbridge`, `loopbridge-onecache`; `ouro-1.4b-thinking/loopbridge-onecache`; plus 6 26.4B sparse-MoE adapters: `tml-d24-varloop/loopbridge-3x2-v2-48m`, `loopbridge-3x1-v2-48m`, `loopbridge-3x1-24m`; `inkling-2-mini/loopbridge-3x2-v2-48m`, `loopbridge-3x2-24m`, `loopbridge-3x1-24m`).
- **Prepared Distillation & Long-Context Eval Data (ModelScope, public, GPQA-free):**
  [`https://modelscope.cn/datasets/torushy/LoopBridge-Data`](https://modelscope.cn/datasets/torushy/LoopBridge-Data)
  (`mix-data-48m.tar.gz`, `bench.tar.gz`, `bench-mini.tar.gz`).
- **Base Models (Hugging Face / ModelScope):**
  `ByteDance/Ouro-2.6B` ([ModelScope](https://modelscope.cn/models/bytedance-community/Ouro-2.6B)),
  `ByteDance/Ouro-1.4B` ([ModelScope](https://modelscope.cn/models/bytedance-community/Ouro-1.4B)),
  `ByteDance/Ouro-2.6B-Thinking` ([ModelScope](https://modelscope.cn/models/bytedance-community/Ouro-2.6B-Thinking)),
  `ByteDance/Ouro-1.4B-Thinking` ([ModelScope](https://modelscope.cn/models/bytedance-community/Ouro-1.4B-Thinking)),
  `thinkingmachines/TML-d24-varloop-research` ([Hugging Face](https://huggingface.co/thinkingmachines/TML-d24-varloop-research)),
  `thinkingmachines/Inkling-2-Mini` ([Hugging Face](https://huggingface.co/thinkingmachines/Inkling-2-Mini)).
- Training mix (48M KL tokens): `ouro/prepare_mix.py` (OpenThoughts, Tulu SFT, LongAlpaca, FineWeb-Edu; base models),
  `ouro/scripts/prepare_think_mix.py` (same samples in the thinking chat template).
- Benchmarks: `ouro/prepare_full_qa.py` (RULER 4K/8K/16K, 13 tasks x 500 rows), `ouro/prepare_helmet.py` (HELMET),
  `ouro/runs_scripts/mk_bench_mini.py` (the 1,108-row HELMET and 260-row RULER subsets). GSM8K, MMLU, MMLU-Pro,
  ARC-Challenge, BBH, HellaSwag, Winogrande, HumanEval+, MBPP+ and MATH-500 come from lm-evaluation-harness
  (`ouro/lmeval_tasks/` adds the zero-shot thinking tasks, evalplus local wrappers and MATH-500). MATH-500 is scored on
  the final answer (last `\boxed{}`, else the Minerva closing sentence) with `math-verify` (`paper_numbers.py boxed`),
  not with lm-eval's strict Minerva sentence match.

The launchers assume the layout `M=/dev/shm/big-small-ouro` (models in `$M/model-2.6b`, …, data in `$M/mix-data-48m`,
benchmarks in `$M/bench`, outputs in `$M/runs`); change `M` at the top of a launcher to move it. GPQA-Diamond is read
from a local CSV (`$M/gpqa-data/`, not redistributed) and IFEval from `$M/ifeval-data/` (task definitions in
`ouro/lmeval_tasks/gpqa_n_shot_local`, `ifeval_local`).

## Provenance

`ouro/results/provenance/` traces paper results to their runs: run manifests, evaluation configurations and software
versions, times, cleaned evaluation logs, and model outputs, together with training configurations, logs, and
checkpoint hashes for the main and ablation models (see [`ouro/results/provenance/README.md`](ouro/results/provenance/README.md)). Regenerate with
`python ouro/scripts/provenance.py RUNS_DIR OUT_DIR` on each evaluation machine.

## Training (one B300 node, 1–8 GPUs per run)

| checkpoint | used for | launcher |
|---|---|---|
| Ouro-2.6B LoopBridge `X26-a-universal` (`arm a`, main method) | Table 1, Figures 1–6, Table 9, Appendix A/B/D/K | `launch_aligned.sh` (seeds 2/3: `ARM=a seed_main.sh`) |
| Ouro-2.6B wide-depth & trainable-prefill adapters `X26-a-wide`, `X26-a-w8`, `X26-a-pf8` | Figure 5, Figure 10, Tables 5–7 (Appendix B) | `launch_variant.sh wide / w8 / prefill` (`SEED=2,3`) |
| Ouro-2.6B LoopBridge ablations `X26-a-lora`, `X26-a-lora-r80`, `X26-a-fixed4` | Table 2a (rank 16 / rank 80 maps-frozen), Appendix L (fixed `m=4` LoRA) | `launch_variant.sh lora` (`RANK=80`), `launch_variant.sh fixed4` |
| Ouro-1.4B and Thinking LoopBridge `O14-a-`, `T26-a-universal` | Table 8, Table 11, Figure 11, Table 13, Figure 7, Table 15 | `launch_aligned_more.sh 14 / think`, `eval_think_aligned.sh`, `seed14.sh`, `think_seed.sh` |
| Ouro-2.6B / 1.4B / Thinking one-cache LoopBridge `X26-S-universal`, `T26-S-universal`, `O14-S-universal` (`arm S`) | Table 2a/b (`tab_build`, `tab_tokens`), Table 12 (`tab:models`), Figure 14 (`fig2_frontier`), Appendix K/L | `launch_s.sh`, `launch_think.sh`, `ablate_tokens.sh` |
| LoRA distillation baseline `T26-Df-native22` (`arm D`, Thinking 2/2) | Table 13, Figures 11 & 14, Table 15, timing | `launch_t26df.sh` |
| LoRA distillation baselines `X26-{E,D,H}48f-*`, `O14-{E,D}48f-*` (base 1/1, 2/2, 3/3) | Table 1, Table 2c (`tab_lr`), Table 11, Figures 9 & 13, Table 14 | `launch_x26_distill.sh ARM GPUS PORT first` (`SIZE=14`, `RANK=80`, `LRS=0.3/3`), `after_x26_distill.sh` |
| Unrestricted key-map variant `X26-V` | Appendix L (unrestricted vs. RoPE-commuting key maps against one-cache LoopBridge) | `launch_v.sh` |
| Lev continuations (untied arm and control, four seeds, 2× control) | Figure 8, Table 16 (Section 5.7 & Appendix M) | `ouro/lev/train_rd.sh`, `ouro/lev/after_train.sh`, `lev_mn.sh`, `lev_mn_seeds.sh` |

## Evaluation

A configuration is a *condition* string: `native<n>` (Original `n/n`: unmodified model, `n` loops for prefill and decode), `a<m><n>`
(or `a<m>x<n>pm` with folded maps for timing: LoopBridge, prefill `m`, decode `n`), or `S<m><n>` (one-cache LoopBridge);
`E`/`D`/`H` are the tied-loop distillation baselines (`1/1`, `2/2`, `3/3`) and `V` is the Appendix L unrestricted key-map variant.

```bash
# lm-evaluation-harness through our runner
./ouro/runs_scripts/lm1.sh GPU X26-a41-math0shot - "path=$M/model-2.6b,condition=a41,checkpoint=$CK,max_length=16384" \
    "--tasks minerva_math500 --num_fewshot 0"
# RULER / HELMET with our scripts (evaluate_bench.py)
MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' ./ouro/runs_scripts/bench3.sh X26-a41f a41 $CK GPU 1
```

## Appendix elements → source

The main text (Figures 1–8, Tables 1–2) is mapped in the table at the top of this file; this table maps every appendix
figure and table to its runs and generators.

| Appendix item | runs (launcher) | computed by |
|---|---|---|
| Table 3 (implementation behind each result) | — | prose table in `paper/latex/main.tex` |
| Appendix A: Table 4 (`tab_time`, per-request time distributions), Figure 9 (`fig_sametime`, paired 1-loop bar chart) | `time_controls.sh`, `time_sweep.sh`, `time_set2.sh`, `time_math0.sh`, `time_helmet.sh` | `paper_numbers.py` `time_breakdown` → `paper_figs.py tab_time`, `fig_sametime` |
| Appendix B: Figure 10 (`fig_depth`), Table 5 (`tab_plateau`), Table 6 (`tab_prefilltrn`), Table 7 (`tab_depthwide`) | `launch_variant.sh prefill / w8 / wide`, `sweep_depth.sh` | `prefill_trainable`, `plateau`, `depth_sweep` → `fig_depth`, `tab_plateau`, `tab_prefilltrn`, `tab_depthwide` |
| Appendix C: Table 8 (`tab_depth14`, Ouro-1.4B prefill depth) | `launch_aligned_more.sh 14` | `depth_sweep_14` → `tab_depth14` |
| Appendix D: Figure 11 (`fig_equal`, Ouro-2.6B-Thinking 4/2), Table 9 (`tab_aligned`, full 4×4 LoopBridge grid) | `eval_think_aligned.sh`, `launch_t26df.sh`, `launch_aligned.sh` | `equal("a")`, `aligned_cells` → `fig_equal`, `tab_aligned` |
| Appendix E: Figure 12 (LoopBridge PyTorch pseudocode) | — | `ouro/adapter_model.py`, `ouro/single_bank.py` |
| Appendix G: training details & 13-gram overlap audit | `ouro/scripts/overlap_audit.py` | `overlap_check` (`ouro/results/overlap/mix-data-48m.json`) |
| Appendix H: Table 10 (`tab_repro`, reproducing Ouro-2.6B 4/4) & MATH-500 exemplar check | `raw_q.sh`, `math0_q.sh ARM=a`, `evalplus_q.sh` + `score_evalplus.py` | `repro`, `CMP` (`Aligned MATH`) → `tab_repro` |
| Appendix I: Table 11 (`tab_main14`, Ouro-1.4B), Table 12 (`tab:models`, fixed-decoder comparison) | `launch_aligned_more.sh`, `launch_s.sh`, `launch_think.sh`, `seed14.sh` | `main_table_14`, `same_weights` → `tab_main14` |
| Appendix J: Table 13 (`tab_equal_26B-T`, Figure 11 in numbers) | `eval_think_aligned.sh`, `launch_t26df.sh` | `equal("a")` → `tab_equal("2.6B-T")` |
| Appendix K: Figure 13 (`fig_batched`), Table 14 (`tab_batched`), Table 15 (`tab_tbatched`), Figure 14 (`fig2_frontier`) | `time_batch.sh`, `batch_set2.sh`, `time_batch_think.sh`, `tput_sb.sh`, `tput_aligned.sh`, `tput_ctx.sh`, `tput_batch.sh`, `e2e_latency.py` | `batched_serving`, `thinking_batched`, `think_frontier`, `throughput`, `appendix_checks` → `fig_batched`, `tab_batched`, `tab_tbatched`, `fig2_frontier` |
| Appendix L: design variants (unrestricted key maps vs. one-cache LoopBridge; fixed-depth `m=4` LoRA) | `launch_v.sh`, `launch_s.sh`, `launch_variant.sh fixed4`, `fixed14.sh` | `main_table`, `CMP` (`Table 6`), `time_breakdown`, `batched_serving` |
| Appendix M: Table 16 (`tab_lev`, Lev decision models, both directions) | `ouro/lev/train_rd.sh`, `ouro/lev/after_train.sh`, `lev_mn.sh`, `lev_mn_seeds.sh` | `lev` → `tab_lev` |
| Appendix N: Figures 15–16 (`fig_inkling_pareto`, `fig_inkling_ablate`), Tables 17–19 (26.4B sparse-MoE `TML-d24-varloop` & `Inkling-2-Mini` in Day-0 vLLM) | `inkling/prepare_data_v3.py`, `prepare_data_v4_gkd.py`, `train_inkling.py`, `prepare_official_exact.py`, `eval_vllm.py`, `bench_vllm_timing.py` | `inkling/make_figs.py` (reads `inkling/results/eval_official_exact.json`, `eval_vllm_*.json`, `vllm_sse_*.json`) |
| Appendix O: from-scratch re-runs & single-bank exactness checks | `verify_q.sh`, `ouro/scripts/verify_compare.py`, `check_aligned_runner.py`, `single_bank_check.py`, `check_replay_scorer.py` | `ouro/results/verify_rerun.txt` |
| Paper PDF and arXiv package | — | in order: `paper_numbers.py compute`, `paper_figs.py`, `inkling/make_figs.py`, `tectonic -X compile main.tex`, `paper/build_arxiv.py` |

### 26.4B Sparse-MoE Day-0 vLLM Commands (`inkling/`)

Requires Thinking Machines' Day-0 vLLM Blackwell environment (`vllm`, `PyTorch 2.14+cu130`, NVIDIA B300):

```bash
# 1. Unit-test official scorers (CPU, no GPU required)
PYTHONPATH=. python inkling/test_scorers.py

# 2. Verify every 26.4B paired bootstrap 95% CI & redraw Figures 15-16 (CPU, no GPU required)
uv run --with numpy --with matplotlib python inkling/make_figs.py

# 3. Prepare exact official 256K-1M evaluation splits (GPQA, LCB v6, MRCR, GraphWalks, LongBench v2)
python inkling/prepare_official_exact.py --model-dir /path/to/TML-d24-varloop-research --out-dir ./eval_official_exact

# 4. Train LoopBridge (e.g., controlled 24M 3/1 or v2 48M 3/2 with --freeze-sink --jsd-beta 0.5 --length-bucket)
torchrun --nproc_per_node=8 -m inkling.train_inkling \
    --model-dir /path/to/TML-d24-varloop-research --data /path/to/train_v4_gkd/train.jsonl \
    --out-dir ./runs/vl_lb_3_2_v2_48m --mode loopbridge --cells 3x2 --lora-rank 128 \
    --kv-map-layers global --kv-map-cross-pass --freeze-sink --jsd-beta 0.5 --length-bucket --target-tokens 48000000

# 5. Serve zero-overhead folded LoopBridge in Day-0 vLLM & evaluate / benchmark throughput (512..256K)
LOOPBRIDGE_CKPT=./runs/vl_lb_3_2_v2_48m/checkpoint-48m.pt LOOPBRIDGE_CELL=3x2 \
    python -m inkling.vllm_loopbridge --model /path/to/TML-d24-varloop-research --port 8000
python inkling/eval_vllm.py --urls http://127.0.0.1:8000 --data-dir ./eval_official_exact --out-dir ./eval_out
python inkling/bench_vllm_timing.py --url http://127.0.0.1:8000 --label vl_lb_3_2_v2_48m --out ./timing.json
```

`paper_numbers.py compute` checks every number quoted in the text against hard-coded expectations and reports any
mismatch with `!!` (must end with `0 printed values differ`).

## Helpers

`grid_worker.sh` (queue runner), `wait_gpus.sh` (start a job when `K` GPUs are free), `lm1.sh` (one
lm-evaluation-harness job), `bench3.sh` (RULER / HELMET shards), `run_mix2.sh` (training), `mk_code_local.py` and
`mk_gpqa_local.py` (local task files), `ouro/lev/levbench_compare.py` (Lev comparisons), `launch_think_o.sh` (shared
answer KV, Thinking). Queue files recording the grid runs are in `ouro/runs_scripts/queues/`, and remaining legacy
training-arm branches are documented in [`docs/DEPRECATED.md`](docs/DEPRECATED.md).

## What is not here

Raw lm-evaluation-harness sample files (16 GB) and checkpoints are not in the repository; the per-item scores are
sufficient to verify every statistic, and the commands above regenerate the rest. Timing depends on the GPU (one NVIDIA
B300 for every timing in the paper).
