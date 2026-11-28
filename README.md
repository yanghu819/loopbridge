<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/hero-light.svg">
  <img alt="LoopBridge: Should Prefill and Decode Loop Equally?" src="docs/assets/hero-light.svg" width="100%">
</picture>

<br/>

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-EB6834?style=flat&logo=adobeacrobatreader&logoColor=white)](paper/loopbridge.pdf)
[![ModelScope Checkpoints](https://img.shields.io/badge/ModelScope-Checkpoints-624AFF?style=flat)](https://modelscope.cn/models/torushy/LoopBridge)
[![ModelScope Dataset](https://img.shields.io/badge/ModelScope-Dataset-624AFF?style=flat)](https://modelscope.cn/datasets/torushy/LoopBridge-Data)
[![Reproducibility](https://img.shields.io/badge/Reproducible-0--diff%20verified-0F766E?style=flat&logo=checkmarx&logoColor=white)](REPRODUCE.md)
[![Python 3.12](https://img.shields.io/badge/Python-3.12%20%C2%B7%20PyTorch-334155?style=flat&logo=pytorch&logoColor=white)](environments/ouro/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-334155?style=flat)](LICENSE)

[**Paper PDF**](paper/loopbridge.pdf) · [**Checkpoints (ModelScope)**](https://modelscope.cn/models/torushy/LoopBridge) · [**Data (ModelScope)**](https://modelscope.cn/datasets/torushy/LoopBridge-Data) · [**Results**](#results) · [**How It Works**](#how-loopbridge-works) · [**Quick Start**](#quick-start) · [**Reproduce**](REPRODUCE.md) · [**Citation**](#citation)

</div>

<br/>

Looped (recurrent-depth) language models reuse one block of layers for the same number of loops during prefill and decode. Yet generation is heavily asymmetric: at batch 1, prefill reads the prompt once in parallel (**<3%** of wall time), while decoding steps token by token (**>97%**).

**LoopBridge** unties prefill depth (*m*) from decode depth (*n* < *m*) with a lightweight adapter (**2.4%** on Ouro-2.6B; **0.017–0.25%** on 26.4B sparse-MoE looped models) that combines cache-foldable prompt KV maps (and optional decoder LoRA). It serves every trained (*m*, *n*) pair, preserves long-context retrieval up to 256K–1M tokens, and **folds once into the prompt KV cache at prefill**—keeping Original *n*/*n*'s per-token decode speed intact in both custom PyTorch runners and Thinking Machines' Day-0 vLLM Blackwell engine.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/teaser_horizontal_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/teaser_horizontal.png">
    <img src="docs/assets/teaser_horizontal.png" alt="LoopBridge: Ouro-2.6B 11-Benchmark Average with Overall TPS and 26.4B Inkling-2 Sparse-MoE Pareto frontier" width="98%">
  </picture>
  <br/>
  <sub><b>More prefill loops (<i>m</i>) and fewer decode loops (<i>n</i>) lift accuracy at high throughput across 2.6B dense and 26.4B sparse-MoE models (one B300).</b> (a) <b>Ouro-2.6B 11-Benchmark Average</b> across all 16 (<i>m</i>, <i>n</i>) pairs with measured overall end-to-end throughput (tok/s, batch 1, 4 CoT tasks; <code>n=3</code>: 400 reqs, others: 800; LoopBridge 4/1 = 27.7, 4/2 = 14.3): raising <i>m</i> to 4 adds <b>+14.0</b> at <code>n=1</code> (<b>4.2×</b> speed) and <b>+5.4</b> at <code>n=2</code> (<b>2.0×</b>), while <i>m</i> &lt; <i>n</i> hurts. (b) <b>10× scale-up to 26.4B Inkling-2</b> (<code>TML-d24-varloop</code> in Day-0 vLLM, 5 reasoning &amp; 256K–1M benchmarks; all trained and <code>n≥2</code> points at 5×avg@4, Orig 1/1 at @1): 4/3 recovers <b>99.0% of 4/4</b>, 4/2 beats 2/2 by <b>+2.8*</b> at <b>1.7×</b> E2E speed (no-train TF: 74.0/68.9), and 3/1 beats Distilled 1/1 by <b>+5.3*</b> (<code>p&lt;1e-4</code>).</sub>
</p>

---

## Results

Every number below is recomputed directly from released per-item scores and timing logs via [`inkling/make_figs.py`](inkling/make_figs.py) (26.4B sparse-MoE) and [`ouro/scripts/paper_numbers.py`](ouro/scripts/paper_numbers.py) (Ouro; `0 printed values differ`).

### 26.4B Sparse-MoE Looped Models in Day-0 vLLM (`TML-d24-varloop` & `Inkling-2-Mini`)

Evaluated in Thinking Machines' Day-0 vLLM Blackwell engine on five exact official splits (`GPQA Diamond` 198, `LiveCodeBench v6` 175, `MRCR v2 8n 256K–512K` 100, `GraphWalks BFS 256K` 100, `LongBench v2 1M` 503; **124,816 four-sample rollouts across 29 archived `5×avg@4` arms**), sharply separating our two zero-decode-overhead modules across **22 variable-depth (`n≥2`, `94,688` rollouts) arms** (complete 4-cell `2×2` causal factorial, matched `r=64` / `r=128` full-LoRA twins, one `0.042%` `0%`-synthetic universal multi-depth adapter, and 3-way Plato's Key ablation), **full `5×avg@4` fixed-depth & `n=1` sweeps** (`Inkling-2-Mini 3/2`: `+5.70 [+4.10, +7.34]` 5-Bench, `+9.36 [+6.95, +11.84]` 3-Long; `TML-varloop 3/1`: `+5.31 [+3.53, +7.14]` 5-Bench, `+8.50 [+5.83, +11.23]` 3-Long, both `p<1e-4`), and Day-0 vLLM systems verification (`9/9` step-0 / `96.8–97.9%` 31-step PyTorch–vLLM parity, `1.25–3.58×` moderate & saturated async Poisson ITL speedup, `0` preemptions under tested load matching equal-`m` KV occupancy):

<p align="center">
  <img src="paper/figs/fig_inkling_main.png" alt="Contextual shift and causal controls on Thinking Machines' two 26.4B sparse-MoE looped models in Day-0 vLLM" width="98%">
  <br/>
  <sub><b>Contextual shift and causal controls on Thinking Machines' two 26.4B sparse-MoE looped models (vLLM, 5 benchmarks to 1M).</b> (a) <b>Variable-depth <code>TML-d24-varloop</code> (<code>n ≥ 2</code>, 94,688 rollouts across 22 full 5×avg@4 arms):</b> matched-data <code>Tied KV 2/2</code> stays flat (<b>67.0%</b>) and 10.5×-larger <code>Distilled 2/2 (r=64, 46.9M, 5.8h)</code> (<b>68.9%*</b>) merely ties zero-shot <code>No-train 4/2</code> (<b>68.9%*</b>, <code>p=0.98</code>), whereas <code>LB 4/2 (r=0, 4.5M)</code> hits <b>69.6%*</b> (<code>+2.7* [p&lt;1e-3]</code> over Tied KV 2/2, tying 9.1×-larger <code>r=64</code> and 15×-larger <code>r=128</code> twins) and one single-weight 0%-synthetic <code>Universal 3/2</code> adapter (<code>11.0M = 0.042%</code>) leads all <code>r=0</code> <code>n=2</code> arms at <b>69.8%*</b> (<code>+2.9* [p=0.0004]</code> / <code>+4.6*</code> long over 2/2); <code>LB 4/3 (r=0)</code> recovers <b>99.0%</b> of Orig 4/4. (b) <b>Fixed-depth <code>Inkling-2-Mini 3/2</code> &amp; un-gated <code>VarLoop 3/1</code> (locked 24M, full 5×avg@4; unadapted baselines @1):</b> no-training 3/2 collapses (2.2%, 93% trunc.), while KV bridge + merged decoder LoRA beats tied distillation by <b>+5.7* (+9.4* long)</b> on <code>Mini 3/2</code> and <b>+5.3* (+8.5* long)</b> on <code>VarLoop 3/1</code> (both <code>p&lt;1e-4</code>).</sub>
</p>

- **Variable-depth backbones (`TML-d24-varloop`, `n ≥ 2`) — KV Bridge alone (`r=0`) suffices:** Training-free untying works natively, and ultra-compact **KV-Only maps** (`r=0`, `4.46M–6.55M` params = `17.0–25.0 MiB` = **`0.017–0.025%`** of 26.4B, frozen decoder) lead `4/3` (`74.1%`, `99.0%` of `4/4` at `1.27×`) and `4/2` (`69.6%`, `1.75×`). Across all four cells of our `2×2` causal factorial (`m ∈ {2,4} × {full, 0% nosynth}`, 12M tokens) and matched LoRA twins: (i) untying `m=2 → 4` at `r=0` adds **`+2.65 [+1.14, +4.16]` 5-Bench (`p < 1e-3`; `+5.49` MRCR, `+3.43` 3-Long)** on the full corpus (`KV-Only 2/2` flat at `67.0%`) and directional `+0.70 [-0.93, +2.27]` (`p=0.40`; `+1.90` 3-Long) on 12M `0% nosynth` (`KV-Only 2/2 0%S = 68.0%`, `p=0.13` vs `2/2`; `LB 4/2 0%S = 68.7%`, revealing an `m × synth` interaction of `+1.96 pp`), while **one `0.042%` `0%`-synthetic 18M `Universal` adapter** (`11.01M`, `6M/cell`) significantly beats `0%`-nosynth `KV-Only 2/2` by **`+1.79 [+0.42, +3.17]` 5-Bench (`p=0.013`) / `+3.08*` 3-Long (`p=0.0016`)** and serves `4/3` (`73.8%`), **`3/2` (`69.8%`, `+2.93*` `p=0.0004` vs `2/2`, `+2.34*` LBv2 `p=0.0072` vs `Distilled 2/2`, 25% cheaper prefill)**, and `4/2` (`68.4%`) without reloading; (ii) scaling tied `2/2` to full `r=64` LoRA (`Distilled 2/2`, `46.9M`, `5.8h`) hits `68.9%` (tying zero-shot `TF 4/2 Tail` `68.9%`, `p=0.98`), whereas `LB 4/2` twins at 12M `r=64` (`40.6M`, `2.13h`, `69.6%`) and 24M `r=128` (`66.8M`, `69.8%`) tie `9.1–15.0×`-smaller `r=0` `LB 4/2 KV-Only` (`LoRA - KV`: `-0.03, p=0.98` and `+0.17, p=0.81`), confirming decoder LoRA is unnecessary at `n ≥ 2`; (iii) in a 0-gradient closed-form **Plato's Key** test (192 unlabeled prompts), orthogonal `Procrustes 4/2` (`68.4%`) beats `Cross-Ridge 4/2` (`60.7%`, `-7.63*`), yet behavioral KL `LB 4/2 KV-Only` significantly beats `Procrustes 4/2` on 3-Long (`+2.45*`) and MRCR 512K (`+6.06*`, `59.2%` vs `41.9%` dark-memory CV).
- **Fixed-depth `Inkling-2-Mini` (`3/2`) & un-gated `n=1` (`3/1`) — Merged Decoder LoRA rescues collapse:** All unadapted `n=2` variants collapse (`2.2%`, `93.2%` trunc. for Anchored TF 3/2; `5.6%` GPQA for Tail TF 3/2 & Orig 2/2; `11.1%` GPQA / `81.8%` trunc. for closed-form `Procrustes 3/2`; `40.8%` on `VarLoop 3/1`), while KV bridge + merged decoder LoRA beats parameter-advantaged tied distillation under locked 24M controls on BOTH backbones at full `5×avg@4` (`+5.70 [+4.10, +7.34]` 5-Bench, `+9.36 [+6.95, +11.84]` 3-Long, `p<1e-4` on `Inkling-2-Mini 3/2`; `+5.31 [+3.53, +7.14]` 5-Bench, `+8.50 [+5.83, +11.23]` 3-Long, `p<1e-4`, 5/5 split sweep & lower micro truncation `12.9%` vs `14.5%` on `VarLoop 3/1`, where the 3-step `5×avg@4` KV-map ladder climbs `48.9 → 49.7 → 50.0` 5-Bench and `26.7 → 29.6 → 31.4` 512K MRCR [`+4.70 [+0.84, +8.53]`, `p=0.017`]; `3.55–4.5×` faster training).
- **All verified 26.4B & Ouro checkpoints** are released on [ModelScope](https://modelscope.cn/models/torushy/LoopBridge).

<details>
<summary><b>Full per-benchmark 26.4B numbers (29 full 5×avg@4 arms = 124,816 rollouts + @1 diagnostics) & isolated 512–256K decode speedup figure</b></summary>
<br/>

| Backbone (26.4B MoE) | Method | Params | GPQA | LCB v6 | MRCR 512K | GW 256K | LBv2 1M | 3-Long | 5-Bench | vLLM Decode (`256K B=2`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TML-d24-varloop` (`@4`) | Original 4/4 (ceiling) | 0 | 82.6 | 76.7 | 94.0 | 56.3 | 64.9 | 71.7 | 74.9 | 1.00× (232 tok/s) |
| `TML-d24-varloop` (`@4`) | Original 3/3 | 0 | 82.3 | 76.0 | 90.4 | 52.2 | 64.7 | 69.1 | 73.1 | 1.27× (294 tok/s) |
| `TML-d24-varloop` (`@4`) | TF 4/3 (Tail / Anchored) | 0 | 83.0 / 83.1 | 75.7 / 74.9 | 91.0 / **93.4** | 50.9 / 52.0 | 65.9 / 66.5 | 69.2 / **70.6** | 73.3 / 74.0 | 1.27× (295 tok/s) |
| `TML-d24-varloop` (`@4`) | **LoopBridge 4/3 KV-Only (`0% nosynth`, `r=0`)** | **6.55M** | 82.2 | 75.7 | 88.2 | 54.5 | **66.9** | 69.8 | 73.5 | **1.27×** |
| `TML-d24-varloop` (`@4`) | **Universal 4/3 KV-Only (`0% nosynth`, 1 wt)** | **11.0M§** | **83.2** | 74.9 | 90.8 | **54.6** | 65.8 | 70.4 | 73.8 | **1.27×** |
| `TML-d24-varloop` (`@4`) | **LoopBridge 4/3 KV-Only (`r=0`, 25 MiB)** | **6.55M** | 82.5 | **76.4** | 91.7 | 54.5 | 65.6 | 70.6 | **74.1** | **1.27× (296 tok/s)** |
| `TML-d24-varloop` (`@4`) | Original 2/2 | 0 | 81.1 | 68.3 | 72.6 | 50.6 | 61.7 | 61.6 | 66.9 | 1.74× (404 tok/s) |
| `TML-d24-varloop` (`@4`) | KV-Only 2/2 (`m=2`, matched `v5_t4`, 100% synth) | 2.36M‡ | 80.7 | 66.7 | 76.7 | 48.6 | 62.2 | 62.5 | 67.0 | 1.74× |
| `TML-d24-varloop` (`@4`) | KV-Only 2/2 (`m=2`, `0% nosynth`, `r=0`) | 2.36M‡ | 79.8 | 70.6 | 76.3 | 49.6 | 63.7 | 63.2 | 68.0 | 1.74× |
| `TML-d24-varloop` (`@4`) | Distilled 2/2 (`m=2`, matched `v5_t4`, `r=64`, 5.8h) | 46.9M | 79.8 | **71.4** | 77.4 | **53.7** | 62.0 | 64.4 | 68.9 | 1.74× |
| `TML-d24-varloop` (`@4`) | Cross-Ridge 4/2 (124s, 0-grad MSE, `r=0`) | 4.46M | 60.6 | 53.9 | 79.0 | 48.1 | 62.2 | 63.1 | 60.7 | 1.75× |
| `TML-d24-varloop` (`@4`) | Procrustes 4/2 (121s, 0-grad 1→1 SVD, `r=0`) | 0.79M‡ | 80.9 | 70.4 | 76.1 | 51.2 | 63.1 | 63.5 | 68.4 | 1.75× |
| `TML-d24-varloop` (`@4`) | TF 4/2 (Anchored / Tail) | 0 | 81.2 / 80.6 | 69.1 / 67.9 | 77.7 / 78.9 | 51.8 / 52.1 | 62.4 / **64.9** | 64.0 / 65.3 | 68.4 / 68.9 | 1.74× (403 tok/s) |
| `TML-d24-varloop` (`@4`) | **Universal 4/2 KV-Only (`0% nosynth`, 1 wt)** | **11.0M§** | **81.7** | 68.9 | 77.3 | 51.4 | 62.6 | 63.8 | 68.4 | **1.75×** |
| `TML-d24-varloop` (`@4`) | **LoopBridge 4/2 KV-Only (`0% nosynth`, `r=0`)** | **4.46M** | 79.4 | 68.7 | 80.0 | 50.7 | 64.6 | 65.1 | 68.7 | **1.75×** |
| `TML-d24-varloop` (`@4`) | **LoopBridge 4/2 Full-LoRA (`r=64`, 12M twin)** | **40.6M** | 80.2 | 69.1 | 81.8 | 52.2 | 64.7 | 66.2 | 69.6 | **1.75×** |
| `TML-d24-varloop` (`@4`) | **LoopBridge 4/2 Full-LoRA (`r=128`, 24M twin)** | **66.8M** | **81.7** | 68.7 | **82.2** | 52.7 | 63.8 | 66.2 | **69.8** | **1.75×** |
| `TML-d24-varloop` (`@4`) | **LoopBridge 4/2 KV-Only (`r=0`, 17 MiB)** | **4.46M** | 80.9 | 69.4 | **82.2†** | 51.9 | 63.7 | 65.9† | 69.6† | **1.75× (406 tok/s)** |
| `TML-d24-varloop` (`@4`) | **Universal 3/2 KV-Only (`0% nosynth`, 1 wt)** | **11.0M§** | 81.1 | 69.0 | 81.8 | 52.7 | 64.4 | **66.3\*** | **69.8\*** | **1.75× (25% less prefill)** |
| `TML-d24-varloop` (`@1`) | Orig 1/1 / TF 3/1 | 0 | 74.7 / 72.7 | 39.4 / 40.6 | 20.7 / 19.5 | 20.0 / 14.4 | 48.9 / 56.7 | 29.9 / 30.2 | 40.8 / 40.8 | 2.76× (642 tok/s) |
| `TML-d24-varloop` (`@4`) | Distilled 1/1 (24M ctrl) | 43.7M | 73.1 | 45.0 | 24.3 | 28.0 | 53.2 | 35.2 | 44.7 | 2.76× |
| `TML-d24-varloop` (`@4`) | LB 3/1, no KV / 1-pass KV (24M ablations) | 32.8 / 33.6M | **74.1** / 71.7 | 44.4 / 43.7 | 26.7 / 29.6 | 39.1 / **42.9** | 60.2 / **60.6** | 42.0 / **44.4** | 48.9 / 49.7 | 2.76× |
| `TML-d24-varloop` (`@4`) | **LoopBridge 3/1 (24M ctrl, 3.55× faster train)** | **34.6M** | 73.7 | **45.4** | **31.4\*** | 39.0\* | **60.6\*** | 43.7\* | **50.0\*** | **2.76× (641 tok/s)** |
| `Inkling-2-Mini` (`@4`) | Original 3/3 (teacher, symmetric 57K cap) | 0 | 84.3 | 78.6 | 50.3 | 58.5 | 65.7 | 58.2 | 67.5 | 1.00× (295 tok/s) |
| `Inkling-2-Mini` (`@4`) | Distilled 2/2 (24M ctrl) | 47.4M | 81.1 | 65.0 | 17.1 | 39.6 | 57.4 | 38.0 | 52.0 | 1.36× (401 tok/s) |
| `Inkling-2-Mini` (`@4`) | **LoopBridge 3/2 (24M ctrl, 4.5× faster train)** | **39.9M** | 81.1 | **65.4** | **28.8\*** | **50.7\*** | **62.7\*** | **47.4\*** | **57.7\*** | **1.37× (403 tok/s)** |
| `Inkling-2-Mini` (`@1`) | TF 3/2 Anchored (93.2% trunc collapse) | 0 | 3.0 | 2.9 | 0.2 | 0.0 | 4.8 | 1.7 | 2.2 | 1.37× |
| `Inkling-2-Mini` (`@1`) | Orig 2/2 / TF 3/2 Tail / Procrustes 3/2 (GPQA) | 0 / 0.79M‡ | 5.6 / 5.6 / 11.1 | – | – | – | – | – | – | 1.37× |
| `Inkling-2-Mini` (`@1`) | **LoopBridge 3/2 v2 (48M hybrid, `r=128`)** | **65.8M** | **82.8** | **66.9** | **29.9** | 49.0 | **63.0** | **47.3** | **58.3** | **1.37× (403 tok/s)** |

<sub><i>(`@4`): 29 archived full 5×avg@4 arms = 124,816 rollouts (22 variable-depth n≥2 arms = 94,688; 3 Inkling-2-Mini arms = 12,912; 4 TML-varloop n=1 control/ablation arms = 17,216); TF 5/2 Anchored = 67.8%. §One single 11.01M (0.042%, 42.0 MiB fp32) 0%-synthetic multi-depth adapter trained 18M tokens (6M/cell) serves 4/3 (73.8%), 4/2 (68.4%), and 3/2 (69.8%, <code>+2.93 [+1.39, +4.46]</code> p=0.0004 and <code>+4.64 [+2.43, +6.91]</code> 3-Long p&lt;1e-4 vs Orig 2/2; <code>+1.79 [+0.42, +3.17]</code> p=0.013 vs 0%-nosynth KV-Only 2/2; <code>+2.34 [+0.60, +4.13]</code> LBv2 p=0.0072 vs Distilled 2/2) without reloading. ‡Non-identity active params out of 4.46M allocated in the 4-slot <code>ParameterDict</code> (2.36M for <code>KV-Only 2/2</code>; 0.79M non-identity [1.31M 1→1 slots] for anchored <code>Procrustes</code>). Matched LoRA twins: 12M <code>r=64 LB 4/2</code> (40.6M, 2.13h) and 24M <code>r=128 LB 4/2</code> (66.8M) tie <code>r=0 LB 4/2 KV</code> (<code>LoRA - KV</code>: <code>-0.03</code> p=0.98 and <code>+0.17</code> p=0.81). 2×2 causal factorial: <code>LB 4/2 KV</code> vs. matched <code>KV-Only 2/2</code> = <code>+2.65 [+1.14, +4.16]</code> 5-Bench (<code>p&lt;1e-3</code>); matched 12M <code>0% nosynth LB 4/2</code> vs. <code>KV-Only 2/2 0% nosynth</code> = directional <code>+0.70 [-0.93, +2.27]</code> (p=0.40; 3-Long <code>+1.90</code> p=0.12), exposing an <code>m × synth</code> interaction (<code>+1.96 pp</code>). †LoopBridge 4/2 KV-Only clears zero vs. Orig 2/2 under both 95% and strict k=3 Holm-Bonferroni 98.33% clustered CIs (<code>+2.78 [+0.82, +4.67]</code> 5-Bench, <code>+4.29 [+1.37, +7.18]</code> 3-Long, <code>+9.54 [+2.91, +16.49]</code> MRCR 512K) plus uncloned 1M LBv2 (<code>+1.99 [+0.25, +3.73]</code>; <code>+3.34 [+0.98, +5.78]</code> on strictly 13-gram-clean N=307). Controlled 24M sweeps at full 5×avg@4 (*95% CI, both <code>p&lt;1e-4</code>): <code>+5.70 [+4.10, +7.34]</code> 5-Bench / <code>+9.36 [+6.95, +11.84]</code> 3-Long on Mini 3/2; <code>+5.31 [+3.53, +7.14]</code> 5-Bench / <code>+8.50 [+5.83, +11.23]</code> 3-Long on VarLoop 3/1 (KV-map ladder: <code>48.9 → 49.7 → 50.0</code> 5-Bench; <code>26.7 → 29.6 → 31.4</code> MRCR, <code>+4.70 [+0.84, +8.53]</code>, p=0.017).</i></sub>

<p align="center">
  <img src="paper/figs/fig_inkling_speed.png" alt="Isolated 1x B300 Day-0 vLLM pure decode speedup across 512 to 256K tokens" width="76%">
</p>
</details>

### Multi-Turn & Agentic Serving (1×B300 Systems Grid, BFCL v4 Multi-Turn, SWE-bench Verified)

In multi-turn chat and tool-use agents, turn *t*−1's shallow-decoded answer *A*<sub>*t*−1</sub> re-enters turn *t*'s prompt. We derive exact KV and sliding-window memory costs (`2+2m` global slots for prefix-cached **Policy APC-D0** / **APC-S** vs. `3+2m+2n` slots + `160–200 MiB` SWA/SConv state for analytical dual-state **Policy I**) and evaluate all four policies across an isolated 1×B300 grid, **BFCL v4 Multi-Turn** (19 arms × 800 eps = **15,200 executable episodes**), and official zero-Docker **SWE-bench Verified** (19 arms, `mini-swe-agent` bash-only):

<p align="center">
  <img src="paper/figs/fig_multiturn_agentic.png" alt="Multi-turn speedup crossover, BFCL v4 Multi-Turn, and official SWE-bench Verified on 26.4B Inkling-2" width="98%">
  <br/>
  <sub><b>Multi-turn and agentic serving on 26.4B Inkling-2 in Day-0 vLLM (bar base: LLM s/ep).</b> (a) <b>1×B300 grid:</b> naive full re-prefill (<code>Policy R</code>) drops to <b>0.52×</b> on long-history short-output turns (crossing 1.0× at <code>L_A ≈ 256</code>), whereas prefix-cached <code>Policy APC-D0</code> sustains <b>1.50–1.76×</b> speedup (83.6% APC, zero dual-KV; incremental 4-pass prefill bounds analytical <code>Policy I</code> at 43–105 ms/turn). (b) <b>BFCL v4 <code>long_context</code>:</b> on <code>TML-varloop</code>, <code>APC-D0</code> preserves Orig 2/2 (46.8% vs. 46.5%), flips <code>Policy R</code> by <b>+8.0*</b> (<code>p=0.0044</code>, 1.77×), and avoids Distilled 2/2's LoRA drift (<b>+6.3*</b>, <code>p=0.0124</code>; and leads <code>effort=low</code> at 47.0% avg vs. 44.5% Orig 2/2); on fixed-depth Mini, LoopBridge rescues 0% collapse and <code>Policy R [I sem.]</code> beats Distilled 2/2 by <b>+7.8*</b> (<code>p=0.0044</code>). (c) <b>Official SWE-bench Verified (19 arms, 96.4% APC):</b> frozen <code>r=0</code> Univ 3/2 (70%, 50 s, 1.78× vs. 4/4) and LB 4/2 D0/APC-S (68%, 50–51 s) match Orig 2/2 (68%, 47 s) while avoiding Distilled 2/2 drift (60%; directional <code>+10.0, p=0.058</code>); at <code>n=1</code>, upgrading shallow <i>A</i><sub><i>t</i>−1</sub> lifts 32% Orig / 40% Distilled 1/1 to <b>70%</b> (<code>+38* / +30*</code>, <code>p&lt;1e-4</code>).</sub>
</p>

### Ouro-2.6B (11-Benchmark Average & Batch-1 Serving)

| Method | Loops (*m*/*n*) | 11-Bench Avg | 3-Seed Mean | 800 CoT Acc | Throughput | Relative TPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Original (Ouro) | 1/1 | 44.2 | — | 27.8% | 28.3 tok/s | 4.2× |
| Training-free | 4/1 | 28.7 | — | — | — | — |
| Distilled r16 / r80 (KD) | 1/1 | 49.5 / 49.3 | 49.6 / 49.2 | 34.2% | 26.6 tok/s | 4.0× |
| **LoopBridge (ours)** | **4/1** | **58.2** | **58.3 ± 0.1** | **45.0%** | **27.7 tok/s** | **4.1×** |
| Original (Ouro) | 2/2 | 63.7 | — | 57.1% | 13.6 tok/s | 2.0× |
| Distilled r16 / r80 (KD) | 2/2 | 66.2 / 65.5 | 66.1 / 65.6 | 59.2% | 13.6 tok/s | 2.0× |
| **LoopBridge (ours)** | **4/2** | **69.1** | **68.6 ± 0.4** | **62.5%** | **14.3 tok/s** | **2.1×** |
| Original (Ouro) | 4/4 | 71.9 | — | 69.1% | 6.7 tok/s | 1.0× |

<sub><i>11 benchmarks: MMLU, MMLU-Pro, BBH, ARC-C, HellaSwag, Winogrande, GSM8K, MATH-500 (zero-shot), HumanEval+, MBPP+, RULER-16K. 800 CoT: 200 timed requests each of GSM8K, MATH-500, BBH, MMLU-Pro on one B300 GPU. LoopBridge 4/1 also leads a time-matched Original 1/1–2/2 mix by <b>+10.5</b> points, and keeps <b>+18.5 / +11.3</b> points at batch 32 continuous batching in 1.18× / 1.19× time.</i></sub>

### Ouro-2.6B-Thinking (8-Benchmark Average & Batch-16 Serving)

| Method | Loops (*m*/*n*) | 8-Bench Avg | GSM8K (b16) | Wall Time / Req | Throughput (b16) | Speedup vs 4/4 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Original (Ouro-Thinking) | 2/2 | 65.5 | 92.9% | 4.04 s | 232 tok/s | 1.9× |
| Distilled r16 (KD) | 2/2 | 67.6 | 91.8% | 3.79 s | 236 tok/s | 1.9× |
| **LoopBridge (ours)** | **4/2** | **70.7** | **94.6%** | **3.19 s** | **247 tok/s** | **2.0×** |
| Original (Ouro-Thinking) | 4/4 | 71.5 | 94.3% | 6.84 s | 122 tok/s | 1.0× |

- **Transfers across scales & families:** Works on **Ouro-1.4B** (`34.7 → 44.5` at 4/1; `44.2 ± 0.2` over 3 seeds vs `40.5` distilled) and **Lev looped decision models** (`+4.4` to `+7.5` at one decision loop across all 13/13 *m* > *n* pairs).
- **Maps matter for long context:** Across 3 seeds on RULER-16K (4/1), LoopBridge scores **63.9 ± 1.0** vs **57.6 ± 1.6** for a parameter-matched rank-80 LoRA and **44.7 ± 4.4** for rank-16 LoRA.
- **Baselines are tuned:** Scaling the learning rate by `0.3×–3×` shifts Distilled baselines by at most **0.9** points; LoopBridge's weakest LR (`56.8`) stays far above the best distilled baseline (`49.5`).

---

## How LoopBridge Works

<p align="center">
  <img src="paper/figs/fig0_method.png" alt="LoopBridge architecture diagram" width="90%">
</p>

1. **Aligned cross-loop mapping.** The frozen backbone prefills the prompt with *m* loops. Decode loop *k* ∈ {1, …, *n*} reads prefill loop `max(m - n + k, 1)` through a complex rotary key scale *c*<sub>*k*</sub> (which commutes with RoPE) and a linear value map *W*<sub>*k*</sub>, plus a shared rank-16 decoder-only LoRA (**63.3M** trainable params = **2.4%** of Ouro-2.6B).
2. **Folded once at prefill (zero decode overhead).** Because *c*<sub>*k*</sub> commutes with RoPE and attention is linear in *V*, both maps fold once into the prompt KV caches right after prefill—matching Original *n*/*n*'s KV footprint and running standard single-call FlashAttention during decoding. *(A lighter one-cache variant shares only the last prefill loop to cut prompt KV memory by n×.)*
3. **One adapter for all (*m*, *n*) pairs.** Distilled from Original 4/4 via forward KL over 48M answer tokens while sampling (*m*, *n*) per micro-batch; at tied *n*/*n*, serving can bypass the adapter for exact Original accuracy.

```python
# LoopBridge core (timed folded form; full exact per-step equivalence in ouro/single_bank.py)
def prefill_and_fold(prompt, m, n):
    h = embed(prompt[:-1])
    for j in range(m):                                 # frozen backbone, m parallel loops
        for l in range(L):
            h, K[j][l], V[j][l] = blocks[l](h)
    for k in range(n):                                 # fold RoPE-commuting key scale & value map once
        j = max(m - n + k, 0)
        for l in range(L):
            Kp[k][l] = complex_mul(K[j][l], c[k][l])   # c_k commutes with RoPE
            Vp[k][l] = V[j][l] @ W[k][l].T
    return Kp, Vp

def decode_step(token, Kp, Vp, kv_ans, n):
    h = embed(token)
    for k in range(n):                                 # n fast decode loops (+ shared decoder LoRA)
        for l in range(L):
            h = blocks_lora[l](h, prompt_kv=(Kp[k][l], Vp[k][l]), answer_kv=kv_ans[k][l])
    return lm_head(h)
```

---

## Key Insights

<p align="center">
  <img src="paper/figs/fig_taskmap_domains.png" alt="Per-domain breakdown across all 16 prefill/decode pairs on Ouro-2.6B" width="96%">
  <br/>
  <sub><b>Per-domain directional asymmetry across all 16 (m, n) pairs.</b> Knowledge (MMLU) and retrieval (RULER-16K) depend heavily on prefill depth <i>m</i>, while multi-step reasoning (BBH, GSM8K) also benefits strongly from decode depth <i>n</i>; running <i>m &lt; n</i> consistently hurts.</sub>
</p>

<p align="center">
  <img src="paper/figs/fig_depth_main.png" alt="Prefill depth sweep up to m=8" width="96%">
  <br/>
  <sub><b>A few prefill loops suffice.</b> Across tasks, seeds, and even trainable prefill, gains saturate within 2–5 prefill loops (around pretrained depth <i>m</i> = 4) before drifting off-manifold.</sub>
</p>

<details>
<summary><b>More figures</b> (wall-time split, paired 1-loop comparison, batch 1/8/32 frontiers, Ouro-2.6B-Thinking 4/2, and Lev decision models)</summary>
<br/>
<table>
  <tr>
    <td align="center" width="50%">
      <img src="paper/figs/fig_time.png" alt="Prefill vs decode wall-time split" width="96%"><br/>
      <sub><b>Prefill is &lt;3% of wall time</b> (batch 1 CoT)</sub>
    </td>
    <td align="center" width="50%">
      <img src="paper/figs/fig_sametime.png" alt="Paired 1-decode-loop comparison" width="94%"><br/>
      <sub><b>Paired 1-loop gain</b> (+17.2 / +10.8 on 800 requests)</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="paper/figs/fig_equal.png" alt="Ouro-2.6B-Thinking 4/2 per benchmark" width="98%"><br/>
      <sub><b>Ouro-2.6B-Thinking 4/2</b> across 8 benchmarks</sub>
    </td>
    <td align="center" width="50%">
      <img src="paper/figs/fig_lev.png" alt="Lev looped decision models" width="98%"><br/>
      <sub><b>Lev decision models</b> (+4.4 to +7.5 at 1 decision loop)</sub>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="paper/figs/fig_batched.png" alt="More accuracy for the wall time across batch 1, batch 8, and batch 32" width="94%"><br/>
      <sub><b>Accuracy vs. wall time across batch 1, batch 8, and batch 32</b> (+10.5 / +9.8 / +12.4 over time-matched Original mix)</sub>
    </td>
  </tr>
</table>
</details>

---

## Quick Start

### 1. Code-only clone (`< 5 MB`)

The full repository (~1.0 GB) bundles per-item scores and provenance logs so every paper number can be verified offline. If you only want the code and paper:

```bash
git clone --filter=blob:none --no-checkout https://github.com/yanghu819/loopbridge
cd loopbridge && git sparse-checkout set --no-cone '/*' '!ouro/results' && git checkout main
```

### 2. Verify every paper number & figure on a CPU (~30s, no GPU needed)

```bash
git clone https://github.com/yanghu819/loopbridge && cd loopbridge

# Recompute every Ouro table cell, paired bootstrap 95% CI, timing ratio & throughput from raw per-item scores
uv run --with numpy --with math-verify python ouro/scripts/paper_numbers.py compute
# Expected final line: wrote ouro/results/paper_numbers.json; 0 printed values differ (0 in the paired comparisons)

# Redraw all 14 paper figures & verify 26.4B sparse-MoE paired bootstrap 95% CIs
uv run --with numpy --with matplotlib python ouro/scripts/paper_figs.py
uv run --with numpy --with matplotlib python inkling/make_figs.py
```

### 3. Download released checkpoints & distillation data (ModelScope)

All 16 core LoopBridge adapters (8 Ouro + 8 26.4B sparse-MoE, including the flagship `loopbridge-4x3-kvonly-12m` and `loopbridge-4x2-kvonly-12m` zero-LoRA checkpoints, every file SHA256-verified) and the GPQA-free 48M-token Ouro distillation mix + RULER/HELMET evaluation archives are publicly hosted on ModelScope:

- **Adapter Checkpoints:** [`torushy/LoopBridge`](https://modelscope.cn/models/torushy/LoopBridge) (`ouro-2.6b`, `ouro-2.6b-thinking`, `ouro-1.4b`, `ouro-1.4b-thinking`, `tml-d24-varloop`, `inkling-2-mini`)
- **Distillation & Eval Data:** [`torushy/LoopBridge-Data`](https://modelscope.cn/datasets/torushy/LoopBridge-Data) (`mix-data-48m.tar.gz`, `bench.tar.gz`, `bench-mini.tar.gz`)
- **Base Models (ModelScope / HF):** [Ouro-2.6B](https://modelscope.cn/models/bytedance-community/Ouro-2.6B) · [Ouro-2.6B-Thinking](https://modelscope.cn/models/bytedance-community/Ouro-2.6B-Thinking) · [Ouro-1.4B](https://modelscope.cn/models/bytedance-community/Ouro-1.4B) · [Ouro-1.4B-Thinking](https://modelscope.cn/models/bytedance-community/Ouro-1.4B-Thinking) · [TML-d24-varloop-research](https://huggingface.co/thinkingmachines/TML-d24-varloop-research) · [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)

```bash
pip install modelscope
modelscope download --model torushy/LoopBridge --local_dir ./checkpoints
modelscope download --dataset torushy/LoopBridge-Data --local_dir ./loopbridge-data
```

### 4. Train & evaluate on GPUs

```bash
cd environments/ouro && uv sync && cd ../..
```

See **[`REPRODUCE.md`](REPRODUCE.md)** for the full table/figure-to-script mapping, data preparation, training launchers, Day-0 vLLM 26.4B MoE evaluation commands, and continuous-batching benchmarks.

---

## Repository Layout

```text
loopbridge/
├── ouro/
│   ├── adapter_model.py        # LoopBridge adapter (RoPE-commuting key scales, value maps, decoder LoRA)
│   ├── single_bank.py          # Folded-cache & exact LSE-merged single-bank inference runners
│   ├── batched.py              # Batched forward KL distillation (LoopBridge & Distilled n/n baselines)
│   ├── train_batched.py        # Multi-GPU DDP training entrypoint
│   ├── model.py                # Frozen Ouro looped backbone & cache wrappers
│   ├── evaluate_bench.py       # RULER & HELMET evaluation runner
│   ├── lmeval_ouro.py          # lm-evaluation-harness integration & zero-shot Thinking tasks
│   ├── lev/                    # Lev looped decision-model untied training & LevBench evaluation
│   ├── scripts/                # paper_numbers.py, paper_figs.py, batch_requests.py, time_requests.py, provenance.py
│   ├── runs_scripts/           # Exact training & evaluation launchers for every Ouro run in the paper
│   └── results/                # Per-item scores (items/), timing (e2e/, throughput-sb4/), lev/, provenance/
├── inkling/
│   ├── model.py                # 26.4B sparse-MoE NoPE global-3 KV maps + decode LoRA + AttnRes adapters
│   ├── vllm_loopbridge.py      # Zero-overhead Day-0 vLLM Blackwell integration (merged decode LoRA + prefill/multi-turn KV fold)
│   ├── train_inkling.py        # 8x B300 DDP length-bucketed KL / generalized JSD trainer
│   ├── prepare_data_v3.py      # Controlled 24M on-policy v3 distillation corpus builder (13-gram decontaminated)
│   ├── prepare_data_v4_gkd.py  # Hybrid verified SFT + on-policy OPD + 16K-52K v4 corpus builder
│   ├── prepare_official_exact.py # Official 256K-1M evaluation split builder (GPQA, LCB v6, MRCR, GraphWalks, LBv2)
│   ├── eval_vllm.py            # Multi-replica Day-0 vLLM official evaluator
│   ├── bench_vllm_timing.py    # Strictly isolated single-GPU SSE decode & E2E microbenchmark (512..256K)
│   ├── bench_multiturn_systems.py # Isolated 1x B300 multi-turn serving grid (APC-D0 vs Policy R vs Orig)
│   ├── eval_bfcl_multiturn.py  # Official BFCL v4 Multi-Turn (base + long_context) evaluator
│   ├── eval_swebench_verified.py # Official zero-Docker SWE-bench Verified (mini-swe-agent bash-only) runner
│   ├── swe_gold_verify.py      # Isolated git-worktree official swebench.harness.grading verifier
│   ├── scoring.py              # Exact official benchmark scorers (+ test_scorers.py, test_model.py, test_vllm_parity.py)
│   ├── make_figs.py            # Recomputes 26.4B & multi-turn paired bootstrap CIs & draws Figures 3, 4, 15
│   └── results/                # Official-exact avg@4, multi-turn B300 grid, BFCL v4, SWE-bench Verified & SSE logs
├── paper/
│   ├── loopbridge.pdf          # Compiled 38-page arXiv preprint (<=8-page ACL main body + Appendices A-P)
│   ├── latex/                  # Single-source ACL LaTeX & bibliography
│   ├── figs/                   # All 15 paper figures (PDF/PNG) & generated LaTeX table fragments
│   └── build_arxiv.py          # Reproducible arXiv tarball & PDF builder
├── environments/ouro/          # Pinned uv environment (pyproject.toml, uv.lock)
├── third_party/ouro/           # Ouro modeling code (Apache-2.0)
├── docs/                       # Release archive notes (ARCHIVE.md), deprecated arms/scripts (DEPRECATED.md), README assets
└── REPRODUCE.md                # End-to-end reproduction guide
```

> The tag `paper-arxiv` points to the exact commit that built [`paper/loopbridge.pdf`](paper/loopbridge.pdf) and all figures/tables. Details on archived checkpoints and pruned legacy branches/scripts are in [`docs/ARCHIVE.md`](docs/ARCHIVE.md) and [`docs/DEPRECATED.md`](docs/DEPRECATED.md).

---

## Citation

```bibtex
@misc{hu2026loopbridge,
  title  = {LoopBridge: Should Prefill and Decode Loop Equally?},
  author = {Hu, Yang},
  year   = {2026},
  url    = {https://github.com/yanghu819/loopbridge}
}
```

## License

Code is released under the [MIT License](LICENSE). Third-party modeling and benchmark code under `third_party/ouro/` and `ouro/licenses/` retain their respective upstream licenses (Apache-2.0 / MIT).
