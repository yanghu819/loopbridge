# Deprecated code

These pieces are **not used by the paper**. They stay only because live code still imports them or because a
launcher in `ouro/runs_scripts/` refers to them. Do not extend them, cite them, or build new experiments on them.
Anything here can be deleted once its importer is cleaned up.

The project was closed on 2026-11-19 without deleting any of it: the modules below are imported by modules the paper
does use, so removing them would mean editing the code that produced the results after the GPUs were released, with no
way to test the edit. The tag `paper-arxiv` marks the commit the arXiv version was built from, and the published code
is the code that ran. [`ARCHIVE.md`](ARCHIVE.md) says where the artifacts went.

| What | Why it is deprecated | Still referenced by |
|---|---|---|
| `ouro/refresh_model.py`, arm `R` in `ouro/train_batched.py` | "refresh" decoding (re-prefill generated chunks); stopped line, it only saves latency and overlaps speculative decoding | `train_batched.py` import |
| `ouro/evaluate_gsm8k.py`, `ouro/evaluate_prefix_suffix.py` | early evaluators (3-shot GSM8K for the first four paths; PG19 NLL); replaced by `evaluate_bench.py` + lm-evaluation-harness | `evaluate_adapter.py` import |
| arms `A`, `B`, `C` (per-cell adapters), `U`, `V`, `W` (earlier universal adapters), `T`/`Q` (two-source map), `J`/`K` (shallower prefill models) | superseded by the single-bank universal loop adapter, arm `S`, which every paper result uses | `adapter_model.py`, `batched.py`, `train_batched.py` branches |
| `ouro/runs_scripts/launch_v.sh` | trains arm `V` (superseded by `S`) | — |
| `ouro/sensenova_bridge.py`, `ouro/sensenova_runner.py`, `ouro/scripts/sn_prepare.py`, `launch_sn.sh`, `launch_sng.sh`, `sn_*.sh` | SenseNova-Loop-9B (a hybrid Gated-DeltaNet looped model): mixed signs, dropped from the paper on 2026-11-12 | `lmeval_ouro.py`, `evaluate_bench.py`, `scripts/e2e_latency.py` imports |
| `ouro/runs_scripts/launch_nbs21.sh`, `nbs_cell.sh` | Nanbeige: same, dropped on 2026-11-12 | — |

Paper arms, for contrast: `a` (aligned caches, the main method), `S` (one stored prompt cache, the memory-saving
variant and the task-map / design-variant appendices), `O`/`P` (shared answer KV variant, appendix), `M` (gated first-loop
branch, appendix), `S` with `--freeze-maps` (maps fixed at identity), `D`/`E`/`H` (LoRA distillation of native n/n).

Moved to `ouro/runs_scripts/deprecated/` (2026-11-07): `a1_fill.sh` (one-off GPU filler), `make_bib.py` (first references.bib generator; the bib now has hand-verified entries).
`math0_q.sh` (zero-shot MATH-500) is back in `ouro/runs_scripts/` (2026-11-10, `ARM=a` for the aligned adapter's
0-shot check in Section 6); its old copy in `deprecated/` was removed.
Deleted: `ouro/scripts/rescore_mbpp_plus.py` (a check of the earlier MBPP+ prompt, superseded by evalplus scoring).

Never in this repository: the training-free untying check on Huginn-0125 (2026-11-17), a recurrent-depth model from
another family. It gave no reliable gain on GSM8K at any of eight prefill/decode settings, so it is not in the paper;
its script and per-item outputs are in the private archive (`a2/extra/pn.tar.gz`, folder `huginn`).

Removed outright (in git history if ever needed): probe and "answers without a final answer" analyses, the
training-free SenseNova appendix, `latest.json`, long-context throughput records, unused per-item score files, old
copies of figures, and Codex reviews of superseded drafts.
