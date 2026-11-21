# Deprecated code

The tag `paper-arxiv` points to the clean release commit the arXiv paper and all figures/tables were built from. All
dropped model lines, broken/unused launchers, non-paper result files, and dead helper scripts have been removed from the
working tree (preserved in git history if ever needed).

## Remaining legacy branches in live code

Only the internal legacy training-arm branches below remain inside the core Ouro modules (so we do not disturb the
tested training/evaluation codepaths that produced the paper's checkpoints):

| What | Why it is deprecated | Where |
|---|---|---|
| arms `A`, `B`, `C` (per-cell adapters), `U`, `W` (earlier universal adapters), `T`/`Q` (two-source map), `J`/`K` (shallower prefill models) | superseded by LoopBridge (`arm a`, aligned caches) and its one-cache variant (`arm S`) | `ouro/adapter_model.py`, `ouro/batched.py`, `ouro/train_batched.py` |

**Active paper arms:** `a` (LoopBridge with aligned caches, the main method), `S` (one-cache LoopBridge, the
memory-saving variant in Table 2a and Appendix K), `V` (unrestricted pre-RoPE key-map control, Appendix L via
`ouro/runs_scripts/launch_v.sh`), `a`/`S` with `--freeze-maps` (maps fixed at identity, Table 2a & Appendix L fixed-depth
`m=4` LoRA), and `D`/`E`/`H` (tied-loop LoRA distillation baselines at `2/2`, `1/1`, `3/3`). Optional code branches for
shared answer KV (`O`/`P`) and a gated first-loop branch (`M`) also remain in `ouro/adapter_model.py`.

## Removed from the repository (available in git history)

- **Dropped non-Ouro model lines:** SenseNova-Loop-9B (`sensenova_bridge.py`, `sensenova_runner.py`, `sn_prepare.py`,
  `launch_sn*.sh`, `sn_*.sh`) and Nanbeige / IQuest-Loop (`looped_bridge.py`, `launch_nbs21.sh`, `nbs_cell.sh`), plus
  their non-paper evaluation files in `ouro/results/`.
- **Stopped / superseded scripts:** refresh decoding (`refresh_model.py`, arm `R`), early standalone evaluators
  (`evaluate_gsm8k.py`, `evaluate_prefix_suffix.py`), speculative decoding check (`spec_agree.py`, `spec_agree.sh`),
  1.4B batched timing (`time_batch14.sh`), `ouro/runs_scripts/deprecated/` (`a1_fill.sh`, `make_bib.py`),
  `rescore_mbpp_plus.py`, and non-paper timing/item files.
- **Internal operational files:** cluster backup scripts (`backup_*.py`), internal handoff/scratch logs, and orphaned
  figure/table artifacts.
- **Never in this repository:** a training-free untying check on Huginn-0125 (no reliable GSM8K gain; kept only in the
  private cold archive described in [`ARCHIVE.md`](ARCHIVE.md)).
