# Archive

The project finished on 2026-11-19. The two 8-GPU boxes it ran on were released, and their working directories
(`/dev/shm`, a tmpfs) were deleted. This file says what survived, and where.

The tag `paper-arxiv` is the commit the arXiv version was built from. The code is frozen there: nothing was removed at
close-out, so the published modules are the ones that produced the results (see [`DEPRECATED.md`](DEPRECATED.md) for
the parts the paper does not use).

## In this repository

Everything the paper's numbers are made of: the source, the launcher for every run, the per-item scores
(`ouro/results/items/`), the per-request timings (`ouro/results/e2e/`), the Lev outputs (`ouro/results/lev/`), the
provenance records (`ouro/results/provenance/`: command, configuration, checkpoint hash, log and outputs of every
run), the figures and the LaTeX source. `python ouro/scripts/paper_numbers.py compute` recomputes every printed value
from them on a CPU, and `ouro/scripts/paper_figs.py` redraws every figure. The environment is pinned in
`environments/ouro/` (`pyproject.toml`, `uv.lock`).

## Private cold archive

The heavy artifacts are in a private ModelScope repository, `torushy/untying-prefill-decode-backup`: 115.8 GB in
10,679 files, uploaded by `ouro/scripts/backup_ms.py` (every two hours while the project ran) and, at close-out, by
`backup_lev.py` and `backup_extra.py`. `a1/` and `a2/` are the two boxes.

| path | what | size |
|---|---|---|
| `a1/train/`, `a2/train/` | one folder per training run: setup, logs, progress and the **final checkpoint** | 95.7 GB |
| `a1/lmeval/`, `a2/lmeval/` | one `.tar.gz` per lm-evaluation-harness run: samples and results | 10.4 GB |
| `a1/bench/`, `a2/bench/` | `evaluate_bench.py` outputs (RULER, HELMET, LongBench) | 0.07 GB |
| `a1/e2e/`, `a2/e2e/` | per-request timing records, including the runs the paper does not use | 0.35 GB |
| `a1/lev/`, `a2/lev/` | the Lev decision-model work: 56 training runs with their adapters, LevBench development outputs, scored rows, logs | 5.1 GB |
| `a1/extra/data/`, `a2/extra/data/` | the distillation corpus (`adapter-distill-200m`), the Thinking and older mixtures, the local benchmark copies (`hf-lmeval`, GPQA, IFEval), the exported per-item scores | 3.7 GB |
| `a1/data/` | the 48M-token training mixture and the `bench` / `bench-mini` benchmark copies | 0.41 GB |
| `a1/extra/env/`, `a2/extra/env/` | `pip freeze`, Python and driver versions, and each base model's config, tokenizer and `modeling_*.py` with the SHA256 of its weight files | 0.04 GB |
| `a1/extra/code-on-box.tar.gz`, `a2/` | the source tree as the runs left it on each box (this repository is a pruned copy of it) | 5 MB |
| `a1/extra/pn.tar.gz`, `a2/` | the paper-number and packaging logs, and the training-free Huginn-0125 check | 4 MB |
| `a1/queue_logs/`, `a2/queue_logs/` | which job ran on which GPU, and when | 4 MB |

Spot-checked after upload: files download, extract, and match the copy on the box byte for byte.

The repository is private. Ask the author for access, or for a single checkpoint.

## Deliberately not kept

- **Base model weights** (Ouro-1.4B / 2.6B and their Thinking versions, Nanbeige-3B, SenseNova-Loop-9B). They are
  public upstream; their config, tokenizer and modeling files and the SHA256 of every weight file are in `extra/env/`,
  so a download can be checked against what ran.
- **Third-party data with its own licence**: LevBench, Lev-train-v2, the released Lev adapters, the Hugging Face
  dataset caches. GPQA items are never redistributed (only per-item scores, keyed by index), and LevBench's locked
  test split was never evaluated.
- **Intermediate training checkpoints.** Only the final checkpoint of each run is archived.
- **Package caches** (`lmeval-pkgs`, `fwcache`), the SenseNova and Nanbeige mixtures (that line is not in the paper),
  and an October restore copy of material that is archived anyway.
