# Archive

The project finished on 2026-11-19. The two 8-GPU boxes it ran on were released, and their working directories
(`/dev/shm`, a tmpfs) were deleted. This file says what survived, and where.

The tag `paper-arxiv` is the commit the arXiv version was built from. Nothing was removed from the code at close-out,
so the modules the runs used are all published (see [`DEPRECATED.md`](DEPRECATED.md) for the parts the paper does not
use). The source tree on each box is archived as it stood when the runs ended, because it is not identical to this
repository: the repository was pruned of old development scripts (296 of them on one box) and then kept growing. On
a1's snapshot, 11 files differ, and the differences are the repository's later additions — docstrings and
`DEPRECATED` headers, HELMET support in `scripts/e2e_latency.py`, `paper_numbers.py` growing from 687 to 2,831 lines
as the paper was written, an `OURO_TIME_START` offset and two opt-in environment flags that default to the behaviour
that ran.

## In this repository

Everything the paper's numbers are made of: the source, the launcher for every run the paper uses, the per-item scores
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
| `a1/extra/data/`, `a2/extra/data/` | the distillation corpus (`adapter-distill-200m`; each box's copy, because their file lists differ), the Thinking and older mixtures, the local benchmark copies, the exported per-item scores | 3.7 GB |
| `a1/data/` | the 48M-token training mixture and the `bench` / `bench-mini` benchmark copies | 0.41 GB |
| `a1/extra/env/`, `a2/extra/env/` | `pip freeze`, Python and driver versions, and each base model's config, tokenizer and `modeling_*.py` with the SHA256 of its weight files | 0.04 GB |
| `a1/extra/code-on-box.tar.gz`, `a2/extra/code-on-box.tar.gz` | the source tree as the runs left it on each box | 5 MB |
| `a1/extra/pn.tar.gz`, `a2/extra/pn.tar.gz` | the paper-number and packaging logs, and the training-free Huginn-0125 check | 4 MB |
| `a1/queue_logs/`, `a2/queue_logs/` | which job ran on which GPU, and when | 4 MB |

Checked before the boxes were wiped: random files from `a1/lmeval/`, `a2/lmeval/`, `a1/train/`, `a2/lev/train/` and
`a1/extra/data/` were downloaded and opened, and every file of one archived Lev run matched the copy on the box by
SHA256. The rest of the 10,679 files was not read back. The two boxes' copies of `mix-data-48m`, `bench` and
`bench-mini` were compared by file list and size before only a1's was archived.

The repository is private, and it stays private: it holds GPQA items, which are never redistributed, and raw outputs
of runs that are not in the paper. Ask the author (an issue on GitHub reaches him) for access or for a single
checkpoint; GPQA data is not shared in any case.

## Deliberately not kept

- **Base model weights** (Ouro-1.4B / 2.6B and their Thinking versions, Nanbeige-3B, SenseNova-Loop-9B). They are
  public upstream; their config, tokenizer and modeling files and the SHA256 of every weight file are in `extra/env/`,
  so a download can be checked against what ran.
- **Third-party data and models with their own licence**: LevBench, Lev-train-v2, the released Lev adapters and the
  Lev base models. LevBench's locked test split was never evaluated. The local benchmark copies that *are* archived
  (`hf-lmeval`, GPQA, IFEval) are there so the evaluations stay reproducible if the upstream sets move; the archive is
  private and GPQA items are never passed on, in the repository or out of it (the public repository holds only
  per-item scores, keyed by index).
- **Intermediate training checkpoints.** Only the final checkpoint of each run is archived.
- **Package and download caches** (`lmeval-pkgs`, the FineWeb download cache), the SenseNova and Nanbeige mixtures
  (that line is not in the paper), and an October restore copy of material that is archived anyway. `pip freeze` and
  the pinned `uv.lock` describe the environment instead; a locally patched package would not be captured by either.
