"""Merge sharded eval JSONLs, export compact per-item scores, and compute paired bootstrap 95% CIs.

Matches ouro/scripts/paper_numbers.py statistical protocol exactly:
  5,000 numpy bootstrap resamples (seed 0), 2.5 / 97.5 percentiles, in points (x100).
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

DEFAULT_BENCHES = [
    "gsm8k",
    "math500",
    "mmlu_pro",
    "arc_c",
    "bbh",
    "ruler16k",
    "helmet",
    "longbench_v2",
]

KEY_PAIRS = [
    # (label, A, B) -> A - B
    ("lb_3_1_vs_native_1_1", "lb_3_1_24m", "native_1_1"),
    ("lb_3_1_vs_tf_3_1", "lb_3_1_24m", "tf_3_1"),
    ("lb_3_1_vs_distill_1_1", "lb_3_1_24m", "distill_1_1_24m"),
    ("lb_3_1_vs_nomap_3_1", "lb_3_1_24m", "lb_3_1_nomap_24m"),
    ("lb_3_1_vs_norouter_3_1", "lb_3_1_24m", "lb_3_1_norouter_24m"),
    ("lb_3_1_vs_native_3_3", "lb_3_1_24m", "native_3_3"),
    ("lb_3_2_vs_native_2_2", "lb_3_2_24m", "native_2_2"),
    ("lb_3_2_vs_tf_3_2", "lb_3_2_24m", "tf_3_2"),
    ("lb_3_2_vs_distill_2_2", "lb_3_2_24m", "distill_2_2_24m"),
    ("lb_3_2_vs_nomap_3_2", "lb_3_2_24m", "lb_3_2_nomap_24m"),
    ("lb_3_2_vs_native_3_3", "lb_3_2_24m", "native_3_3"),
    ("distill_1_1_vs_native_1_1", "distill_1_1_24m", "native_1_1"),
    ("distill_2_2_vs_native_2_2", "distill_2_2_24m", "native_2_2"),
    ("tf_3_2_vs_native_2_2", "tf_3_2", "native_2_2"),
    ("tf_3_1_vs_native_1_1", "tf_3_1", "native_1_1"),
]


def load_bench_rows(run_dir: Path, bench: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    gz = run_dir / f"{bench}.items.jsonl.gz"
    single = run_dir / f"{bench}.jsonl"
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["id"]] = r
        return rows
    pilot = run_dir / f"{bench}.pilot.jsonl"
    files = [p for p in (pilot, single) if p.exists()] + sorted(run_dir.glob(f"{bench}.shard*.jsonl"))
    for p in files:
        for line in p.open():
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def score_of(r: dict) -> float:
    return float(r["correct"]) if "correct" in r else float(r["score"])


def paired_bootstrap_ci(
    a_rows: dict[str, dict],
    b_rows: dict[str, dict],
    n_boot: int = 5000,
    seed: int = 0,
) -> dict:
    common = sorted(set(a_rows) & set(b_rows))
    if not common:
        return {}
    diffs = np.array([100.0 * (score_of(a_rows[k]) - score_of(b_rows[k])) for k in common], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boots = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": len(common),
        "diff": round(float(diffs.mean()), 2),
        "ci95": [round(float(lo), 2), round(float(hi), 2)],
    }


def macro_paired_bootstrap_ci(
    a_by_bench: dict[str, dict[str, dict]],
    b_by_bench: dict[str, dict[str, dict]],
    benches: list[str],
    n_boot: int = 5000,
    seed: int = 0,
) -> dict:
    """Stratified paired bootstrap for unweighted macro-average across benchmarks."""
    if not all(b in a_by_bench and b in b_by_bench for b in benches):
        return {}
    rng = np.random.default_rng(seed)
    boot_sum = np.zeros(n_boot, dtype=np.float64)
    point_sum = 0.0
    total_n = 0
    for b in benches:
        common = sorted(set(a_by_bench[b]) & set(b_by_bench[b]))
        if not common:
            return {}
        diffs = np.array(
            [100.0 * (score_of(a_by_bench[b][k]) - score_of(b_by_bench[b][k])) for k in common],
            dtype=np.float64,
        )
        idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
        boot_sum += diffs[idx].mean(axis=1)
        point_sum += float(diffs.mean())
        total_n += len(common)
    boots = boot_sum / len(benches)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": total_n,
        "benches": len(benches),
        "diff": round(point_sum / len(benches), 2),
        "ci95": [round(float(lo), 2), round(float(hi), 2)],
    }


def export_compact_items(run_rows: dict[str, dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=9) as f:
        for k in sorted(run_rows):
            r = run_rows[k]
            compact = {
                "id": r["id"],
                "task": r.get("task", ""),
                "score": score_of(r),
                "generated_tokens": r.get("generated_tokens", 0),
                "stop_reason": r.get("stop_reason", ""),
            }
            if "length_bucket" in r:
                compact["length_bucket"] = r["length_bucket"]
            f.write(json.dumps(compact, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--benches", nargs="+", default=DEFAULT_BENCHES)
    parser.add_argument("--ref", default="native_3_3", help="Reference run directory name for paired CIs")
    parser.add_argument("--merge", action="store_true", help="Write merged <bench>.jsonl in each run dir")
    parser.add_argument("--export-items", type=Path, default=None, help="Export compact per-item .items.jsonl.gz tree")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runs = sorted(p.name for p in args.eval_root.iterdir() if p.is_dir())
    table: dict[str, dict] = {}
    all_rows: dict[str, dict[str, dict]] = {}
    for run in runs:
        rdir = args.eval_root / run
        table[run] = {}
        all_rows[run] = {}
        for b in args.benches:
            rows = load_bench_rows(rdir, b)
            if not rows:
                continue
            if args.merge and not (rdir / f"{b}.jsonl").exists():
                (rdir / f"{b}.jsonl").write_text("".join(json.dumps(rows[k], ensure_ascii=False) + "\n" for k in sorted(rows)))
            if args.export_items:
                export_compact_items(rows, args.export_items / run / f"{b}.items.jsonl.gz")
            vals = [score_of(r) for r in rows.values()]
            toks = [r.get("generated_tokens", 0) for r in rows.values()]
            table[run][b] = {
                "n": len(vals),
                "score": round(100.0 * sum(vals) / len(vals), 2),
                "avg_gen_tok": round(sum(toks) / len(toks), 1),
            }
            all_rows[run][b] = rows

    present_benches = [b for b in args.benches if any(b in table[r] for r in runs)]
    means: dict[str, float] = {}
    header = "| Condition | " + " | ".join(present_benches) + " | Mean |"
    sep = "| :--- | " + " | ".join(["---:"] * (len(present_benches) + 1)) + " |"
    print(header)
    print(sep)
    for run in runs:
        if not table[run]:
            continue
        cells = []
        scs = []
        for b in present_benches:
            if b in table[run]:
                sc = table[run][b]["score"]
                n = table[run][b]["n"]
                cells.append(f"{sc:.2f} ({n})")
                scs.append(sc)
            else:
                cells.append("—")
        if len(scs) == len(present_benches):
            mval = round(sum(scs) / len(scs), 2)
            means[run] = mval
            mean_str = f"**{mval:.2f}**"
        elif scs:
            mean_str = f"{sum(scs)/len(scs):.2f}*"
        else:
            mean_str = "—"
        print(f"| `{run}` | " + " | ".join(cells) + f" | {mean_str} |")

    cis_vs_ref: dict[str, dict] = {}
    if args.ref in all_rows:
        for run in runs:
            if run == args.ref:
                continue
            cis_vs_ref[run] = {}
            for b in present_benches:
                if b in all_rows[run] and b in all_rows[args.ref]:
                    cis_vs_ref[run][b] = paired_bootstrap_ci(all_rows[run][b], all_rows[args.ref][b])
            macro = macro_paired_bootstrap_ci(all_rows[run], all_rows[args.ref], present_benches)
            if macro:
                cis_vs_ref[run]["macro_mean"] = macro

    key_cis: dict[str, dict] = {}
    for label, a_name, b_name in KEY_PAIRS:
        if a_name not in all_rows or b_name not in all_rows:
            continue
        pair_out: dict[str, dict] = {}
        for b in present_benches:
            if b in all_rows[a_name] and b in all_rows[b_name]:
                pair_out[b] = paired_bootstrap_ci(all_rows[a_name][b], all_rows[b_name][b])
        macro = macro_paired_bootstrap_ci(all_rows[a_name], all_rows[b_name], present_benches)
        if macro:
            pair_out["macro_mean"] = macro
        if pair_out:
            key_cis[label] = pair_out

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "benches": present_benches,
                    "scores": table,
                    "means": means,
                    "paired_ci_vs_ref": cis_vs_ref,
                    "key_paired_cis": key_cis,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
