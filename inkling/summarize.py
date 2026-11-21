"""Merge sharded eval JSONLs and compute benchmark summary tables + paired bootstrap 95% CIs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


def load_bench_rows(run_dir: Path, bench: str) -> dict[str, dict]:
    rows = {}
    files = sorted(run_dir.glob(f"{bench}.shard*.jsonl"))
    single = run_dir / f"{bench}.jsonl"
    if single.exists():
        files = [single]
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
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    common = sorted(set(a_rows) & set(b_rows))
    if not common:
        return {}
    diffs = [100.0 * (score_of(a_rows[k]) - score_of(b_rows[k])) for k in common]
    mean_d = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    boots = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return {"n": n, "diff": round(mean_d, 2), "ci95": [round(lo, 2), round(hi, 2)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--benches", nargs="+", default=["gsm8k", "math500", "mmlu_pro", "arc_c", "bbh", "ruler16k", "helmet", "longbench_v2"])
    parser.add_argument("--ref", default="native_3_3", help="Reference run directory name for paired CIs")
    parser.add_argument("--merge", action="store_true", help="Write merged <bench>.jsonl in each run dir")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runs = sorted(p.name for p in args.eval_root.iterdir() if p.is_dir())
    table = {}
    all_rows = {}
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
            vals = [score_of(r) for r in rows.values()]
            toks = [r.get("generated_tokens", 0) for r in rows.values()]
            table[run][b] = {
                "n": len(vals),
                "score": round(100.0 * sum(vals) / len(vals), 2),
                "avg_gen_tok": round(sum(toks) / len(toks), 1),
            }
            all_rows[run][b] = rows

    # Print markdown table
    present_benches = [b for b in args.benches if any(b in table[r] for r in runs)]
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
        mean_str = f"**{sum(scs)/len(scs):.2f}**" if len(scs) == len(present_benches) else (f"{sum(scs)/len(scs):.2f}*" if scs else "—")
        print(f"| `{run}` | " + " | ".join(cells) + f" | {mean_str} |")

    cis = {}
    if args.ref in all_rows:
        for run in runs:
            if run == args.ref:
                continue
            cis[run] = {}
            for b in present_benches:
                if b in all_rows[run] and b in all_rows[args.ref]:
                    cis[run][b] = paired_bootstrap_ci(all_rows[run][b], all_rows[args.ref][b])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"scores": table, "paired_ci_vs_ref": cis}, indent=2) + "\n")


if __name__ == "__main__":
    main()
