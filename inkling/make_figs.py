"""Generate publication figures (PDF + PNG) and LaTeX tables for Inkling-2-Mini LoopBridge.

Matches ouro/scripts/paper_figs.py typography, palette, and marker conventions:
  - Orange (#EB6834) squares: LoopBridge (filled = n=1, hollow = n=2)
  - Blue   (#2A78D6) triangles: Distilled (filled = n=1, hollow = n=2)
  - Grey   (#4D4D4D / #9A9A9A) circles: Original (filled = n=1, hollow = n=2, star/dash = 3/3 teacher)

Usage:
  python inkling/make_figs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RES = ROOT / "results"
OUT = REPO / "paper" / "figs"

OURS, DIST = "#EB6834", "#2A78D6"
NAT, NAT_LIGHT, INK, GRID = "#4D4D4D", "#9A9A9A", "#1A1A1A", "#E6E6E6"

plt.rcParams.update(
    {
        "font.size": 8,
        "font.family": "DejaVu Sans",
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": NAT,
        "xtick.color": NAT,
        "ytick.color": NAT,
        "axes.labelcolor": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
)

BENCH_ORDER = [
    ("gsm8k", "GSM8K"),
    ("math500", "MATH"),
    ("mmlu_pro", "MMLU-P"),
    ("arc_c", "ARC-C"),
    ("bbh", "BBH"),
    ("ruler16k", "RULER"),
    ("helmet", "HELMET"),
    ("longbench_v2", "LB-v2"),
]

# Trainable parameter counts (from training setup.json manifests; 26.4B total backbone)
PARAM_LABELS = {
    "native_3_3": ("0", "0\\%"),
    "native_2_2": ("0", "0\\%"),
    "native_1_1": ("0", "0\\%"),
    "tf_3_2": ("0", "0\\%"),
    "tf_3_1": ("0", "0\\%"),
    "lb_3_2_24m": ("43.2M", "0.17\\%"),
    "lb_3_2_nomap_24m": ("38.5M", "0.15\\%"),
    "distill_2_2_24m": ("48.2M", "0.18\\%"),
    "lb_3_1_24m": ("36.4M", "0.14\\%"),
    "lb_3_1_nomap_24m": ("34.0M", "0.13\\%"),
    "lb_3_1_norouter_24m": ("31.1M", "0.12\\%"),
    "distill_1_1_24m": ("42.2M", "0.16\\%"),
}


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _score(scores: dict, run: str, bench: str) -> float | None:
    return scores.get(run, {}).get(bench, {}).get("score")


def _mean(scores: dict, run: str) -> float | None:
    vals = [_score(scores, run, b) for b, _ in BENCH_ORDER]
    if any(v is None for v in vals):
        return None
    return float(np.mean(vals))


def make_figure(summary: dict, timing: dict | None) -> None:
    scores = summary["scores"]
    has_timing = timing is not None and bool(timing.get("results"))

    if has_timing:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(6.75, 2.75), gridspec_kw={"width_ratios": [1.75, 1.0], "wspace": 0.24}
        )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(4.6, 2.75))
        ax2 = None

    # --- Panel (a): 8 benchmarks + Mean ---
    keys = [b for b, _ in BENCH_ORDER] + ["__mean__"]
    labels = [lab for _, lab in BENCH_ORDER] + ["Mean"]
    xs = np.arange(len(keys), dtype=float)
    # Small visual gap before Mean
    xs[-1] += 0.35

    def series(run: str):
        out = []
        for k in keys:
            v = _mean(scores, run) if k == "__mean__" else _score(scores, run, k)
            out.append(np.nan if v is None else v)
        return np.array(out, dtype=float)

    t33 = series("native_3_3")
    n11, d11, l31 = series("native_1_1"), series("distill_1_1_24m"), series("lb_3_1_24m")
    n22, d22, l32 = series("native_2_2"), series("distill_2_2_24m"), series("lb_3_2_24m")

    # Subtle separator before Mean
    ax1.axvline(xs[-2] + 0.68, color="#D8D8D8", lw=0.7, ls="--", zorder=0)

    # Original 3/3 reference horizontal ticks at each benchmark
    for x, y in zip(xs, t33):
        if np.isfinite(y):
            ax1.plot([x - 0.30, x + 0.30], [y, y], color=INK, lw=1.35, zorder=2)

    # Vertical range stems for n=1 (left offset -0.13) and n=2 (right offset +0.13)
    for x, y_lo, y_hi in zip(xs - 0.13, n11, l31):
        if np.isfinite(y_lo) and np.isfinite(y_hi):
            ax1.plot([x, x], [y_lo, y_hi], color="#F5B89B", lw=1.6, zorder=1)
    for x, y_lo, y_hi in zip(xs + 0.13, n22, l32):
        if np.isfinite(y_lo) and np.isfinite(y_hi):
            ax1.plot([x, x], [y_lo, y_hi], color="#D6D6D6", lw=1.2, zorder=1)

    # n=1 (filled markers, x - 0.13)
    ax1.scatter(xs - 0.13, n11, s=22, color=NAT_LIGHT, marker="o", zorder=3)
    ax1.scatter(xs - 0.13, d11, s=24, color=DIST, marker="^", zorder=4)
    ax1.scatter(xs - 0.13, l31, s=28, color=OURS, marker="s", zorder=5)

    # n=2 (hollow markers, x + 0.13)
    ax1.scatter(xs + 0.13, n22, s=22, facecolors="white", edgecolors=NAT, lw=1.0, marker="o", zorder=3)
    ax1.scatter(xs + 0.13, d22, s=24, facecolors="white", edgecolors=DIST, lw=1.0, marker="^", zorder=4)
    ax1.scatter(xs + 0.13, l32, s=28, facecolors="white", edgecolors=OURS, lw=1.2, marker="s", zorder=5)

    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, rotation=24, ha="right")
    ax1.set_ylabel("Accuracy / Score (%)")
    ax1.set_ylim(-2, 102)
    ax1.set_xlim(-0.55, xs[-1] + 0.55)
    ax1.grid(axis="y", color=GRID, lw=0.5)
    ax1.set_axisbelow(True)
    ax1.set_title("(a) Inkling-2-Mini (26B MoE) across 8 benchmarks", fontsize=8.5, color=INK, pad=5)

    # --- Panel (b): Real B300 Throughput vs Accuracy frontier ---
    if ax2 is not None:
        # Use B=16, L=512 (batched serving decode tok/s) — falls back to B=1, L=512 if absent
        rows = timing["results"]
        pick = [r for r in rows if r["batch_size"] == 16 and r["prompt_len"] == 512]
        if not pick:
            pick = [r for r in rows if r["batch_size"] == 1 and r["prompt_len"] == 512]
        by_lbl = {r["label"]: r for r in pick}
        bs_used = pick[0]["batch_size"] if pick else 16

        def tps_of(label: str) -> float | None:
            r = by_lbl.get(label)
            return r["decode_agg_tok_per_s"] if r else None

        nat_pts = []
        for lbl, run, short in [
            ("Original 3/3", "native_3_3", "3/3"),
            ("Original 2/2", "native_2_2", "2/2"),
            ("Original 1/1", "native_1_1", "1/1"),
        ]:
            x, y = tps_of(lbl), _mean(scores, run)
            if x is not None and y is not None:
                nat_pts.append((x, y, short))

        lb_pts = []
        for lbl, run, short, filled in [
            ("LoopBridge 3/2 (folded)", "lb_3_2_24m", "LB 3/2", False),
            ("LoopBridge 3/1 (folded)", "lb_3_1_24m", "LB 3/1", True),
        ]:
            x, y = tps_of(lbl), _mean(scores, run)
            if x is not None and y is not None:
                lb_pts.append((x, y, short, filled))

        dist_pts = []
        for lbl, run, short, filled in [
            ("Original 2/2", "distill_2_2_24m", "Dist 2/2", False),
            ("Original 1/1", "distill_1_1_24m", "Dist 1/1", True),
        ]:
            x, y = tps_of(lbl), _mean(scores, run)
            if x is not None and y is not None:
                dist_pts.append((x, y, short, filled))

        if nat_pts:
            ax2.plot([p[0] for p in nat_pts], [p[1] for p in nat_pts], "-", color=NAT_LIGHT, lw=1.4, zorder=2)
            for x, y, short in nat_pts:
                if short == "3/3":
                    ax2.scatter(x, y, s=30, color=INK, marker="o", zorder=4)
                    ax2.annotate("3/3", (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7.2, color=INK, fontweight="bold")
                elif short == "2/2":
                    ax2.scatter(x, y, s=28, facecolors="white", edgecolors=NAT, lw=1.1, marker="o", zorder=4)
                    ax2.annotate("2/2", (x, y), xytext=(5, -9), textcoords="offset points", fontsize=7.0, color=NAT)
                else:
                    ax2.scatter(x, y, s=28, color=NAT_LIGHT, marker="o", zorder=4)
                    ax2.annotate("1/1", (x, y), xytext=(-3, 6), textcoords="offset points", fontsize=7.0, color=NAT, ha="right")

        if dist_pts:
            ax2.plot([p[0] for p in dist_pts], [p[1] for p in dist_pts], "--", color=DIST, lw=1.1, alpha=0.8, zorder=2)
            for x, y, short, filled in dist_pts:
                if filled:
                    ax2.scatter(x, y, s=30, color=DIST, marker="^", zorder=4)
                    ax2.annotate(short, (x, y), xytext=(-5, -10), textcoords="offset points", fontsize=6.9, color=DIST, ha="right")
                else:
                    ax2.scatter(x, y, s=30, facecolors="white", edgecolors=DIST, lw=1.1, marker="^", zorder=4)
                    ax2.annotate(short, (x, y), xytext=(4, -10), textcoords="offset points", fontsize=6.9, color=DIST, ha="left")

        if lb_pts:
            ax2.plot([p[0] for p in lb_pts], [p[1] for p in lb_pts], "-", color=OURS, lw=1.7, zorder=3)
            for x, y, short, filled in lb_pts:
                if filled:
                    ax2.scatter(x, y, s=34, color=OURS, marker="s", zorder=5)
                    ax2.annotate(short, (x, y), xytext=(-4, -12), textcoords="offset points", fontsize=7.2, color=OURS, fontweight="bold", ha="right")
                else:
                    ax2.scatter(x, y, s=34, facecolors="white", edgecolors=OURS, lw=1.3, marker="s", zorder=5)
                    ax2.annotate(short, (x, y), xytext=(4, 5), textcoords="offset points", fontsize=7.2, color=OURS, fontweight="bold", ha="left")

        ax2.set_xlabel(f"Decode throughput (tok/s, B={bs_used}, 1x B300)")
        ax2.set_ylabel("8-Benchmark Mean (%)")
        ax2.set_ylim(-2, 86)
        if nat_pts or lb_pts:
            xs_all = [p[0] for p in nat_pts] + [p[0] for p in lb_pts]
            span = max(xs_all) - min(xs_all) + 1.0
            ax2.set_xlim(min(xs_all) - 0.14 * span, max(xs_all) + 0.14 * span)
        ax2.grid(color=GRID, lw=0.5)
        ax2.set_axisbelow(True)
        ax2.set_title("(b) Accuracy vs. B300 decode speed", fontsize=8.5, color=INK, pad=5)

    # Shared legend on top
    handles = [
        plt.Line2D([], [], color=INK, lw=1.5, label="Original 3/3 (teacher)"),
        plt.Line2D([], [], color=OURS, marker="s", ms=4.8, lw=0, label="LoopBridge (filled: 3/1, hollow: 3/2)"),
        plt.Line2D([], [], color=DIST, marker="^", ms=4.8, lw=0, label="Distilled (filled: 1/1, hollow: 2/2)"),
        plt.Line2D([], [], color=NAT_LIGHT, marker="o", ms=4.5, lw=0, label="Original (filled: 1/1, hollow: 2/2)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        fontsize=7.3,
        bbox_to_anchor=(0.5, 1.11),
        columnspacing=1.2,
        handletextpad=0.4,
        frameon=False,
    )
    save(fig, "fig_inkling")


def make_main_table(summary: dict, param_overrides: dict[str, tuple[str, str]] | None = None) -> None:
    scores = summary["scores"]
    pmap = dict(PARAM_LABELS)
    if param_overrides:
        pmap.update(param_overrides)
    t33_mean = _mean(scores, "native_3_3")

    rows_spec = [
        ("native_3_3", "Original 3/3 (teacher)", False),
        ("midrule", "", False),
        ("native_2_2", "Original 2/2", False),
        ("tf_3_2", "Training-free 3/2", False),
        ("distill_2_2_24m", "Distilled 2/2 (24M)", False),
        ("lb_3_2_24m", "\\textbf{LoopBridge 3/2 (24M)}", True),
        ("lb_3_2_nomap_24m", "\\quad w/o KV maps (maps frozen)", False),
        ("midrule", "", False),
        ("native_1_1", "Original 1/1", False),
        ("tf_3_1", "Training-free 3/1", False),
        ("distill_1_1_24m", "Distilled 1/1 (24M)", False),
        ("lb_3_1_24m", "\\textbf{LoopBridge 3/1 (24M)}", True),
        ("lb_3_1_nomap_24m", "\\quad w/o KV maps (maps frozen)", False),
        ("lb_3_1_norouter_24m", "\\quad w/o MoE router delta", False),
    ]

    lines = []
    for key, label, hi in rows_spec:
        if key == "midrule":
            lines.append("\\midrule")
            continue
        if key not in scores:
            continue
        p_str, pct_str = pmap.get(key, ("—", "—"))
        p_col = "0" if p_str == "0" else f"{p_str} ({pct_str})"
        vals = [_score(scores, key, b) for b, _ in BENCH_ORDER]
        cells = [f"{v:.1f}" if v is not None else "--" for v in vals]
        mval = _mean(scores, key)
        if mval is not None:
            m_str = f"\\textbf{{{mval:.2f}}}" if hi else f"{mval:.2f}"
            rec_str = f"\\textbf{{{100.0 * mval / t33_mean:.1f}\\%}}" if (hi and t33_mean) else (f"{100.0 * mval / t33_mean:.1f}\\%" if t33_mean else "--")
        else:
            m_str, rec_str = "--", "--"
        prefix = "\\rowcolor{oursbg}\n" if hi else ""
        lines.append(f"{prefix}{label} & {p_col} & " + " & ".join(cells) + f" & {m_str} & {rec_str} \\\\")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tab_inkling_main.tex").write_text("\n".join(lines) + "\n")


def make_timing_table(timing: dict) -> None:
    rows = timing["results"]
    by_key = {(r["batch_size"], r["prompt_len"], r["label"]): r for r in rows}
    cfgs = [(1, 512), (1, 4096), (1, 16384), (16, 512), (16, 2048), (32, 512)]
    arms = [
        ("Original 3/3", "Original 3/3", False),
        ("Original 2/2", "Original 2/2", False),
        ("LoopBridge 3/2 (folded)", "\\textbf{LoopBridge 3/2}", True),
        ("Original 1/1", "Original 1/1", False),
        ("LoopBridge 3/1 (folded)", "\\textbf{LoopBridge 3/1}", True),
    ]
    lines = []
    for ci, (bs, plen) in enumerate(cfgs):
        ref = by_key.get((bs, plen, "Original 3/3"))
        if ref is None:
            continue
        if ci > 0:
            lines.append("\\midrule")
        for ai, (arm_key, arm_lbl, hi) in enumerate(arms):
            r = by_key.get((bs, plen, arm_key))
            if r is None:
                continue
            cfg_col = f"{bs} / {plen:,}" if ai == 0 else ""
            dec_sp = r["decode_agg_tok_per_s"] / ref["decode_agg_tok_per_s"]
            e2e_sp = r["e2e_agg_tok_per_s"] / ref["e2e_agg_tok_per_s"]
            map_str = "--" if r["map_fold_ms"] == 0 else f"{r['map_fold_ms']:.1f}"
            prefix = "\\rowcolor{oursbg}\n" if hi else ""
            lines.append(
                f"{prefix}{cfg_col} & {arm_lbl} & {r['prefill_ms']:.0f} & {map_str} & "
                f"{r['decode_ms_per_tok']:.2f} & {r['decode_agg_tok_per_s']:.1f} & "
                f"{dec_sp:.2f}$\\times$ & {e2e_sp:.2f}$\\times$ \\\\"
            )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tab_inkling_time.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    summary = json.loads((RES / "summary.json").read_text())
    timing_path = RES / "timing_b300.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else None
    prov_path = RES / "provenance.json"
    param_overrides = None
    if prov_path.exists():
        prov = json.loads(prov_path.read_text())
        param_overrides = {}
        total_backbone = prov.get("backbone_total_params", 26_400_000_000)
        for run_key, info in prov.get("runs", {}).items():
            n_p = info.get("trainable_parameters")
            if n_p is not None:
                param_overrides[run_key] = (f"{n_p / 1e6:.1f}M", f"{100.0 * n_p / total_backbone:.2f}\\%")

    make_figure(summary, timing)
    make_main_table(summary, param_overrides)
    if timing:
        make_timing_table(timing)
    print("Generated fig_inkling.{pdf,png}, tab_inkling_main.tex" + (", tab_inkling_time.tex" if timing else ""))


if __name__ == "__main__":
    main()
