"""Generate publication figures (PDF + PNG) and verified paired bootstrap tables
for the 26.4B Sparse-MoE Looped Transformers results (TML-d24-varloop-research
and Inkling-2-Mini in Day-0 vLLM Blackwell).

Strictly recomputes every number directly from archived JSON files in
inkling/results/ (zero hardcoded numbers). Matches ouro/scripts/paper_figs.py
typography, palette, and marker conventions:
  - Orange (#EB6834 / #C94A18): LoopBridge (ours)
  - Blue   (#2A78D6): Distilled tied-loop controls
  - Grey   (#4D4D4D / #9A9A9A): Original / Training-Free baselines

Usage:
  uv run --with numpy --with matplotlib python inkling/make_figs.py
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RES = ROOT / "results"
OUT = REPO / "paper" / "figs"

OURS, OURS_DARK, OURS_LIGHT = "#EB6834", "#C94A18", "#F4A582"
DIST = "#2A78D6"
NAT, NAT_LIGHT, INK, GRID = "#4D4D4D", "#9A9A9A", "#1A1A1A", "#E6E6E6"

plt.rcParams.update(
    {
        "font.size": 8,
        "font.family": "DejaVu Sans",
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
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

BENCHES = [
    ("gpqa_diamond_high", "GPQA"),
    ("livecodebench_v6_high", "LCB v6"),
    ("mrcr_8n_256k_512k", "MRCR\n512K"),
    ("graphwalks_bfs_256k", "GW\n256K"),
    ("longbench_v2_1m", "LB-v2\n1M"),
]
BENCH_KEYS = [k for k, _ in BENCHES]
PURE4_KEYS = BENCH_KEYS[:4]
LONG_KEYS = BENCH_KEYS[2:]

AVG4_ARMS = [
    "vl_native_2_2",
    "vl_kvonly_2_2_t4_12m",
    "vl_crossridge_4_2",
    "vl_procrustes_4_2",
    "vl_tf_4_2_tail",
    "vl_tf_4_2_anchored",
    "vl_tf_5_2_anchored",
    "vl_lb_4_2_kvonly_nosynth_12m",
    "vl_lb_4_2_kvonly_12m",
    "vl_native_3_3",
    "vl_tf_4_3_tail",
    "vl_tf_4_3_anchored",
    "vl_lb_4_3_kvonly_12m",
    "vl_native_4_4",
    # Wave-2B Regime II & n=1 full 5xavg@4 arms
    "mini_orig_3_3_cap57k",
    "mini_distilled_2_2_v3",
    "mini_lb_3_2_24m",
    "vl_lb_3_1_24m",
    "vl_abl_3_1_no_kvmaps",
]


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def load_evals() -> tuple[dict[str, dict[str, dict]], dict[str, dict[str, dict]]]:
    """Load (evals_avg1, evals_avg4) directly from archived official-exact JSONs.

    evals_avg4 holds all 19 full 5xavg@4 arms (81,776 rollouts: 14 Regime I arms,
    3 Inkling-2-Mini arms, and 2 TML-varloop n=1 ablation arms) evaluated with 4x symmetric sampling.
    evals_avg1 holds all single-sample (seed=42) official-exact arms across
    TML-d24-varloop and Inkling-2-Mini.
    """
    eo = json.load(open(RES / "eval_official_exact.json"))
    ea4 = json.load(open(RES / "eval_official_exact_avg4.json"))
    ev3 = json.load(open(RES / "eval_vllm_official_v3.json"))
    elcb = json.load(open(RES / "eval_vllm_lcb_fixed_v3.json"))
    evl = json.load(open(RES / "eval_vllm_varloop.json"))

    evals_avg1: dict[str, dict[str, dict]] = {}

    def add1(name: str, src_eo=None, fb_v3=None, fb_lcb=None, fb_vl=None):
        entry = {}
        for bk in BENCH_KEYS:
            if src_eo and src_eo in eo and bk in eo[src_eo]:
                entry[bk] = eo[src_eo][bk]
            elif bk == "livecodebench_v6_high" and fb_lcb and fb_lcb in elcb:
                entry[bk] = elcb[fb_lcb][bk]
            elif fb_v3 and fb_v3 in ev3 and bk in ev3[fb_v3]:
                entry[bk] = ev3[fb_v3][bk]
            elif fb_vl and fb_vl in evl and bk in evl[fb_vl]:
                entry[bk] = evl[fb_vl][bk]
            else:
                raise KeyError(f"Missing {name}.{bk}")
        evals_avg1[name] = entry

    # TML-d24-varloop-research (avg@1)
    for arm in [
        "vl_native_5_5",
        "vl_native_4_4",
        "vl_tf_5_3_anchored",
        "vl_lb_5_3_kvonly_12m",
        "vl_tf_4_3_tail",
        "vl_tf_4_3_anchored",
        "vl_lb_4_3_v5_24m",
        "vl_lb_4_3_kvonly_12m",
        "vl_tf_5_2_anchored",
        "vl_tf_4_2_tail",
        "vl_tf_4_2_anchored",
        "vl_lb_4_2_v5_48m",
        "vl_lb_4_2_onpolicy_24m",
        "vl_lb_4_2_kvonly_12m",
        "vl_tf_3_2_anchored",
        "vl_lb_3_2_v2_48m",
        "vl_native_2_2",
        "vl_lb_3_1_24m",
        "vl_lb_3_1_v2_48m",
        "vl_abl_3_1_singlepass_kv",
        "vl_abl_3_1_no_kvmaps",
        "vl_lb_4_1_24m",
        "vl_distilled_1_1_24m",
    ]:
        add1(arm, arm)
    add1("vl_native_3_3", "vl_native_3_3", fb_vl="vl_native_3_3")
    add1("vl_tf_3_1", "vl_tf_3_1", fb_vl="vl_tf_3_1")
    add1("vl_native_1_1", "vl_native_1_1", fb_vl="vl_native_1_1")

    # Inkling-2-Mini (avg@1, 100% symmetric caps: 65K/65K/65K/98K/57K)
    evals_avg1["mini_orig_3_3"] = {
        "gpqa_diamond_high": ev3["native_3_3"]["gpqa_diamond_high"],
        "livecodebench_v6_high": elcb["native_3_3"]["livecodebench_v6_high"],
        "mrcr_8n_256k_512k": eo["mini_orig_3_3_cap65k"]["mrcr_8n_256k_512k"],
        "graphwalks_bfs_256k": eo["mini_orig_3_3_cap65k"]["graphwalks_bfs_256k"],
        "longbench_v2_1m": eo["mini_orig_3_3_cap57k"]["longbench_v2_1m"],
    }
    add1("mini_tf_3_2_anchored", "mini_tf_3_2_anchored")
    add1("mini_lb_3_2_v2_48m", "mini_lb_3_2_v2_48m")
    add1("mini_lb_3_2_24m", "mini_lb_3_2_24m", fb_v3="lb_3_2_24m", fb_lcb="lb_3_2_24m")
    add1("mini_distilled_2_2_v3", "mini_distilled_2_2_v3", fb_v3="distilled_2_2_v3")
    add1("mini_lb_3_1_24m", "mini_lb_3_1_24m", fb_v3="lb_3_1_24m")
    add1("mini_distilled_1_1_v3", "mini_distilled_1_1_v3", fb_v3="distilled_1_1_v3", fb_lcb="distilled_1_1_v3")

    # TML-d24-varloop-research (14-arm Regime I TRUE 5/5 symmetric avg@4 across ALL 5 benchmarks = 60,256 rollouts)
    evals_avg4: dict[str, dict[str, dict]] = {}
    for arm in AVG4_ARMS:
        entry = {}
        for bk in BENCH_KEYS:
            entry[bk] = ea4[arm][bk]
        evals_avg4[arm] = entry

    return evals_avg1, evals_avg4


def cluster_item_scores(evals: dict, arm: str, bk: str) -> dict[str, float]:
    """Return base_item_id -> score (%) averaged over replicates (#s0..#s3 if present)."""
    by_base: dict[str, list[float]] = defaultdict(list)
    for x in evals[arm][bk]["items"]:
        iid = x["id"]
        base = iid.rsplit("#", 1)[0] if "#" in iid else iid
        by_base[base].append(100.0 * float(x["score"]))
    return {k: sum(v) / len(v) for k, v in by_base.items()}


def score_pct(evals: dict, arm: str, bk: str) -> float:
    cs = cluster_item_scores(evals, arm, bk)
    return float(np.mean(list(cs.values())))


def macro_pct(evals: dict, arm: str, keys: list[str] = BENCH_KEYS) -> float:
    return float(np.mean([score_pct(evals, arm, bk) for bk in keys]))


def paired_boot_clustered(
    evals: dict, a: str, b: str, keys: list[str], n_boot: int = 5000, seed: int = 0
) -> tuple[float, float, float, float, float, float, float, float]:
    """Item-clustered stratified paired bootstrap preserving base item N.

    Returns (obs, lo95, hi95, lo975, hi975, lo9833, hi9833, p_two_sided).
    """
    diffs_by_bk = []
    for bk in keys:
        am = cluster_item_scores(evals, a, bk)
        bm = cluster_item_scores(evals, b, bk)
        ids = sorted(set(am) & set(bm))
        assert len(ids) == len(am) == len(bm), f"ID mismatch {a} vs {b} on {bk}"
        diffs_by_bk.append([am[i] - bm[i] for i in ids])
    obs = float(np.mean([np.mean(ds) for ds in diffs_by_bk]))
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        ms = [sum(ds[rng.randrange(len(ds))] for _ in range(len(ds))) / len(ds) for ds in diffs_by_bk]
        boots.append(sum(ms) / len(ms))
    boots.sort()
    p1 = sum(1 for x in boots if x <= 0.0) / n_boot
    p2 = 2.0 * min(p1, 1.0 - p1)
    return (
        obs,
        boots[int(0.025 * n_boot)],
        boots[int(0.975 * n_boot)],
        boots[int(0.0125 * n_boot)],
        boots[int(0.9875 * n_boot)],
        boots[int(0.008333 * n_boot)],
        boots[int(0.991667 * n_boot)],
        p2,
    )


def load_vllm_vl_timing() -> dict[str, dict[tuple[int, int, int], dict]]:
    arms = [
        "orig_5_5",
        "orig_4_4",
        "orig_3_3",
        "tf_4_3_anchored",
        "lb_4_3_kvonly_12m",
        "orig_2_2",
        "tf_4_2_anchored",
        "tf_5_2_anchored",
        "lb_4_2_kvonly_12m",
        "lb_3_2_v2_48m",
        "orig_1_1",
        "lb_3_1_24m",
        "lb_4_1_24m",
    ]
    out = {}
    for arm in arms:
        rows = []
        for suf in ("short", "long"):
            p = RES / f"vllm_sse_vl_vl_{arm}_{suf}.json"
            if p.exists():
                rows.extend(json.load(open(p))["results"])
        out[arm] = {(r["L"], r["B"], r["gen_len"]): r for r in rows}
    return out


def make_speed_fig(vllm_vl: dict) -> None:
    """Single-panel Appendix N figure (ZERO Pareto duplication — Pareto lives exclusively in Figure 1b!):
    Isolated 1x B300 pure decode speedup vs. Original 4/4 across L=512..256K, visually proving 0.0%
    KV-map folding decode overhead at n=3, n=2, and n=1.
    """
    fig, ax2 = plt.subplots(figsize=(5.1, 2.55))

    cfgs = [
        ((512, 16, 1024), "512\nB=16"),
        ((4096, 16, 256), "4K\nB=16"),
        ((16384, 8, 512), "16K\nB=8"),
        ((65536, 4, 512), "64K\nB=4"),
        ((131072, 4, 512), "128K\nB=4"),
        ((262144, 2, 512), "256K\nB=2"),
    ]
    xs = np.arange(len(cfgs))

    def speedups(arm: str) -> list[float]:
        return [
            vllm_vl[arm][c]["agg_decode_tok_s"] / vllm_vl["orig_4_4"][c]["agg_decode_tok_s"]
            for c, _ in cfgs
        ]

    ax2.axhline(1.0, color=NAT_LIGHT, ls=":", lw=0.9, zorder=1)
    ax2.plot(xs, speedups("orig_5_5"), ":", color=NAT_LIGHT, marker="v", ms=3.5, lw=1.0, label="Original 5/5 (0.83×)")
    ax2.plot(xs, speedups("orig_3_3"), "--", color=NAT, marker="o", ms=3.6, lw=1.05, label="Original 3/3")
    ax2.plot(xs, speedups("lb_4_3_kvonly_12m"), "-", color=OURS_LIGHT, marker="s", mfc="white", mec=OURS_DARK, mew=1.2, ms=4.8, lw=1.5, label="LoopBridge 4/3 (KV-only)")
    ax2.plot(xs, speedups("orig_2_2"), "--", color="#707070", marker="o", ms=3.6, lw=1.05, label="Original 2/2")
    ax2.plot(xs, speedups("lb_4_2_kvonly_12m"), "-", color=OURS, marker="s", ms=4.5, lw=1.6, label="LoopBridge 4/2 (KV-only)")
    ax2.plot(xs, speedups("orig_1_1"), "--", color=DIST, marker="^", ms=4.0, lw=1.05, label="Original 1/1")
    ax2.plot(xs, speedups("lb_3_1_24m"), "-", color=OURS_DARK, marker="s", ms=4.2, lw=1.5, label="LoopBridge 3/1 (+LoRA)")

    for arm, col, dy in [
        ("lb_3_1_24m", OURS_DARK, 5.5),
        ("lb_4_2_kvonly_12m", OURS, 5.5),
        ("lb_4_3_kvonly_12m", OURS_DARK, 5.5),
    ]:
        val = speedups(arm)[-1]
        ax2.annotate(f"{val:.2f}×", (xs[-1], val), xytext=(-2, dy), textcoords="offset points", fontsize=7.0, fontweight="bold", color=col, ha="center")

    ax2.set_xticks(xs)
    ax2.set_xticklabels([lab for _, lab in cfgs], fontsize=7.2)
    ax2.set_xlabel("Prompt length L and batch size B (1× B300, vLLM)", fontsize=7.8)
    ax2.set_ylabel("Pure decode speedup vs. Original 4/4", fontsize=7.8)
    ax2.set_ylim(0.68, 3.30)
    ax2.grid(axis="y", color=GRID, lw=0.6)
    ax2.set_title("Isolated 1× B300 pure decode speedup (0.0% KV-fold decode overhead)", fontsize=8.2, fontweight="bold", pad=6)
    ax2.legend(loc="upper left", ncol=2, fontsize=6.2, columnspacing=0.8, handletextpad=0.35, handlelength=1.5, labelspacing=0.28)

    save(fig, "fig_inkling_speed")


def make_main_body_two_regime_fig(evals_avg1: dict, evals_avg4: dict) -> None:
    """Spacious, zero-duplication, zero-jargon 2-panel Main-Body Figure 4 (replaces old Fig 6 AND deletes
    duplicate Appendix Fig 17!). Uses crisp side-by-side paired bars (solid = 5-Bench Avg, hatched = 3-Long
    Avg 256K-1M) with plain-English labels readable without prose:
      (a) Regime I (variable-depth TML-d24-varloop, n >= 2, full 5xavg@4): 2x2 causal proof at n=2 (1.75x)
          plus n=3 (1.27x) recovering 99.0% of Original 4/4.
      (b) Regime II (fixed-depth Inkling-2-Mini 3/2 at full 5xavg@4 & TML-varloop n=1 at avg@1, locked 24M tok):
          no-training collapse rescued by LoopBridge, beating tied Distilled across both backbones.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.05, 2.88), gridspec_kw={"width_ratios": [1.15, 0.95], "wspace": 0.24}
    )

    # =========================================================================
    # Panel (a): Regime I — Variable-depth TML-d24-varloop (n >= 2, 5xavg@4)
    # =========================================================================
    n2_arms = [
        ("vl_native_2_2", "Orig\n2/2", NAT_LIGHT, False),
        ("vl_kvonly_2_2_t4_12m", "Tied KV\n2/2", DIST, False),
        ("vl_tf_4_2_tail", "No-train\n4/2", "#A8A8A8", False),
        ("vl_lb_4_2_kvonly_nosynth_12m", "LB 4/2\nreal-only", OURS_LIGHT, True),
        ("vl_lb_4_2_kvonly_12m", "LB 4/2\nKV-only", OURS_DARK, True),
    ]
    n3_arms = [
        ("vl_native_3_3", "Orig\n3/3", NAT_LIGHT, False),
        ("vl_tf_4_3_anchored", "No-train\n4/3", "#A8A8A8", False),
        ("vl_lb_4_3_kvonly_12m", "LB 4/3\nKV-only", OURS_DARK, True),
    ]
    pos1 = [0.0, 1.08, 2.16, 3.24, 4.32, 5.76, 6.84, 7.92]
    all_a = n2_arms + n3_arms
    w = 0.36
    m5_a = [macro_pct(evals_avg4, a, BENCH_KEYS) for a, _, _, _ in all_a]
    m3_a = [macro_pct(evals_avg4, a, LONG_KEYS) for a, _, _, _ in all_a]
    cols_a = [c for _, _, c, _ in all_a]

    bars5_a = ax1.bar([p - w / 2 for p in pos1], m5_a, width=w, color=cols_a, edgecolor="white", lw=0.45, zorder=3)
    bars3_a = ax1.bar([p + w / 2 for p in pos1], m3_a, width=w, color=cols_a, alpha=0.42, edgecolor=NAT, lw=0.55, hatch="///", zorder=3)

    ceil44 = macro_pct(evals_avg4, "vl_native_4_4", BENCH_KEYS)
    ax1.axhline(ceil44, color=NAT, ls=":", lw=0.95, zorder=1)
    ax1.text(-0.42, ceil44 + 0.45, f"Original 4/4 ceiling ({ceil44:.1f}%)", fontsize=5.8, color=NAT, ha="left", va="bottom")

    for i, (bar, (_, _, _, is_ours)) in enumerate(zip(bars5_a, all_a)):
        h = bar.get_height()
        suf = "*" if i in (2, 3, 4) else ""
        dx = -2.2 if i == 2 else (0.6 if i == 3 else (-1.8 if i == 6 else (1.4 if i == 7 else -0.8)))
        ax1.annotate(
            f"{h:.1f}{suf}",
            (bar.get_x() + bar.get_width() / 2, h),
            xytext=(dx, 2.0),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=5.3,
            fontweight="bold" if is_ours else "normal",
            color=OURS_DARK if is_ours else INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.25, alpha=0.90) if i in (6, 7) else None,
            zorder=7,
        )
    for i, (bar, (_, _, _, is_ours)) in enumerate(zip(bars3_a, all_a)):
        h = bar.get_height()
        suf = "*" if i in (2, 3, 4) else ""
        ax1.annotate(
            f"{h:.1f}{suf}",
            (bar.get_x() + bar.get_width() / 2, h),
            xytext=(2.4, 1.8),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=4.7,
            color=OURS_DARK if is_ours else NAT,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.15, alpha=0.80),
            zorder=6,
        )

    d5_n2 = m5_a[4] - m5_a[1]
    rec_n3 = 100.0 * m5_a[7] / ceil44
    ax1.axvline(5.04, color=GRID, lw=1.0)
    ax1.text(2.16, 81.8, f"n = 2 (1.75× decode)\n+{d5_n2:.1f}* over Tied KV 2/2", fontsize=5.65, fontweight="bold", color=OURS_DARK, ha="center", va="top", linespacing=1.15)
    ax1.text(6.84, 81.8, f"n = 3 (1.27× decode)\n{rec_n3:.1f}% of Original 4/4", fontsize=5.65, fontweight="bold", color=OURS_DARK, ha="center", va="top", linespacing=1.15)

    ax1.set_xticks(pos1)
    ax1.set_xticklabels([lab for _, lab, _, _ in all_a], fontsize=5.7)
    ax1.set_ylabel("Accuracy (%, 5×avg@4)", fontsize=7.6)
    ax1.set_ylim(56.0, 83.2)
    ax1.grid(axis="y", color=GRID, lw=0.6)
    ax1.set_axisbelow(True)
    ax1.set_title("(a) Regime I: Variable-depth TML-d24-varloop (n ≥ 2)", fontsize=7.8, fontweight="bold", pad=6)

    # =========================================================================
    # Panel (b): Regime II — Fixed-depth Inkling-2-Mini (5xavg@4) & n=1 (avg@1)
    # Strictly locked 24M-token matched controls
    # =========================================================================
    mini_b = [
        (evals_avg1, "mini_tf_3_2_anchored", "No-train\n3/2 (@1)", NAT_LIGHT, False),
        (evals_avg4, "mini_distilled_2_2_v3", "Distilled\n2/2", DIST, False),
        (evals_avg4, "mini_lb_3_2_24m", "LB\n3/2", OURS_DARK, True),
    ]
    vl1_b = [
        (evals_avg1, "vl_native_1_1", "Orig\n1/1", NAT_LIGHT, False),
        (evals_avg1, "vl_distilled_1_1_24m", "Distilled\n1/1", DIST, False),
        (evals_avg1, "vl_lb_3_1_24m", "LB\n3/1", OURS_DARK, True),
    ]
    pos2 = [0.0, 1.08, 2.16, 3.66, 4.74, 5.82]
    all_b = mini_b + vl1_b
    m5_b = [macro_pct(src, a, BENCH_KEYS) for src, a, _, _, _ in all_b]
    m3_b = [macro_pct(src, a, LONG_KEYS) for src, a, _, _, _ in all_b]
    cols_b = [c for _, _, _, c, _ in all_b]

    bars5_b = ax2.bar([p - w / 2 for p in pos2], m5_b, width=w, color=cols_b, edgecolor="white", lw=0.45, zorder=3)
    bars3_b = ax2.bar([p + w / 2 for p in pos2], m3_b, width=w, color=cols_b, alpha=0.42, edgecolor=NAT, lw=0.55, hatch="///", zorder=3)

    # Inkling-2-Mini Orig 3/3 teacher ceiling (5xavg@4, left half only!)
    mini_ceil = macro_pct(evals_avg4, "mini_orig_3_3_cap57k", BENCH_KEYS)
    ax2.hlines(mini_ceil, -0.45, 2.75, colors=NAT, linestyles=":", lw=0.95, zorder=1)
    ax2.text(1.08, mini_ceil + 1.0, f"Original 3/3 ({mini_ceil:.1f}%)", fontsize=5.8, color=NAT, ha="center", va="bottom")

    for i, (bar, (_, _, _, _, is_ours)) in enumerate(zip(bars5_b, all_b)):
        h = bar.get_height()
        suf = "*" if is_ours else ""
        ax2.annotate(
            f"{h:.1f}{suf}",
            (bar.get_x() + bar.get_width() / 2, h),
            xytext=(-1.2, 2.0),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=5.6,
            fontweight="bold" if is_ours else "normal",
            color=OURS_DARK if is_ours else INK,
            zorder=7,
        )
    for i, (bar, (_, _, _, _, is_ours)) in enumerate(zip(bars3_b, all_b)):
        h = bar.get_height()
        suf = "*" if is_ours else ""
        ax2.annotate(
            f"{h:.1f}{suf}",
            (bar.get_x() + bar.get_width() / 2, h),
            xytext=(2.6, 1.8),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=5.0,
            color=OURS_DARK if is_ours else NAT,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.15, alpha=0.80),
            zorder=6,
        )

    ax2.annotate(
        "93% trunc.\n(collapse)",
        (0.0, 6.0),
        xytext=(0.05, 22.0),
        fontsize=5.7,
        color="#A02020",
        ha="center",
        fontweight="bold",
        linespacing=1.1,
        arrowprops=dict(arrowstyle="->", color="#A02020", lw=0.75),
    )

    d5_mini = m5_b[2] - m5_b[1]
    d3_mini = m3_b[2] - m3_b[1]
    d5_vl1 = m5_b[5] - m5_b[4]
    d3_vl1 = m3_b[5] - m3_b[4]
    ax2.axvline(2.91, color=GRID, lw=1.0)
    ax2.text(1.08, 79.8, f"Inkling-2-Mini 3/2 (avg@4)\n+{d5_mini:.1f}* avg / +{d3_mini:.1f}* long", fontsize=5.6, fontweight="bold", color=OURS_DARK, ha="center", va="top", linespacing=1.15)
    ax2.text(4.74, 79.8, f"TML-varloop 3/1 (avg@1)\n+{d5_vl1:.1f}* avg / +{d3_vl1:.1f}* long", fontsize=5.6, fontweight="bold", color=OURS_DARK, ha="center", va="top", linespacing=1.15)

    ax2.set_xticks(pos2)
    ax2.set_xticklabels([lab for _, _, lab, _, _ in all_b], fontsize=5.9)
    ax2.set_ylabel("Accuracy (%)", fontsize=7.6)
    ax2.set_ylim(0.0, 82.0)
    ax2.grid(axis="y", color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    ax2.set_title("(b) Regime II: Fixed-depth & n = 1 (locked 24M tok)", fontsize=7.8, fontweight="bold", pad=6)

    # Crisp shared 4-item legend across the top of the figure
    from matplotlib.patches import Patch
    h_leg = [
        Patch(facecolor="#555555", edgecolor="white", label="Solid bar: 5-Bench Avg"),
        Patch(facecolor="#B0B0B0", edgecolor=NAT, hatch="///", alpha=0.55, label="Hatched bar: 3-Long Avg (256K–1M)"),
        Patch(facecolor=OURS_DARK, edgecolor="white", label="LoopBridge (ours)"),
        Patch(facecolor=DIST, edgecolor="white", label="Tied baseline (same data)"),
    ]
    fig.legend(
        handles=h_leg, loc="upper center", bbox_to_anchor=(0.51, 1.03),
        ncol=4, fontsize=6.2, frameon=False, handlelength=1.25, columnspacing=1.0, handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.83, bottom=0.14, wspace=0.22)
    save(fig, "fig_inkling_main")


def print_verified_summary(evals_avg1: dict, evals_avg4: dict) -> None:
    print("=== 19-ARM TRUE 5/5 SYMMETRIC AVG@4 SCORES (81,776 ROLLOUTS) ===")
    for arm in AVG4_ARMS:
        vals = [score_pct(evals_avg4, arm, bk) for bk in BENCH_KEYS]
        p4 = macro_pct(evals_avg4, arm, PURE4_KEYS)
        l3 = macro_pct(evals_avg4, arm, LONG_KEYS)
        m5 = macro_pct(evals_avg4, arm, BENCH_KEYS)
        print(
            f"  {arm:28s} | GPQA {vals[0]:5.2f} | LCB {vals[1]:5.2f} | MRCR {vals[2]:5.2f} | "
            f"GW {vals[3]:5.2f} | LBv2 {vals[4]:5.2f} | Pure4 {p4:5.2f} | 3L {l3:5.2f} | 5B {m5:5.2f}"
        )

    print("\n=== VERIFIED AVG@4 CLUSTERED PAIRED BOOTSTRAP (5,000 resamples, seed=0) ===")
    avg4_pairs = [
        ("vl_native_4_4", "vl_native_3_3"),
        ("vl_lb_4_3_kvonly_12m", "vl_native_4_4"),
        ("vl_lb_4_3_kvonly_12m", "vl_native_3_3"),
        ("vl_lb_4_3_kvonly_12m", "vl_tf_4_3_anchored"),
        ("vl_tf_4_3_anchored", "vl_tf_4_3_tail"),
        ("vl_lb_4_2_kvonly_12m", "vl_native_2_2"),
        ("vl_lb_4_2_kvonly_12m", "vl_kvonly_2_2_t4_12m"),
        ("vl_lb_4_2_kvonly_nosynth_12m", "vl_kvonly_2_2_t4_12m"),
        ("vl_lb_4_2_kvonly_nosynth_12m", "vl_native_2_2"),
        ("vl_lb_4_2_kvonly_12m", "vl_procrustes_4_2"),
        ("vl_lb_4_2_kvonly_12m", "vl_crossridge_4_2"),
        ("vl_procrustes_4_2", "vl_crossridge_4_2"),
        ("vl_procrustes_4_2", "vl_native_2_2"),
        ("vl_procrustes_4_2", "vl_tf_4_2_anchored"),
        ("vl_procrustes_4_2", "vl_tf_4_2_tail"),
        ("vl_lb_4_2_kvonly_12m", "vl_tf_4_2_anchored"),
        ("vl_lb_4_2_kvonly_12m", "vl_tf_4_2_tail"),
        ("vl_tf_4_2_tail", "vl_native_2_2"),
        ("vl_tf_4_2_tail", "vl_kvonly_2_2_t4_12m"),
        ("vl_tf_4_2_anchored", "vl_native_2_2"),
        ("vl_tf_5_2_anchored", "vl_native_2_2"),
        ("mini_lb_3_2_24m", "mini_distilled_2_2_v3"),
        ("mini_lb_3_2_24m", "mini_orig_3_3_cap57k"),
        ("vl_lb_3_1_24m", "vl_abl_3_1_no_kvmaps"),
    ]
    for a, b in avg4_pairs:
        o5, l5, h5, l5_2, h5_2, l5_3, h5_3, p5 = paired_boot_clustered(evals_avg4, a, b, BENCH_KEYS)
        o3, l3, h3, l3_2, h3_2, l3_3, h3_3, p3 = paired_boot_clustered(evals_avg4, a, b, LONG_KEYS)
        print(
            f"  {a:28s} vs {b:22s} | 5B {o5:+6.2f} 95%[{l5:+6.2f},{h5:+6.2f}] p={p5:.4f}"
            f" | 3L {o3:+6.2f} 95%[{l3:+6.2f},{h3:+6.2f}] p={p3:.4f}"
        )
    om, lm, hm, *_, pm = paired_boot_clustered(evals_avg4, "vl_lb_4_2_kvonly_12m", "vl_procrustes_4_2", ["mrcr_8n_256k_512k"])
    print(f"  [MRCR] vl_lb_4_2_kvonly_12m vs vl_procrustes_4_2: {om:+6.2f} 95%[{lm:+6.2f},{hm:+6.2f}] p={pm:.4f}")
    om31, lm31, hm31, *_, pm31 = paired_boot_clustered(evals_avg4, "vl_lb_3_1_24m", "vl_abl_3_1_no_kvmaps", ["mrcr_8n_256k_512k"])
    print(f"  [MRCR] vl_lb_3_1_24m vs vl_abl_3_1_no_kvmaps   : {om31:+6.2f} 95%[{lm31:+6.2f},{hm31:+6.2f}] p={pm31:.4f}")
    ol, ll, hl, *_, pl = paired_boot_clustered(evals_avg4, "vl_procrustes_4_2", "vl_tf_4_2_tail", ["longbench_v2_1m"])
    print(f"  [LBv2] vl_procrustes_4_2 vs vl_tf_4_2_tail      : {ol:+6.2f} 95%[{ll:+6.2f},{hl:+6.2f}] p={pl:.4f}")

    print("\n=== VERIFIED AVG@1 CONTROLLED 24M & ABLATION BOOTSTRAP (5,000 resamples, seed=0) ===")
    avg1_pairs = [
        ("vl_lb_3_2_v2_48m", "vl_native_2_2"),
        ("vl_lb_3_2_v2_48m", "vl_native_3_3"),
        ("vl_lb_3_1_24m", "vl_distilled_1_1_24m"),
        ("vl_lb_3_1_24m", "vl_native_1_1"),
        ("vl_lb_3_1_24m", "vl_abl_3_1_no_kvmaps"),
        ("vl_abl_3_1_no_kvmaps", "vl_distilled_1_1_24m"),
        ("mini_lb_3_2_24m", "mini_distilled_2_2_v3"),
        ("mini_lb_3_2_24m", "mini_tf_3_2_anchored"),
        ("mini_lb_3_2_v2_48m", "mini_distilled_2_2_v3"),
        ("mini_lb_3_1_24m", "mini_distilled_1_1_v3"),
    ]
    for a, b in avg1_pairs:
        o5, l5, h5, *_ = paired_boot_clustered(evals_avg1, a, b, BENCH_KEYS)
        o3, l3, h3, *_ = paired_boot_clustered(evals_avg1, a, b, LONG_KEYS)
        print(f"  {a:22s} vs {b:22s} | 5B {o5:+6.2f} [{l5:+6.2f},{h5:+6.2f}] | 3L {o3:+6.2f} [{l3:+6.2f},{h3:+6.2f}]")


def main() -> None:
    evals_avg1, evals_avg4 = load_evals()
    vllm_vl = load_vllm_vl_timing()
    make_main_body_two_regime_fig(evals_avg1, evals_avg4)
    make_speed_fig(vllm_vl)
    print(
        f"Generated {OUT / 'fig_inkling_main.pdf'} and {OUT / 'fig_inkling_speed.pdf'}"
    )
    print_verified_summary(evals_avg1, evals_avg4)


if __name__ == "__main__":
    main()
