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
]


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def load_evals() -> tuple[dict[str, dict[str, dict]], dict[str, dict[str, dict]]]:
    """Load (evals_avg1, evals_avg4) directly from archived official-exact JSONs.

    evals_avg4 holds all 14 full 5xavg@4 TML-d24-varloop Regime I arms (60,256 rollouts,
    including the 2x2 causal factorial controls and 3-way Plato's Key ablation) evaluated with 4x symmetric sampling.
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

    Returns (obs, lo95, hi95, lo975, hi975, lo9833, hi9833, p_one_sided).
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
    return (
        obs,
        boots[int(0.025 * n_boot)],
        boots[int(0.975 * n_boot)],
        boots[int(0.0125 * n_boot)],
        boots[int(0.9875 * n_boot)],
        boots[int(0.008333 * n_boot)],
        boots[int(0.991667 * n_boot)],
        p1,
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


def make_pareto_and_speed_fig(evals_avg1: dict, evals_avg4: dict, vllm_vl: dict) -> None:
    """Two-panel figure:
    (a) TML-d24-varloop multi-depth Pareto frontier (5-bench macro vs Day-0 vLLM
        CoT E2E throughput at L=512, B=16, G=1024 on 1x B300): showcases n=3
        (LB 4/3 KV-Only = 99.8% of Orig 4/4), n=2 (LB 4/2 KV-Only = +2.6* Holm-sig
        vs Orig 2/2), and n=1 (LB 3/1 = +7.8* vs Distilled 1/1).
    (b) Isolated 1x B300 pure decode speedup vs Orig 4/4 across L=512..256K,
        visually proving 0.0% KV-fold decode overhead at n=3, n=2, and n=1.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.85, 2.75), gridspec_kw={"width_ratios": [1.08, 1.02], "wspace": 0.30}
    )

    # --- Panel (a): Accuracy vs E2E throughput (L=512, B=16, G=1024) ---
    cfg_cot = (512, 16, 1024)
    e2e = {arm: vllm_vl[arm][cfg_cot]["agg_e2e_tok_s"] / 1000.0 for arm in vllm_vl}

    # Native curve: 4/4 -> 3/3 -> 2/2 (pure avg@4) + dashed extension to 1/1 (@1)
    nat4_pts = [
        ("orig_4_4", "vl_native_4_4", "Orig 4/4", (-2, 6), "left"),
        ("orig_3_3", "vl_native_3_3", "Orig 3/3", (-5, -11), "center"),
        ("orig_2_2", "vl_native_2_2", "Orig 2/2", (5, -10), "left"),
    ]
    nx4 = [e2e[t] for t, _, _, _, _ in nat4_pts]
    ny4 = [macro_pct(evals_avg4, m) for _, m, _, _, _ in nat4_pts]
    x11, y11 = e2e["orig_1_1"], macro_pct(evals_avg1, "vl_native_1_1")

    ax1.plot(nx4, ny4, "--", color=NAT_LIGHT, lw=1.1, zorder=1)
    ax1.plot([nx4[-1], x11], [ny4[-1], y11], ":", color=NAT_LIGHT, lw=1.0, zorder=1)
    ax1.scatter(nx4, ny4, s=34, color=NAT, edgecolor="white", linewidth=0.5, zorder=3, label="Original $n/n$ (avg@4)")
    ax1.scatter([x11], [y11], s=32, facecolor="white", edgecolor=NAT, linewidth=1.0, zorder=3)
    ax1.annotate("Orig 1/1 (@1)", (x11, y11), xytext=(-6, -3), textcoords="offset points", fontsize=6.2, color=NAT, ha="right", va="center")

    for (_, _, lab, (dx, dy), ha), x, y in zip(nat4_pts, nx4, ny4):
        ax1.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.4, color=NAT, ha=ha)

    # Faint horizontal ceiling reference line from Orig 4/4
    ax1.axhline(ny4[0], color=NAT_LIGHT, ls=":", lw=0.7, alpha=0.65, zorder=0)

    # Zero-training TF points at n=3 (Anchored 4/3) & n=2 (Tail 4/2)
    x_tf43, y_tf43 = e2e["tf_4_3_anchored"], macro_pct(evals_avg4, "vl_tf_4_3_anchored")
    x_tf42, y_tf42 = e2e["tf_4_2_anchored"], macro_pct(evals_avg4, "vl_tf_4_2_tail")
    ax1.scatter(
        [x_tf43, x_tf42],
        [y_tf43, y_tf42],
        s=32,
        marker="D",
        facecolor="white",
        edgecolor=NAT,
        linewidth=1.0,
        zorder=4,
        label="Training-free 4/$n$ (0 MB)",
    )

    # Distilled 1/1 (24M, @1)
    y_d11 = macro_pct(evals_avg1, "vl_distilled_1_1_24m")
    ax1.scatter([x11], [y_d11], s=38, marker="^", color=DIST, edgecolor="white", linewidth=0.5, zorder=4, label="Distilled 1/1 (24M, @1)")
    ax1.annotate(
        f"Dist. 1/1 ({y_d11:.1f})",
        (x11, y_d11),
        xytext=(-13, 0),
        textcoords="offset points",
        fontsize=6.0,
        color=DIST,
        ha="right",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.88),
        zorder=6,
    )

    # LoopBridge flagship tiers: 4/3 KV-Only, 4/2 KV-Only, 3/1 (24M)
    x_43, y_43 = e2e["lb_4_3_kvonly_12m"], macro_pct(evals_avg4, "vl_lb_4_3_kvonly_12m")
    x_42, y_42 = e2e["lb_4_2_kvonly_12m"], macro_pct(evals_avg4, "vl_lb_4_2_kvonly_12m")
    x_31, y_31 = e2e["lb_3_1_24m"], macro_pct(evals_avg1, "vl_lb_3_1_24m")

    # Vertical gain arrows
    ax1.annotate("", xy=(x_42, y_42 - 0.8), xytext=(nx4[2], ny4[2] + 0.8), arrowprops=dict(arrowstyle="->", color=OURS, lw=0.9))
    ax1.annotate("", xy=(x_31, y_31 - 0.9), xytext=(x11, y_d11 + 0.9), arrowprops=dict(arrowstyle="->", color=OURS, lw=0.9))

    ax1.scatter(
        [x_43, x_42],
        [y_43, y_42],
        s=48,
        marker="s",
        facecolor=OURS_DARK,
        edgecolor="white",
        linewidth=0.6,
        zorder=6,
        label="LB 4/$n$ KV-Only (0.02%)",
    )
    ax1.scatter(
        [x_31],
        [y_31],
        s=46,
        marker="s",
        facecolor="white",
        edgecolor=OURS,
        linewidth=1.5,
        zorder=5,
        label="LB 3/1 (24M, @1)",
    )

    rec_43 = 100.0 * y_43 / ny4[0]
    d5_42 = y_42 - ny4[2]
    ax1.annotate(
        f"LB 4/3 KV-Only ({y_43:.1f})\n{rec_43:.1f}% of 4/4",
        (x_43, y_43),
        xytext=(6, 13),
        textcoords="offset points",
        fontsize=6.3,
        fontweight="bold",
        color=OURS_DARK,
        ha="left",
        arrowprops=dict(arrowstyle="-", color=OURS_DARK, lw=0.6, shrinkA=0, shrinkB=3),
    )
    ax1.annotate(
        f"LB 4/2 KV-Only ({y_42:.1f}*)\n+{d5_42:.1f}* vs 2/2",
        (x_42, y_42),
        xytext=(15, 2),
        textcoords="offset points",
        fontsize=6.3,
        fontweight="bold",
        color=OURS_DARK,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=OURS_DARK, lw=0.6, shrinkA=0, shrinkB=4),
    )
    ax1.annotate(
        f"LB 3/1 ({y_31:.1f}*)\n+7.8* vs Dist.",
        (x_31, y_31),
        xytext=(-4, 9),
        textcoords="offset points",
        fontsize=6.2,
        fontweight="bold",
        color=OURS,
        ha="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.90),
        zorder=6,
    )

    ax1.set_xlabel("Day-0 vLLM E2E throughput (k tok/s, $L$=512, $B$=16)")
    ax1.set_ylabel("5-benchmark macro accuracy (%)")
    ax1.set_xlim(1.68, 4.72)
    ax1.set_ylim(36.5, 85.2)
    ax1.grid(axis="y", color=GRID, lw=0.6)
    ax1.set_title("(a) TML-d24-varloop Pareto frontier", fontsize=8.0, fontweight="bold", pad=6)
    ax1.legend(loc="lower left", fontsize=5.7, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.28)

    # --- Panel (b): Pure decode speedup vs Orig 4/4 across 512..256K ---
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
    ax2.plot(xs, speedups("orig_5_5"), ":", color=NAT_LIGHT, marker="v", ms=3.3, lw=0.9, label="Orig 5/5 (0.83x)")
    ax2.plot(xs, speedups("orig_3_3"), "--", color=NAT, marker="o", ms=3.5, lw=1.0, label="Orig 3/3")
    ax2.plot(xs, speedups("lb_4_3_kvonly_12m"), "-", color=OURS_LIGHT, marker="s", mfc="white", mec=OURS_DARK, mew=1.1, ms=4.5, lw=1.4, label="LB 4/3 KV-Only")
    ax2.plot(xs, speedups("orig_2_2"), "--", color="#707070", marker="o", ms=3.5, lw=1.0, label="Orig 2/2")
    ax2.plot(xs, speedups("lb_4_2_kvonly_12m"), "-", color=OURS, marker="s", ms=4.3, lw=1.5, label="LB 4/2 KV-Only")
    ax2.plot(xs, speedups("orig_1_1"), "--", color=DIST, marker="^", ms=3.8, lw=1.0, label="Orig 1/1")
    ax2.plot(xs, speedups("lb_3_1_24m"), "-", color=OURS_DARK, marker="s", ms=4.0, lw=1.4, label="LB 3/1 ($r$=64)")

    for arm, col, dy in [
        ("lb_3_1_24m", OURS_DARK, 5),
        ("lb_4_2_kvonly_12m", OURS, 5),
        ("lb_4_3_kvonly_12m", OURS_DARK, 5),
    ]:
        val = speedups(arm)[-1]
        ax2.annotate(f"{val:.2f}x", (xs[-1], val), xytext=(-2, dy), textcoords="offset points", fontsize=6.7, fontweight="bold", color=col, ha="center")

    ax2.set_xticks(xs)
    ax2.set_xticklabels([lab for _, lab in cfgs], fontsize=6.8)
    ax2.set_ylabel("Pure decode speedup vs. Orig 4/4")
    ax2.set_ylim(0.68, 3.32)
    ax2.grid(axis="y", color=GRID, lw=0.6)
    ax2.set_title("(b) Isolated 1x B300 decode speedup", fontsize=8.0, fontweight="bold", pad=6)
    ax2.legend(loc="upper left", ncol=2, fontsize=5.7, columnspacing=0.6, handletextpad=0.3, handlelength=1.4, labelspacing=0.25)

    save(fig, "fig_inkling_pareto")


def make_ablate_and_regime_fig(evals_avg1: dict, evals_avg4: dict) -> None:
    """Two-panel figure proving the Two-Regime Law on 26.4B sparse-MoEs:
    (a) Regime I (variable-depth TML-d24-varloop, n >= 2, symmetric avg@4,
        60,256 rollouts across 14 arms): n=2 (1.75x) and n=3 (1.27x) ladders comparing Orig n/n,
        matched KV-Only 2/2, zero-training Tail/Anchored, and pure-prompt KV-Only (17-25 MiB, r=0).
    (b) Regime II (fixed-depth Inkling-2-Mini 3/2 & un-gated VarLoop 3/1, avg@1):
        training-free collapse (Mini TF 3/2 = 2.2%, 93% trunc; VarLoop TF 3/1 = 40.8%)
        rescued by parametric LoopBridge, sweeping tied Distilled 5/5 on both backbones.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.95, 2.75), gridspec_kw={"width_ratios": [1.06, 1.04], "wspace": 0.25}
    )

    # --- Panel (a): Regime I symmetric avg@4 ladders + 2x2 causal dissociation ---
    n2_arms = [
        ("vl_native_2_2", "Orig\n2/2", NAT_LIGHT),
        ("vl_kvonly_2_2_t4_12m", "KV\n2/2", DIST),
        ("vl_tf_4_2_tail", "TF\nTail", "#9A9A9A"),
        ("vl_lb_4_2_kvonly_nosynth_12m", "4/2\n0%S", OURS_LIGHT),
        ("vl_lb_4_2_kvonly_12m", "4/2\nKV", OURS_DARK),
    ]
    n3_arms = [
        ("vl_native_3_3", "Orig\n3/3", NAT_LIGHT),
        ("vl_tf_4_3_anchored", "TF\nAnch.", "#9A9A9A"),
        ("vl_lb_4_3_kvonly_12m", "4/3\nKV", OURS_DARK),
    ]
    pos1 = [0.0, 1.06, 2.12, 3.18, 4.24, 5.66, 6.72, 7.78]
    all_a = n2_arms + n3_arms
    w = 0.35
    m5_a = [macro_pct(evals_avg4, a, BENCH_KEYS) for a, _, _ in all_a]
    m3_a = [macro_pct(evals_avg4, a, LONG_KEYS) for a, _, _ in all_a]
    cols_a = [c for _, _, c in all_a]

    bars5_a = ax1.bar([p - w / 2 for p in pos1], m5_a, width=w, color=cols_a, edgecolor="white", lw=0.4, label="5-Bench Macro")
    bars3_a = ax1.bar([p + w / 2 for p in pos1], m3_a, width=w, color=cols_a, alpha=0.42, edgecolor=NAT, lw=0.55, hatch="//", label="3-Long (256K-1M)")

    ceil44 = macro_pct(evals_avg4, "vl_native_4_4", BENCH_KEYS)
    ax1.axhline(ceil44, color=NAT, ls=":", lw=0.9, zorder=1)
    ax1.text(-0.35, ceil44 + 0.45, f"Orig 4/4 ceiling ({ceil44:.1f}%)", fontsize=5.7, color=NAT, ha="left", va="bottom")

    for i, bar in enumerate(bars5_a):
        h = bar.get_height()
        suf = "*" if i in (2, 3, 4) else ""
        dx = -2.2 if i == 2 else (0.5 if i == 3 else (-1.2 if i not in (6, 7) else (-2.0 if i == 6 else 1.2)))
        dy = 2.0 if i not in (6, 7) else 2.2
        ax1.annotate(
            f"{h:.1f}{suf}",
            (bar.get_x() + bar.get_width() / 2, h),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.0,
            fontweight="bold" if i in (3, 4, 7) else "normal",
            color=OURS_DARK if i in (3, 4, 7) else INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.35, alpha=0.92) if i in (6, 7) else None,
            zorder=7,
        )
    for i, bar in enumerate(bars3_a):
        h = bar.get_height()
        suf = "*" if i in (2, 3, 4) else ""
        ax1.annotate(
            f"{h:.1f}{suf}",
            (bar.get_x() + bar.get_width() / 2, h),
            xytext=(2.2, 1.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=4.5,
            color=OURS_DARK if i in (3, 4, 7) else NAT,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.15, alpha=0.80),
            zorder=5,
        )

    ax1.axvline(4.95, color=GRID, lw=1.0)
    ax1.text(2.12, 77.6, "$n=2$ 2x2 causal (1.75x)", fontsize=6.1, fontweight="bold", color=INK, ha="center")
    ax1.text(6.72, 77.6, "$n=3$ (1.27x)", fontsize=6.1, fontweight="bold", color=INK, ha="center")

    ax1.set_xticks(pos1)
    ax1.set_xticklabels([lab for _, lab, _ in all_a], fontsize=5.9)
    ax1.set_ylabel("Macro accuracy (%, avg@4)")
    ax1.set_ylim(56.0, 81.8)
    ax1.grid(axis="y", color=GRID, lw=0.6)
    ax1.set_title("(a) Regime I: varloop $n\\geq 2$ (avg@4)", fontsize=8.0, fontweight="bold", pad=6)
    ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 1.00), fontsize=5.6, handletextpad=0.35, ncol=2)

    # --- Panel (b): Regime II TF collapse & 24M controlled 5/5 sweeps ---
    mini_arms = [
        ("mini_tf_3_2_anchored", "TF 3/2\nAnch.", NAT_LIGHT),
        ("mini_distilled_2_2_v3", "Dist.\n2/2", DIST),
        ("mini_lb_3_2_24m", "LB\n3/2", OURS),
        ("mini_lb_3_2_v2_48m", "LB v2\n3/2", OURS_DARK),
    ]
    vl1_arms = [
        ("vl_native_1_1", "Orig\n1/1", NAT_LIGHT),
        ("vl_tf_3_1", "TF\n3/1", "#B5B5B5"),
        ("vl_distilled_1_1_24m", "Dist.\n1/1", DIST),
        ("vl_lb_3_1_24m", "LB\n3/1", OURS_DARK),
    ]
    pos2 = [0.0, 1.05, 2.10, 3.15, 4.65, 5.70, 6.75, 7.80]
    all_b = mini_arms + vl1_arms
    m5_b = [macro_pct(evals_avg1, a, BENCH_KEYS) for a, _, _ in all_b]
    m3_b = [macro_pct(evals_avg1, a, LONG_KEYS) for a, _, _ in all_b]
    cols_b = [c for _, _, c in all_b]

    bars5_b = ax2.bar([p - w / 2 for p in pos2], m5_b, width=w, color=cols_b, edgecolor="white", lw=0.4)
    bars3_b = ax2.bar([p + w / 2 for p in pos2], m3_b, width=w, color=cols_b, alpha=0.42, edgecolor=NAT, lw=0.55, hatch="//")

    for i, bar in enumerate(bars5_b):
        h = bar.get_height()
        suf = "*" if i in (2, 7) else ""
        ax2.annotate(
            f"{h:.1f}{suf}",
            (bar.get_x() + bar.get_width() / 2, h),
            xytext=(-1.0, 1.8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.4,
            fontweight="bold" if i in (2, 3, 7) else "normal",
            color=OURS_DARK if i in (2, 3, 7) else INK,
        )
    for i, bar in enumerate(bars3_b):
        h = bar.get_height()
        ax2.annotate(
            f"{h:.1f}",
            (bar.get_x() + bar.get_width() / 2, h),
            xytext=(2.6, 1.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=4.9,
            color=OURS_DARK if i in (2, 3, 7) else NAT,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.15, alpha=0.78),
            zorder=5,
        )

    ax2.annotate(
        "93%\ntrunc",
        (0.0, 5.0),
        xytext=(-0.05, 19.5),
        fontsize=5.6,
        color="#A02020",
        ha="center",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#A02020", lw=0.7),
    )

    ax2.axvline(3.90, color=GRID, lw=1.0)
    ax2.text(1.58, 65.5, "Inkling-2-Mini (fixed-3)", fontsize=6.3, fontweight="bold", color=INK, ha="center")
    ax2.text(6.22, 65.5, "TML-varloop ($n=1$)", fontsize=6.3, fontweight="bold", color=INK, ha="center")

    ax2.set_xticks(pos2)
    ax2.set_xticklabels([lab for _, lab, _ in all_b], fontsize=5.9)
    ax2.set_ylabel("Macro accuracy (%, avg@1)")
    ax2.set_ylim(0, 71.0)
    ax2.grid(axis="y", color=GRID, lw=0.6)
    ax2.set_title("(b) Regime II: TF collapse & 24M sweeps", fontsize=8.0, fontweight="bold", pad=6)

    save(fig, "fig_inkling_ablate")


def make_main_body_triptych_fig(evals_avg1: dict, evals_avg4: dict, vllm_vl: dict) -> None:
    """Spacious 3-panel Main-Body hero figure (replaces dense Table 3 in Section 5.8
    and serves as the lead 26.4B visual in README.md).

    Presents BOTH Thinking Machines 26.4B sparse-MoE looped models cleanly at a glance:
      (a) TML-d24-varloop Pareto frontier (5-Bench Macro vs Day-0 vLLM CoT E2E tok/s),
      (b) TML-d24-varloop Regime I ladders (n=2 at 1.75x & n=3 at 1.27x; Orig vs TF vs LB KV-Only),
      (c) Regime II collapse & locked-24M rescue across Inkling-2-Mini (fixed-3, 3/2) &
          TML-d24-varloop (n=1, 3/1) with 5-Bench & 3-Long callouts.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(7.30, 2.68), gridspec_kw={"width_ratios": [1.22, 1.00, 1.05], "wspace": 0.34}
    )

    # =========================================================================
    # Panel (a): TML-d24-varloop Pareto frontier (5-Bench Macro vs vLLM E2E)
    # =========================================================================
    cfg_cot = (512, 16, 1024)
    e2e = {arm: vllm_vl[arm][cfg_cot]["agg_e2e_tok_s"] / 1000.0 for arm in vllm_vl}

    nat4_pts = [
        ("orig_4_4", "vl_native_4_4", "Orig 4/4", (0, 6), "left"),
        ("orig_3_3", "vl_native_3_3", "Orig 3/3", (-4, -10.5), "center"),
        ("orig_2_2", "vl_native_2_2", "Orig 2/2", (5, -9.5), "left"),
    ]
    nx4 = [e2e[t] for t, _, _, _, _ in nat4_pts]
    ny4 = [macro_pct(evals_avg4, m) for _, m, _, _, _ in nat4_pts]
    x11, y11 = e2e["orig_1_1"], macro_pct(evals_avg1, "vl_native_1_1")
    y_d11 = macro_pct(evals_avg1, "vl_distilled_1_1_24m")

    ax1.plot(nx4, ny4, "--", color=NAT_LIGHT, lw=1.05, zorder=1)
    ax1.plot([nx4[-1], x11], [ny4[-1], y11], ":", color=NAT_LIGHT, lw=0.95, zorder=1)
    ax1.axhline(ny4[0], color=NAT_LIGHT, ls=":", lw=0.65, alpha=0.6, zorder=0)

    ax1.scatter(nx4, ny4, s=28, color=NAT, edgecolor="white", lw=0.5, zorder=3, label="Original $n/n$")
    ax1.scatter([x11], [y11], s=26, facecolor="white", edgecolor=NAT, lw=0.9, zorder=3)
    ax1.annotate("Orig 1/1", (x11, y11), xytext=(-5, -2.5), textcoords="offset points", fontsize=5.6, color=NAT, ha="right", va="center")
    for (_, _, lab, (dx, dy), ha), x, y in zip(nat4_pts, nx4, ny4):
        ax1.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=5.8, color=NAT, ha=ha)

    # Zero-training TF points (high zorder so hollow diamonds stay visible next to LB squares)
    x_tf43, y_tf43 = e2e["tf_4_3_anchored"], macro_pct(evals_avg4, "vl_tf_4_3_anchored")
    x_tf42, y_tf42 = e2e["tf_4_2_anchored"], macro_pct(evals_avg4, "vl_tf_4_2_tail")
    ax1.scatter([x_tf43, x_tf42], [y_tf43, y_tf42], s=26, marker="D", facecolor="none", edgecolor=INK, lw=1.0, zorder=7, label="Training-free 4/$n$")

    # Distilled 1/1
    ax1.scatter([x11], [y_d11], s=32, marker="^", color=DIST, edgecolor="white", lw=0.45, zorder=4, label="Distilled 1/1")
    ax1.annotate(
        "Dist. 1/1",
        (x11, y_d11),
        xytext=(-11, 0),
        textcoords="offset points",
        fontsize=5.5,
        color=DIST,
        ha="right",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.3, alpha=0.88),
        zorder=6,
    )

    # LoopBridge frontier
    x_43, y_43 = e2e["lb_4_3_kvonly_12m"], macro_pct(evals_avg4, "vl_lb_4_3_kvonly_12m")
    x_42, y_42 = e2e["lb_4_2_kvonly_12m"], macro_pct(evals_avg4, "vl_lb_4_2_kvonly_12m")
    x_31, y_31 = e2e["lb_3_1_24m"], macro_pct(evals_avg1, "vl_lb_3_1_24m")

    ax1.annotate("", xy=(x_42, y_42 - 0.9), xytext=(nx4[2], ny4[2] + 0.9), arrowprops=dict(arrowstyle="->", color=OURS, lw=0.85))
    ax1.annotate("", xy=(x_31, y_31 - 1.1), xytext=(x11, y_d11 + 1.1), arrowprops=dict(arrowstyle="->", color=OURS, lw=0.85))

    ax1.scatter([x_43, x_42], [y_43, y_42], s=40, marker="s", facecolor=OURS_DARK, edgecolor="white", lw=0.5, zorder=6, label="LB 4/$n$ KV (0.02%)")
    ax1.scatter([x_31], [y_31], s=38, marker="s", facecolor="white", edgecolor=OURS, lw=1.35, zorder=5, label="LB 3/1 (24M)")

    ax1.annotate(
        f"LB 4/3 KV ({y_43:.1f})\n1.27x decode",
        (x_43, y_43),
        xytext=(6, 11),
        textcoords="offset points",
        fontsize=5.6,
        fontweight="bold",
        color=OURS_DARK,
        ha="left",
        arrowprops=dict(arrowstyle="-", color=OURS_DARK, lw=0.55, shrinkA=0, shrinkB=3),
    )
    ax1.annotate(
        f"LB 4/2 KV ({y_42:.1f}*)\n1.75x decode",
        (x_42, y_42),
        xytext=(7, 5),
        textcoords="offset points",
        fontsize=5.6,
        fontweight="bold",
        color=OURS_DARK,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=OURS_DARK, lw=0.55, shrinkA=0, shrinkB=3),
    )
    ax1.annotate(
        f"LB 3/1 ({y_31:.1f}*)\n2.76x decode",
        (x_31, y_31),
        xytext=(-3, 9),
        textcoords="offset points",
        fontsize=5.5,
        fontweight="bold",
        color=OURS,
        ha="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.45, alpha=0.90),
        zorder=6,
    )

    ax1.set_xlabel("Day-0 vLLM E2E throughput (k tok/s)", fontsize=7.2)
    ax1.set_ylabel("5-bench macro accuracy (%)", fontsize=7.2)
    ax1.set_xlim(1.65, 4.92)
    ax1.set_ylim(36.0, 86.5)
    ax1.grid(axis="y", color=GRID, lw=0.55)
    ax1.set_title("(a) TML-varloop Pareto (vLLM)", fontsize=7.5, fontweight="bold", pad=6)
    ax1.legend(loc="lower left", fontsize=5.1, handletextpad=0.25, borderaxespad=0.15, labelspacing=0.22)

    # =========================================================================
    # Panel (b): TML-d24-varloop Regime I + 2x2 Causal Double Dissociation (avg@4)
    # Restores zero-training TF 4/2 Tail (68.9%) right next to 4/2 0%S & 4/2 KV!
    # =========================================================================
    g_n2 = [
        ("vl_native_2_2", "Orig\n2/2", NAT_LIGHT, False),
        ("vl_kvonly_2_2_t4_12m", "KV\n2/2", DIST, False),
        ("vl_tf_4_2_tail", "TF\nTail", "#B8B8B8", False),
        ("vl_lb_4_2_kvonly_nosynth_12m", "4/2\n0%S", OURS_LIGHT, True),
        ("vl_lb_4_2_kvonly_12m", "4/2\nKV", OURS_DARK, True),
    ]
    g_n3 = [
        ("vl_native_3_3", "Orig\n3/3", NAT_LIGHT, False),
        ("vl_tf_4_3_anchored", "TF\nAnch.", "#B8B8B8", False),
        ("vl_lb_4_3_kvonly_12m", "4/3\nKV", OURS_DARK, True),
    ]
    pos_b = [0.0, 0.86, 1.72, 2.58, 3.44, 4.66, 5.52, 6.38]
    all_b = g_n2 + g_n3
    vals_5b = [macro_pct(evals_avg4, a, BENCH_KEYS) for a, _, _, _ in all_b]
    vals_3l = [macro_pct(evals_avg4, a, LONG_KEYS) for a, _, _, _ in all_b]
    cols_b = [c for _, _, c, _ in all_b]

    bars_b = ax2.bar(pos_b, vals_5b, width=0.58, color=cols_b, edgecolor="white", lw=0.45, zorder=2, label="Bars: 5-Bench macro")
    ax2.scatter(pos_b, vals_3l, s=21, marker="D", facecolor="white", edgecolor=INK, lw=0.85, zorder=5, label="Diamonds: 3-Long (256K-1M)")
    ax2.plot(pos_b[:5], vals_3l[:5], "-", color=INK, lw=0.6, alpha=0.55, zorder=4)
    ax2.plot(pos_b[5:], vals_3l[5:], "-", color=INK, lw=0.6, alpha=0.55, zorder=4)

    ceil44 = macro_pct(evals_avg4, "vl_native_4_4", BENCH_KEYS)
    ax2.axhline(ceil44, color=NAT, ls=":", lw=0.85, zorder=1)
    ax2.text(-0.35, ceil44 + 0.35, f"Orig 4/4 ({ceil44:.1f}%)", fontsize=5.0, color=NAT, ha="left", va="bottom")

    for i, (bar, v5, v3, (_, _, _, is_ours)) in enumerate(zip(bars_b, vals_5b, vals_3l, all_b)):
        suf = "*" if i in (2, 3, 4) else ""
        is_dark = i in (4, 7)
        dx = -2.2 if i == 2 else (1.5 if i == 3 else (-1.8 if i == 6 else (2.2 if i == 7 else 0.0)))
        dy = 3.2 if i == 2 else (2.0 if i not in (6, 7) else (2.2 if i == 6 else 4.2))
        ax2.annotate(
            f"{v5:.1f}{suf}",
            (bar.get_x() + bar.get_width() / 2, v5),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=4.4,
            fontweight="bold" if is_ours else "normal",
            color=OURS_DARK if is_ours else INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.25, alpha=0.92) if i in (2, 3, 6, 7) else None,
            zorder=7,
        )
        ax2.annotate(
            f"{v3:.1f}",
            (pos_b[i], v3),
            xytext=(0, -7.8),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=4.0,
            fontweight="bold" if is_ours else "normal",
            color="white" if (is_dark or i == 1) else INK,
        )

    d5_n2 = vals_5b[4] - vals_5b[1]
    d3_n2 = vals_3l[4] - vals_3l[1]
    rec_n3 = 100.0 * vals_5b[7] / ceil44
    ax2.axvline(4.05, color=GRID, lw=0.9)
    ax2.text(1.72, 80.7, f"$n=2$ causal (1.75x)\n+{d5_n2:.1f}* 5B / +{d3_n2:.1f}* 3L", fontsize=4.75, fontweight="bold", color=OURS_DARK, ha="center", va="top")
    ax2.text(5.52, 80.7, f"$n=3$ (1.27x)\n{rec_n3:.1f}% of 4/4", fontsize=4.8, fontweight="bold", color=OURS_DARK, ha="center", va="top")

    ax2.set_xticks(pos_b)
    ax2.set_xticklabels([lab for _, lab, _, _ in all_b], fontsize=4.7)
    ax2.set_ylim(58.5, 81.8)
    ax2.grid(axis="y", color=GRID, lw=0.55)
    ax2.set_title("(b) Regime I & 2x2 causal ($n\\geq 2$)", fontsize=7.4, fontweight="bold", pad=6)

    # =========================================================================
    # Panel (c): Regime II — Inkling-2-Mini (fixed-3) & TML-varloop (n=1)
    # Strictly locked 24M matched controls (no uncontrolled v2 rows in main!)
    # =========================================================================
    mini_c = [
        ("mini_tf_3_2_anchored", "TF\n3/2", NAT_LIGHT, False),
        ("mini_distilled_2_2_v3", "Dist.\n2/2", DIST, False),
        ("mini_lb_3_2_24m", "LB\n3/2", OURS_DARK, True),
    ]
    vl1_c = [
        ("vl_native_1_1", "Orig\n1/1", NAT_LIGHT, False),
        ("vl_distilled_1_1_24m", "Dist.\n1/1", DIST, False),
        ("vl_lb_3_1_24m", "LB\n3/1", OURS_DARK, True),
    ]
    pos_c = [0.0, 0.96, 1.92, 3.36, 4.32, 5.28]
    all_c = mini_c + vl1_c
    c_5b = [macro_pct(evals_avg1, a, BENCH_KEYS) for a, _, _, _ in all_c]
    c_3l = [macro_pct(evals_avg1, a, LONG_KEYS) for a, _, _, _ in all_c]
    cols_c = [c for _, _, c, _ in all_c]

    bars_c = ax3.bar(pos_c, c_5b, width=0.62, color=cols_c, edgecolor="white", lw=0.5, zorder=2)
    ax3.scatter(pos_c, c_3l, s=26, marker="D", facecolor="white", edgecolor=INK, lw=0.95, zorder=5)
    ax3.plot(pos_c[:3], c_3l[:3], "-", color=INK, lw=0.7, alpha=0.55, zorder=4)
    ax3.plot(pos_c[3:], c_3l[3:], "-", color=INK, lw=0.7, alpha=0.55, zorder=4)

    # Inkling-2-Mini Orig 3/3 teacher ceiling line (left half only!)
    mini_ceil = macro_pct(evals_avg1, "mini_orig_3_3", BENCH_KEYS)
    ax3.hlines(mini_ceil, -0.40, 2.45, colors=NAT, linestyles=":", lw=0.85, zorder=1)
    ax3.text(0.96, mini_ceil + 1.2, f"Orig 3/3 ({mini_ceil:.1f}%)", fontsize=5.2, color=NAT, ha="center", va="bottom")

    for i, (bar, v5, v3, (_, _, _, is_ours)) in enumerate(zip(bars_c, c_5b, c_3l, all_c)):
        suf = "*" if is_ours else ""
        ax3.annotate(
            f"{v5:.1f}{suf}",
            (bar.get_x() + bar.get_width() / 2, v5),
            xytext=(0, 2.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.6,
            fontweight="bold" if is_ours else "normal",
            color=OURS_DARK if is_ours else INK,
        )
        if i != 0:
            ax3.annotate(
                f"{v3:.1f}",
                (pos_c[i], v3),
                xytext=(0, -8.5),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=4.4,
                fontweight="bold" if is_ours else "normal",
                color="white" if is_ours or i in (1, 4) else INK,
            )

    ax3.annotate(
        "93%\ntrunc",
        (0.0, 5.2),
        xytext=(0.0, 23.0),
        fontsize=5.3,
        color="#A02020",
        ha="center",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#A02020", lw=0.7),
    )

    ax3.axvline(2.64, color=GRID, lw=0.9)
    ax3.text(0.88, 82.8, "Inkling-2-Mini 3/2\n+5.1* / +7.7* 3L", fontsize=5.0, fontweight="bold", color=OURS_DARK, ha="center", va="top")
    ax3.text(4.40, 82.8, "TML-varloop 3/1\n+7.8* / +11.7* 3L", fontsize=5.0, fontweight="bold", color=OURS_DARK, ha="center", va="top")

    ax3.set_xticks(pos_c)
    ax3.set_xticklabels([lab for _, lab, _, _ in all_c], fontsize=5.6)
    ax3.set_ylim(0.0, 85.0)
    ax3.grid(axis="y", color=GRID, lw=0.55)
    ax3.set_title("(c) Regime II: Mini & $n=1$ (24M)", fontsize=7.5, fontweight="bold", pad=6)

    # Clean shared legend centered cleanly beneath Panels (b) & (c) — zero bar overlap!
    handles, labels = ax2.get_legend_handles_labels()
    # Order: Bars first, Diamonds second
    order = sorted(range(len(labels)), key=lambda idx: 0 if labels[idx].startswith("Bars") else 1)
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower center",
        bbox_to_anchor=(0.665, -0.055),
        ncol=2,
        fontsize=6.0,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.2,
    )

    save(fig, "fig_inkling_main")


def print_verified_summary(evals_avg1: dict, evals_avg4: dict) -> None:
    print("=== 14-ARM REGIME I TRUE 5/5 SYMMETRIC AVG@4 SCORES (60,256 ROLLOUTS) ===")
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
    make_main_body_triptych_fig(evals_avg1, evals_avg4, vllm_vl)
    make_pareto_and_speed_fig(evals_avg1, evals_avg4, vllm_vl)
    make_ablate_and_regime_fig(evals_avg1, evals_avg4)
    print(
        f"Generated {OUT / 'fig_inkling_main.pdf'}, {OUT / 'fig_inkling_pareto.pdf'}, "
        f"and {OUT / 'fig_inkling_ablate.pdf'}"
    )
    print_verified_summary(evals_avg1, evals_avg4)


if __name__ == "__main__":
    main()
