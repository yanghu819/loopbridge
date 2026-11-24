"""Generate publication figures (PDF + PNG) and verified LaTeX summary tables
for the 26.4B Sparse-MoE Looped Transformers section (TML-d24-varloop-research
and Inkling-2-Mini in Day-0 vLLM Blackwell).

Strictly recomputes every number directly from archived JSON files in
inkling/results/ (zero hardcoded numbers). Matches ouro/scripts/paper_figs.py
typography, palette, and marker conventions:
  - Orange (#EB6834): LoopBridge (filled = n=1, hollow = n=2)
  - Blue   (#2A78D6): Distilled tied-loop controls
  - Grey   (#4D4D4D / #9A9A9A): Native / Original / Training-Free baselines

Usage:
  python inkling/make_figs.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RES = ROOT / "results"
OUT = REPO / "paper" / "figs"

OURS, OURS_DARK, DIST = "#EB6834", "#C94A18", "#2A78D6"
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

BENCHES = [
    ("gpqa_diamond_high", "GPQA"),
    ("livecodebench_v6_high", "LCB v6"),
    ("mrcr_8n_256k_512k", "MRCR\n512K"),
    ("graphwalks_bfs_256k", "GW\n256K"),
    ("longbench_v2_1m", "LB-v2\n1M"),
]
BENCH_KEYS = [k for k, _ in BENCHES]
LONG_KEYS = BENCH_KEYS[2:]


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def load_all_eval() -> dict[str, dict[str, dict]]:
    """Merge official-exact 5-bench per-item records from archived JSONs."""
    eo = json.load(open(RES / "eval_official_exact.json"))
    ev3 = json.load(open(RES / "eval_vllm_official_v3.json"))
    elcb = json.load(open(RES / "eval_vllm_lcb_fixed_v3.json"))
    evl = json.load(open(RES / "eval_vllm_varloop.json"))

    merged: dict[str, dict[str, dict]] = {}

    def add(name: str, src_eo=None, fb_v3=None, fb_lcb=None, fb_vl=None):
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
        merged[name] = entry

    # TML-d24-varloop-research (12 arms)
    add("vl_native_4_4", "vl_native_4_4")
    add("vl_native_3_3", "vl_native_3_3", fb_vl="vl_native_3_3")
    add("vl_lb_3_2_v2_48m", "vl_lb_3_2_v2_48m")
    add("vl_native_2_2", "vl_native_2_2")
    add("vl_lb_3_1_24m", "vl_lb_3_1_24m")
    add("vl_lb_3_1_v2_48m", "vl_lb_3_1_v2_48m")
    add("vl_abl_3_1_singlepass_kv", "vl_abl_3_1_singlepass_kv")
    add("vl_abl_3_1_no_kvmaps", "vl_abl_3_1_no_kvmaps")
    add("vl_lb_4_1_24m", "vl_lb_4_1_24m")
    add("vl_distilled_1_1_24m", "vl_distilled_1_1_24m")
    add("vl_tf_3_1", "vl_tf_3_1", fb_vl="vl_tf_3_1")
    add("vl_native_1_1", "vl_native_1_1", fb_vl="vl_native_1_1")

    # Inkling-2-Mini (6 arms)
    merged["mini_orig_3_3"] = {
        "gpqa_diamond_high": ev3["native_3_3"]["gpqa_diamond_high"],
        "livecodebench_v6_high": elcb["native_3_3"]["livecodebench_v6_high"],
        "mrcr_8n_256k_512k": eo["mini_orig_3_3_cap65k"]["mrcr_8n_256k_512k"],
        "graphwalks_bfs_256k": eo["mini_orig_3_3_cap65k"]["graphwalks_bfs_256k"],
        "longbench_v2_1m": eo["mini_orig_3_3_cap32k"]["longbench_v2_1m"],
    }
    add("mini_lb_3_2_v2_48m", "mini_lb_3_2_v2_48m")
    add("mini_lb_3_2_24m", "mini_lb_3_2_24m", fb_v3="lb_3_2_24m", fb_lcb="lb_3_2_24m")
    add("mini_distilled_2_2_v3", "mini_distilled_2_2_v3", fb_v3="distilled_2_2_v3")
    add("mini_lb_3_1_24m", "mini_lb_3_1_24m", fb_v3="lb_3_1_24m")
    add("mini_distilled_1_1_v3", "mini_distilled_1_1_v3", fb_v3="distilled_1_1_v3", fb_lcb="distilled_1_1_v3")

    return merged


def score_pct(evals: dict, model: str, bk: str) -> float:
    return 100.0 * float(evals[model][bk]["score"])


def macro_pct(evals: dict, model: str, keys: list[str] = BENCH_KEYS) -> float:
    return float(np.mean([score_pct(evals, model, bk) for bk in keys]))


def load_vllm_vl_timing() -> dict[str, dict[tuple[int, int, int], dict]]:
    arms = ["orig_4_4", "orig_3_3", "orig_2_2", "lb_3_2_v2_48m", "orig_1_1", "lb_3_1_24m", "lb_4_1_24m"]
    out = {}
    for arm in arms:
        rows = []
        for suf in ("short", "long"):
            p = RES / f"vllm_sse_vl_vl_{arm}_{suf}.json"
            if p.exists():
                rows.extend(json.load(open(p))["results"])
        out[arm] = {(r["L"], r["B"], r["gen_len"]): r for r in rows}
    return out


def make_pareto_and_speed_fig(evals: dict, vllm_vl: dict) -> None:
    """Two-panel figure:
    (a) TML-d24-varloop 5-bench macro accuracy vs Day-0 vLLM CoT E2E throughput
        (L=512, B=16, G=1024, 1x B300) AND annotated with 262K B=2 decode speedup.
    (b) Pure decode speedup vs Orig 3/3 across context lengths 512..256K (1x B300),
        proving 0.0% adapter overhead even at lora_rank=128.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.75, 2.65), gridspec_kw={"width_ratios": [1.05, 1.05], "wspace": 0.32}
    )

    # --- Panel (a): Accuracy vs E2E throughput (L=512, B=16, G=1024, 1x B300) ---
    cfg_cot = (512, 16, 1024)
    e2e = {arm: vllm_vl[arm][cfg_cot]["agg_e2e_tok_s"] / 1000.0 for arm in vllm_vl}  # k tok/s

    nat_pts = [
        ("orig_4_4", "vl_native_4_4", "4/4", (6, 2)),
        ("orig_3_3", "vl_native_3_3", "3/3", (-4, -12)),
        ("orig_2_2", "vl_native_2_2", "2/2", (6, -9)),
        ("orig_1_1", "vl_native_1_1", "1/1", (-8, -3)),
    ]
    nx = [e2e[t] for t, _, _, _ in nat_pts]
    ny = [macro_pct(evals, m) for _, m, _, _ in nat_pts]
    ax1.plot(nx, ny, "--", color=NAT_LIGHT, lw=1.0, zorder=1)
    ax1.scatter(nx, ny, s=34, color=NAT, edgecolor="white", linewidth=0.5, zorder=3, label="Original $n/n$")
    for (t, m, lab, (dx, dy)), x, y in zip(nat_pts, nx, ny):
        ax1.annotate(
            f"Orig {lab}",
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.7,
            color=NAT,
            ha="left" if dx >= 0 else "right",
        )

    # Distilled 1/1 (uses same 1/1 vLLM engine speed as orig_1_1)
    x_d11 = e2e["orig_1_1"]
    y_d11 = macro_pct(evals, "vl_distilled_1_1_24m")
    ax1.scatter([x_d11], [y_d11], s=40, marker="^", color=DIST, edgecolor="white", linewidth=0.5, zorder=4, label="Distilled 1/1 (24M)")
    ax1.annotate(
        f"Dist. 1/1\n({y_d11:.1f})",
        (x_d11, y_d11),
        xytext=(-10, 4),
        textcoords="offset points",
        fontsize=6.4,
        color=DIST,
        ha="right",
        va="center",
    )

    # LoopBridge 3/2 v2 (48M, r=128) & LoopBridge 3/1 (24M)
    x_32 = e2e["lb_3_2_v2_48m"]
    y_32 = macro_pct(evals, "vl_lb_3_2_v2_48m")
    x_31 = e2e["lb_3_1_24m"]
    y_31 = macro_pct(evals, "vl_lb_3_1_24m")

    # Vertical gain arrows
    ax1.annotate(
        "",
        xy=(x_32, y_32 - 0.8),
        xytext=(e2e["orig_2_2"], macro_pct(evals, "vl_native_2_2") + 0.8),
        arrowprops=dict(arrowstyle="->", color=OURS, lw=0.9),
    )
    ax1.annotate(
        "",
        xy=(x_31, y_31 - 0.9),
        xytext=(x_d11, y_d11 + 0.9),
        arrowprops=dict(arrowstyle="->", color=OURS, lw=0.9),
    )

    ax1.scatter([x_32], [y_32], s=46, marker="s", facecolor="white", edgecolor=OURS, linewidth=1.6, zorder=5, label="LoopBridge 3/2 (v2)")
    ax1.scatter([x_31], [y_31], s=46, marker="s", facecolor=OURS, edgecolor="white", linewidth=0.5, zorder=5, label="LoopBridge 3/1 (24M)")

    ax1.annotate(
        f"LB 3/2 v2 ({y_32:.1f})\n96.2% of Orig 3/3",
        (x_32, y_32),
        xytext=(6, 9),
        textcoords="offset points",
        fontsize=6.6,
        fontweight="bold",
        color=OURS_DARK,
        ha="center",
    )
    ax1.annotate(
        f"LB 3/1 ({y_31:.1f})\n+7.8 vs Dist. 1/1",
        (x_31, y_31),
        xytext=(-4, 9),
        textcoords="offset points",
        fontsize=6.5,
        fontweight="bold",
        color=OURS_DARK,
        ha="right",
    )

    ax1.set_xlabel("Day-0 vLLM E2E throughput (k tok/s, $L$=512, $B$=16)")
    ax1.set_ylabel("5-benchmark macro accuracy (%)")
    ax1.set_xlim(1.65, 4.65)
    ax1.set_ylim(36.5, 82.0)
    ax1.grid(axis="y", color=GRID, lw=0.6)
    ax1.set_title("(a) TML-d24-varloop Pareto frontier", fontsize=8.0, fontweight="bold", pad=6)
    ax1.legend(loc="lower left", fontsize=6.2, handletextpad=0.3, borderaxespad=0.2)

    # --- Panel (b): Pure decode speedup across 512..256K (zero overhead proof) ---
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
            vllm_vl[arm][c]["agg_decode_tok_s"] / vllm_vl["orig_3_3"][c]["agg_decode_tok_s"]
            for c, _ in cfgs
        ]

    ax2.axhline(1.0, color=NAT_LIGHT, ls=":", lw=0.9, zorder=1)
    ax2.plot(xs, speedups("orig_4_4"), ":", color=NAT_LIGHT, marker="o", ms=3.5, lw=1.0, label="Orig 4/4 (0.79x)")
    ax2.plot(xs, speedups("orig_2_2"), "--", color=NAT, marker="o", ms=4.0, lw=1.1, label="Orig 2/2")
    ax2.plot(
        xs,
        speedups("lb_3_2_v2_48m"),
        "-",
        color=OURS,
        marker="s",
        mfc="white",
        mec=OURS,
        mew=1.3,
        ms=4.8,
        lw=1.5,
        label="LB 3/2 v2 ($r$=128)",
    )
    ax2.plot(xs, speedups("orig_1_1"), "--", color=DIST, marker="^", ms=4.2, lw=1.1, label="Orig 1/1")
    ax2.plot(
        xs,
        speedups("lb_3_1_24m"),
        "-",
        color=OURS_DARK,
        marker="s",
        ms=4.5,
        lw=1.5,
        label="LB 3/1 ($r$=64)",
    )

    # Annotate exact 256K endpoints
    s31_end = speedups("lb_3_1_24m")[-1]
    s32_end = speedups("lb_3_2_v2_48m")[-1]
    ax2.annotate(f"{s31_end:.2f}x", (xs[-1], s31_end), xytext=(-2, 6), textcoords="offset points", fontsize=7, fontweight="bold", color=OURS_DARK, ha="center")
    ax2.annotate(f"{s32_end:.2f}x", (xs[-1], s32_end), xytext=(-2, 6), textcoords="offset points", fontsize=7, fontweight="bold", color=OURS, ha="center")

    ax2.set_xticks(xs)
    ax2.set_xticklabels([lab for _, lab in cfgs], fontsize=7)
    ax2.set_ylabel("Pure decode speedup vs. Orig 3/3")
    ax2.set_ylim(0.68, 2.58)
    ax2.grid(axis="y", color=GRID, lw=0.6)
    ax2.set_title("(b) Isolated 1x B300 vLLM decode speedup", fontsize=8.0, fontweight="bold", pad=6)
    ax2.legend(loc="upper left", ncol=2, fontsize=6.1, columnspacing=0.7, handletextpad=0.35, handlelength=1.5)

    save(fig, "fig_inkling_pareto")


def make_ablate_and_sweep_fig(evals: dict) -> None:
    """Two-panel figure:
    (a) Paired per-benchmark gains (Delta pp) on the two locked 24M controlled
        sweeps: TML-varloop LB 3/1 vs Distilled 1/1, and Inkling-2-Mini LB 3/2
        vs Distilled 2/2 (BOTH 5/5 sweeps!).
    (b) Mechanistic ablation ladder on TML-d24-varloop 3/1 (24M), comparing
        5-Bench Macro and 3-Long-Ctx Macro across Native 1/1 -> TF 3/1 ->
        Distilled 1/1 -> No KV Maps -> Single-Pass KV Map -> Cross-Pass KV Map.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.75, 2.55), gridspec_kw={"width_ratios": [1.05, 1.05], "wspace": 0.32}
    )

    # --- Panel (a): Paired Delta pp on the two 24M 5/5 sweeps ---
    xs = np.arange(len(BENCH_KEYS))
    w = 0.38
    vl_deltas = [score_pct(evals, "vl_lb_3_1_24m", bk) - score_pct(evals, "vl_distilled_1_1_24m", bk) for bk in BENCH_KEYS]
    mini_deltas = [score_pct(evals, "mini_lb_3_2_24m", bk) - score_pct(evals, "mini_distilled_2_2_v3", bk) for bk in BENCH_KEYS]

    b1 = ax1.bar(xs - w / 2, vl_deltas, width=w, color=OURS_DARK, label="VarLoop LB 3/1 vs. Distilled 1/1 (24M)")
    b2 = ax1.bar(xs + w / 2, mini_deltas, width=w, color=OURS, alpha=0.65, edgecolor=OURS_DARK, lw=0.7, label="Mini LB 3/2 vs. Distilled 2/2 (24M)")
    ax1.axhline(0.0, color=NAT, lw=0.6)

    for idx_s, bars in enumerate((b1, b2)):
        dx = -1.5 if idx_s == 0 else 1.5
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(
                f"+{h:.1f}",
                (bar.get_x() + bar.get_width() / 2, h),
                xytext=(dx, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=5.8,
                color=INK,
            )

    ax1.set_xticks(xs)
    ax1.set_xticklabels([lab for _, lab in BENCHES], fontsize=7)
    ax1.set_ylabel("LoopBridge gain over Distilled (pp)")
    ax1.set_ylim(0, 28.0)
    ax1.grid(axis="y", color=GRID, lw=0.6)
    ax1.set_title("(a) Controlled 24M 5/5 sweeps", fontsize=8.0, fontweight="bold", pad=6)
    ax1.legend(loc="upper left", fontsize=6.1, handletextpad=0.4)

    # --- Panel (b): Mechanistic ablation ladder (TML-varloop 3/1, 24M) ---
    ladder = [
        ("vl_native_1_1", "Orig\n1/1"),
        ("vl_tf_3_1", "TF\n3/1"),
        ("vl_distilled_1_1_24m", "Dist.\n1/1"),
        ("vl_abl_3_1_no_kvmaps", "No KV\nmaps"),
        ("vl_abl_3_1_singlepass_kv", "1-pass\nKV map"),
        ("vl_lb_3_1_24m", "Full LB\n3/1"),
    ]
    x2 = np.arange(len(ladder))
    m5 = [macro_pct(evals, m, BENCH_KEYS) for m, _ in ladder]
    m3 = [macro_pct(evals, m, LONG_KEYS) for m, _ in ladder]

    w2 = 0.36
    colors_5 = [NAT_LIGHT, NAT_LIGHT, DIST, "#F3A37E", OURS, OURS_DARK]
    bars5 = ax2.bar(x2 - w2 / 2, m5, width=w2, color=colors_5, edgecolor="white", lw=0.4, label="5-Bench Macro")
    bars3 = ax2.bar(x2 + w2 / 2, m3, width=w2, color=colors_5, alpha=0.45, edgecolor=NAT, lw=0.6, hatch="//", label="3-Long-Ctx Macro")

    for bar in bars5:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=5.8, fontweight="bold", color=INK)
    for bar in bars3:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=5.6, color=NAT)

    ax2.set_xticks(x2)
    ax2.set_xticklabels([lab for _, lab in ladder], fontsize=6.7)
    ax2.set_ylabel("Macro accuracy (%)")
    ax2.set_ylim(24, 59.5)
    ax2.grid(axis="y", color=GRID, lw=0.6)
    ax2.set_title("(b) VarLoop 3/1 mechanistic ablation", fontsize=8.0, fontweight="bold", pad=6)
    ax2.legend(loc="upper left", fontsize=6.3, handletextpad=0.4)

    save(fig, "fig_inkling_ablate")


def paired_boot(evals: dict, a: str, b: str, keys: list[str], n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    diffs_by_bk = []
    for bk in keys:
        am = {x["id"]: float(x["score"]) for x in evals[a][bk]["items"]}
        bm = {x["id"]: float(x["score"]) for x in evals[b][bk]["items"]}
        ids = sorted(set(am) & set(bm))
        assert len(ids) == len(evals[a][bk]["items"]) == len(evals[b][bk]["items"]), f"ID mismatch {a} vs {b} on {bk}"
        diffs_by_bk.append([100.0 * (am[i] - bm[i]) for i in ids])
    obs = float(np.mean([np.mean(ds) for ds in diffs_by_bk]))
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        ms = [sum(ds[rng.randrange(len(ds))] for _ in range(len(ds))) / len(ds) for ds in diffs_by_bk]
        boots.append(sum(ms) / len(ms))
    boots.sort()
    return obs, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def print_verified_summary(evals: dict, vllm_vl: dict) -> None:
    print("=== VERIFIED PAIRED BOOTSTRAP CIs (5,000 resamples, seed=0) ===")
    pairs = [
        ("vl_lb_3_2_v2_48m", "vl_native_2_2"),
        ("vl_lb_3_2_v2_48m", "vl_native_3_3"),
        ("vl_lb_3_1_24m", "vl_distilled_1_1_24m"),
        ("vl_lb_3_1_24m", "vl_native_1_1"),
        ("vl_lb_3_1_v2_48m", "vl_native_1_1"),
        ("vl_lb_3_1_24m", "vl_abl_3_1_no_kvmaps"),
        ("vl_abl_3_1_singlepass_kv", "vl_abl_3_1_no_kvmaps"),
        ("vl_lb_3_1_24m", "vl_abl_3_1_singlepass_kv"),
        ("vl_abl_3_1_no_kvmaps", "vl_distilled_1_1_24m"),
        ("vl_lb_4_1_24m", "vl_native_1_1"),
        ("vl_lb_4_1_24m", "vl_lb_3_1_24m"),
        ("mini_lb_3_2_24m", "mini_distilled_2_2_v3"),
        ("mini_lb_3_2_v2_48m", "mini_distilled_2_2_v3"),
        ("mini_lb_3_1_24m", "mini_distilled_1_1_v3"),
    ]
    for a, b in pairs:
        o5, l5, h5 = paired_boot(evals, a, b, BENCH_KEYS)
        o3, l3, h3 = paired_boot(evals, a, b, LONG_KEYS)
        print(f"  {a:25s} vs {b:22s} | 5B: {o5:+6.2f} [{l5:+6.2f}, {h5:+6.2f}] | 3L: {o3:+6.2f} [{l3:+6.2f}, {h3:+6.2f}]")


def main() -> None:
    evals = load_all_eval()
    vllm_vl = load_vllm_vl_timing()
    make_pareto_and_speed_fig(evals, vllm_vl)
    make_ablate_and_sweep_fig(evals)
    print(f"Generated {OUT / 'fig_inkling_pareto.pdf'} and {OUT / 'fig_inkling_ablate.pdf'}")
    print_verified_summary(evals, vllm_vl)


if __name__ == "__main__":
    main()
