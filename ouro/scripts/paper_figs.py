"""Paper figures (PDF for LaTeX, PNG for README) and generated LaTeX tables, all from results/paper_numbers.json.

usage: python scripts/paper_figs.py  ->  paper/figs/fig*.{pdf,png}, paper/figs/tab_*.tex
One palette everywhere (see OURS / DIST / ... below); each main-text figure makes one point, the full data are in
appendix figures and tables.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]                 # ouro/
OUT = ROOT.parent / "paper" / "figs"                       # paper/figs at the repository root
# One meaning per color across the paper (palette checked for colour-blind separation with the dataviz validator):
# methods: Ours/LoopBridge orange (squares), Distilled blue (triangles), Original grey (circles); decode depth inside
# one method (Figure 10): n = 1 violet, n = 2 green. Filled markers: one decode loop; hollow: two.
OURS, DIST = "#EB6834", "#2A78D6"
N1, N2 = "#6250D6", "#008300"
NAT, NAT_LIGHT, INK, GRID = "#4D4D4D", "#9A9A9A", "#1A1A1A", "#E6E6E6"
plt.rcParams.update({
    "font.size": 8, "font.family": "DejaVu Sans", "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "axes.edgecolor": NAT, "xtick.color": NAT, "ytick.color": NAT, "axes.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42})

PN = json.loads((ROOT / "results" / "paper_numbers.json").read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)



def ci(lo, hi):
    """95% interval in points: one decimal, two when one would round a nonzero endpoint to 0.0 (significance is decided
    on the unrounded values)."""
    f = lambda x: f"{x:.2f}" if x != 0 and abs(x) < 0.05 else f"{x + 0.0:.1f}"
    return rf"\ci{{{f(lo)}}}{{{f(hi)}}}"


def frontier(ax, nat, ours, title, xlog=False, lines=True):
    """The appendix figure style: Original (grey) and Ours (orange) as lines of (x, y, label) points."""
    for pts, col, dy, mk in ((nat, NAT_LIGHT, -10, "o"), (ours, OURS, 5, "s")):
        pts = [p for p in pts if np.isfinite(p[1])]
        if not pts:
            continue
        x, y = [p[0] for p in pts], [p[1] for p in pts]
        if lines:
            ax.plot(x, y, "-", color=col, lw=1.4, zorder=2)
        ax.scatter(x, y, s=18, color=col, marker=mk, zorder=3)
        for xi, yi, lab, *off in pts:                 # optional 4th element: this label's vertical offset
            if lab:
                ax.annotate(lab, (xi, yi), xytext=(0, off[0] if off else dy), textcoords="offset points", fontsize=8,
                            color=OURS if col == OURS else NAT, ha="center")
    ys = [p[1] for p in nat + ours if np.isfinite(p[1])]
    lo, hi = min(ys), max(ys); pad = 0.14 * (hi - lo + 1)
    ax.set_ylim(lo - pad, hi + pad)
    if xlog:
        ax.set_xscale("log"); ax.minorticks_off()
    ax.set_title(title, fontsize=8.5, color=INK, pad=4)
    ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def legend(fig, ours, labels, y=1.1, native="Original n/n (tied)", ncol=3, ours_ls="-"):
    h = [plt.Line2D([], [], color=OURS, marker="s", ms=4.2, lw=1.4 if ours_ls != "none" else 0, ls=ours_ls),
         plt.Line2D([], [], color=NAT_LIGHT, marker="o", ms=4, lw=1.4)]
    names = [ours, native]
    if labels:
        h.append(plt.Line2D([], [], ls="none")); names.append(labels)
    fig.legend(h, names, loc="upper center", ncol=ncol, fontsize=8, bbox_to_anchor=(0.5, y),
               handlelength=1.6, frameon=False)


def _load_inkling_teaser_data():
    """Load verified 26.4B Inkling-2 numbers dynamically from inkling/results/ (zero hardcoding)."""
    ires = ROOT.parent / "inkling" / "results"
    ea4 = json.loads((ires / "eval_official_exact_avg4.json").read_text())
    eo = json.loads((ires / "eval_official_exact.json").read_text())
    ev3 = json.loads((ires / "eval_vllm_official_v3.json").read_text())
    elcb = json.loads((ires / "eval_vllm_lcb_fixed_v3.json").read_text())
    evl = json.loads((ires / "eval_vllm_varloop.json").read_text())
    bks = ["gpqa_diamond_high", "livecodebench_v6_high", "mrcr_8n_256k_512k", "graphwalks_bfs_256k", "longbench_v2_1m"]

    def m5(src, arm):
        return float(np.mean([100.0 * src[arm][b]["score"] for b in bks]))

    def m5_mix(arm, fb_v3=None, fb_lcb=None, fb_vl=None):
        vals = []
        for b in bks:
            if arm in eo and b in eo[arm]:
                vals.append(100.0 * eo[arm][b]["score"])
            elif b == "livecodebench_v6_high" and fb_lcb and fb_lcb in elcb:
                vals.append(100.0 * elcb[fb_lcb][b]["score"])
            elif fb_v3 and fb_v3 in ev3 and b in ev3[fb_v3]:
                vals.append(100.0 * ev3[fb_v3][b]["score"])
            else:
                vals.append(100.0 * evl[fb_vl][b]["score"])
        return float(np.mean(vals))

    def e2e_cot(arm):
        rows = json.loads((ires / f"vllm_sse_vl_vl_{arm}_short.json").read_text())["results"]
        r = next(x for x in rows if (x["L"], x["B"], x["gen_len"]) == (512, 16, 1024))
        return r["agg_e2e_tok_s"] / 1000.0

    def dec_256k(arm):
        rows = json.loads((ires / f"vllm_sse_vl_vl_{arm}_long.json").read_text())["results"]
        r = next(x for x in rows if (x["L"], x["B"], x["gen_len"]) == (262144, 2, 512))
        return r["agg_decode_tok_s"]

    d44 = dec_256k("orig_4_4")
    return {
        "nat4": [(e2e_cot(t), m5(ea4, k), lab) for t, k, lab in (
            ("orig_4_4", "vl_native_4_4", "4/4"),
            ("orig_3_3", "vl_native_3_3", "3/3"),
            ("orig_2_2", "vl_native_2_2", "2/2"),
        )],
        "orig_11": (e2e_cot("orig_1_1"), m5_mix("vl_native_1_1", fb_vl="vl_native_1_1")),
        "dist_11": (e2e_cot("orig_1_1"), m5(ea4, "vl_distilled_1_1_24m")),
        "lb_43": (e2e_cot("lb_4_3_kvonly_12m"), m5(ea4, "vl_lb_4_3_kvonly_12m"), dec_256k("lb_4_3_kvonly_12m") / d44),
        "lb_42": (e2e_cot("lb_4_2_kvonly_12m"), m5(ea4, "vl_lb_4_2_kvonly_12m"), dec_256k("lb_4_2_kvonly_12m") / d44),
        "lb_31": (e2e_cot("lb_3_1_24m"), m5(ea4, "vl_lb_3_1_24m"), dec_256k("lb_3_1_24m") / d44),
        "tf_43": (e2e_cot("tf_4_3_anchored"), m5(ea4, "vl_tf_4_3_anchored")),
        "tf_42": (e2e_cot("tf_4_2_anchored"), m5(ea4, "vl_tf_4_2_tail")),
        "e2e_42_vs_44": e2e_cot("lb_4_2_kvonly_12m") / e2e_cot("orig_4_4"),
        "mrcr_gain_42": 100.0 * (ea4["vl_lb_4_2_kvonly_12m"]["mrcr_8n_256k_512k"]["score"] - ea4["vl_native_2_2"]["mrcr_8n_256k_512k"]["score"]),
        "mini_tf": m5_mix("mini_tf_3_2_anchored"),
        "mini_dist": m5(ea4, "mini_distilled_2_2_v3"),
        "mini_lb": m5(ea4, "mini_lb_3_2_24m"),
    }


def fig_tps_teaser():
    """Figure 1 (Page-1 Hero): Two spacious, zero-jargon panels spanning 10x in scale:
    Top (a): Ouro-2.6B 11-Benchmark Average across all 16 (m, n) prefill/decode pairs (promoted from old Fig 3),
             with measured overall end-to-end throughput (tok/s, batch 1, 1x B300) displayed cleanly under n=1..4.
    Bottom (b): 10x scale-up to 26.4B Thinking Machines Inkling-2 (TML-d24-varloop in vLLM, 5 benchmarks to 1M):
                ultra-clean 3-series Pareto frontier (LoopBridge vs. Original n/n vs. Distilled 1/1)."""
    rows = PN.get("aligned_cells", [])
    if not rows:
        return drop("fig_tps_teaser")
    INKL = _load_inkling_teaser_data()

    # Exact 11-benchmark averages across all 16 (m, n) pairs
    avg_cell = {f"native{n}": float(np.mean([r[f"native{n}"] for r in rows])) for n in range(1, 5)}
    for m in range(1, 5):
        for n in range(1, 5):
            avg_cell[f"a{m}{n}"] = float(np.mean([r[f"a{m}{n}"] for r in rows]))

    # Exact measured overall end-to-end throughput (tok/s) across the 4 CoT tasks for n=1..4
    def e2e_tps(cond):
        rs = [_time(t, cond) for t in TIMED]
        if None in rs:
            return None
        return sum(r["out_tokens"] for r in rs) / sum(r["total_s"] for r in rs)

    tps_n = {n: e2e_tps(f"native{n}st") for n in range(1, 5)}
    tps_base = tps_n[4]

    M_COLS = ["#FBD8C8", "#F7AC88", "#F1824F", OURS]
    M_EDGE = ["#D49A7E", "#D67E56", "#C85C2D", "#B54417"]

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(3.36, 4.18), gridspec_kw=dict(height_ratios=[1.16, 1.00])
    )

    # =========================================================================
    # Panel (a): Ouro-2.6B 11-Benchmark Average (m=1..4 x n=1..4) + Overall TPS
    # =========================================================================
    xs = np.arange(1, 5)
    bw = 0.185
    offs = np.array([-1.5, -0.5, 0.5, 1.5]) * bw

    for m in range(1, 5):
        vals = [avg_cell[f"a{m}{n}"] for n in range(1, 5)]
        ax0.bar(
            xs + offs[m - 1], vals, width=bw * 0.92,
            color=M_COLS[m - 1], edgecolor=M_EDGE[m - 1],
            linewidth=0.45 if m < 4 else 0.65, zorder=3,
        )

    pill = dict(fc="white", ec="none", alpha=0.90, pad=0.35)
    for idx, n in enumerate(range(1, 5)):
        nv = avg_cell[f"native{n}"]
        a4 = avg_cell[f"a4{n}"]
        ax0.hlines(nv, xs[idx] - 0.41, xs[idx] + 0.41, color=NAT, lw=1.35, ls="--", zorder=5)
        y_lab = max(a4, nv) + 2.0 if n >= 3 else a4 + 1.5
        ax0.text(
            xs[idx] + offs[3], y_lab, f"{a4:.1f}",
            ha="center", va="bottom", fontsize=6.6, color=OURS, fontweight="bold", zorder=6,
        )
        if n == 1:
            ax0.text(xs[idx], nv - 2.3, f"Orig {nv:.1f}", ha="center", va="top", fontsize=5.8, color=NAT, zorder=6, bbox=pill)
            ax0.text(xs[idx] + offs[3], a4 + 8.6, "+14.0", ha="center", va="bottom", fontsize=6.8, color=OURS, fontweight="bold", zorder=6)
        elif n == 2:
            ax0.text(xs[idx] - 0.40, nv - 1.5, f"{nv:.1f}", ha="left", va="top", fontsize=5.8, color=NAT, zorder=6, bbox=pill)
            ax0.text(xs[idx] + offs[3], a4 + 8.6, "+5.4", ha="center", va="bottom", fontsize=6.8, color=OURS, fontweight="bold", zorder=6)
        else:
            ax0.text(xs[idx] - 0.40, nv - 1.5, f"{nv:.1f}", ha="left", va="top", fontsize=5.8, color=NAT, zorder=6, bbox=pill)

    # X-axis labels with Overall TPS directly beneath each decode column n=1..4
    xlabels = [
        f"n = {n}\n{tps_n[n]:.1f} tok/s\n({tps_n[n] / tps_base:.1f}×)"
        for n in range(1, 5)
    ]
    ax0.set_xticks(xs)
    ax0.set_xticklabels(xlabels, fontsize=6.1, linespacing=1.18)
    ax0.tick_params(axis="x", length=0, pad=2.5, colors=INK)
    # Subtle left caption aligned cleanly with the tok/s & speedup rows
    ax0.text(
        0.43, -0.105, "Overall\nTPS:", transform=ax0.get_xaxis_transform(),
        ha="right", va="top", fontsize=5.5, color="#1D5C63", fontweight="bold", linespacing=1.15,
    )

    ax0.set_xlim(0.42, 4.56)
    ax0.set_ylim(0, 88.5)
    ax0.set_yticks([0, 20, 40, 60, 80])
    ax0.set_ylabel("11-bench avg (%)", fontsize=7.0)
    ax0.set_title("(a) Ouro-2.6B: 11-Benchmark Average & Overall TPS", fontsize=7.5, color=INK, fontweight="bold", loc="left", pad=14.0)
    ax0.grid(axis="y", color="#EDEDED", lw=0.55)
    ax0.set_axisbelow(True)
    ax0.tick_params(axis="y", labelsize=6.6)

    # Crisp single-row legend right between title and bars
    h0 = [plt.Rectangle((0, 0), 1, 1, fc=M_COLS[i], ec=M_EDGE[i], lw=0.45) for i in range(4)]
    h0.append(plt.Line2D([], [], color=NAT, lw=1.35, ls="--"))
    l0 = ["m=1", "m=2", "m=3", "m=4 prefill", "Original n/n"]
    ax0.legend(
        h0, l0, loc="upper center", bbox_to_anchor=(0.50, 1.155),
        ncol=5, fontsize=5.6, handlelength=0.95, columnspacing=0.55, handletextpad=0.25, frameon=False,
    )

    # =========================================================================
    # Panel (b): 26.4B Inkling-2 (TML-d24-varloop in vLLM, 5 Benchmarks to 1M)
    # Ultra-clean 3-series frontier: ZERO colliding diamonds, ZERO sticky-note!
    # =========================================================================
    nat4 = INKL["nat4"]
    nx4, ny4 = [p[0] for p in nat4], [p[1] for p in nat4]
    x11, y11 = INKL["orig_11"]
    xd11, yd11 = INKL["dist_11"]
    x43, y43, _ = INKL["lb_43"]
    x42, y42, _ = INKL["lb_42"]
    x31, y31, _ = INKL["lb_31"]

    ax1.plot(nx4 + [x11], ny4 + [y11], "--", color=NAT_LIGHT, lw=1.15, zorder=1)
    ax1.scatter(nx4 + [x11], ny4 + [y11], s=26, color=NAT, edgecolor="white", lw=0.5, zorder=3, label="Original n/n (tied)")
    ax1.axhline(ny4[0], color=NAT_LIGHT, lw=0.65, ls=":", alpha=0.75, zorder=0)

    nat_offs = {"4/4": (0, -10.5), "3/3": (-4, -10.5), "2/2": (-8, -9.5)}
    for x, y, lab in nat4:
        ox, oy = nat_offs[lab]
        ax1.annotate(lab, (x, y), xytext=(ox, oy), textcoords="offset points", fontsize=6.3, color=NAT, ha="center")
    ax1.annotate("1/1 (@1)", (x11, y11), xytext=(-6.0, -1.5), textcoords="offset points", fontsize=6.0, color=NAT, ha="right", va="center")

    # Distilled 1/1
    ax1.scatter([xd11], [yd11], marker="^", s=30, color=DIST, edgecolor="white", lw=0.45, zorder=4, label="Distilled 1/1")
    ax1.annotate(
        "Distilled 1/1", (xd11, yd11), xytext=(-22.0, 0.0), textcoords="offset points",
        fontsize=6.0, color=DIST, ha="right", va="center",
        bbox=dict(fc="white", ec="none", alpha=0.88, pad=0.35), zorder=6,
    )

    # Ours LoopBridge (4/3, 4/2, 3/1)
    ax1.plot([x43, x42, x31], [y43, y42, y31], "-", color=OURS, lw=1.65, zorder=3)
    ax1.scatter([x43, x42, x31], [y43, y42, y31], marker="s", s=34, color=OURS, edgecolor="white", lw=0.5, zorder=5, label="LoopBridge m/n (ours)")

    # Crisp vertical gain arrows
    ax1.annotate("", xy=(x42, y42 - 1.3), xytext=(nx4[2], ny4[2] + 1.3),
                 arrowprops=dict(arrowstyle="-|>", color=OURS, lw=0.95, mutation_scale=7.0), zorder=4)
    ax1.annotate("", xy=(x31, y31 - 1.1), xytext=(x31, yd11 - 0.2),
                 arrowprops=dict(arrowstyle="-|>", color=OURS, lw=0.95, mutation_scale=5.8), zorder=4)

    rec43 = 100.0 * y43 / ny4[0]
    ax1.annotate(
        f"4/3: {y43:.1f}% ({rec43:.0f}% of 4/4)",
        (x43, y43), xytext=(8, 8.5), textcoords="offset points",
        fontsize=6.1, color=OURS, fontweight="bold", ha="left", va="bottom",
        arrowprops=dict(arrowstyle="-", color=OURS, lw=0.55, shrinkA=0, shrinkB=3.0),
    )
    ax1.annotate(
        f"4/2: {y42:.1f}% (+{y42 - ny4[2]:.1f} vs 2/2)\n{INKL['e2e_42_vs_44']:.1f}× E2E throughput",
        (x42, y42), xytext=(9, 4.0), textcoords="offset points",
        fontsize=5.95, color=OURS, fontweight="bold", ha="left", va="center", linespacing=1.12,
    )
    ax1.annotate(
        f"3/1: {y31:.1f}%\n(+{y31 - yd11:.1f} vs Dist.)",
        (x31, y31), xytext=(8, 9.5), textcoords="offset points",
        fontsize=5.95, color=OURS, fontweight="bold", ha="center", va="bottom", linespacing=1.10,
        zorder=6,
    )

    ax1.set_title("(b) 26.4B Inkling-2 Sparse-MoE (vLLM, 5 Benches to 1M)", fontsize=7.5, color=INK, fontweight="bold", loc="left", pad=4.0)
    ax1.set_xlabel("Overall E2E throughput (k tokens/s, 1× B300)", fontsize=6.8, color=INK, labelpad=2.0)
    ax1.set_ylabel("5-bench avg (%)", fontsize=7.0)
    ax1.set_xlim(1.65, 5.18)
    ax1.set_ylim(36.0, 86.0)
    ax1.set_xticks([2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    ax1.set_yticks([40, 50, 60, 70, 80])
    ax1.grid(color="#EDEDED", lw=0.55)
    ax1.set_axisbelow(True)
    ax1.tick_params(labelsize=6.6)
    ax1.legend(loc="lower left", fontsize=6.0, handlelength=1.2, handletextpad=0.35, labelspacing=0.3, borderaxespad=0.3)

    fig.subplots_adjust(left=0.16, right=0.98, top=0.92, bottom=0.08, hspace=0.54)
    save(fig, "fig_tps_teaser")


def fig0():
    """Figure 2 (Why & How LoopBridge Works): Top motivation bar showing <3% prefill vs >97% decode wall time
    (measured from paper_numbers.json), paired with the de-jargonized 3-stage LoopBridge flow unified across
    dense RoPE (Ouro) and 26.4B sparse-MoE NoPE (Inkling-2)."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    # Measured mean prefill share on batch-1 CoT tasks (Original 1/1)
    trows = [r for t in TIMED if (r := _time(t, "native1st")) is not None]
    p_mean = float(np.mean([r["prefill_share"] for r in trows])) if trows else 1.6
    p_max = float(max(r["prefill_share"] for r in trows)) if trows else 2.6

    fig, ax = plt.subplots(figsize=(6.9, 2.90))
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 47.0)
    ax.axis("off")
    LIGHT = "#FDEBE3"
    GREY = "#F2F2F2"

    # --- Top motivation strip: Why untie? Prefill <3% vs Decode >97% wall time (zero overlap!) ---
    ax.text(1.0, 45.4, "Why untie?  Batch-1 CoT wall-time split (4 tasks, 1× B300):", ha="left", va="center", fontsize=6.8, color=INK, fontweight="bold")
    bar_x, bar_y, bar_w, bar_h = 1.0, 40.3, 137.5, 3.3
    p_w = max(2.8, bar_w * (p_mean / 100.0))   # visually legible sliver representing actual ~1.6%
    ax.add_patch(Rectangle((bar_x, bar_y), p_w, bar_h, fc=OURS, ec=OURS, lw=0.6))
    ax.add_patch(Rectangle((bar_x + p_w, bar_y), bar_w - p_w, bar_h, fc="#E4E4E4", ec=NAT_LIGHT, lw=0.6))
    ax.text(bar_x + p_w + 1.8, bar_y + bar_h / 2,
            f"Prefill: <{p_max:.0f}% (mean {p_mean:.1f}%)   |   Decode: >{100 - p_max:.0f}% of total runtime — extra prefill loops cost almost nothing",
            ha="left", va="center", fontsize=5.80, color=INK, fontweight="bold")
    ax.plot([1.0, 138.5], [38.3, 38.3], color="#E0E0E0", lw=0.7)

    def box(x, y, w, h, text, fc=GREY, ec=NAT_LIGHT, lw=0.75, size=6.3, color=INK, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, fontweight=weight, linespacing=1.18)

    def arrow(x0, y0, x1, y1, c=NAT, lw=0.85):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=c, mutation_scale=8, shrinkA=1, shrinkB=1))

    dots = lambda x, y: ax.text(x, y, "$\\vdots$", ha="center", va="center", fontsize=9.5, color=NAT)
    for x, t in ((12.5, "1. Prefill: once, $m$ loops"), (57.5, "2. LoopBridge: once per prompt"), (107.0, "3. Decode: $n$ loops per token")):
        ax.text(x, 35.6, t, ha="center", va="center", fontsize=7.4, color=INK, fontweight="bold")

    # Stage 1: Prefill (frozen backbone)
    box(1.0, 26.2, 23, 4.5, "prefill loop $m$")
    dots(12.5, 22.5); arrow(12.5, 24.1, 12.5, 25.9)
    box(1.0, 15.8, 23, 4.5, "prefill loop $j(1)$")
    dots(12.5, 12.1); arrow(12.5, 13.7, 12.5, 15.5)
    box(1.0, 5.8, 23, 4.5, "prefill loop $1$")
    ax.text(12.5, 2.0, "Frozen backbone reads prompt\nonce in parallel ($m > n$)", ha="center", va="center", fontsize=5.8, color=NAT, linespacing=1.18)

    # Stage 2: LoopBridge prompt KV-map folding (once per prompt)
    box(34.5, 25.7, 46, 5.5, "mapped cache $n$ ($K, V$)\nOuro: $c_n{\\odot}K,\\, W_n V$  |  Inkling-2: $I{+} \\Delta_{n,p}$", fc=LIGHT, ec=OURS, lw=1.0, size=6.0)
    dots(57.5, 22.5)
    box(34.5, 15.3, 46, 5.5, "mapped cache $1$ ($K, V$)\nOuro: $c_1{\\odot}K,\\, W_1 V$  |  Inkling-2: $I{+} \\Delta_{1,p}$", fc=LIGHT, ec=OURS, lw=1.0, size=6.0)
    ax.text(57.5, 8.6, "Folded once after prefill → 0% decode overhead\nOuro: all RoPE layers  |  Inkling-2: 3 global layers",
            ha="center", va="center", fontsize=5.55, color=OURS, linespacing=1.28)
    arrow(24.5, 28.4, 34.0, 28.4, OURS); arrow(24.5, 18.0, 34.0, 18.0, OURS)
    ax.annotate("", xy=(34.0, 16.4), xytext=(24.5, 8.8),
                arrowprops=dict(arrowstyle="-|>", lw=0.75, ls="--", color=OURS, alpha=0.65, mutation_scale=7))
    ax.text(29.3, 29.9, "$K,V$", ha="center", fontsize=6.4, color=NAT); ax.text(29.3, 19.5, "$K,V$", ha="center", fontsize=6.4, color=NAT)

    # Stage 3: Decode (native speed)
    box(89.5, 26.2, 35, 4.5, "decode loop $n$ (frozen or +LoRA)", ec=OURS, lw=0.95, size=6.0)
    dots(107.0, 22.5); arrow(107.0, 24.1, 107.0, 25.9)
    box(89.5, 15.8, 35, 4.5, "decode loop $1$ (frozen or +LoRA)", ec=OURS, lw=0.95, size=6.0)
    box(100.5, 6.4, 13, 3.8, "token", fc="white", size=6.3)
    arrow(107.0, 10.5, 107.0, 15.5)
    arrow(81.0, 28.4, 89.0, 28.4, OURS); arrow(81.0, 18.0, 89.0, 18.0, OURS)
    arrow(125.0, 28.4, 129.8, 28.4); ax.text(130.8, 28.4, "next\ntoken", ha="left", va="center", fontsize=6.4, color=INK)
    ax.text(107.0, 2.1, "Variable-depth MoEs ($n\\geq 2$): frozen decoder (0.02% KV maps)\nFixed-depth / $n=1$: merged decoder LoRA (native speed)",
            ha="center", va="center", fontsize=5.7, color=NAT, linespacing=1.18)
    save(fig, "fig0_method")


def fig2():
    """Ouro-2.6B-Thinking: average accuracy over the eight benchmarks of Figure 11 against decode throughput
    (paper_numbers.json, think_frontier)."""
    F = {r["config"]: (r["tokens_per_s"], r["avg"]) for r in PN.get("think_frontier", [])}
    if len(F) < 5:
        return
    nat = [(*F["native 4/4"], "4/4"), (*F["native 2/2"], "2/2", -11)]
    o22, o42 = F["adapter 2/2"], F["adapter 4/2"]
    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    frontier(ax, nat, [(*o22, "2/2", -11), (*o42, "4/2", 6)], "", lines=False)
    ax.plot([p[0] for p in nat], [p[1] for p in nat], "-", color=NAT_LIGHT, lw=1.4, zorder=2)
    ax.annotate("", xy=(o42[0], o42[1] - 0.25), xytext=(o22[0], o22[1] + 0.25),
                arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.4, mutation_scale=9), zorder=4)
    ax.text(o42[0] + 8, (o22[1] + o42[1]) / 2, f"+{round(o42[1], 1) - round(o22[1], 1):.1f}", fontsize=7.5,
            color=OURS, fontweight="bold", ha="left", va="center")
    x, y = F["distilled 2/2"]
    ax.scatter([x], [y], s=26, color=DIST, marker="^", zorder=4)
    ax.annotate("Distilled 2/2\n(baseline)", (x, y), xytext=(-6, 7), textcoords="offset points", fontsize=7.0,
                color=DIST, ha="center", va="bottom", linespacing=1.05)
    ax.set_xlim(80, 470)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.55, hi + 0.25)
    ax.set_xlabel("decode tokens/s, best batch, 8K prompt", fontsize=8)
    ax.set_ylabel("average accuracy (%)", fontsize=8)
    h = [plt.Line2D([], [], color=OURS, marker="s", ms=4.2, lw=0, ls="none"),
         plt.Line2D([], [], color=NAT_LIGHT, marker="o", ms=4.0, lw=1.4)]
    fig.legend(h, ["Ours (LoopBridge, one cache)", "Original n/n (tied)"], loc="upper center", ncol=2, fontsize=7.0,
               bbox_to_anchor=(0.54, 1.005), handlelength=1.4, columnspacing=0.9, frameon=False)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.18, top=0.87)
    save(fig, "fig2_frontier")


def fig_equal():
    """Ouro-2.6B-Thinking: native 2/2 and 4/4 and Distilled 2/2 against LoopBridge at prefill 4 / decode 2."""
    from matplotlib.ticker import MaxNLocator
    E = {e["bench"]: e for e in PN["equal_aligned"] if e.get("model", "2.6B-T") == "2.6B-T"}   # LoopBridge (per-loop caches)
    benches = ["GSM8K", "MATH-500", "MMLU-Pro", "ARC-C", "BBH", "HumanEval+", "MBPP+", "HELMET"]
    fig, axes = plt.subplots(2, 4, figsize=(6.9, 3.45))
    for ax, b in zip(axes.flat, benches):
        e = E[b]
        below = e["ours"] < e["native2"]   # ours under Original 2/2: its label goes below, 2/2's above
        close = abs(e["ours"] - e["native2"]) < 1.2
        x_n2, x_o2 = (1.90, 2.10) if close else (2.0, 2.0)
        frontier(ax, [(x_n2, e["native2"], "2/2", 6 if below else -10), (4, e["native4"], "4/4")],
                 [(x_o2, e["ours"], "4/2", -12 if below else 6)], b)
        ax.set_xlim(1.35, 4.45); ax.set_xticks([2, 4])
        lo, hi = ax.get_ylim()
        if hi - lo < 6:   # at least 6 points per panel, so differences within noise are not drawn as large gaps
            mid = (lo + hi) / 2
            ax.set_ylim(mid - 3, mid + 3)
        low = e["ours"] if below else e["native2"]   # the point whose label goes below it: keep the label off the axis
        lo, hi = ax.get_ylim()
        span = hi - lo
        ax.set_ylim(min(lo, low - 0.32 * span), max(hi, max(e["ours"], e["native4"]) + 0.30 * span))
        if "dist2" in e:   # Distilled 2/2, drawn a little left of n = 2: off the two points there and off the grey line
            ax.scatter([1.62], [e["dist2"]], marker="^", s=20, color=DIST, zorder=3)
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, e["dist2"] - 0.06 * (hi - lo)), max(hi, e["dist2"] + 0.06 * (hi - lo)))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(b, fontsize=8.5, color=INK, pad=6)
    fig.supxlabel("decode loops n", fontsize=8.5, y=0.015); fig.supylabel("score (%)", fontsize=8.5, x=0.005)
    h = [plt.Line2D([], [], color=OURS, marker="s", ms=4.5, ls="none"),
         plt.Line2D([], [], color=NAT_LIGHT, marker="o", ms=4, lw=1.4),
         plt.Line2D([], [], color=DIST, marker="^", ms=4.5, ls="none")]
    fig.legend(h, ["Ours (LoopBridge 4/2)", "Original n/n (tied)", "Distilled 2/2 (baseline)"], loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.01), handlelength=1.6, frameon=False)
    fig.subplots_adjust(wspace=0.36, hspace=0.50, left=0.07, right=0.99, bottom=0.125, top=0.86)
    save(fig, "fig_equal")


def fig_lev():
    """Decision models (Lev): accuracy against the number of decision loops n. Grey: Original n/n continued with the same
    data and updates as ours; orange: the context prefilled with all of the backbone's loops (fixed m, no selection)."""
    L = PN.get("lev", [])
    def series(label, which):
        return {r["depth"]: r[which] for r in L if r["label"] == label}
    models = [m for m in ("Lev-1.4B", "Lev-2.6B", "Lev-3B") if series(f"{m} untied vs same-training control", "a")]
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, len(models), figsize=(6.9, 1.75), squeeze=False)
    for ax, m in zip(axes[0], models):
        top = 2 if m == "Lev-3B" else 4
        by_m = {top: series(f"{m} untied vs same-training control", "a")}
        by_m.update({k: series(f"{m} prefill {k} vs same-training control", "a") for k in range(1, top)})
        by_m = {k: v for k, v in by_m.items() if v}
        nat = series(f"{m} untied vs same-training control", "b")
        ns = sorted(nat)
        ours = [(n, by_m[top][n], "") for n in ns if n in by_m[top] and n < top]   # fixed m = all loops; untied (m > n) only
        frontier(ax, [(n, nat[n], "") for n in ns], ours, m)
        g1 = next((r["diff"] for r in L if r["label"] == f"{m} untied vs same-training control" and r["depth"] == 1), None)
        if g1 is not None and 1 in nat and 1 in by_m[top]:
            y0, y1 = nat[1], by_m[top][1]
            ax.annotate("", xy=(1, y1 - 0.7), xytext=(1, y0 + 0.7),
                        arrowprops=dict(arrowstyle="-|>", color=OURS, lw=0.85, mutation_scale=7), zorder=4)
            ax.annotate(f"+{g1:.1f}", (1, y1), xytext=(-2 if top == 4 else 0, 9 if top == 4 else 6), textcoords="offset points",
                        fontsize=6.9, color=OURS, fontweight="bold", ha="center", va="bottom")
        if m == "Lev-2.6B":
            ax.set_yticks([48, 54, 60, 66]); ax.set_ylim(45.5, 69.5)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.set_xticks(ns); ax.set_xlim(ns[0] - (0.55 if top == 4 else 0.4), ns[-1] + 0.4)
    axes[0][0].set_ylabel("accuracy (%)", fontsize=8.5)
    fig.supxlabel("decision loops n", fontsize=8.5, y=0.0)
    legend(fig, "Ours: context prefilled with all loops (m = 4; Lev-3B: 2)", "", y=1.08,
           native="Original n/n, continued (baseline)", ncol=2)
    fig.subplots_adjust(wspace=0.28, bottom=0.2, top=0.8)
    save(fig, "fig_lev")


def tab_equal(model="1.4B-T"):
    """Table 13: Figure 11 in numbers, with Distilled 2/2 and the average over the eight benchmarks (LaTeX rows from
    paper_numbers.json)."""
    def pm(a, b, iv):   # the gap as the difference of the shown values; the paired interval as computed
        d = round(round(a, 1) - round(b, 1), 1)
        sign = r"\up{%.1f}" if d > 0 else r"\dn{%.1f}" if d < 0 else "%.1f"   # a zero gap carries no sign
        return sign % abs(d) + ("" if iv is None else ci(iv[1], iv[2]))
    rows = [e for e in PN["equal_aligned"] if e.get("model") == model]   # LoopBridge (per-loop caches), as Figure 11
    v = lambda e, k: f"{e[k]:.1f}" if k in e else "--"   # same22: the same adapter reading 2 prefill loops
    lines = [rf"{e['bench']} & {e['n']:,} & {e['native2']:.1f} & {v(e, 'dist2')} & {v(e, 'same22')} & {e['ours']:.1f} & "
             rf"{e['native4']:.1f} & {pm(e['ours'], e['native2'], e['vs2'])} & "
             + (pm(e["ours"], e["dist2"], e["vsd"]) if "dist2" in e else "--") + r" \\" for e in rows]
    if rows and all(k in e for e in rows for k in ("dist2", "same22")):   # the unweighted mean of the eight benchmarks
        m = {k: round(float(np.mean([e[k] for e in rows])), 1) for k in ("native2", "dist2", "same22", "ours", "native4")}
        lines += [r"\midrule", rf"average & & {m['native2']:.1f} & {m['dist2']:.1f} & {m['same22']:.1f} & {m['ours']:.1f} & "
                  rf"{m['native4']:.1f} & {pm(m['ours'], m['native2'], None)} & {pm(m['ours'], m['dist2'], None)} \\"]
    (OUT / f"tab_equal_{model.replace('.', '')}.tex").write_text("\n".join(lines) + "\n")


def fig_aligned(panels=None, name="fig_taskmap_domains", ncols=4):
    """Appendix D domain breakdown: 1x4 horizontal strip of four representative benchmarks (MMLU, BBH, GSM8K,
    RULER-16K) across all 16 (m, n) prefill/decode pairs vs. Original n/n dashed (the 11-benchmark average now
    leads the paper in Figure 1a, eliminating duplication)."""
    rows = PN.get("aligned_cells", [])
    if not rows:
        return drop(name)
    T = {r["bench"]: r for r in rows}

    M_COLS = ["#FBD8C8", "#F7AC88", "#F1824F", OURS]
    M_EDGE = ["#D49A7E", "#D67E56", "#C85C2D", "#B54417"]

    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.88), sharey=True)
    dom_specs = [("MMLU", "MMLU (knowledge)"), ("BBH 3-shot", "BBH (reasoning)"),
                 ("GSM8K", "GSM8K (math)"), ("RULER-16K", "RULER-16K (retrieval)")]

    xs = np.arange(1, 5)
    bw = 0.185
    offs = np.array([-1.5, -0.5, 0.5, 1.5]) * bw

    for idx_ax, (ax, (key, title)) in enumerate(zip(axes, dom_specs)):
        cell = T[key]
        for m in range(1, 5):
            vals = [cell[f"a{m}{n}"] for n in range(1, 5)]
            ax.bar(xs + offs[m - 1], vals, width=bw * 0.92, color=M_COLS[m - 1], edgecolor=M_EDGE[m - 1],
                   linewidth=0.45 if m < 4 else 0.6, zorder=3)
        for idx, n in enumerate(range(1, 5)):
            nv = cell[f"native{n}"]
            ax.hlines(nv, xs[idx] - 0.40, xs[idx] + 0.40, color=NAT, lw=1.25, ls="--", zorder=5)
        ax.set_title(title, fontsize=7.8, color=INK, fontweight="bold", pad=4)
        ax.set_xticks(xs); ax.set_xticklabels([f"n={n}" for n in range(1, 5)], fontsize=7.2)
        ax.set_xlim(0.45, 4.55)
        ax.set_ylim(0, 88)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.set_xlabel("decode loops n", fontsize=7.4)
        ax.grid(axis="y", color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=7.2)
        if idx_ax > 0:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("accuracy (%)", fontsize=8.0)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=M_COLS[i], ec=M_EDGE[i], lw=0.5) for i in range(4)]
    handles.append(plt.Line2D([], [], color=NAT, lw=1.4, ls="--"))
    labels = ["LoopBridge: m=1", "m=2", "m=3", "m=4 (prefill loops)", "Original n/n"]
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=7.3, bbox_to_anchor=(0.52, 1.01),
               handlelength=1.4, columnspacing=1.0, frameon=False)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.77, bottom=0.22, wspace=0.14)
    save(fig, name)


def tab_lev():
    """Appendix table: Lev decision models, every trained prefill count m minus the same-training control (native n/n)
    at every number of decision loops n, both directions (LaTeX rows from paper_numbers.json, lev)."""
    both = next((r for r in PN.get("lev", []) if r["label"] == "both directions"), None)
    if not both:
        return
    G = {(g["model"], g["m"], g["n"]): g["diff"] for g in both["grid"]}
    def cell(v):
        if v is None:
            return "--"
        d, lo, hi = v
        sig = "$^*$" if lo > 0 or hi < 0 else ""
        return (r"\up{%.1f}" if d >= 0 else r"\dn{%.1f}") % abs(d) + sig + ci(lo, hi)
    lines = []
    for mi, model in enumerate(dict.fromkeys(g["model"] for g in both["grid"])):
        if mi and lines:
            lines.append(r"\midrule")
        ms = sorted({m for (mo, m, _) in G if mo == model})
        for m in ms:
            lines.append(rf"{model} & {m} & " + " & ".join(cell(G.get((model, m, n))) for n in range(1, 5)) + r" \\")
    (OUT / "tab_lev.tex").write_text("\n".join(lines) + "\n")


def tab_main(key="main_table", name="tab_main"):
    """Table 1: main results, fixed configurations grouped by decode loops (accuracy on the items all rows share; bold:
    best of each group; LaTeX rows from paper_numbers.json, main_table)."""
    T = {r["bench"]: r for r in PN.get(key, [])}
    benches = list(T)
    groups = [(1, ["native 1/1", "training-free 4/1", "LoRA-distilled 1/1", "distilled r80 1/1", "aligned adapter 4/1"]),
              (2, ["native 2/2", "training-free 4/2", "LoRA-distilled 2/2", "distilled r80 2/2", "aligned adapter 4/2"]),
              (4, ["native 4/4"])]   # fixed-depth LoRA moved to Appendix L; maps-frozen and one-cache rows in Table 2a
    if name == "tab_main14":   # Ouro-1.4B: the one-decode-loop comparison of Section 5.7
        groups = [g for g in groups if g[0] != 2]
    show = {"native 1/1": (r"Original {\scriptsize(Ouro)}", "1/1"), "native 2/2": (r"Original {\scriptsize(Ouro)}", "2/2"), "native 4/4": (r"Original {\scriptsize(Ouro)}", "4/4"),
            "training-free 4/1": ("Training-free", "4/1"), "training-free 4/2": ("Training-free", "4/2"),
            "LoRA-distilled 1/1": (r"Distilled r16 {\scriptsize(KD)}", "1/1"), "LoRA-distilled 2/2": (r"Distilled r16 {\scriptsize(KD)}", "2/2"),
            "distilled r80 1/1": (r"Distilled r80 {\scriptsize(KD)}", "1/1"), "distilled r80 2/2": (r"Distilled r80 {\scriptsize(KD)}", "2/2"),
            "fixed-4 LoRA 4/1": (r"Ours, fixed $m$", "4/1"), "fixed-4 LoRA 4/2": (r"Ours, fixed $m$", "4/2"),
            "aligned LoRA-only 4/1": ("LoRA all pairs, maps frozen", "4/1"), "aligned LoRA-only 4/2": ("LoRA all pairs, maps frozen", "4/2"),
            "loop adapter 4/1": ("LoopBridge, one cache", "4/1"), "loop adapter 4/2": ("LoopBridge, one cache", "4/2"),
            "aligned adapter 4/1": (r"LoopBridge {\scriptsize(ours)}", "4/1"), "aligned adapter 4/2": (r"LoopBridge {\scriptsize(ours)}", "4/2")}
    lines = []
    for n, labels in groups:
        labels = [l for l in labels if any(T[b].get(l) is not None for b in benches)]   # rows not run for this model
        vals = {l: [T[b].get(l) for b in benches] for l in labels}
        avg = {l: (sum(float(f"{x:.1f}") for x in v) / len(v) if None not in v else None) for l, v in vals.items()}   # mean of the printed cells
        cols = [[vals[l][j] for l in labels] for j in range(len(benches))] + [[avg[l] for l in labels]]
        best = [max((x for x in c if x is not None), default=None) for c in cols]
        for i, l in enumerate(labels):
            cells = [c[i] for c in cols]
            fmt = lambda x, b: "--" if x is None else (rf"\textbf{{{x:.1f}}}" if len(labels) > 1 and f"{x:.1f}" == f"{b:.1f}" else f"{x:.1f}")   # printed ties all bold
            shade = r"\rowcolor{oursbg} " if l.startswith(("aligned adapter", "fixed-4 LoRA")) else ""   # both adapters are Ours
            lines.append(f"{shade}{show[l][0]} & {show[l][1]} & " + " & ".join(fmt(x, b) for x, b in zip(cells, best)) + r" \\")
        if n < 4:
            lines.append(r"\midrule")
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def tab_lr():
    """Main-text ablation (c): the whole learning-rate schedule scaled by 0.3, 1 and 3 for the distilled control
    (Distilled 1/1) and for LoopBridge (4/1), 11-benchmark average (paper_numbers.lr_check)."""
    L = PN.get("lr_check", {})
    if not L:
        return (OUT / "tab_lr.tex").unlink(missing_ok=True)
    f = lambda d, k: "--" if d.get(k) is None else f"{d[k]:.1f}"
    lines = [f"Distilled 1/1 & {f(L['1'], 'lr0.3')} & {f(L['1'], 'lr1')} & {f(L['1'], 'lr3')} \\\\",
             f"LoopBridge 4/1 & {f(L['ours a41'], 'lr0.3')} & \\cellcolor{{oursbg}}{f(L['ours a41'], 'lr1')} & "   # default shaded
             f"{f(L['ours a41'], 'lr3')} \\\\"]
    (OUT / "tab_lr.tex").write_text("\n".join(lines) + "\n")


def tab_tokens():
    """Main-text ablation (b), transposed: 11-benchmark average of the one-cache LoopBridge at 4/1 and 4/2 against the
    number of answer tokens it is trained on (rank 16; paper_numbers.ablations)."""
    A = {(r["label"], r["cell"]): r["avg"] for r in PN.get("ablations", [])}
    toks = [("6M tokens", "6M"), ("12M tokens", "12M"), ("24M tokens", "24M"), ("48M tokens", "48M")]
    f = lambda x: "--" if x is None else f"{x:.1f}"
    lines = ["$m/n$ & " + " & ".join(t for _, t in toks) + r" \\", r"\midrule"]
    for cell, name in (("S41", "4/1"), ("S42", "4/2")):
        vals = [f(A.get((lab, cell))) for lab, _ in toks]
        vals[-1] = r"\cellcolor{oursbg}" + vals[-1]
        lines.append(f"{name} & " + " & ".join(vals) + r" \\")
    (OUT / "tab_tokens.tex").write_text("\n".join(lines) + "\n")


def tab_repro():
    """Table 10 (Appendix H): native Ouro-2.6B 4/4 under our evaluation and as reported by Ouro (paper_numbers.json, repro)."""
    from decimal import Decimal, ROUND_HALF_UP   # 90.85 -> 90.9 (published values have two decimals)
    f = lambda x: "--" if x is None else str(Decimal(repr(x)).quantize(Decimal("0.1"), ROUND_HALF_UP))
    rows = [f"{r['bench']} & {f(r['ours'])} & {f(r['reported'])} & {r['items']:,} \\\\" for r in PN.get("repro", [])]
    (OUT / "tab_repro.tex").write_text("\n".join(rows) + "\n")


def tab_aligned():
    """Appendix: the aligned loop adapter's full map on the main-table benchmarks, every prefill m / decode n cell and
    native n/n (each column on the items all available configurations share), plus the 11-benchmark average; bold: the
    best m at each n (paper_numbers.json, aligned_cells)."""
    rows = PN.get("aligned_cells", [])
    short = {"MMLU-Pro": "MMLU-P", "MATH-500": "MATH", "BBH 3-shot": "BBH", "HumanEval+": "HE+",   # the column names of Table 1
             "HellaSwag": "HellaSw.", "Winogrande": "Winogr.", "RULER-16K": "RULER"}
    hdr = " & ".join(rf"\multicolumn{{1}}{{c}}{{{short.get(r['bench'], r['bench'])}}}" for r in rows)
    lines = [rf"configuration & {hdr} & \multicolumn{{1}}{{c}}{{Avg.}} \\", r"\midrule"]
    f = lambda x, b: "--" if x is None else (rf"\textbf{{{x:.1f}}}" if b else f"{x:.1f}")
    avg_of = lambda vals: None if any(v is None for v in vals) else sum(float(f"{v:.1f}") for v in vals) / len(vals)
    for n in range(1, 5):   # all 16 (m, n) cells for m, n in 1..4
        best = {r["bench"]: max((r[f"a{m}{n}"] for m in range(1, 5) if r[f"a{m}{n}"] is not None), default=None)
                for r in rows}
        avgs = {m: avg_of([r[f"a{m}{n}"] for r in rows]) for m in range(1, 5)}
        best_avg = max((v for v in avgs.values() if v is not None), default=None)
        nat_vals = [r[f"native{n}"] for r in rows]
        lines.append(rf"Original {n}/{n} & " + " & ".join(f(v, False) for v in nat_vals) + f" & {f(avg_of(nat_vals), False)}" + r" \\")
        for m in range(1, 5):
            cells = [f(r[f"a{m}{n}"], r[f"a{m}{n}"] is not None and f"{r[f'a{m}{n}']:.1f}" == f"{best[r['bench']]:.1f}") for r in rows]
            acell = f(avgs[m], avgs[m] is not None and f"{avgs[m]:.1f}" == f"{best_avg:.1f}")
            lines.append(rf"LoopBridge {m}/{n} & " + " & ".join(cells) + f" & {acell}" + r" \\")
        if n < 4:
            lines.append(r"\midrule")
    (OUT / "tab_aligned.tex").write_text("\n".join(lines) + "\n")


TIMED = ["GSM8K", "MATH-500", "BBH", "MMLU-Pro"]
SWEEP_NAME = {"GSM8K": "GSM8K", "MATH-500": "MATH-500", "BBH": "BBH 3-shot", "MMLU-Pro": "MMLU-Pro"}


def drop(name):
    """No data yet: remove a stale figure so that the LaTeX source shows its placeholder instead."""
    for ext in ("pdf", "png"):
        (OUT / f"{name}.{ext}").unlink(missing_ok=True)


def _time(task, cond, model="Ouro-2.6B"):
    return next((r for r in PN.get("time_breakdown", []) if r["task"] == task and r["cond"] == cond
                 and r.get("model", "Ouro-2.6B") == model), None)


def fig_time():
    """Figure 2: on generation tasks the time of a request is almost all decoding. Share of the time per request spent
    in prefill and decode for the released model at one loop (Original 1/1), base model, 200 requests per task; label:
    prefill share and total seconds. The other configurations and HELMET are in Section 5.1 and Table 4 (Appendix A)."""
    rows = [(t, _time(t, "native1st")) for t in TIMED]
    rows = [(t, r) for t, r in rows if r]
    if not rows:
        return drop("fig_time")
    rows.sort(key=lambda tr: tr[1]["prefill_share"])
    fig, ax = plt.subplots(figsize=(3.3, 0.32 * len(rows) + 0.55))
    for y, (t, r) in enumerate(rows):
        p = r["prefill_share"]
        ax.barh(y, p, color=INK, height=0.62, zorder=3)
        ax.barh(y, 100 - p, left=p, color="#DADADA", height=0.62, zorder=2)
        ax.text(p + 1.5, y, f"prefill {p:.1f}%", va="center", ha="left", fontsize=7.5, color=INK, zorder=4)
        ax.text(98.5, y, f"{r['total_s']:.1f} s / request", va="center", ha="right", fontsize=7, color=NAT, zorder=4)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([t for t, _ in rows], fontsize=8)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("share of total time (batch 1)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=INK), plt.Rectangle((0, 0), 1, 1, color="#DADADA")], ["prefill", "decode"],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5, handlelength=1.2)
    save(fig, "fig_time")


def fig_depth():
    """Figure 10 (Appendix B, full data behind Figure 5): per timed task, accuracy (full evaluation set) against prefill
    loops m at decode n = 1 (violet) and n = 2 (green), for m <= 4 (solid) and wide m <= 16 (dashed)."""
    rows = PN.get("depth_sweep", [])
    tasks = [t for t in TIMED if any(r["bench"] == SWEEP_NAME[t] for r in rows)]
    if not tasks:
        return drop("fig_depth")
    timed = False   # accuracy only (time against m: Table 4 and Section 5.5)
    nrow = 2 if timed else 1   # the time row only once per-request timings exist
    fig, axes = plt.subplots(nrow, len(tasks), figsize=(1.75 * len(tasks), 1.3 + 1.25 * nrow), squeeze=False)
    for j, t in enumerate(tasks):
        for n, col in ((1, N1), (2, N2)):
            for wide, ls in ((False, "-"), (True, "--")):
                r = next((r for r in rows if r["bench"] == SWEEP_NAME[t] and r["n"] == n and r["adapter"] == ("wide" if wide else "m<=4")), None)
                if r:
                    ms = sorted(int(m) for m in r["quality"] if int(m) <= 8)   # beyond 8: outside the main training range
                    axes[0, j].plot(ms, [r["quality"][str(m)] if str(m) in r["quality"] else r["quality"][m] for m in ms],
                                    ls, color=col, marker="s", ms=2.5, lw=1.2)
            pts = [(m, _time(t, f"a{m}x{n}sb")) for m in (1, 2, 3, 4, 6, 8, 12, 16)]
            pts = [(m, r) for m, r in pts if r]
            if pts and timed:
                axes[1, j].plot([m for m, _ in pts], [r["total_s"] for _, r in pts], "-", color=col, marker="s", ms=2.5, lw=1.2)
                axes[1, j].plot([m for m, _ in pts], [r["prefill_s"] for _, r in pts], ":", color=col, lw=1)
        axes[0, j].set_title(t, fontsize=8.5, color=INK, pad=4)
        for i in range(nrow):
            axes[i, j].set_xscale("log", base=2); axes[i, j].set_xticks([1, 2, 3, 4, 6, 8]); axes[i, j].set_xticklabels(["1", "2", "3", "4", "6", "8"])
            axes[i, j].minorticks_off()
            axes[i, j].axvline(4, color="#BDBDBD", lw=0.8, zorder=0)
            axes[i, j].grid(color="#EDEDED", lw=0.5); axes[i, j].set_axisbelow(True)
        axes[nrow - 1, j].set_xlabel("prefill loops m", fontsize=7.5)
    axes[0, 0].set_ylabel("accuracy (%)")
    if timed:
        axes[1, 0].set_ylabel("s / request")
    h = [plt.Line2D([], [], color=N1, lw=1.2), plt.Line2D([], [], color=N2, lw=1.2),
         plt.Line2D([], [], color=NAT, ls="-", lw=1.2), plt.Line2D([], [], color=NAT, ls="--", lw=1.2)]
    fig.legend(h, ["decode n = 1", "decode n = 2", "trained on m ≤ 4", "trained on m ≤ 16"], loc="upper center", ncol=4,
               fontsize=7.3, bbox_to_anchor=(0.52, 0.99), columnspacing=1.1, frameon=False)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.78, bottom=0.19, wspace=0.28)
    save(fig, "fig_depth")


DEPTH_MAIN = ["GSM8K", "MMLU-Pro", "HumanEval+", "RULER-16K"]   # math, knowledge, code, long context


def fig_depth_main(mmax=8, n=1):
    """Figure 5: with one decode loop, two to five prefill loops buy most of the accuracy gain. Adapter trained on m in
    {1,...,16} (no extrapolation), four benchmarks from different domains (math, knowledge, code, long context), m up to
    `mmax`; label: gain from m = 1 to the best m. Two decode loops and all tasks: Figure 10 / Table 7."""
    rows = [r for r in PN.get("depth_sweep", []) if r["adapter"] == "wide" and r["n"] == n]
    benches = [b for b in DEPTH_MAIN if any(r["bench"] == b for r in rows)]
    if not benches:
        return drop("fig_depth_main")
    YTICKS = {"GSM8K": [36, 40, 44, 48], "MMLU-Pro": [24, 34, 44, 54],
              "HumanEval+": [48, 54, 60, 66], "RULER-16K": [52, 56, 60, 64]}
    fig, axes = plt.subplots(1, len(benches), figsize=(6.9, 1.72), squeeze=False)
    for j, b in enumerate(benches):
        ax = axes[0, j]
        ax.axvspan(2, 5, color="#F3F3F3", zorder=0, lw=0)   # where the gain peaks (two to five loops, Section 5.4)
        ax.axvline(4, color="#9E9E9E", lw=0.85, ls=":", zorder=1)
        r = next(r for r in rows if r["bench"] == b)
        q = {int(m): v for m, v in r["quality"].items() if int(m) <= mmax}
        ms = sorted(q)
        ax.plot(ms, [q[m] for m in ms], "-s", color=OURS, lw=1.7, ms=3.8, mec="white", mew=0.6, zorder=3)
        best = max(ms, key=lambda m: q[m])
        ax.hlines(q[1], 1, best, color=NAT_LIGHT, lw=0.8, ls=(0, (2, 2)), zorder=2)
        ax.annotate("", (best, q[best] - 0.3), xytext=(best, q[1]), arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.1,
                    mutation_scale=8, shrinkA=0, shrinkB=3), zorder=4)
        ax.set_title(b, fontsize=8.1, color=INK, pad=4, loc="left")
        gain = round(round(q[best], 1) - round(q[1], 1), 1)   # from the values Table 7 shows (paper_numbers depth_gains)
        ax.set_title(f"+{gain:.1f}", fontsize=8.4, color=OURS, fontweight="bold", pad=4, loc="right")
        ticks = [m for m in (1, 2, 3, 4, 5, 6, 8) if m <= mmax]
        ax.set_xticks(ticks); ax.set_xticklabels([str(m) for m in ticks]); ax.minorticks_off()
        ax.set_xlim(0.55, mmax + 0.45)
        ax.set_xlabel("prefill loops m", fontsize=7.6)
        ax.grid(axis="y", color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
        yt = YTICKS[b]
        ax.set_yticks(yt)
        span = yt[-1] - yt[0]
        ax.set_ylim(yt[0] - 0.08 * span, yt[-1] + 0.08 * span)
        ax.tick_params(labelsize=7.4)
        if j == 0:
            ax.text(4.18, yt[0] + 0.03 * span, "pretrained\n(m = 4)", fontsize=5.8, color=NAT, ha="left", va="bottom", linespacing=1.05)
    axes[0, 0].set_ylabel("accuracy (%)", fontsize=8.2)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.84, bottom=0.25, wspace=0.34)
    save(fig, "fig_depth_main")


def fig_sametime():
    """Figure 9 (Appendix A): paired comparison at one decode loop over the 800 timed requests (batch 1, one B300, two
    passes): Original 1/1, Distilled 1/1 baseline, and LoopBridge 4/1 with folded maps."""
    tasks = [t for t in TIMED if _time(t, "native1st")]
    bars = [("Original 1/1\n(tied)", "native1st", NAT_LIGHT), ("Distilled 1/1\n(baseline)", "D1st", DIST),
            ("LoopBridge 4/1\n(ours)", "a4x1pm", OURS)]
    pts = []
    for lab, cond, col in bars:
        rs = [_time(t, cond) for t in tasks]
        if not tasks or None in rs:
            return drop("fig_sametime")
        pts.append((lab, float(np.mean([r["quality"] for r in rs])), float(np.mean([r["total_s"] for r in rs])), col))
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    x = np.arange(len(pts))
    ax.bar(x, [p[1] for p in pts], width=0.52, color=[p[3] for p in pts], zorder=3)
    for xi, (lab, q, t, col) in zip(x, pts):
        ax.text(xi, q + 0.8, f"{q:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold")
        if col == OURS:   # gain over Original (grey) and over Distilled (blue), as the difference of the shown values
            ax.text(xi, q + 10.6, f"+{round(q, 1) - round(pts[0][1], 1):.1f}", ha="center", va="bottom", fontsize=7.8, color=NAT,
                    fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.25), zorder=5)
            ax.text(xi, q + 5.8, f"+{round(q, 1) - round(pts[1][1], 1):.1f}", ha="center", va="bottom", fontsize=7.8, color=DIST,
                    fontweight="bold")
    ax.hlines(pts[0][1], x[0] - 0.26, x[-1] + 0.26, color=NAT, lw=0.7, ls=(0, (2, 2)), alpha=0.6, zorder=4)
    ax.hlines(pts[1][1], x[1] - 0.26, x[-1] + 0.26, color=DIST, lw=0.7, ls=(0, (2, 2)), alpha=0.6, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([f"{lab}\n{t:.1f} s" for lab, q, t, col in pts], fontsize=7.0)
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 66)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlim(-0.55, len(pts) - 0.45)
    ax.spines["bottom"].set_color(NAT); ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig_sametime")


def fig_budget(name="fig_budget"):
    """Figure 6: accuracy against measured time per request on the same 800 replayed requests (batch 1, one B300, mean
    of two passes; paper_numbers time_breakdown). Both curves connect n = 1 and n = 2 to show the accuracy-latency
    frontier (any request mix lies on the segment). The arrow marks the time-matched comparison of Section 5.5: at
    LoopBridge 4/1's mean time, an Original 1/1-2/2 mix lies on the grey segment (time_matched)."""
    tasks = [t for t in TIMED if _time(t, "native1st")]
    if not tasks:
        return drop(name)

    def point(cond):
        rs = [_time(t, cond) for t in tasks]
        return None if None in rs else (float(np.mean([r["total_s"] for r in rs])), float(np.mean([r["quality"] for r in rs])))

    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    nat = [(point(f"native{d}st"), f"Original {d}/{d}") for d in (1, 2)]
    ax.plot([p[0][0] for p in nat], [p[0][1] for p in nat], "-", color=NAT_LIGHT, marker="o", ms=4.4, lw=1.5, zorder=2)
    ax.annotate("Original 1/1", nat[0][0], xytext=(4, -11), textcoords="offset points", fontsize=6.8, color=NAT, ha="center")
    ax.annotate("Original 2/2", nat[1][0], xytext=(5, -16), textcoords="offset points", fontsize=6.8, color=NAT, ha="center")
    ours = [point("a4x1pm"), point("a4x2pm")]
    if None not in ours:
        ax.plot([p[0] for p in ours], [p[1] for p in ours], "-", color=OURS, marker="s", ms=5.0, lw=1.7, zorder=4)
        ax.annotate("Ours 4/1", ours[0], xytext=(-4, 7), textcoords="offset points", fontsize=7.0, color=OURS, fontweight="bold", ha="center")
        ax.annotate("Ours 4/2", ours[1], xytext=(-6, 7), textcoords="offset points", fontsize=7.0, color=OURS, fontweight="bold", ha="center")
    # time-matched comparison at LoopBridge 4/1 (Section 5.5): the grey segment at the same mean time
    (x1, y1), (x2, y2) = nat[0][0], nat[1][0]
    xo, yo = ours[0]
    ym = y1 + (xo - x1) / (x2 - x1) * (y2 - y1)
    ax.annotate("", (xo, yo - 1.3), xytext=(xo, ym + 0.9), arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.1, mutation_scale=8), zorder=5)
    ax.text(xo - 0.35, (yo + ym) / 2 + 0.4, f"+{yo - ym:.1f} at\nsame time", fontsize=6.7, color=OURS, fontweight="bold", va="center", ha="right")
    ax.set_xlim(8.8, 22.8); ax.set_ylim(21.0, 71.0)
    ax.set_xlabel("seconds per request"); ax.set_ylabel("mean accuracy (%)")
    ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    L = plt.Line2D
    h = [L([], [], color=OURS, marker="s", ms=4.5, lw=1.7),
         L([], [], color=NAT_LIGHT, marker="o", ms=4.4, lw=1.5)]
    l = ["Ours (LoopBridge 4/n)", "Original n/n (tied)"]
    fig.legend(h, l, fontsize=6.9, loc="upper center", bbox_to_anchor=(0.54, 1.01), frameon=False, handlelength=1.6, ncol=2, columnspacing=1.2)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.87, bottom=0.18)
    save(fig, name)


def fig_batched(name="fig_batched"):
    """Unified Appendix K wall-time frontier across batch 1, batch 8, and batch 32 (absorbs old Fig 5 so batch-1
    and continuous-batching wall-time frontiers live cleanly in one 3-panel figure with zero duplication)."""
    B = PN.get("batched_serving", {})
    tasks = [t for t in TIMED if _time(t, "native1st")]
    sizes = [b for b in (8, 32) if all(f"{c}@{b}" in B for c in ("native1st", "a4x1pm", "D1st", "native2st"))]
    if not sizes or not tasks:
        return drop(name)
    all_bs = [1] + sizes
    fig, axes = plt.subplots(1, len(all_bs), figsize=(6.9, 2.55), sharey=True, squeeze=False)

    def pt(c, b):
        if b == 1:
            rs = [_time(t, c) for t in tasks]
            if None in rs:
                return None
            return (float(np.mean([r["total_s"] for r in rs])), float(np.mean([r["quality"] for r in rs])))
        return (B[f"{c}@{b}"]["s_per_request"], B[f"{c}@{b}"]["acc"]) if f"{c}@{b}" in B else None

    for idx, (ax, b) in enumerate(zip(axes[0], all_bs)):
        nat = [(pt(f"native{d}st", b), f"{d}/{d}") for d in (1, 2) if pt(f"native{d}st", b)]
        ax.plot([p[0][0] for p in nat], [p[0][1] for p in nat], "-", color=NAT_LIGHT, marker="o", ms=4.0, lw=1.45, zorder=2)
        for (x, y), lab in nat:
            ax.annotate(lab, (x, y), xytext=(3 if lab == "1/1" else -4, -10.5), textcoords="offset points", fontsize=7.0, color=NAT, ha="center")
        for c, m, col, sz in (("D1st", "^", DIST, 28), ("a4x1pm", "s", OURS, 32)):
            p = pt(c, b)
            if p:
                ax.scatter(*p, marker=m, s=sz, zorder=4, color=col)
        p80 = pt("D1r80st", b)
        if p80:
            ax.scatter(*p80, marker="x", s=26, zorder=4, color=DIST, lw=1.2)
        if b == 1:
            p42 = pt("a4x2pm", 1)
            p41 = pt("a4x1pm", 1)
            if p41 and p42:
                ax.plot([p41[0], p42[0]], [p41[1], p42[1]], "-", color=OURS, lw=1.55, zorder=3)
                ax.scatter(*p42, marker="s", s=32, zorder=4, color=OURS)
                ax.annotate("4/2", p42, xytext=(-4, 6.5), textcoords="offset points", fontsize=7.0, color=OURS, fontweight="bold", ha="center")
        p41 = pt("a4x1pm", b)
        if p41 and len(nat) == 2:
            (x1, y1), (x2, y2) = nat[0][0], nat[1][0]
            ym = y1 + (p41[0] - x1) / (x2 - x1) * (y2 - y1)
            ax.annotate("", (p41[0], p41[1] - 1.4), xytext=(p41[0], ym + 1.0),
                        arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.1, mutation_scale=8), zorder=5)
            ax.text(p41[0] - 0.04 * (x2 - x1), (p41[1] + ym) / 2 + 0.8, f"+{p41[1] - ym:.1f}",
                    fontsize=7.0, color=OURS, fontweight="bold", ha="right", va="center")
            ax.annotate("4/1", p41, xytext=(0, 6.5), textcoords="offset points", fontsize=7.0, color=OURS, fontweight="bold", ha="center")
        ax.set_ylim(20, 68.5)
        ax.set_yticks([20, 30, 40, 50, 60])
        if idx > 0:
            ax.tick_params(labelleft=False)
        xlo, xhi = ax.get_xlim(); ax.set_xlim(xlo - 0.08 * (xhi - xlo), xhi + 0.08 * (xhi - xlo))
        ax.set_title(f"batch {b}" + (" (single stream)" if b == 1 else " (continuous)"), fontsize=8.2, color=INK, fontweight="bold", pad=4)
        ax.set_xlabel("seconds per request", fontsize=8.0); ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.4)
    axes[0, 0].set_ylabel("mean accuracy (%)", fontsize=8.2)
    L = plt.Line2D
    h = [L([], [], color=OURS, marker="s", ms=4.5, lw=1.4),
         L([], [], color=NAT_LIGHT, marker="o", ms=4.2, lw=1.4),
         L([], [], color=DIST, marker="^", ms=4.5, ls="none"),
         L([], [], color=DIST, marker="x", ms=4.5, lw=1.2, ls="none")]
    l = ["LoopBridge (ours)", "Original n/n (tied)", "Distilled 1/1 (r16)", "Distilled 1/1 (r80)"]
    fig.legend(h, l, fontsize=7.2, loc="upper center", bbox_to_anchor=(0.52, 1.01), frameon=False, handlelength=1.5, ncol=4, columnspacing=1.0)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.80, bottom=0.18, wspace=0.12)
    save(fig, name)


def tab_batched():
    """Appendix: the batched serving numbers behind fig_batched: accuracy of the served answers (batch 32), wall seconds
    per request at batch 8 and 32, and the prefill share (%) of the wall time at batch 32."""
    B = PN.get("batched_serving", {})
    rows = [("Original 1/1", "native1st"), ("Distilled 1/1", "D1st"), ("Distilled 1/1 r80", "D1r80st"),
            ("LoopBridge 4/1", "a4x1pm")]
    lines = []
    for r in rows:
        if r is None:
            lines.append(r"\midrule"); continue
        lab, c = r
        if f"{c}@32" not in B:
            continue
        sh = r"\rowcolor{oursbg} " if c.startswith(("a4", "f4")) else ""
        t = " & ".join(f"{B[f'{c}@{b}']['s_per_request']:.3f}" if f"{c}@{b}" in B else "--" for b in (8, 32))   # 3 decimals: ratios in Sec 5.5 follow from the printed times
        lines.append(rf"{sh}{lab} & {B[f'{c}@32']['acc']:.1f} & {t} & {B[f'{c}@32']['prefill_pct']:.0f} \\")
    (OUT / "tab_batched.tex").write_text("\n".join(lines) + "\n")


def fig_tbatched(name="fig_tbatched"):
    """Figure 7: Ouro-2.6B-Thinking served in batches of 16 and 32 (paper_numbers thinking_batched): accuracy of the
    served answers against wall seconds per request on all 1,319 GSM8K problems for Original 2/2, Original 4/4, and
    LoopBridge 4/2; horizontal arrow marks LoopBridge 4/2's wall-time reduction relative to Original 4/4."""
    T = PN.get("thinking_batched", {})
    tasks = [("gsmthink", b, f"GSM8K, batch {b}") for b in (16, 32)
             if all(f"gsmthink {c}@{b}" in T for c in ("native2st", "native4st", "D2st", "a4x2pm"))]
    if not tasks:
        return drop(name)
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.3, 2.3), sharey=True, squeeze=False)
    XTICKS = {16: [3, 4, 5, 6, 7], 32: [2.0, 2.5, 3.0, 3.5]}
    for idx, (ax, (t, bs, lab)) in enumerate(zip(axes[0], tasks)):
        pt = lambda c: (T[f"{t} {c}@{bs}"]["s_per_request"], T[f"{t} {c}@{bs}"]["acc"])
        (x2, y2), (x4, y4), (xo, yo) = pt("native2st"), pt("native4st"), pt("a4x2pm")
        ax.plot([x2, x4], [y2, y4], "-", color=NAT_LIGHT, marker="o", ms=4.0, lw=1.4, zorder=2)
        ax.annotate("2/2", (x2, y2), xytext=(4, -10), textcoords="offset points", fontsize=7.0, color=NAT, ha="center")
        ax.annotate("4/4", (x4, y4), xytext=(-2, -10), textcoords="offset points", fontsize=7.0, color=NAT, ha="center")
        ax.scatter(xo, yo, marker="s", s=32, zorder=4, color=OURS)
        ax.annotate("4/2", (xo, yo), xytext=(0, 7), textcoords="offset points", fontsize=7.2, color=OURS, fontweight="bold", ha="center")
        ax.axhline(y4, color=NAT_LIGHT, lw=0.7, ls=":", zorder=1)
        r = T.get(f"{t} a4x2pm vs native4st@{bs}")
        if r:
            span = x4 - xo
            ax.annotate("", xy=(xo + 0.14 * span, y4 + 0.14), xytext=(x4 - 0.08 * span, y4 + 0.14),
                        arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.1, mutation_scale=8), zorder=4)
            ax.text((xo + x4) / 2 + 0.10 * span, y4 + 0.36, rf"{r['ratio']:.2f}$\times$ time", fontsize=6.8,
                    color=OURS, fontweight="bold", ha="center", va="bottom")
        ax.set_ylim(91.5, 95.8)
        ax.set_yticks([92, 93, 94, 95])
        if idx > 0:
            ax.tick_params(labelleft=False)
        xpad = 0.24 * (x4 - xo)
        ax.set_xlim(xo - xpad * 1.15, x4 + xpad * 0.85)
        ax.set_xticks(XTICKS[bs])
        ax.set_title(lab, fontsize=7.8, color=INK, fontweight="bold", pad=4)
        ax.set_xlabel("seconds per request", fontsize=7.4); ax.tick_params(labelsize=6.9)
        ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("accuracy (%)", fontsize=7.6)
    L = plt.Line2D
    h = [L([], [], color=OURS, marker="s", ms=4.5, ls="none"),
         L([], [], color=NAT_LIGHT, marker="o", ms=4.2, lw=1.4)]
    fig.legend(h, ["Ours (LoopBridge 4/2)", "Original n/n (tied)"], fontsize=6.9, loc="upper center",
               bbox_to_anchor=(0.54, 1.01), frameon=False, handlelength=1.5, ncol=2, columnspacing=1.2)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.79, bottom=0.19, wspace=0.12)
    save(fig, name)


def tab_tbatched():
    """Table 15 (Appendix K): Ouro-2.6B-Thinking served in batches (paper_numbers thinking_batched): accuracy and wall
    seconds per request on all 1,319 GSM8K problems (batch 16 and 32)."""
    T = PN.get("thinking_batched", {})
    rows = [("Original 2/2", "native2st"), ("Original 4/4", "native4st"), ("Distilled 2/2", "D2st"),
            ("LoopBridge 4/2", "a4x2pm")]
    lines = []
    for lab, c in rows:
        g16, g32 = T.get(f"gsmthink {c}@16"), T.get(f"gsmthink {c}@32")
        if not g16:
            continue
        cell = lambda r, k, fmt: fmt % r[k] if r else "--"
        sh = r"\rowcolor{oursbg} " if c == "a4x2pm" else ""
        lines.append(rf"{sh}{lab} & {cell(g16, 'acc', '%.1f')} & {cell(g32, 'acc', '%.1f')} & {cell(g16, 's_per_request', '%.2f')} & "
                     rf"{cell(g32, 's_per_request', '%.2f')} \\")
    (OUT / "tab_tbatched.tex").write_text("\n".join(lines) + "\n")


def tab_time():
    """Appendix: per-request distributions behind Figure 2 (200 requests per task, batch 1)."""
    rows = [r for r in PN.get("time_breakdown", []) if "total_median" in r]
    confs = [("a4x1pm", "LoopBridge 4/1"), ("a4x2pm", "LoopBridge 4/2"), ("native1st", "Original 1/1"), ("native2st", "Original 2/2"),
             ("native4st", "Original 4/4")]
    lines = []
    for ti, t in enumerate(TIMED):
        if ti and lines:
            lines.append(r"\midrule")
        for c, lab in confs:
            r = next((r for r in rows if r["task"] == t and r["cond"] == c), None)
            if r:
                sh = r"\rowcolor{oursbg} " if c.startswith("a") else ""
                lines.append(f"{sh}{t} & {lab} & {r['in_median']:,.0f} & {r['out_median']:,.0f} & {r['out_p90']:,.0f} & "
                             f"{r['total_median']:.1f} & {r['total_p90']:.1f} & {r['share_median']:.2f} & {r['share_p90']:.2f} & "
                             f"{r['prefill_share']:.1f} \\\\")   # last: share of the summed time (Figure 2, Section 5.1)
    (OUT / "tab_time.tex").write_text("\n".join(lines) + "\n")


def tab_plateau():
    """Appendix: stopping depth of the prefill gain per benchmark and decode depth (paper_numbers.plateau)."""
    lines = []
    for ki, (key, name) in enumerate((("plateau_extrapolated", r"$m{\le}4$"), ("plateau", r"$m{\le}16$"))):
        if ki and lines:
            lines.append(r"\midrule")
        for r in PN.get(key, []):
            lines.append(f"{name} & {r['bench']} & {r['n']} & {r['m_star']} & {r['test_m1']:.1f} & {r['test_at_star']:.1f} \\\\")
    (OUT / "tab_plateau.tex").write_text("\n".join(lines) + "\n")


def tab_depthwide():
    """Appendix: the depth sweep behind Figure 5, the adapter trained on m in {1,2,3,4,6,8,12,16} (paper_numbers
    depth_sweep, adapter "wide"), accuracy per benchmark and decode depth against prefill loops m; best m in bold."""
    rows = [r for r in PN.get("depth_sweep", []) if r["adapter"] == "wide"]
    if not rows:
        return (OUT / "tab_depthwide.tex").unlink(missing_ok=True)
    ms = [1, 2, 3, 4, 5, 6, 8, 12, 16]
    order = ["MMLU", "MMLU-Pro", "BBH 3-shot", "ARC-C", "GSM8K", "MATH-500", "HumanEval+", "MBPP+", "RULER-16K"]
    lines = []
    for n in (1, 2):
        if n == 2 and lines:
            lines.append(r"\midrule")
        for b in order:
            r = next((r for r in rows if r["n"] == n and r["bench"] == b), None)
            if r is None:
                continue
            q = {int(k): v for k, v in r["quality"].items()}
            best = max(q.values())
            cell = lambda m: "--" if m not in q else (f"\\textbf{{{q[m]:.1f}}}" if f"{q[m]:.1f}" == f"{best:.1f}" else f"{q[m]:.1f}")
            lines.append(f"{b.replace(' 3-shot', '')} & {n} & " + " & ".join(cell(m) for m in ms) + r" \\")
    (OUT / "tab_depthwide.tex").write_text("\n".join(lines) + "\n")


def tab_depth14():
    """Appendix: Ouro-1.4B depth sweep (paper_numbers.depth_sweep_14), accuracy per benchmark and decode depth against
    prefill loops m; the best m of each row in bold; '--' where that depth was not evaluated."""
    rows = PN.get("depth_sweep_14", [])
    if not rows:
        return (OUT / "tab_depth14.tex").unlink(missing_ok=True)
    ms = [1, 2, 3, 4]   # the adapter's training range (m, n <= 4)
    lines = []
    prev_n = None
    for r in rows:
        if prev_n is not None and r["n"] != prev_n:
            lines.append(r"\midrule")
        prev_n = r["n"]
        q = {int(k): v for k, v in r["quality"].items() if int(k) in ms}
        best = max(q.values())
        cell = lambda m: "--" if m not in q else (f"\\textbf{{{q[m]:.1f}}}" if f"{q[m]:.1f}" == f"{best:.1f}" else f"{q[m]:.1f}")
        lines.append(f"{r['bench']} & {r['n']} & " + " & ".join(cell(m) for m in ms) + " \\\\")
    (OUT / "tab_depth14.tex").write_text("\n".join(lines) + "\n")


def tab_prefilltrn():
    """Appendix (v2): prefill depth sweep with the prefill frozen or trained (paper_numbers.prefill_trainable), accuracy
    per benchmark, decode depth and arm against prefill loops m, items both arms share; the best m of each row in bold;
    last column the stopping depth m* of the plateau rule ('--' where the rule finds none)."""
    P = PN.get("prefill_trainable", {})
    if not P.get("curves"):
        return (OUT / "tab_prefilltrn.tex").unlink(missing_ok=True)
    ms = [1, 2, 3, 4, 5, 6, 8, 12, 16]
    lines = []
    prev_n = None
    for idx, r in enumerate(P["curves"]):
        if prev_n is not None and r["n"] != prev_n:
            lines.append(r"\midrule")
        elif idx > 0:
            lines.append(r"\addlinespace[2pt]")
        prev_n = r["n"]
        for arm, q in r["quality"].items():
            q = {int(k): v for k, v in q.items()}
            best = max(q.values())
            cell = lambda m: "--" if m not in q else (f"\\textbf{{{q[m]:.1f}}}" if f"{q[m]:.1f}" == f"{best:.1f}" else f"{q[m]:.1f}")
            star = P.get("stops", {}).get(f"{r['n']}/{r['bench']}", {}).get(arm)
            lines.append(f"{r['bench']} & {r['n']} & {arm.split()[0]} & " + " & ".join(cell(m) for m in ms)
                         + f" & {star if star is not None else '--'} \\\\")
    (OUT / "tab_prefilltrn.tex").write_text("\n".join(lines) + "\n")


def tab_build():
    """Main text, Table 2a: LoopBridge architectural ablation at prefill 4 / decode 1 (every row serves all trained
    (m, n) pairs): KV maps vs decoder-only LoRA (rank 16 and parameter-matched rank 80) and one-cache variant."""
    T = PN["main_table"]
    avg = lambda l: (None if any(r.get(l) is None for r in T) else sum(r[l] for r in T) / len(T))
    one = lambda l, b: next((r.get(l) for r in T if r["bench"] == b), None)
    f = lambda x: "--" if x is None else f"{x:.1f}"
    build = [("LoRA r16, all $(m,n)$", "aligned LoRA-only 4/1", False, False),
             ("LoopBridge (LoRA r16 + KV maps)", "aligned adapter 4/1", True, True),
             ("one-cache LoopBridge", "loop adapter 4/1", True, False)]
    lines = []
    R = PN.get("lora80", {}).get("a41")   # rank-80 all-(m,n) LoRA with frozen maps (63M params, matching LoopBridge)
    for name, l, has_maps, default in build:
        mark = "\\cmark" if has_maps else "\\xmark"
        row = f"{name} & {mark} & {f(avg(l))} & {f(one(l, 'RULER-16K'))} \\\\"
        lines.append(("\\rowcolor{oursbg} " + row) if default else row)
        if l == "aligned LoRA-only 4/1" and R:
            lines.append(f"LoRA r80, all $(m,n)$ & \\xmark & "
                         f"{f(R['avg']['LoRA r80, maps frozen'])} & {f(R['ruler']['LoRA r80, maps frozen'])} \\\\")
    (OUT / "tab_build.tex").write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    # Zero-duplication figure set (Fig 1-3 + deduplicated appendix figures) + 16 generated tables
    fig_tps_teaser(); fig_time(); fig0(); fig_aligned(); fig_depth_main(); fig_tbatched(); fig_lev()
    fig_sametime(); fig_depth(); fig_equal(); fig_batched(); fig2()
    tab_main(); tab_build(); tab_tokens(); tab_lr(); tab_time(); tab_prefilltrn(); tab_plateau(); tab_depthwide()
    tab_depth14(); tab_aligned(); tab_repro(); tab_main("main_table_14", "tab_main14"); tab_equal("2.6B-T")
    tab_batched(); tab_tbatched(); tab_lev()
    for f in OUT.glob("tab_*.tex"):   # one name per benchmark in every table: BBH is 3-shot throughout (Appendix H)
        f.write_text(f.read_text().replace("BBH 3-shot", "BBH"))
    print("\n".join(sorted(p.name for p in OUT.iterdir())))
