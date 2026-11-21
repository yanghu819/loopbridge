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


def legend(fig, ours, labels, y=1.1, native="Original n/n", ncol=3):
    h = [plt.Line2D([], [], color=NAT_LIGHT, marker="o", ms=4, lw=1.4), plt.Line2D([], [], color=OURS, marker="s", ms=4, lw=1.4)]
    names = [native, ours]
    if labels:
        h.append(plt.Line2D([], [], ls="none")); names.append(labels)
    fig.legend(h, names, loc="upper center", ncol=ncol, fontsize=8, bbox_to_anchor=(0.5, y),
               handlelength=1.6, frameon=False)


def fig_tps_teaser():
    """Figure 1 (teaser): accuracy against measured end-to-end throughput (output tokens/s) on the exact same scored
    requests. Top (a): Ouro-2.6B base, 4 chain-of-thought tasks (800 requests, batch 1, mean of two passes). Bottom (b):
    Ouro-2.6B-Thinking, all 1,319 GSM8K problems served with continuous batching (batch 16, mean of three passes)."""
    tasks = [t for t in TIMED if _time(t, "native1st")]
    T = PN.get("thinking_batched", {})
    if not tasks or "gsmthink a4x2pm@16" not in T:
        return drop("fig_tps_teaser")

    def base_pt(cond):
        rs = [_time(t, cond) for t in tasks]
        if None in rs:
            return None
        tps = sum(r["out_tokens"] for r in rs) / sum(r["total_s"] for r in rs)
        acc = float(np.mean([r["quality"] for r in rs]))
        return tps, acc

    def think_pt(cond):
        r = T[f"gsmthink {cond}@16"]
        return r["tokens"] / r["s_per_request"], r["acc"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.3, 3.35), gridspec_kw=dict(height_ratios=[1.14, 0.90]))

    # --- Panel (a): Ouro-2.6B Base, 4 CoT tasks, batch 1 (all points on the exact same 800 requests) ---
    nat_b = [(base_pt(f"native{d}st"), f"{d}/{d}") for d in (4, 2, 1)]
    ax1.plot([p[0][0] for p in nat_b], [p[0][1] for p in nat_b], "-", color=NAT_LIGHT, marker="o", ms=4.4, lw=1.5, zorder=2)
    nat_offs = {"4/4": (0, -11), "2/2": (-8, -10), "1/1": (0, -10)}
    for (x, y), lab in nat_b:
        ox, oy = nat_offs[lab]
        ax1.annotate(lab, (x, y), xytext=(ox, oy), textcoords="offset points", fontsize=6.9, color=NAT, ha="center")

    lb2, lb1 = base_pt("a4x2pm"), base_pt("a4x1pm")
    ax1.plot([lb2[0], lb1[0]], [lb2[1], lb1[1]], "-", color=OURS, lw=1.7, zorder=3)
    for (x, y) in (lb2, lb1):
        ax1.scatter(x, y, marker="s", s=32, color=OURS, zorder=4)
    ax1.annotate("4/2", lb2, xytext=(6, 2), textcoords="offset points", fontsize=7.0, color=OURS, fontweight="bold", ha="left", va="bottom")
    ax1.annotate("4/1", lb1, xytext=(0, 6), textcoords="offset points", fontsize=7.0, color=OURS, fontweight="bold", ha="center")

    # Gain callouts (printed-value differences: 45.0 - 27.8 = +17.2; TPS ratio: 14.3 / 6.7 = 2.1x)
    g1 = round(lb1[1], 1) - round(nat_b[2][0][1], 1)
    ax1.hlines(lb1[1], lb1[0] + 0.45, nat_b[2][0][0] + 0.45, color=OURS, lw=0.75, ls=":", zorder=3)
    ax1.annotate("", xy=(nat_b[2][0][0], lb1[1] - 0.5), xytext=(nat_b[2][0][0], nat_b[2][0][1] + 2.4),
                 arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.05, mutation_scale=8), zorder=4)
    ax1.text(nat_b[2][0][0] + 0.65, (lb1[1] + nat_b[2][0][1]) / 2, f"+{g1:.1f}", fontsize=7.1, color=OURS,
             fontweight="bold", ha="left", va="center")
    ax1.text(14.3, 71.8, rf"{lb2[0] / nat_b[0][0][0]:.1f}$\times$ TPS of 4/4", fontsize=7.0, color=OURS,
             fontweight="bold", ha="left", va="center")

    ax1.set_title("(a) Ouro-2.6B, 4 CoT tasks (batch 1)", fontsize=8.0, color=INK, fontweight="bold", loc="left", pad=3)
    ax1.set_xlabel("output tokens/s", fontsize=7.1, color=NAT, labelpad=1.5)
    ax1.set_ylabel("mean accuracy (%)", fontsize=7.8)
    ax1.set_xlim(4.5, 33.4); ax1.set_ylim(19.5, 76.5)
    ax1.set_xticks([5, 10, 15, 20, 25, 30])
    ax1.grid(color="#EDEDED", lw=0.5); ax1.set_axisbelow(True)
    ax1.tick_params(labelsize=7.2)

    # --- Panel (b): Ouro-2.6B-Thinking, GSM8K (all 1,319), batch 16 ---
    n4, n2, o42 = think_pt("native4st"), think_pt("native2st"), think_pt("a4x2pm")
    ax2.plot([n4[0], n2[0]], [n4[1], n2[1]], "-", color=NAT_LIGHT, marker="o", ms=4.2, lw=1.5, zorder=2)
    ax2.axhline(n4[1], color=NAT_LIGHT, lw=0.7, ls=":", zorder=1)
    ax2.annotate("Original 4/4", n4, xytext=(0, -11), textcoords="offset points", fontsize=6.8, color=NAT, ha="left")
    ax2.annotate("Original 2/2", n2, xytext=(-4, -10), textcoords="offset points", fontsize=6.8, color=NAT, ha="right")
    ax2.scatter(*o42, marker="s", s=34, color=OURS, zorder=4)
    ax2.annotate("Ours 4/2", o42, xytext=(0, 6), textcoords="offset points", fontsize=7.0, color=OURS, fontweight="bold", ha="center")

    # Horizontal speedup arrow at Original 4/4 accuracy level
    ax2.annotate("", xy=(o42[0] - 5, n4[1] + 0.14), xytext=(n4[0] + 5, n4[1] + 0.14),
                 arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.1, mutation_scale=8), zorder=4)
    ax2.text((n4[0] + o42[0]) / 2 - 4, n4[1] + 0.36, rf"{o42[0] / n4[0]:.1f}$\times$ TPS, +{o42[1] - n4[1]:.1f} acc",
             fontsize=7.0, color=OURS, fontweight="bold", ha="center", va="bottom")

    ax2.set_title("(b) Ouro-2.6B-Thinking, GSM8K (batch 16)", fontsize=8.0, color=INK, fontweight="bold", loc="left", pad=3)
    ax2.set_xlabel("end-to-end throughput (output tokens/s)", fontsize=7.8)
    ax2.set_ylabel("accuracy (%)", fontsize=7.8)
    ax2.set_xlim(105, 272); ax2.set_ylim(91.7, 95.9)
    ax2.set_yticks([92, 93, 94, 95])
    ax2.grid(color="#EDEDED", lw=0.5); ax2.set_axisbelow(True)
    ax2.tick_params(labelsize=7.2)

    L = plt.Line2D
    h = [L([], [], color=OURS, marker="s", ms=4.5, lw=1.6),
         L([], [], color=NAT_LIGHT, marker="o", ms=4.4, lw=1.5)]
    l = ["Ours (LoopBridge 4/n)", "Original n/n (tied)"]
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=6.9, bbox_to_anchor=(0.52, 1.005),
               handlelength=1.5, columnspacing=1.2, frameon=False)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.90, bottom=0.105, hspace=0.54)
    save(fig, "fig_tps_teaser")


def fig0():
    """Figure 3 (method, timed form): the frozen model prefills the prompt once with m loops; LoopBridge maps the prompt
    KV of prefill loop max(m-n+k, 1) once per prompt into a cache for decode loop k (key scale per rotary pair, value
    map); every token then runs n decode loops (frozen model + LoRA), loop k attending to its own cache and answer KV."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(6.9, 2.05))
    ax.set_xlim(0, 100); ax.set_ylim(0, 31); ax.axis("off")
    LIGHT = "#FDEBE3"   # light tint of OURS: the adapter's parts
    GREY = "#F2F2F2"

    def box(x, y, w, h, text, fc=GREY, ec=NAT_LIGHT, lw=0.7, size=7.2, color=INK, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, fontweight=weight)

    def arrow(x0, y0, x1, y1, c=NAT, lw=0.8):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=c, mutation_scale=8, shrinkA=1, shrinkB=1))

    dots = lambda x, y: ax.text(x, y, "$\\vdots$", ha="center", va="center", fontsize=10, color=NAT)
    for x, t in ((11.5, "Prefill: once, m loops"), (47.5, "LoopBridge: once per prompt"), (85.5, "Decode: n loops per token")):
        ax.text(x, 29.2, t, ha="center", va="center", fontsize=7.5, color=INK, fontweight="bold")
    # prefill column (frozen model)
    box(1.5, 21, 20, 4, "prefill loop m")
    dots(11.5, 18.0)
    box(1.5, 11, 20, 4, "prefill loop m-n+1")
    dots(11.5, 8.2)
    box(1.5, 1.5, 20, 4, "prefill loop 1")
    # maps column (adapter)
    box(34.5, 20.6, 26, 4.8, "cache n:  $c_n \\odot K$,  $W_n V$", fc=LIGHT, ec=OURS, lw=1.0)
    dots(47.5, 18.0)
    box(34.5, 10.6, 26, 4.8, "cache 1:  $c_1 \\odot K$,  $W_1 V$", fc=LIGHT, ec=OURS, lw=1.0)
    ax.text(47.5, 5.6, "key scale per rotary pair, value map;\nfolded into the prompt cache once", ha="center",
            va="center", fontsize=6.8, color=OURS)
    arrow(21.9, 23, 34.1, 23, OURS); arrow(21.9, 13, 34.1, 13, OURS)
    ax.text(28.0, 24.4, "K, V", ha="center", fontsize=6.8, color=NAT); ax.text(28.0, 14.4, "K, V", ha="center", fontsize=6.8, color=NAT)
    # decode column (frozen model + LoRA)
    box(72.5, 21, 26, 4, "decode loop n (+ LoRA)", ec=OURS, lw=0.9)
    dots(85.5, 17.3)
    box(72.5, 11, 26, 4, "decode loop 1 (+ LoRA)", ec=OURS, lw=0.9)
    box(79.5, 1.5, 12, 3.6, "token", fc="white")
    arrow(85.5, 5.4, 85.5, 10.6); arrow(85.5, 19.0, 85.5, 20.6)
    arrow(60.9, 23, 72.1, 23, OURS); arrow(60.9, 13, 72.1, 13, OURS)
    arrow(99.2, 23, 103.2, 23); ax.text(104.0, 23, "next\ntoken", ha="left", va="center", fontsize=6.6, color=INK)
    ax.text(93.2, 5.6, "loop k attends to cache k\nand to the answer's KV", ha="left", va="center", fontsize=6.3, color=NAT)
    ax.set_xlim(0, 115)
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
    ax.set_xlabel("decode tokens/s, best batch, 8K prompt", fontsize=8)
    ax.set_ylabel("average accuracy (%)", fontsize=8)
    legend(fig, "Ours (LoopBridge, one cache)", "", y=1.02, ncol=2)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.18, top=0.80)
    save(fig, "fig2_frontier")


def fig_equal():
    """Ouro-2.6B-Thinking: native 2/2 and 4/4 and Distilled 2/2 against LoopBridge at prefill 4 / decode 2."""
    E = {e["bench"]: e for e in PN["equal_aligned"] if e.get("model", "2.6B-T") == "2.6B-T"}   # LoopBridge (per-loop caches)
    benches = ["GSM8K", "MATH-500", "MMLU-Pro", "ARC-C", "BBH", "HumanEval+", "MBPP+", "HELMET"]
    fig, axes = plt.subplots(2, 4, figsize=(6.9, 3.4))
    for ax, b in zip(axes.flat, benches):
        e = E[b]
        below = e["ours"] < e["native2"]   # ours under Original 2/2: its label goes below, 2/2's above
        frontier(ax, [(2, e["native2"], "2/2", 5 if below else -10), (4, e["native4"], "4/4")],
                 [(2, e["ours"], "4/2", -12 if below else 5)], b)
        ax.set_xlim(1.5, 4.5); ax.set_xticks([2, 3, 4])
        lo, hi = ax.get_ylim()
        if hi - lo < 6:   # at least 6 points per panel, so differences within noise are not drawn as large gaps
            mid = (lo + hi) / 2
            ax.set_ylim(mid - 3, mid + 3)
        low = e["ours"] if below else e["native2"]   # the point whose label goes below it: keep the label off the axis
        lo, hi = ax.get_ylim()
        span = hi - lo
        ax.set_ylim(min(lo, low - 0.28 * span), max(hi, max(e["ours"], e["native4"]) + 0.24 * span))
        if "dist2" in e:   # Distilled 2/2, drawn a little left of n = 2: off the two points there and off the grey line
            ax.scatter([1.7], [e["dist2"]], marker="^", s=20, color=DIST, zorder=3)
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, e["dist2"] - 0.06 * (hi - lo)), max(hi, e["dist2"] + 0.06 * (hi - lo)))
        ax.set_title(b, fontsize=8.5, color=INK, pad=6)
    fig.supxlabel("decode loops n", fontsize=8.5, y=0.015); fig.supylabel("score (%)", fontsize=8.5, x=0.005)
    h = [plt.Line2D([], [], color=NAT_LIGHT, marker="o", ms=4, lw=1.4), plt.Line2D([], [], color=OURS, marker="s", ms=4.5, ls="none"),
         plt.Line2D([], [], color=DIST, marker="^", ms=4.5, ls="none")]
    fig.legend(h, ["Original n/n (tied)", "Ours, prefill 4 / decode 2", "Distilled 2/2 (baseline)"], loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.01), handlelength=1.6, frameon=False)
    fig.subplots_adjust(wspace=0.35, hspace=0.48, left=0.07, right=0.99, bottom=0.125, top=0.865)
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


def fig_aligned(panels=None, name="fig_taskmap", ncols=4):
    """Figure 4: full 4x4 (m, n) grouped 2D bar chart of LoopBridge across m in {1..4} and n in {1..4}, against
    Original n/n (dashed horizontal marker per decode group). Left: 11-benchmark average; right 2x2: four representative
    domains (MMLU, BBH, GSM8K, RULER-16K). Every cell of all 11 benchmarks is in Appendix Table tab:alignedcells."""
    rows = PN.get("aligned_cells", [])
    if not rows:
        return drop(name)
    T = {r["bench"]: r for r in rows}
    avg_cell = {f"native{n}": float(np.mean([r[f"native{n}"] for r in rows])) for n in range(1, 5)}
    for m in range(1, 5):
        for n in range(1, 5):
            avg_cell[f"a{m}{n}"] = float(np.mean([r[f"a{m}{n}"] for r in rows]))

    M_COLS = ["#FBD8C8", "#F7AC88", "#F1824F", OURS]
    M_EDGE = ["#D49A7E", "#D67E56", "#C85C2D", "#B54417"]

    fig = plt.figure(figsize=(6.9, 3.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.48, 1.0, 1.0], wspace=0.26, hspace=0.48,
                          left=0.065, right=0.99, top=0.84, bottom=0.14)
    ax_avg = fig.add_subplot(gs[:, 0])
    dom_specs = [("MMLU", "MMLU", gs[0, 1]), ("BBH 3-shot", "BBH", gs[0, 2]),
                 ("GSM8K", "GSM8K", gs[1, 1]), ("RULER-16K", "RULER-16K", gs[1, 2])]

    xs = np.arange(1, 5)
    bw = 0.185
    offs = np.array([-1.5, -0.5, 0.5, 1.5]) * bw

    def draw_bars(ax, cell, title, big=False):
        for m in range(1, 5):
            vals = [cell[f"a{m}{n}"] for n in range(1, 5)]
            ax.bar(xs + offs[m - 1], vals, width=bw * 0.92, color=M_COLS[m - 1], edgecolor=M_EDGE[m - 1],
                   linewidth=0.45 if m < 4 else 0.6, zorder=3)
        for idx, n in enumerate(range(1, 5)):
            nv = cell[f"native{n}"]
            ax.hlines(nv, xs[idx] - 0.40, xs[idx] + 0.40, color="#262626", lw=1.35 if big else 1.15,
                      ls="--", zorder=5)
            if big:
                a4 = cell[f"a4{n}"]
                y_lab = max(a4, nv) + 1.4 if n >= 3 else a4 + 1.4
                ax.text(xs[idx] + offs[3], y_lab, f"{a4:.1f}", ha="center", va="bottom",
                        fontsize=6.8, color=OURS, fontweight="bold", zorder=6)
                pill = dict(fc="white", ec="none", alpha=0.88, pad=0.4)
                if n == 1:
                    ax.text(xs[idx], nv - 2.2, f"Orig {nv:.1f}", ha="center", va="top",
                            fontsize=6.1, color="#262626", zorder=6, bbox=pill)
                elif n == 2:
                    ax.text(xs[idx] - 0.39, nv - 1.2, f"{nv:.1f}", ha="left", va="top",
                            fontsize=6.0, color="#262626", zorder=6, bbox=pill)
                else:
                    ax.text(xs[idx] - 0.39, nv - 1.1, f"{nv:.1f}", ha="left", va="top",
                            fontsize=6.0, color="#262626", zorder=6, bbox=pill)
        ax.set_title(title, fontsize=8.5 if big else 8.0, color=INK, fontweight="bold" if big else "normal", pad=4)
        ax.set_xticks(xs); ax.set_xticklabels([f"n={n}" for n in range(1, 5)], fontsize=7.5 if big else 7.2)
        ax.set_xlim(0.45, 4.55)
        if big:
            ymax = max([cell[f"native{n}"] for n in range(1, 5)] + [cell[f"a{m}{n}"] for m in range(1, 5) for n in range(1, 5)])
            ax.set_ylim(0, ymax * 1.20)
        else:
            ax.set_ylim(0, 88)
            ax.set_yticks([0, 25, 50, 75])
        ax.grid(axis="y", color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=7.2)

    draw_bars(ax_avg, avg_cell, "11-Benchmark Average", big=True)
    ax_avg.set_ylabel("accuracy (%)", fontsize=8.5)
    ax_avg.set_xlabel("decode loops n", fontsize=8.2)
    # Highlight the asymmetry directly on the 11-benchmark average panel
    ax_avg.annotate("+14.0", xy=(1 + offs[3], avg_cell["a41"] + 6.4), ha="center", va="bottom",
                    fontsize=7.2, color=OURS, fontweight="bold")
    ax_avg.annotate("+5.4", xy=(2 + offs[3], avg_cell["a42"] + 6.4), ha="center", va="bottom",
                    fontsize=7.2, color=OURS, fontweight="bold")

    for key, title, slot in dom_specs:
        ax = fig.add_subplot(slot)
        draw_bars(ax, T[key], title, big=False)
        if slot in (gs[1, 1], gs[1, 2]):
            ax.set_xlabel("decode loops n", fontsize=7.5)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=M_COLS[i], ec=M_EDGE[i], lw=0.5) for i in range(4)]
    handles.append(plt.Line2D([], [], color="#262626", lw=1.35, ls="--"))
    labels = ["LoopBridge: m=1", "m=2", "m=3", "m=4 (prefill loops)", "Original n/n"]
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.52, 0.99),
               handlelength=1.4, columnspacing=1.0, frameon=False)
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
            "training-free 4/1": (r"Training-free {\scriptsize(baseline)}", "4/1"), "training-free 4/2": (r"Training-free {\scriptsize(baseline)}", "4/2"),
            "LoRA-distilled 1/1": (r"Distilled {\scriptsize(KD baseline)}", "1/1"), "LoRA-distilled 2/2": (r"Distilled {\scriptsize(KD baseline)}", "2/2"),
            "distilled r80 1/1": (r"Distilled r80 {\scriptsize(KD base.)}", "1/1"), "distilled r80 2/2": (r"Distilled r80 {\scriptsize(KD base.)}", "2/2"),
            "fixed-4 LoRA 4/1": (r"Ours, fixed $m$", "4/1"), "fixed-4 LoRA 4/2": (r"Ours, fixed $m$", "4/2"),
            "aligned LoRA-only 4/1": ("LoRA all pairs, maps frozen", "4/1"), "aligned LoRA-only 4/2": ("LoRA all pairs, maps frozen", "4/2"),
            "loop adapter 4/1": ("LoopBridge, one cache", "4/1"), "loop adapter 4/2": ("LoopBridge, one cache", "4/2"),
            "aligned adapter 4/1": ("Ours, LoopBridge", "4/1"), "aligned adapter 4/2": ("Ours, LoopBridge", "4/2")}
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
    lines = ["tokens & " + " & ".join(t for _, t in toks) + r" \\", r"\midrule"]
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
    short = {"MATH-500": "MATH", "BBH 3-shot": "BBH", "HumanEval+": "HE+",   # the column names of Table 1
             "HellaSwag": "HellaSw.", "Winogrande": "Winogr.", "RULER-16K": "RULER"}
    lines = [" & " + " & ".join(short.get(r["bench"], r["bench"]) for r in rows) + r" & Avg. \\", r"\midrule"]
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
            lines.append(rf"{m}\,/\,{n} & " + " & ".join(cells) + f" & {acell}" + r" \\")
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
    """Figure 10 (Appendix D, full data behind Figure 5): per timed task, accuracy (full evaluation set) against prefill
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
            axes[i, j].set_xscale("log", base=2); axes[i, j].set_xticks([1, 2, 4, 8]); axes[i, j].set_xticklabels(["1", "2", "4", "8"])
            axes[i, j].axvline(4, color="#BDBDBD", lw=0.8, zorder=0)
            axes[i, j].grid(color="#EDEDED", lw=0.5); axes[i, j].set_axisbelow(True)
        axes[nrow - 1, j].set_xlabel("prefill loops m", fontsize=7.5)
    axes[0, 0].set_ylabel("accuracy (%)")
    if timed:
        axes[1, 0].set_ylabel("s / request")
    h = [plt.Line2D([], [], color=N1, lw=1.2), plt.Line2D([], [], color=N2, lw=1.2),
         plt.Line2D([], [], color=NAT, ls="--", lw=1.2)]
    fig.legend(h, ["decode n = 1", "decode n = 2", "trained on m ≤ 16"], loc="upper center", ncol=3,
               fontsize=7.5, bbox_to_anchor=(0.52, 0.99), frameon=False)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.78, bottom=0.19, wspace=0.28)
    save(fig, "fig_depth")


DEPTH_MAIN = ["GSM8K", "MMLU-Pro", "HumanEval+", "RULER-16K"]   # math, knowledge, code, long context


def fig_depth_main(mmax=8, n=1):
    """Figure 5: with one decode loop, two to five prefill loops buy most of the accuracy gain. Adapter trained on m in
    {1,...,16} (no extrapolation), four benchmarks from different domains (math, knowledge, code, long context), m up to
    `mmax`; label: gain from m = 1 to the best m. Two decode loops and all tasks: Figure 10 / Table 7."""
    from matplotlib.ticker import MaxNLocator
    rows = [r for r in PN.get("depth_sweep", []) if r["adapter"] == "wide" and r["n"] == n]
    benches = [b for b in DEPTH_MAIN if any(r["bench"] == b for r in rows)]
    if not benches:
        return drop("fig_depth_main")
    fig, axes = plt.subplots(1, len(benches), figsize=(6.9, 1.55), squeeze=False)
    for j, b in enumerate(benches):
        ax = axes[0, j]
        ax.axvspan(2, 5, color="#F3F3F3", zorder=0, lw=0)   # where the gain peaks (two to five loops, Section 5.4)
        r = next(r for r in rows if r["bench"] == b)
        q = {int(m): v for m, v in r["quality"].items() if int(m) <= mmax}
        ms = sorted(q)
        ax.plot(ms, [q[m] for m in ms], "-o", color=OURS, lw=1.8, ms=4.2, mec="white", mew=0.8, zorder=3)
        best = max(ms, key=lambda m: q[m])
        ax.hlines(q[1], 1, best, color=NAT_LIGHT, lw=0.8, ls=(0, (2, 2)), zorder=2)
        ax.annotate("", (best, q[best]), xytext=(best, q[1]), arrowprops=dict(arrowstyle="->", color=INK, lw=0.9,
                    shrinkA=0, shrinkB=4), zorder=4)
        ax.set_title(b, fontsize=8.5, color=INK, pad=3, loc="left")
        gain = round(round(q[best], 1) - round(q[1], 1), 1)   # from the values Table 7 shows (paper_numbers depth_gains)
        ax.set_title(f"+{gain:.1f}", fontsize=9, color=OURS, fontweight="bold", pad=3, loc="right")
        ax.set_xscale("log", base=2); ticks = [m for m in (1, 2, 3, 4, 6, 8, 12, 16) if m <= mmax]
        ax.set_xticks(ticks); ax.set_xticklabels([str(m) for m in ticks]); ax.minorticks_off()
        ax.set_xlabel("prefill loops m", fontsize=7.5)
        ax.grid(axis="y", color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.10 * (hi - lo))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    axes[0, 0].set_ylabel("accuracy (%)")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_depth_main")


def fig_sametime():
    """Figure 9 (Appendix D): paired comparison at one decode loop over the 800 timed requests (batch 1, one B300, two
    passes): Original 1/1, Distilled 1/1 baseline, and LoopBridge 4/1 with folded maps."""
    tasks = [t for t in TIMED if _time(t, "native1st")]
    bars = [("Original\n1/1", "native1st", NAT_LIGHT), ("Distilled 1/1\n(baseline)", "D1st", DIST),
            ("Ours, LoopBridge\n4/1", "a4x1pm", OURS)]
    pts = []
    for lab, cond, col in bars:
        rs = [_time(t, cond) for t in tasks]
        if not tasks or None in rs:
            return drop("fig_sametime")
        pts.append((lab, float(np.mean([r["quality"] for r in rs])), float(np.mean([r["total_s"] for r in rs])), col))
    fig, ax = plt.subplots(figsize=(3.3, 1.95))
    x = np.arange(len(pts))
    ax.bar(x, [p[1] for p in pts], width=0.52, color=[p[3] for p in pts], zorder=3)
    for xi, (lab, q, t, col) in zip(x, pts):
        ax.text(xi, q + 0.8, f"{q:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold")
        if col == OURS:   # gain over Original (grey) and over Distilled (blue), as the difference of the shown values
            ax.text(xi, q + 10.2, f"+{round(q, 1) - round(pts[0][1], 1):.1f}", ha="center", va="bottom", fontsize=7.8, color=NAT,
                    fontweight="bold")
            ax.text(xi, q + 5.6, f"+{round(q, 1) - round(pts[1][1], 1):.1f}", ha="center", va="bottom", fontsize=7.8, color=DIST,
                    fontweight="bold")
    ax.hlines(pts[0][1], x[0] - 0.26, x[-1] + 0.26, color=NAT, lw=0.7, ls=(0, (2, 2)), zorder=4)
    ax.hlines(pts[1][1], x[1] - 0.26, x[-1] + 0.26, color=DIST, lw=0.7, ls=(0, (2, 2)), zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([f"{lab}\n{t:.1f} s" for lab, q, t, col in pts], fontsize=7.0)
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, max(p[1] for p in pts) * 1.42)
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
    fig.legend(h, l, fontsize=6.8, loc="lower center", frameon=False, handlelength=1.6, ncol=2, columnspacing=1.2)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.27)
    save(fig, name)


def fig_batched(name="fig_batched"):
    """Figure 13 (Appendix K): Figure 6 measured with batched serving (batch_requests.py, continuous batching): mean
    accuracy of the served answers against wall seconds per request at batch 8 and 32 (paper_numbers batched_serving)."""
    B = PN.get("batched_serving", {})
    sizes = [b for b in (8, 32) if all(f"{c}@{b}" in B for c in ("native1st", "a4x1pm", "D1st", "native2st"))]   # batch 1: Figure 6
    if not sizes:
        return drop(name)
    fig, axes = plt.subplots(1, len(sizes), figsize=(2.4 * len(sizes), 2.65), squeeze=False)
    pt = lambda c, b: (B[f"{c}@{b}"]["s_per_request"], B[f"{c}@{b}"]["acc"]) if f"{c}@{b}" in B else None
    for ax, b in zip(axes[0], sizes):
        nat = [(pt(f"native{d}st", b), f"{d}/{d}") for d in (1, 2) if pt(f"native{d}st", b)]
        ax.plot([p[0][0] for p in nat], [p[0][1] for p in nat], "-", color=NAT_LIGHT, marker="o", ms=3.5, lw=1.4, zorder=2)
        for (x, y), lab in nat:
            ax.annotate(lab, (x, y), xytext=(2 if lab == "1/1" else -4, -10), textcoords="offset points", fontsize=6.7, color=NAT, ha="center")
        for n, fill in ((1, None),):   # one decode loop (the regime of Section 5.5)
            for c, m, col, sz in ((f"D{n}st", "^", DIST, 24), (f"a4x{n}pm", "s", OURS, 28)):
                p = pt(c, b)
                if p:
                    ax.scatter(*p, marker=m, s=sz, zorder=4, facecolors=fill or col, edgecolors=col, lw=0.9)
            p80 = pt(f"D{n}r80st", b)
            if p80:
                ax.scatter(*p80, marker="x", s=24, zorder=4, color=DIST, lw=1.2)
        p41 = pt("a4x1pm", b)
        if p41 and len(nat) == 2:
            (x1, y1), (x2, y2) = nat[0][0], nat[1][0]
            ym = y1 + (p41[0] - x1) / (x2 - x1) * (y2 - y1)
            ax.annotate("", (p41[0], p41[1] - 1.3), xytext=(p41[0], ym + 0.9),
                        arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.05, mutation_scale=7.5), zorder=5)
            ax.text(p41[0] - 0.04 * (x2 - x1), (p41[1] + ym) / 2 + 0.8, f"+{p41[1] - ym:.1f}",
                    fontsize=6.5, color=OURS, fontweight="bold", ha="right", va="center")
            ax.annotate("4/1", p41, xytext=(0, 6), textcoords="offset points", fontsize=6.9, color=OURS, fontweight="bold", ha="center")
        lo = min(B[k]["acc"] for k in B if k.endswith(f"@{b}") and "acc" in B[k])
        ax.set_ylim(lo - 7.5, None)   # room for the 1/1 label under its point
        xlo, xhi = ax.get_xlim(); ax.set_xlim(xlo - 0.05 * (xhi - xlo), xhi + 0.05 * (xhi - xlo))
        ax.set_title(f"batch {b}", fontsize=8.5, color=INK, pad=4)
        ax.set_xlabel("seconds per request"); ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("mean accuracy (%)")
    L = plt.Line2D
    h = [L([], [], color=OURS, marker="s", ms=4.2, ls="none"),
         L([], [], color=DIST, marker="^", ms=4.5, ls="none"), L([], [], color=DIST, marker="x", ms=4.5, lw=1.2, ls="none"),
         L([], [], color=NAT_LIGHT, marker="o", ms=4, lw=1.4)]
    l = ["Ours (LoopBridge 4/1)", "Distilled 1/1 (baseline)", "Distilled r80 (baseline)", "Original n/n (tied)"]
    fig.legend(h, l, fontsize=6.5, loc="lower center", frameon=False, handlelength=1.6, ncol=2, columnspacing=1.2)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.33, wspace=0.28)
    save(fig, name)


def tab_batched():
    """Appendix: the batched serving numbers behind fig_batched: accuracy of the served answers (batch 32), wall seconds
    per request at batch 1, 8 and 32, and the prefill share of the wall time at batch 32."""
    B = PN.get("batched_serving", {})
    rows = [("Original 1/1", "native1st"), ("Distilled 1/1", "D1st"), ("Distilled 1/1, rank 80", "D1r80st"),
            ("Ours, LoopBridge, 4/1", "a4x1pm")]
    lines = []
    for r in rows:
        if r is None:
            lines.append(r"\midrule"); continue
        lab, c = r
        if f"{c}@32" not in B:
            continue
        sh = r"\rowcolor{oursbg} " if c.startswith(("a4", "f4")) else ""
        t = " & ".join(f"{B[f'{c}@{b}']['s_per_request']:.3f}" if f"{c}@{b}" in B else "--" for b in (8, 32))   # 3 decimals: ratios in Sec 5.5 follow from the printed times
        lines.append(rf"{sh}{lab} & {B[f'{c}@32']['acc']:.1f} & {t} & {B[f'{c}@32']['prefill_pct']:.0f}\% \\")
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
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.3, 2.25), squeeze=False)
    for ax, (t, bs, lab) in zip(axes[0], tasks):
        pt = lambda c: (T[f"{t} {c}@{bs}"]["s_per_request"], T[f"{t} {c}@{bs}"]["acc"])
        (x2, y2), (x4, y4), (xo, yo) = pt("native2st"), pt("native4st"), pt("a4x2pm")
        ax.plot([x2, x4], [y2, y4], "-", color=NAT_LIGHT, marker="o", ms=3.8, lw=1.4, zorder=2)
        ax.annotate("2/2", (x2, y2), xytext=(4, -9), textcoords="offset points", fontsize=6.6, color=NAT, ha="center")
        ax.annotate("4/4", (x4, y4), xytext=(-1, -9), textcoords="offset points", fontsize=6.6, color=NAT, ha="center")
        ax.scatter(xo, yo, marker="s", s=30, zorder=4, color=OURS)
        ax.annotate("4/2", (xo, yo), xytext=(0, 6), textcoords="offset points", fontsize=6.9, color=OURS, fontweight="bold", ha="center")
        ax.axhline(y4, color=NAT_LIGHT, lw=0.7, ls=":", zorder=1)
        r = T.get(f"{t} a4x2pm vs native4st@{bs}")
        if r:
            span = x4 - xo
            ax.annotate("", xy=(xo + 0.12 * span, y4 + 0.14), xytext=(x4 - 0.08 * span, y4 + 0.14),
                        arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.1, mutation_scale=8), zorder=4)
            ax.text((xo + x4) / 2 + 0.04 * span, y4 + 0.36, rf"{r['ratio']:.2f}$\times$ time", fontsize=6.6,
                    color=OURS, fontweight="bold", ha="center", va="bottom")
        lo, hi = min(y2, y4, yo), max(y2, y4, yo)
        ax.set_ylim(lo - 1.3, hi + 1.1)
        xpad = 0.24 * (x4 - xo)
        ax.set_xlim(xo - xpad * 1.15, x4 + xpad * 0.85)
        ax.set_title(lab, fontsize=7.6, color=INK, fontweight="bold", pad=3)
        ax.set_xlabel("seconds per request", fontsize=7.2); ax.tick_params(labelsize=6.6)
        ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("accuracy (%)", fontsize=7.2)
    L = plt.Line2D
    h = [L([], [], color=OURS, marker="s", ms=4.4, ls="none"),
         L([], [], color=NAT_LIGHT, marker="o", ms=4.0, lw=1.4)]
    fig.legend(h, ["Ours (LoopBridge 4/2)", "Original n/n (tied)"], fontsize=6.6, loc="lower center",
               frameon=False, handlelength=1.5, ncol=2, columnspacing=1.2)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.89, bottom=0.30, wspace=0.30)
    save(fig, name)


def tab_tbatched():
    """Table 15 (Appendix K): Ouro-2.6B-Thinking served in batches (paper_numbers thinking_batched): accuracy and wall
    seconds per request on all 1,319 GSM8K problems (batch 16 and 32)."""
    T = PN.get("thinking_batched", {})
    rows = [("Original 2/2", "native2st"), ("Original 4/4", "native4st"), ("Distilled 2/2", "D2st"),
            ("Ours, LoopBridge, 4/2", "a4x2pm")]
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
                lines.append(f"{t} & {lab} & {r['in_median']:.0f} & {r['out_median']:.0f} & {r['out_p90']:.0f} & "
                             f"{r['total_median']:.1f} & {r['total_p90']:.1f} & {r['share_median']:.2f} & {r['share_p90']:.2f} & "
                             f"{r['prefill_share']:.1f} \\\\")   # last: share of the summed time (Figure 2, Section 5.1)
    (OUT / "tab_time.tex").write_text("\n".join(lines) + "\n")


def tab_plateau():
    """Appendix: stopping depth of the prefill gain per benchmark and decode depth (paper_numbers.plateau)."""
    lines = []
    for ki, (key, name) in enumerate((("plateau_extrapolated", r"trained $\le4$"), ("plateau", r"trained $\le16$"))):
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
            cell = lambda m: "--" if m not in q else (f"\\textbf{{{q[m]:.1f}}}" if q[m] == best else f"{q[m]:.1f}")
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
        cell = lambda m: "--" if m not in q else (f"\\textbf{{{q[m]:.1f}}}" if q[m] == best else f"{q[m]:.1f}")
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
    for r in P["curves"]:
        if prev_n is not None and r["n"] != prev_n:
            lines.append(r"\midrule")
        prev_n = r["n"]
        for arm, q in r["quality"].items():
            q = {int(k): v for k, v in q.items()}
            best = max(q.values())
            cell = lambda m: "--" if m not in q else (f"\\textbf{{{q[m]:.1f}}}" if q[m] == best else f"{q[m]:.1f}")
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
    # 13 figures (Figures 1-11, 13-14; Figure 12 is pseudocode) + 16 generated tables (Tables 1-2, 4-11, 13-16)
    fig_tps_teaser(); fig_time(); fig0(); fig_aligned(); fig_depth_main(); fig_budget(); fig_tbatched(); fig_lev()
    fig_sametime(); fig_depth(); fig_equal(); fig_batched(); fig2()
    tab_main(); tab_build(); tab_tokens(); tab_lr(); tab_time(); tab_prefilltrn(); tab_plateau(); tab_depthwide()
    tab_depth14(); tab_aligned(); tab_repro(); tab_main("main_table_14", "tab_main14"); tab_equal("2.6B-T")
    tab_batched(); tab_tbatched(); tab_lev()
    for f in OUT.glob("tab_*.tex"):   # one name per benchmark in every table: BBH is 3-shot throughout (Appendix H)
        f.write_text(f.read_text().replace("BBH 3-shot", "BBH"))
    print("\n".join(sorted(p.name for p in OUT.iterdir())))
