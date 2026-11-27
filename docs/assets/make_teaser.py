"""Generate light & dark horizontal 1x2 README hero charts matching Figure 1 of the paper directly from archived JSONs:
  (a) Ouro-2.6B 11-Benchmark Average across all 16 (m, n) prefill/decode pairs + Overall TPS under n=1..4,
  (b) 10x scale-up to 26.4B Thinking Machines Inkling-2 (TML-d24-varloop in vLLM, 5 benchmarks to 1M).

usage: uv run --with numpy --with matplotlib python docs/assets/make_teaser.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ouro" / "scripts"))
import paper_figs  # noqa: E402

PN = paper_figs.PN
INKL = paper_figs._load_inkling_teaser_data()
OUT_DIR = Path(__file__).resolve().parent
TIMED = ["GSM8K", "MATH-500", "BBH", "MMLU-Pro"]

THEMES = {
    "light": dict(
        out="teaser_horizontal.png",
        bg="#FFFFFF", ours="#EB6834", dist="#2A78D6", nat="#4D4D4D", nat_light="#9A9A9A",
        ink="#1A1A1A", spine="#4D4D4D", grid="#EDEDED", pill="#FFFFFF", teal="#1D5C63",
        m_cols=["#FBD8C8", "#F7AC88", "#F1824F", "#EB6834"],
        m_edge=["#D49A7E", "#D67E56", "#C85C2D", "#B54417"],
    ),
    "dark": dict(
        out="teaser_horizontal_dark.png",
        bg="#0D1117", ours="#F27A49", dist="#60A5FA", nat="#CBD5E1", nat_light="#64748B",
        ink="#F8FAFC", spine="#475569", grid="#1E293B", pill="#0D1117", teal="#5EEAD4",
        m_cols=["#4A2A1E", "#7C3D24", "#B85328", "#F27A49"],
        m_edge=["#6B3D2C", "#9E5233", "#D96A3B", "#FFA07A"],
    ),
}


def _time(task, cond):
    return next(r for r in PN["time_breakdown"]
                if r["task"] == task and r["cond"] == cond and r.get("model", "Ouro-2.6B") == "Ouro-2.6B")


def draw(theme):
    c = THEMES[theme]
    plt.rcParams.update({
        "font.size": 8.4, "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.75, "axes.edgecolor": c["spine"],
        "axes.facecolor": c["bg"], "figure.facecolor": c["bg"],
        "xtick.color": c["nat"], "ytick.color": c["nat"], "axes.labelcolor": c["ink"],
        "savefig.dpi": 300,
    })

    rows = PN["aligned_cells"]
    avg_cell = {f"native{n}": float(np.mean([r[f"native{n}"] for r in rows])) for n in range(1, 5)}
    for m in range(1, 5):
        for n in range(1, 5):
            avg_cell[f"a{m}{n}"] = float(np.mean([r[f"a{m}{n}"] for r in rows]))

    def e2e_tps(cond):
        rs = [_time(t, cond) for t in TIMED]
        return sum(r["out_tokens"] for r in rs) / sum(r["total_s"] for r in rs)

    tps_n = {n: e2e_tps(f"native{n}st") for n in range(1, 5)}
    tps_base = tps_n[4]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.2, 3.35), gridspec_kw=dict(width_ratios=[1.05, 1.05], wspace=0.25))

    # --- Panel (a): Ouro-2.6B 11-Benchmark Average + Overall TPS ---
    xs = np.arange(1, 5)
    bw = 0.185
    offs = np.array([-1.5, -0.5, 0.5, 1.5]) * bw
    for m in range(1, 5):
        vals = [avg_cell[f"a{m}{n}"] for n in range(1, 5)]
        ax0.bar(xs + offs[m - 1], vals, width=bw * 0.92, color=c["m_cols"][m - 1], edgecolor=c["m_edge"][m - 1],
                lw=0.5 if m < 4 else 0.75, zorder=3)

    pill = dict(fc=c["pill"], ec="none", alpha=0.90, pad=0.4)
    for idx, n in enumerate(range(1, 5)):
        nv, a4 = avg_cell[f"native{n}"], avg_cell[f"a4{n}"]
        ax0.hlines(nv, xs[idx] - 0.41, xs[idx] + 0.41, color=c["nat"], lw=1.6, ls="--", zorder=5)
        y_lab = max(a4, nv) + 2.0 if n >= 3 else a4 + 1.5
        ax0.text(xs[idx] + offs[3], y_lab, f"{a4:.1f}", ha="center", va="bottom", fontsize=7.6, color=c["ours"], fontweight="bold", zorder=6)
        if n == 1:
            ax0.text(xs[idx], nv - 2.3, f"Orig {nv:.1f}", ha="center", va="top", fontsize=6.8, color=c["nat"], zorder=6, bbox=pill)
            ax0.text(xs[idx] + offs[3], a4 + 8.8, "+14.0", ha="center", va="bottom", fontsize=7.8, color=c["ours"], fontweight="bold", zorder=6)
        elif n == 2:
            ax0.text(xs[idx] - 0.40, nv - 1.6, f"{nv:.1f}", ha="left", va="top", fontsize=6.8, color=c["nat"], zorder=6, bbox=pill)
            ax0.text(xs[idx] + offs[3], a4 + 8.8, "+5.4", ha="center", va="bottom", fontsize=7.8, color=c["ours"], fontweight="bold", zorder=6)
        else:
            ax0.text(xs[idx] - 0.40, nv - 1.6, f"{nv:.1f}", ha="left", va="top", fontsize=6.8, color=c["nat"], zorder=6, bbox=pill)

    xlabels = [f"n = {n}\n{tps_n[n]:.1f} tok/s\n({tps_n[n] / tps_base:.1f}×)" for n in range(1, 5)]
    ax0.set_xticks(xs)
    ax0.set_xticklabels(xlabels, fontsize=7.3, linespacing=1.2)
    ax0.tick_params(axis="x", length=0, pad=3.5, colors=c["ink"])
    ax0.text(0.43, -0.105, "Overall\nTPS:", transform=ax0.get_xaxis_transform(),
             ha="right", va="top", fontsize=6.7, color=c["teal"], fontweight="bold", linespacing=1.15)
    ax0.set_xlim(0.42, 4.56); ax0.set_ylim(0, 89.0); ax0.set_yticks([0, 20, 40, 60, 80])
    ax0.set_ylabel("11-bench avg (%)", fontsize=8.2, color=c["ink"])
    ax0.set_title("(a) Ouro-2.6B: 11-Benchmark Average & Overall TPS", fontsize=8.8, color=c["ink"], fontweight="bold", loc="left", pad=17.0)
    ax0.grid(axis="y", color=c["grid"], lw=0.6); ax0.set_axisbelow(True)
    ax0.tick_params(axis="y", labelsize=7.5)

    h0 = [plt.Rectangle((0, 0), 1, 1, fc=c["m_cols"][i], ec=c["m_edge"][i], lw=0.55) for i in range(4)]
    h0.append(plt.Line2D([], [], color=c["nat"], lw=1.6, ls="--"))
    leg0 = ax0.legend(h0, ["m=1", "m=2", "m=3", "m=4 prefill", "Original n/n"], loc="upper center",
                      bbox_to_anchor=(0.50, 1.135), ncol=5, fontsize=6.8, handlelength=1.1, columnspacing=0.8, handletextpad=0.3, frameon=False)
    for t in leg0.get_texts():
        t.set_color(c["ink"])

    # --- Panel (b): 26.4B Inkling-2 Sparse-MoE (vLLM, 5 Benchmarks to 1M) ---
    nat4 = INKL["nat4"]
    nx4, ny4 = [p[0] for p in nat4], [p[1] for p in nat4]
    x11, y11 = INKL["orig_11"]
    xd11, yd11 = INKL["dist_11"]
    x43, y43, _ = INKL["lb_43"]
    x42, y42, _ = INKL["lb_42"]
    x31, y31, _ = INKL["lb_31"]

    ax1.plot(nx4 + [x11], ny4 + [y11], "--", color=c["nat_light"], lw=1.35, zorder=1)
    ax1.scatter(nx4 + [x11], ny4 + [y11], s=36, color=c["nat"], edgecolor=c["bg"], lw=0.6, zorder=3, label="Original n/n (tied)")
    ax1.axhline(ny4[0], color=c["nat_light"], lw=0.7, ls=":", alpha=0.75, zorder=0)
    for x, y, lab in nat4:
        ox, oy = {"4/4": (0, -12.0), "3/3": (-4, -12.0), "2/2": (-9, -11.0)}[lab]
        ax1.annotate(lab, (x, y), xytext=(ox, oy), textcoords="offset points", fontsize=7.4, color=c["nat"], ha="center")
    ax1.annotate("1/1", (x11, y11), xytext=(-7.0, -1.5), textcoords="offset points", fontsize=7.2, color=c["nat"], ha="right", va="center")

    ax1.scatter([xd11], [yd11], marker="^", s=40, color=c["dist"], edgecolor=c["bg"], lw=0.5, zorder=4, label="Distilled 1/1")
    ax1.annotate("Distilled 1/1", (xd11, yd11), xytext=(-15.0, 0.5), textcoords="offset points", fontsize=7.1, color=c["dist"], ha="right", va="center")

    ax1.plot([x43, x42, x31], [y43, y42, y31], "-", color=c["ours"], lw=1.9, zorder=3)
    ax1.scatter([x43, x42, x31], [y43, y42, y31], marker="s", s=44, color=c["ours"], edgecolor=c["bg"], lw=0.6, zorder=5, label="LoopBridge m/n (ours)")

    ax1.annotate("", xy=(x42, y42 - 1.4), xytext=(nx4[2], ny4[2] + 1.4),
                 arrowprops=dict(arrowstyle="-|>", color=c["ours"], lw=1.1, mutation_scale=8.5), zorder=4)
    ax1.annotate("", xy=(x31, y31 - 1.5), xytext=(x31, yd11 + 0.6),
                 arrowprops=dict(arrowstyle="-|>", color=c["ours"], lw=1.1, mutation_scale=8.0), zorder=4)

    rec43 = 100.0 * y43 / ny4[0]
    ax1.annotate(f"4/3: {y43:.1f}% ({rec43:.0f}% of 4/4)", (x43, y43), xytext=(10, 10.0), textcoords="offset points",
                 fontsize=7.3, color=c["ours"], fontweight="bold", ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-", color=c["ours"], lw=0.65, shrinkA=0, shrinkB=3.5))
    ax1.annotate(f"4/2: {y42:.1f}% (+{y42 - ny4[2]:.1f} vs 2/2)\n{INKL['e2e_42_vs_44']:.1f}× E2E throughput",
                 (x42, y42), xytext=(11, 4.5), textcoords="offset points", fontsize=7.1, color=c["ours"],
                 fontweight="bold", ha="left", va="center", linespacing=1.15)
    ax1.annotate(f"3/1: {y31:.1f}%\n(+{y31 - yd11:.1f} vs Dist.)", (x31, y31), xytext=(10, 11.0), textcoords="offset points",
                 fontsize=7.1, color=c["ours"], fontweight="bold", ha="center", va="bottom", linespacing=1.12,
                 zorder=6)

    ax1.set_title("(b) 26.4B Inkling-2 Sparse-MoE (vLLM, 5 Benches to 1M)", fontsize=8.8, color=c["ink"], fontweight="bold", loc="left", pad=8.0)
    ax1.set_xlabel("Overall E2E throughput (k tokens/s, 1× B300)", fontsize=8.0, color=c["ink"], labelpad=4.0)
    ax1.set_ylabel("5-bench avg (%)", fontsize=8.2, color=c["ink"])
    ax1.set_xlim(1.65, 5.18); ax1.set_ylim(36.0, 86.0)
    ax1.set_xticks([2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    ax1.set_yticks([40, 50, 60, 70, 80])
    ax1.grid(color=c["grid"], lw=0.6); ax1.set_axisbelow(True)
    ax1.tick_params(labelsize=7.5)
    leg1 = ax1.legend(loc="lower left", fontsize=7.2, handlelength=1.4, handletextpad=0.4, labelspacing=0.35, frameon=False)
    for t in leg1.get_texts():
        t.set_color(c["ink"])

    fig.subplots_adjust(left=0.06, right=0.99, top=0.83, bottom=0.22, wspace=0.22)
    out = OUT_DIR / c["out"]
    fig.savefig(out, bbox_inches="tight", pad_inches=0.08, facecolor=c["bg"])
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    for t in ("light", "dark"):
        draw(t)
