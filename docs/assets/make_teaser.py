"""Generate light & dark horizontal 1x3 README teaser charts directly from archived JSONs:
  (a) 26.4B Thinking Machines Inkling-2 (TML-d24-varloop in Day-0 vLLM + Inkling-2-Mini inset),
  (b) Ouro-2.6B (4 CoT tasks, batch 1),
  (c) Ouro-2.6B-Thinking (GSM8K, batch 16).

usage: uv run --with numpy --with matplotlib python docs/assets/make_teaser.py
"""
import json
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
        ink="#1A1A1A", spine="#4D4D4D", grid="#EDEDED", pill="#FFFFFF", box="#FEF3EE",
    ),
    "dark": dict(
        out="teaser_horizontal_dark.png",
        bg="#0D1117", ours="#F27A49", dist="#60A5FA", nat="#CBD5E1", nat_light="#64748B",
        ink="#F8FAFC", spine="#475569", grid="#1E293B", pill="#0D1117", box="#1F1612",
    ),
}


def _time(task, cond):
    return next(r for r in PN["time_breakdown"]
                if r["task"] == task and r["cond"] == cond and r.get("model", "Ouro-2.6B") == "Ouro-2.6B")


def draw(theme):
    c = THEMES[theme]
    plt.rcParams.update({
        "font.size": 8.2, "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "axes.edgecolor": c["spine"],
        "axes.facecolor": c["bg"], "figure.facecolor": c["bg"],
        "xtick.color": c["nat"], "ytick.color": c["nat"], "axes.labelcolor": c["ink"],
        "savefig.dpi": 300,
    })

    def base_pt(cond):
        rs = [_time(t, cond) for t in TIMED]
        return sum(r["out_tokens"] for r in rs) / sum(r["total_s"] for r in rs), float(np.mean([r["quality"] for r in rs]))

    T = PN["thinking_batched"]
    think_pt = lambda cond: (T[f"gsmthink {cond}@16"]["tokens"] / T[f"gsmthink {cond}@16"]["s_per_request"],
                             T[f"gsmthink {cond}@16"]["acc"])

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10.8, 2.85), gridspec_kw=dict(width_ratios=[1.25, 1.0, 0.96]))

    # --- Panel (a): 26.4B Inkling-2 (Day-0 vLLM) ---
    nat4 = INKL["nat4"]
    nx4, ny4 = [p[0] for p in nat4], [p[1] for p in nat4]
    x11, y11 = INKL["orig_11"]
    xd11, yd11 = INKL["dist_11"]
    x43, y43, sp43 = INKL["lb_43"]
    x42, y42, sp42 = INKL["lb_42"]
    x31, y31, _ = INKL["lb_31"]
    xtf43, ytf43 = INKL["tf_43"]
    xtf42, ytf42 = INKL["tf_42"]

    ax0.plot(nx4, ny4, "-", color=c["nat_light"], marker="o", ms=4.5, lw=1.6, zorder=2)
    ax0.plot([nx4[-1], x11], [ny4[-1], y11], ":", color=c["nat_light"], lw=1.2, zorder=1)
    ax0.scatter([x11], [y11], s=26, facecolor=c["bg"], edgecolor=c["nat"], lw=1.1, zorder=3)
    ax0.axhline(ny4[0], color=c["nat_light"], lw=0.7, ls=":", zorder=0)
    for x, y, lab in nat4:
        ox, oy = {"4/4": (0, -11.5), "3/3": (-4, -11.5), "2/2": (-7, -10.5)}[lab]
        ax0.annotate(lab, (x, y), xytext=(ox, oy), textcoords="offset points", fontsize=7.0, color=c["nat"], ha="center")
    ax0.annotate("1/1", (x11, y11), xytext=(-6, -1), textcoords="offset points", fontsize=6.8, color=c["nat"], ha="right", va="center")

    ax0.scatter([xd11], [yd11], marker="^", s=30, color=c["dist"], zorder=4)
    ax0.annotate("Dist. 1/1", (xd11, yd11), xytext=(5.5, 0.5), textcoords="offset points", fontsize=6.6, color=c["dist"], ha="left", va="center")
    ax0.scatter([xtf43, xtf42], [ytf43, ytf42], marker="D", s=26, facecolor="none", edgecolor=c["ink"], lw=1.0, zorder=7)

    ax0.plot([x43, x42], [y43, y42], "-", color=c["ours"], lw=1.8, zorder=3)
    ax0.scatter([x43, x42], [y43, y42], marker="s", s=34, color=c["ours"], zorder=5)
    ax0.scatter([x31], [y31], marker="s", s=32, facecolor=c["bg"], edgecolor=c["ours"], lw=1.5, zorder=5)

    ax0.annotate("", xy=(x42, y42 - 1.3), xytext=(nx4[2], ny4[2] + 1.3),
                 arrowprops=dict(arrowstyle="-|>", color=c["ours"], lw=1.0, mutation_scale=7.5), zorder=4)
    ax0.annotate("", xy=(x31, y31 - 1.6), xytext=(xd11, yd11 + 1.8),
                 arrowprops=dict(arrowstyle="-|>", color=c["ours"], lw=1.0, mutation_scale=7.5), zorder=4)

    rec43 = 100.0 * y43 / ny4[0]
    ax0.annotate(f"4/3 KV ({rec43:.0f}% of 4/4)", (x43, y43), xytext=(8, 7), textcoords="offset points",
                 fontsize=6.9, color=c["ours"], fontweight="bold", ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-", color=c["ours"], lw=0.6, shrinkA=0, shrinkB=2.5))
    ax0.annotate(f"4/2 KV (+{y42 - ny4[2]:.1f}*, +{INKL['mrcr_gain_42']:.1f} MRCR)\n{INKL['e2e_42_vs_44']:.2f}x E2E ({sp42:.2f}x dec.)",
                 (x42, y42), xytext=(8, 1), textcoords="offset points", fontsize=6.5, color=c["ours"],
                 fontweight="bold", ha="left", va="center", linespacing=1.12)
    ax0.annotate(f"3/1 (+{y31 - yd11:.1f}* vs Dist.)", (x31, y31), xytext=(0, 7.5), textcoords="offset points",
                 fontsize=6.6, color=c["ours"], fontweight="bold", ha="center", va="bottom",
                 bbox=dict(fc=c["pill"], ec="none", alpha=0.85, pad=0.25), zorder=6)

    mini_txt = (f"Fixed-3 Inkling-2-Mini 3/2 (26.4B):\n"
                f"TF collapses ({INKL['mini_tf']:.1f}%) → LB {INKL['mini_lb']:.1f}%\n"
                f"(+{INKL['mini_lb'] - INKL['mini_dist']:.1f}* vs Dist. 2/2)")
    ax0.text(1.72, 40.0, mini_txt, fontsize=6.1, color=c["ink"], ha="left", va="bottom", linespacing=1.18,
             bbox=dict(boxstyle="round,pad=0.3", fc=c["box"], ec=c["ours"], lw=0.7), zorder=5)

    ax0.set_title("(a) 26.4B Inkling-2 (Day-0 vLLM, 5 benches to 1M)", fontsize=8.4, color=c["ink"], fontweight="bold", loc="left", pad=6)
    ax0.set_xlabel("Day-0 vLLM E2E throughput (k tokens/s)", fontsize=7.8, color=c["ink"], labelpad=3.5)
    ax0.set_ylabel("5-bench acc. (%)", fontsize=7.8, color=c["ink"])
    ax0.set_xlim(1.65, 5.20); ax0.set_ylim(37.5, 85.5)
    ax0.set_xticks([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    ax0.set_yticks([40, 50, 60, 70, 80])
    ax0.grid(color=c["grid"], lw=0.6); ax0.set_axisbelow(True)
    ax0.tick_params(labelsize=7.2)

    # --- Panel (b): Ouro-2.6B Base, 4 CoT tasks, batch 1 ---
    nat_b = [(base_pt(f"native{d}st"), f"{d}/{d}") for d in (4, 2, 1)]
    ax1.plot([p[0][0] for p in nat_b], [p[0][1] for p in nat_b], "-", color=c["nat_light"], marker="o", ms=4.5, lw=1.6, zorder=2)
    for (x, y), lab in nat_b:
        ox, oy = {"4/4": (0, -12), "2/2": (-8, -11), "1/1": (0, -11)}[lab]
        ax1.annotate(lab, (x, y), xytext=(ox, oy), textcoords="offset points", fontsize=7.2, color=c["nat"], ha="center")

    lb2, lb1 = base_pt("a4x2pm"), base_pt("a4x1pm")
    ax1.plot([lb2[0], lb1[0]], [lb2[1], lb1[1]], "-", color=c["ours"], lw=1.8, zorder=3)
    for x, y in (lb2, lb1):
        ax1.scatter(x, y, marker="s", s=34, color=c["ours"], zorder=4)
    ax1.annotate("4/2", lb2, xytext=(5, -2), textcoords="offset points", fontsize=7.4, color=c["ours"], fontweight="bold", ha="left", va="top")
    ax1.annotate("4/1", lb1, xytext=(0, 6), textcoords="offset points", fontsize=7.4, color=c["ours"], fontweight="bold", ha="center")

    g1 = round(lb1[1], 1) - round(nat_b[2][0][1], 1)
    ax1.annotate("", xy=(nat_b[2][0][0], lb1[1] - 0.8), xytext=(nat_b[2][0][0], nat_b[2][0][1] + 2.6),
                 arrowprops=dict(arrowstyle="-|>", color=c["ours"], lw=1.15, mutation_scale=8.5), zorder=4)
    ax1.text(nat_b[2][0][0] + 0.7, (lb1[1] + nat_b[2][0][1]) / 2, f"+{g1:.1f}", fontsize=7.6, color=c["ours"],
             fontweight="bold", ha="left", va="center")

    x44, y44 = nat_b[0][0]
    ax1.vlines(x44, y44 + 2.2, 74.5, color=c["nat_light"], lw=0.7, ls=":", zorder=1)
    ax1.annotate("", xy=(lb2[0], 73.5), xytext=(x44 + 0.35, 73.5),
                 arrowprops=dict(arrowstyle="-|>", color=c["ours"], lw=1.1, mutation_scale=8.0), zorder=4)
    ax1.text(lb2[0] + 0.6, 73.5, rf"{lb2[0] / x44:.1f}$\times$ TPS of 4/4", fontsize=7.2, color=c["ours"],
             fontweight="bold", ha="left", va="center")

    ax1.set_title("(b) Ouro-2.6B, 4 CoT tasks (batch 1)", fontsize=8.4, color=c["ink"], fontweight="bold", loc="left", pad=6)
    ax1.set_xlabel("end-to-end throughput (output tokens/s)", fontsize=7.8, color=c["ink"], labelpad=3.5)
    ax1.set_ylabel("mean acc. (%)", fontsize=7.8, color=c["ink"])
    ax1.set_xlim(4.5, 33.8); ax1.set_ylim(19.5, 79.0)
    ax1.set_xticks([5, 10, 15, 20, 25, 30])
    ax1.grid(color=c["grid"], lw=0.6); ax1.set_axisbelow(True)
    ax1.tick_params(labelsize=7.2)

    # --- Panel (c): Ouro-2.6B-Thinking, GSM8K (all 1,319), batch 16 ---
    n4, n2, o42 = think_pt("native4st"), think_pt("native2st"), think_pt("a4x2pm")
    ax2.plot([n4[0], n2[0]], [n4[1], n2[1]], "-", color=c["nat_light"], marker="o", ms=4.5, lw=1.6, zorder=2)
    ax2.axhline(n4[1], color=c["nat_light"], lw=0.75, ls=":", zorder=1)
    ax2.annotate("Original 4/4", n4, xytext=(0, -13), textcoords="offset points", fontsize=7.2, color=c["nat"], ha="left")
    ax2.annotate("Original 2/2", n2, xytext=(-4, -11), textcoords="offset points", fontsize=7.2, color=c["nat"], ha="right")
    ax2.scatter(*o42, marker="s", s=36, color=c["ours"], zorder=4)
    ax2.annotate("Ours 4/2", o42, xytext=(0, 6.5), textcoords="offset points", fontsize=7.4, color=c["ours"], fontweight="bold",
                 ha="center", bbox=dict(fc=c["pill"], ec="none", alpha=0.88, pad=0.35), zorder=5)

    ax2.annotate("", xy=(o42[0] - 5, n4[1] + 0.14), xytext=(n4[0] + 5, n4[1] + 0.14),
                 arrowprops=dict(arrowstyle="-|>", color=c["ours"], lw=1.15, mutation_scale=8.5), zorder=4)
    ax2.text((n4[0] + o42[0]) / 2 - 4, n4[1] + 0.36, rf"{o42[0] / n4[0]:.1f}$\times$ TPS, +{o42[1] - n4[1]:.1f} acc",
             fontsize=7.4, color=c["ours"], fontweight="bold", ha="center", va="bottom")

    ax2.set_title("(c) Ouro-2.6B-Thinking, GSM8K (batch 16)", fontsize=8.4, color=c["ink"], fontweight="bold", loc="left", pad=6)
    ax2.set_xlabel("end-to-end throughput (output tokens/s)", fontsize=7.8, color=c["ink"], labelpad=3.5)
    ax2.set_ylabel("accuracy (%)", fontsize=7.8, color=c["ink"])
    ax2.set_xlim(105, 274); ax2.set_ylim(91.7, 95.9)
    ax2.set_yticks([92, 93, 94, 95])
    ax2.grid(color=c["grid"], lw=0.6); ax2.set_axisbelow(True)
    ax2.tick_params(labelsize=7.2)

    L = plt.Line2D
    h = [L([], [], color=c["ours"], marker="s", ms=4.6, lw=1.6),
         L([], [], color=c["nat_light"], marker="o", ms=4.4, lw=1.5),
         L([], [], color=c["ink"], marker="D", mfc="none", ms=4.2, lw=0),
         L([], [], color=c["dist"], marker="^", ms=4.6, lw=0)]
    leg = fig.legend(h, ["Ours (LoopBridge)", "Original n/n (tied)", "Training-free 4/n", "Distilled n/n (a)"],
                     loc="upper center", ncol=4, fontsize=7.8, bbox_to_anchor=(0.51, 1.02),
                     handlelength=1.4, columnspacing=1.3, frameon=False)
    for t in leg.get_texts():
        t.set_color(c["ink"])
    fig.subplots_adjust(left=0.055, right=0.99, top=0.81, bottom=0.18, wspace=0.25)
    out = OUT_DIR / c["out"]
    fig.savefig(out, bbox_inches="tight", pad_inches=0.08, facecolor=c["bg"])
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    for t in ("light", "dark"):
        draw(t)
