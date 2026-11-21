"""Generate light & dark horizontal 1x2 README teaser charts directly from ouro/results/paper_numbers.json.

usage: uv run --with numpy --with matplotlib python docs/assets/make_teaser.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PN = json.loads((ROOT / "ouro" / "results" / "paper_numbers.json").read_text())
OUT_DIR = Path(__file__).resolve().parent
TIMED = ["GSM8K", "MATH-500", "BBH", "MMLU-Pro"]

THEMES = {
    "light": dict(
        out="teaser_horizontal.png",
        bg="#FFFFFF", ours="#EB6834", nat="#4D4D4D", nat_light="#9A9A9A",
        ink="#1A1A1A", spine="#4D4D4D", grid="#EDEDED", pill="#FFFFFF",
    ),
    "dark": dict(
        out="teaser_horizontal_dark.png",
        bg="#0D1117", ours="#F27A49", nat="#CBD5E1", nat_light="#64748B",
        ink="#F8FAFC", spine="#475569", grid="#1E293B", pill="#0D1117",
    ),
}


def _time(task, cond):
    return next(r for r in PN["time_breakdown"]
                if r["task"] == task and r["cond"] == cond and r.get("model", "Ouro-2.6B") == "Ouro-2.6B")


def draw(theme):
    c = THEMES[theme]
    plt.rcParams.update({
        "font.size": 8.5, "font.family": "DejaVu Sans",
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.75))

    # --- Panel (a): Ouro-2.6B Base, 4 CoT tasks, batch 1 ---
    nat_b = [(base_pt(f"native{d}st"), f"{d}/{d}") for d in (4, 2, 1)]
    ax1.plot([p[0][0] for p in nat_b], [p[0][1] for p in nat_b], "-", color=c["nat_light"], marker="o", ms=5.0, lw=1.7, zorder=2)
    for (x, y), lab in nat_b:
        ox, oy = {"4/4": (0, -13), "2/2": (-8, -12), "1/1": (0, -12)}[lab]
        ax1.annotate(lab, (x, y), xytext=(ox, oy), textcoords="offset points", fontsize=7.8, color=c["nat"], ha="center")

    lb2, lb1 = base_pt("a4x2pm"), base_pt("a4x1pm")
    ax1.plot([lb2[0], lb1[0]], [lb2[1], lb1[1]], "-", color=c["ours"], lw=1.9, zorder=3)
    for x, y in (lb2, lb1):
        ax1.scatter(x, y, marker="s", s=38, color=c["ours"], zorder=4)
    ax1.annotate("4/2", lb2, xytext=(6, 4), textcoords="offset points", fontsize=8.0, color=c["ours"], fontweight="bold", ha="left")
    ax1.annotate("4/1", lb1, xytext=(0, 7), textcoords="offset points", fontsize=8.0, color=c["ours"], fontweight="bold", ha="center")

    g1 = round(lb1[1], 1) - round(nat_b[2][0][1], 1)
    ax1.annotate("", xy=(nat_b[2][0][0], lb1[1] - 0.8), xytext=(nat_b[2][0][0], nat_b[2][0][1] + 2.6),
                 arrowprops=dict(arrowstyle="-|>", color=c["ours"], lw=1.25, mutation_scale=9), zorder=4)
    ax1.text(nat_b[2][0][0] + 0.8, (lb1[1] + nat_b[2][0][1]) / 2, f"+{g1:.1f}", fontsize=8.2, color=c["ours"],
             fontweight="bold", ha="left", va="center")

    x44, y44 = nat_b[0][0]
    ax1.vlines(x44, y44 + 2.2, 73.5, color=c["nat_light"], lw=0.75, ls=":", zorder=1)
    ax1.annotate("", xy=(lb2[0], 72.5), xytext=(x44 + 0.35, 72.5),
                 arrowprops=dict(arrowstyle="-|>", color=c["ours"], lw=1.15, mutation_scale=8.5), zorder=4)
    ax1.text(lb2[0] + 0.65, 72.5, rf"{lb2[0] / x44:.1f}$\times$ TPS of 4/4", fontsize=7.9, color=c["ours"],
             fontweight="bold", ha="left", va="center")

    ax1.set_title("(a) Ouro-2.6B, 4 CoT tasks (batch 1)", fontsize=9.0, color=c["ink"], fontweight="bold", loc="left", pad=6)
    ax1.set_xlabel("end-to-end throughput (output tokens/s)", fontsize=8.4, color=c["ink"], labelpad=4)
    ax1.set_ylabel("mean accuracy (%)", fontsize=8.4, color=c["ink"])
    ax1.set_xlim(4.5, 33.8); ax1.set_ylim(19.5, 79.0)
    ax1.set_xticks([5, 10, 15, 20, 25, 30])
    ax1.grid(color=c["grid"], lw=0.6); ax1.set_axisbelow(True)
    ax1.tick_params(labelsize=7.8)

    # --- Panel (b): Ouro-2.6B-Thinking, GSM8K (all 1,319), batch 16 ---
    n4, n2, o42 = think_pt("native4st"), think_pt("native2st"), think_pt("a4x2pm")
    ax2.plot([n4[0], n2[0]], [n4[1], n2[1]], "-", color=c["nat_light"], marker="o", ms=5.0, lw=1.7, zorder=2)
    ax2.axhline(n4[1], color=c["nat_light"], lw=0.8, ls=":", zorder=1)
    ax2.annotate("Original 4/4", n4, xytext=(0, -14), textcoords="offset points", fontsize=7.8, color=c["nat"], ha="left")
    ax2.annotate("Original 2/2", n2, xytext=(-4, -12), textcoords="offset points", fontsize=7.8, color=c["nat"], ha="right")
    ax2.scatter(*o42, marker="s", s=40, color=c["ours"], zorder=4)
    ax2.annotate("Ours 4/2", o42, xytext=(0, 7), textcoords="offset points", fontsize=8.0, color=c["ours"], fontweight="bold",
                 ha="center", bbox=dict(fc=c["pill"], ec="none", alpha=0.88, pad=0.4), zorder=5)

    ax2.annotate("", xy=(o42[0] - 5, n4[1] + 0.14), xytext=(n4[0] + 5, n4[1] + 0.14),
                 arrowprops=dict(arrowstyle="-|>", color=c["ours"], lw=1.25, mutation_scale=9), zorder=4)
    ax2.text((n4[0] + o42[0]) / 2 - 4, n4[1] + 0.36, rf"{o42[0] / n4[0]:.1f}$\times$ TPS, +{o42[1] - n4[1]:.1f} acc",
             fontsize=8.0, color=c["ours"], fontweight="bold", ha="center", va="bottom")

    ax2.set_title("(b) Ouro-2.6B-Thinking, GSM8K (batch 16)", fontsize=9.0, color=c["ink"], fontweight="bold", loc="left", pad=6)
    ax2.set_xlabel("end-to-end throughput (output tokens/s)", fontsize=8.4, color=c["ink"], labelpad=4)
    ax2.set_ylabel("accuracy (%)", fontsize=8.4, color=c["ink"])
    ax2.set_xlim(105, 274); ax2.set_ylim(91.7, 95.9)
    ax2.set_yticks([92, 93, 94, 95])
    ax2.grid(color=c["grid"], lw=0.6); ax2.set_axisbelow(True)
    ax2.tick_params(labelsize=7.8)

    L = plt.Line2D
    h = [L([], [], color=c["ours"], marker="s", ms=5.0, lw=1.8),
         L([], [], color=c["nat_light"], marker="o", ms=4.8, lw=1.7)]
    leg = fig.legend(h, ["Ours (LoopBridge 4/n)", "Original n/n (tied)"], loc="upper center", ncol=2, fontsize=8.2,
                     bbox_to_anchor=(0.52, 1.01), handlelength=1.8, columnspacing=1.6, frameon=False)
    for t in leg.get_texts():
        t.set_color(c["ink"])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.81, bottom=0.18, wspace=0.25)
    out = OUT_DIR / c["out"]
    fig.savefig(out, bbox_inches="tight", pad_inches=0.08, facecolor=c["bg"])
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    for t in ("light", "dark"):
        draw(t)
