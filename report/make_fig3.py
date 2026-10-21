"""Fig. 3: (a) excerpt of the time-scaled spike replay at the three receivers, (b) modeled bus delay
using the chip-measured delivery. Reads runs/spike_replay_20261019_235342.npz and sim/replay_excerpt.json."""
import json, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "runs").exists() else HERE.parent
BLUE, ORNG, GREY, INK = "#2a62b8", "#e07b14", "#8a909a", "#222222"
SEND = {0: "#7a5bc8", 1: BLUE, 2: "#2e8b57", 3: ORNG}            # C1 purple, C2 blue, C3 green, C4 orange
TIA = 220e3
d = np.load(ROOT / "runs/spike_replay_20261019_235342.npz"); y, sender, det = d["y"], d["sender"], d["detected"]
rp = json.loads((ROOT / "sim/replay_excerpt.json").read_text())

fig = plt.figure(figsize=(6.8, 2.25))
gs = fig.add_gridspec(3, 2, width_ratios=[1.75, 1], wspace=0.28, hspace=0.25, left=0.07, right=0.985, bottom=0.17, top=0.86)
# ---- (a) replay excerpt
T0, slot, rate = 0.5, 0.015, 800.0
t = np.arange(y.shape[1]) / rate
k0, k1 = 120, 165
m = (t >= T0 + k0*slot) & (t < T0 + k1*slot)
dark = np.median(y[:, t < 0.4], axis=1)
axes = []
for i, (j, lab) in enumerate(((2, "C3"), (3, "C4"), (4, "C5"))):
    ax = fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None); axes.append(ax)
    for k in range(k0, k1):
        if sender[k, j] >= 0:
            ax.axvspan(k, k + 0.4, color=SEND[int(sender[k, j])], alpha=0.25, lw=0)
    ax.plot((t[m] - T0) / slot, (y[j, m] - dark[j]) / TIA * 1e9 / 1e3, color=INK, lw=0.6)
    ax.set_ylabel(f"{lab}\n(µA)", fontsize=6.2, labelpad=2); ax.tick_params(labelsize=5.8, length=2)
    ax.set_xlim(k0, k1)
    for k in range(k0, k1):
        if k % 5 == 0: ax.axvline(k, color="0.8", lw=0.4, ls=":", zorder=0)
    if i < 2: plt.setp(ax.get_xticklabels(), visible=False)
for k in range(k0, k1, 5):
    axes[0].text(k + 2.5, axes[0].get_ylim()[1] * 1.02, rp["windows"][k // 5]["setting"], fontsize=6, color=GREY,
                 ha="center", va="bottom")
axes[2].set_xlabel("SNN time step (2 ms, replayed as 15 ms slots)", fontsize=6.4, labelpad=2)
axes[0].set_title("(a) Spike replay", fontsize=7, loc="left", pad=17)
handles = [plt.Rectangle((0, 0), 1, 1, color=SEND[s], alpha=0.35) for s in (0, 1, 2, 3)]
axes[0].legend(handles, ["from C1", "from C2", "from C3", "from C4"], ncol=4, fontsize=5.4, frameon=False,
               loc="lower right", bbox_to_anchor=(1.0, 1.27), handlelength=1.0, columnspacing=0.8)

# ---- (b) bus delay
P = np.array([s["packets"] for s in rp["steps"]], float); total = P.sum((1, 2))
opt = np.array([sum(P[k, sender[k, j], j] for j in range(5) if sender[k, j] >= 0 and det[k, j]) for k in range(len(total))])
lam = total.mean()
def p99(arr, cap):
    x, q = 0.0, []
    for a in arr:
        x = max(0.0, x + a - cap); q.append(x)
    return np.percentile(np.array(q) / cap * 2, 99)
f = np.array([0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
bo = np.array([p99(total, x*lam) for x in f]); wl = np.array([p99(total - opt, x*lam) for x in f])
ax = fig.add_subplot(gs[:, 1])
ax.plot(f, bo, "o-", color=GREY, ms=2.5, lw=1.0, label="bus only")
ax.plot(f, wl, "o-", color=BLUE, ms=2.5, lw=1.2, label="bus + optical lanes")
ax.set_yscale("symlog", linthresh=0.1); ax.set_ylim(0, 2e4)
ax.set(xlabel="bus capacity (× mean traffic)", ylabel="99th-percentile delay (ms)")
ax.xaxis.label.set_size(6.4); ax.yaxis.label.set_size(6.4); ax.tick_params(labelsize=5.8, length=2)
ax.set_title("(b) Modeled bus delay", fontsize=7, loc="left", pad=17)
ax.legend(fontsize=5.8, frameon=False, loc="upper right")
i15 = list(f).index(1.5)
ax.annotate(f"{bo[i15]:.0f} ms", (1.5, bo[i15]), (1.75, bo[i15]*4), fontsize=5.8, color=GREY, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
ax.annotate(f"{wl[i15]:.1f} ms", (1.5, wl[i15]), (2.05, 0.6), fontsize=5.8, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.5))
fig.savefig(HERE / "fig3.pdf"); fig.savefig(HERE / "fig3.png", dpi=300)
print("p99 at 1.5x: bus only %.1f ms, with lanes %.2f ms" % (bo[i15], wl[i15]))
