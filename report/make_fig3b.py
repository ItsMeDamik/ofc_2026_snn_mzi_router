"""Fig. 3(b) standalone: modeled 99th-percentile bus delay vs capacity, using chip-measured delivery."""
import json, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "runs").exists() else HERE.parent
BLUE, GREY = "#2a62b8", "#8a909a"
d = np.load(ROOT / "runs/spike_replay_20261019_235342.npz"); sender, det = d["sender"], d["detected"]
rp = json.loads((ROOT / "sim/replay_excerpt.json").read_text())
P = np.array([s["packets"] for s in rp["steps"]], float); total = P.sum((1, 2))
opt = np.array([sum(P[k, sender[k, j], j] for j in range(5) if sender[k, j] >= 0 and det[k, j]) for k in range(len(total))])
lam = total.mean()
def p99(arr, cap):
    x, q = 0.0, []
    for a in arr:
        x = max(0.0, x + a - cap); q.append(x)
    return np.percentile(np.array(q) / cap * 2, 99)          # queue drain time in ms (2 ms steps)
f = np.array([0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
bo = np.array([p99(total, x*lam) for x in f]); wl = np.array([p99(total - opt, x*lam) for x in f])
fig, ax = plt.subplots(figsize=(2.6, 1.95))
ax.plot(f, bo, "o-", color=GREY, ms=2.5, lw=1.0, label="bus only")
ax.plot(f, wl, "o-", color=BLUE, ms=2.5, lw=1.2, label="bus + optical lanes")
ax.set_yscale("symlog", linthresh=0.1); ax.set_ylim(0, 2e4)
ax.set(xlabel="bus capacity (× mean traffic)", ylabel="99th-percentile delay (ms)")
ax.legend(fontsize=5.8, frameon=False, loc="upper right")
i = list(f).index(1.5)
ax.annotate(f"{bo[i]:.0f} ms", (1.5, bo[i]), (1.75, bo[i]*4), fontsize=5.8, color=GREY, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
ax.annotate(f"{wl[i]:.1f} ms", (1.5, wl[i]), (2.05, 0.6), fontsize=5.8, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.5))
fig.tight_layout(pad=0.2); fig.savefig(HERE / "fig3b.pdf"); fig.savefig(HERE / "fig3b.png", dpi=300)
