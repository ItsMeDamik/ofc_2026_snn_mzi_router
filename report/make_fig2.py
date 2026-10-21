"""Fig. 2: measured routing (settings A, B) and route-switching speed, from the lab run files.
Lives in report/; reads ../runs and ../sim, writes fig2.pdf next to itself."""
import json, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
RUNS, SIM = ROOT / "runs", ROOT / "sim"
BLUE, ORNG, GREY, RED = "#2a62b8", "#e07b14", "#8a909a", "#c43a4a"

R = json.loads((SIM / "routes_rows2-6.json").read_text())
d = np.load(RUNS / "route_check_20261019_230840.npz"); nets = list(d["pd_nets"])
def matrix(s):
    I = d[f"{s}_I_nA"]
    inp = np.array([I[k, nets.index(p)] for k, p in enumerate(R["input_pd"])])
    return I[:, [nets.index(p) for p in R["receiver_pd"]]] / inp[:, None]   # rows: sender, cols: receiver

fig = plt.figure(figsize=(6.8, 2.05))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.42, left=0.07, right=0.975, bottom=0.2, top=0.88)
names = [f"C{k+1}" for k in range(5)]
for i, (s, lanes) in enumerate([("A", {1: BLUE, 3: ORNG}), ("B", {0: BLUE, 2: ORNG})]):
    ax = fig.add_subplot(gs[i]); T = matrix(s)
    ax.imshow(T, cmap="Greys", vmin=0, vmax=0.6)
    for (r, c), v in np.ndenumerate(T):
        ax.text(c, r, f"{abs(v):.2f}", ha="center", va="center", fontsize=5.6, color="white" if v > 0.3 else "#222")
    for src, ds in R["settings"][s]["routes"].items():
        for x in ds:
            ax.add_patch(plt.Rectangle((x-0.5, int(src)-0.5), 1, 1, fill=False, ec=lanes[int(src)], lw=1.4))
    ax.set_xticks(range(5), names); ax.set_yticks(range(5), names)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlabel("receiver (photodiode)", fontsize=6.3, labelpad=2)
    if i == 0: ax.set_ylabel("sender (laser)", fontsize=6.3, labelpad=2)
    routes = ", ".join(f"C{int(a)+1}→" + "+".join(f"C{x+1}" for x in b) for a, b in R["settings"][s]["routes"].items())
    ax.set_title(f"({'ab'[i]}) Setting {s}: {routes}", fontsize=6.8, loc="left", pad=4)

# switching
a = np.load(RUNS / "switch_timing_20261019_231107.npz"); b = np.load(RUNS / "switch_timing_20261019_231635.npz")
def avg(Y):
    out = []
    for y in Y:
        lo, hi = np.median(y[:5000]), np.median(y[-5000:]); n = (y - lo) / (hi - lo)
        k = np.argmax(n > 0.5); out.append(n[k-25:k+50])
    return np.mean(out, 0)
t = np.arange(-25, 50) * 4.0                                   # 250 kS/s -> 4 us/sample
ax = fig.add_subplot(gs[2])
ax.plot(t, avg(a["2_2_A->C"]), color=BLUE, lw=1.3, label="MZI 2_2, BAR→CROSS")
ax.plot(t, avg(a["4_2_A->D"]), color=ORNG, lw=1.3, label="MZI 4_2, BAR→CROSS")
ax.plot(t, avg(b["laser_laser_on"]), color=GREY, lw=1.0, ls="--", label="laser step (detector limit)")
for yv in (0.1, 0.9): ax.axhline(yv, color="0.85", lw=0.5, zorder=0)
ax.set(xlim=(-100, 200), xticks=[-100, 0, 100, 200], ylim=(-0.05, 1.15), xlabel="time (µs)", ylabel="normalized power")
ax.set_title("(c) Route switching, avg. of 20 edges", fontsize=6.8, loc="left", pad=4)
ax.text(60, 0.42, "10–90%: 32–36 µs\ndetector: 20 µs", fontsize=6, color="#333")
ax.legend(fontsize=5.6, frameon=False, loc="lower right")
fig.savefig(HERE / "fig2.pdf"); fig.savefig(HERE / "fig2.png", dpi=300)
print("saved")
