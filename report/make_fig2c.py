"""Fig. 2(c) standalone: route switching edges (avg. of 20). Writes fig2c.pdf (with detector-limit
reference) and fig2c_nolaser.pdf (MZI edges only)."""
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" if (HERE / "runs").exists() else HERE.parent / "runs"
BLUE, ORNG, GREY = "#2a62b8", "#e07b14", "#8a909a"
a = np.load(RUNS / "switch_timing_20261019_231107.npz"); b = np.load(RUNS / "switch_timing_20261019_231635.npz")
def avg(Y):
    out = []
    for y in Y:
        lo, hi = np.median(y[:5000]), np.median(y[-5000:]); n = (y - lo) / (hi - lo)
        k = np.argmax(n > 0.1); out.append(n[k-25:k+50])   # t = 0 at the 10% point
    return np.mean(out, 0)
t = np.arange(-25, 50) * 4.0
for name, laser in (("fig2c", True), ("fig2c_nolaser", False)):
    fig, ax = plt.subplots(figsize=(2.6, 1.9))
    ax.plot(t, avg(a["2_2_A->C"]), color=BLUE, lw=1.3, label="MZI 2_2, BAR→CROSS")
    ax.plot(t, avg(a["4_2_A->D"]), color=ORNG, lw=1.3, label="MZI 4_2, BAR→CROSS")
    if laser: ax.plot(t, avg(b["laser_laser_on"]), color=GREY, lw=1.0, ls="--", label="laser step (detector)")
    for yv in (0.1, 0.9): ax.axhline(yv, color="0.85", lw=0.5, zorder=0)
    ax.set(xlim=(-50, 200), xticks=[-50, 0, 50, 100, 150, 200], ylim=(-0.05, 1.18 if laser else 1.1),
           xlabel="time (µs)", ylabel="normalized power")
    ax.text(55, 0.45, "10–90%: 32–36 µs" + ("\ndetector: 20 µs" if laser else ""), fontsize=6, color="#333")
    ax.legend(fontsize=5.6, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.2); fig.savefig(HERE / f"{name}.pdf"); fig.savefig(HERE / f"{name}.png", dpi=300)
