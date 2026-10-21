"""Figures for report.html, drawn from the measured runs in ../runs and the simulation in ../sim.

    python make_figures.py        # writes figs/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
RUNS, SIM, OUT = ROOT / "runs", ROOT / "sim", Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
TIA = 220e3
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
RED, BLUE, GREY, ORANGE, GREEN = "#c8314a", "#2f6db5", "#7b8597", "#d97a12", "#2e8b57"


def nA(volts, dark):
    return (dark - volts) / TIA * 1e9


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, fc="#ffffff", ec="#333333", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else None)


def arrow(ax, x0, y0, x1, y1, color="#333333", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, lw=1.4, ls=ls))


# ---------------------------------------------------------------- 1. concept
def concept():
    fig, ax = plt.subplots(figsize=(9, 4.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.6)
    names = ["C1 inputs (top)", "C2 inputs (bottom)", "C3 hidden-1 (a)", "C4 hidden-1 (b)", "C5 hidden-2 + out"]
    for i, n in enumerate(names):
        y = 4.3 - i * 0.8
        box(ax, 0.1, y - 0.28, 2.0, 0.56, n + "\nsends", fs=8)
        box(ax, 7.9, y - 0.28, 2.0, 0.56, n + "\nreceives", fs=8)
        ax.plot([2.1, 2.9], [y, y], color=RED, lw=1.5); ax.plot(2.5, y, "o", color=RED, ms=6)
        ax.plot([7.1, 7.9], [y, y], color=RED, lw=1.5); ax.plot(7.5, y, "s", color="#333", ms=6, mfc="white")
    box(ax, 2.9, 0.9, 4.2, 3.9, "", fc="#f2f4f8")
    ax.text(5.0, 2.85, "MZI router on the chip\n(heaters choose where\neach laser's light goes)", ha="center", va="center", fontsize=10)
    ax.text(2.5, 4.75, "laser", ha="center", fontsize=8, color=RED); ax.text(7.5, 4.75, "photodiode", ha="center", fontsize=8)
    ax.plot([0.4, 9.6], [0.35, 0.35], color=GREY, lw=4)
    ax.text(5.0, 0.05, "shared electronic bus: carries every spike packet NOT sent by light", ha="center", fontsize=9, color=GREY)
    box(ax, 4.0, 5.0, 2.0, 0.5, "controller", ec=ORANGE, fs=9)
    arrow(ax, 5.0, 5.0, 5.0, 4.8, color=ORANGE, ls="--")
    save(fig, "f01_concept")


# ---------------------------------------------------------------- 2. setup
def setup():
    fig, ax = plt.subplots(figsize=(9, 3.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    box(ax, 0.1, 1.4, 1.7, 1.2, "Lab PC\n(Python scripts)", fc="#eef3fb", bold=True)
    box(ax, 2.5, 2.8, 2.0, 0.8, "USB-3114\n(laser voltages 0-10 V)", fs=8)
    box(ax, 2.5, 1.6, 2.0, 0.8, "Pico + AD5370 DAC\n(heater voltages)", fs=8)
    box(ax, 2.5, 0.4, 2.0, 0.8, "NI USB-6210\n(reads 16 detectors)", fs=8)
    box(ax, 5.2, 2.8, 1.5, 0.8, "6 lasers\n(fibres)", fs=8)
    box(ax, 7.3, 0.4, 2.5, 3.2, "Photonic chip\n6 rows, 13 MZIs\n(5 columns)\n\nheaters on each MZI\n16 on-chip detectors", fc="#f2f4f8", fs=8)
    for y in (3.2, 2.0, 0.8):
        arrow(ax, 1.8, 2.0, 2.5, y)
    arrow(ax, 4.5, 3.2, 5.2, 3.2); arrow(ax, 6.7, 3.2, 7.3, 3.2, color=RED)
    arrow(ax, 4.5, 2.0, 7.3, 2.0, color=ORANGE)
    arrow(ax, 7.3, 0.8, 4.5, 0.8, color=BLUE)
    ax.text(7.0, 3.35, "light", color=RED, fontsize=8, ha="center"); ax.text(5.9, 2.1, "heater voltages", color=ORANGE, fontsize=8, ha="center")
    ax.text(5.9, 0.9, "detector voltages", color=BLUE, fontsize=8, ha="center")
    ax.text(0.95, 0.9, "Mac: write scripts\nand analyse\n<- git push/pull ->", ha="center", fontsize=7.5, color=GREY)
    save(fig, "f02_setup")


# ---------------------------------------------------------------- 3. mesh + zig-zag
MZI_ALL = {"1_1": (1, 1), "3_1": (3, 1), "5_1": (5, 1), "2_2": (2, 2), "4_2": (4, 2), "1_3": (1, 3), "3_3": (3, 3),
           "5_3": (5, 3), "2_4": (2, 4), "4_4": (4, 4), "1_5": (1, 5), "3_5": (3, 5), "5_5": (5, 5)}


def zigzag(row):
    rows = [row]
    for c in range(1, 6):
        for top in (row, row - 1):
            if (top, c) in MZI_ALL.values():
                row = top + 1 if row == top else top
                break
        rows.append(row)
    return rows


def mesh():
    d = np.load(RUNS / "check_setup_20261019_174652.npz")
    I2, nets = d["step2_I_nA"], list(d["pd_nets"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    X = lambda c: 1.0 + 1.6 * (c - 0.5)
    for r in range(1, 7):
        ax.plot([0.2, X(5.5) + 0.3], [-r, -r], color="#d3d9e3", lw=2)
        ax.text(0.0, -r, f"row {r}", ha="right", va="center", fontsize=8)
    for name, (top, c) in MZI_ALL.items():
        color = "#f6d3da" if name == "1_5" else "white"
        ax.add_patch(FancyBboxPatch((X(c) - 0.25, -top - 1 - 0.18), 0.5, 1.36, boxstyle="round,pad=0.02", fc=color, ec="#555"))
        ax.text(X(c), -top - 0.5, name, ha="center", va="center", fontsize=7)
    for k, col in ((2, RED), (6, BLUE)):
        rows = zigzag(k)
        xs, ys = [0.3], [-rows[0]]
        for c in range(1, 6):
            xs += [X(c) - 0.25, X(c) + 0.25]; ys += [-rows[c - 1], -rows[c]]
        xs.append(X(5.5) + 0.3); ys.append(-rows[-1])
        ax.plot(xs, ys, color=col, lw=2.2, alpha=0.85, label=f"laser {k} (enters row {k}, exits row {rows[-1]})")
    ax.set_xlim(-0.6, X(5.5) + 0.6); ax.set_ylim(-6.6, -0.4); ax.axis("off")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False, bbox_to_anchor=(1.0, -0.12))
    ax.set_title("Mesh layout; with all heaters at 0 V light zig-zags (every MZI near CROSS)\n(pink: MZI 1_5, faulty)", fontsize=9.5)
    ax = axs[1]
    pred = [zigzag(k)[-1] for k in range(1, 7)]
    outs = [nets.index(f"PD_{r}_5") for r in range(1, 7)]
    M = I2[:, outs] / I2[:, outs].sum(1, keepdims=True)
    ax.imshow(M, cmap="Greys", vmin=0, vmax=0.9)
    for k in range(6):
        ax.add_patch(plt.Rectangle((pred[k] - 1 - 0.5, k - 0.5), 1, 1, fill=False, ec=RED, lw=2))
        for c in range(6):
            ax.text(c, k, f"{100 * M[k, c]:.0f}", ha="center", va="center", fontsize=7, color="white" if M[k, c] > 0.45 else "black")
    ax.set_xticks(range(6), [f"PD_{r}_5" for r in range(1, 7)], rotation=45, fontsize=8)
    ax.set_yticks(range(6), [f"laser {k}" for k in range(1, 7)], fontsize=8)
    ax.set_title("Measured: % of each laser's output light per\noutput detector (red box = predicted by the zig-zag)", fontsize=9.5)
    save(fig, "f03_mesh_zigzag")


# ---------------------------------------------------------------- 4. lasers
def lasers():
    d = np.load(RUNS / "laser_calibration_20261019_223545.npz")
    V, I = d["curve_v"], d["curve_I_nA"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for k in range(1, 6):
        for ax in axs:
            ax.plot(V, np.clip(I[k, :, k], 0.05, None), "o-", ms=3, label=f"laser {k + 1}")
    axs[0].set(xlabel="Laser control voltage (V)", ylabel="Light at its input detector (nA)", title="Laser power curves")
    axs[1].set(xlabel="Laser control voltage (V)", ylabel="nA (log scale)", yscale="log", title="Same, log scale: lasing starts at 7.1-7.5 V")
    axs[1].axvspan(7.1, 7.5, color="0.9")
    axs[0].legend(fontsize=8, frameon=False)
    save(fig, "f04_lasers")


# ---------------------------------------------------------------- 5. heaters
def heaters():
    d = np.load(RUNS / "heater_sweep_20261019_221707.npz"); cal = json.loads((RUNS / "heater_sweep_20261019_221707.json").read_text())
    nets, V, dark = list(d["pd_nets"]), d["heater_v"], d["dark"]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.7))
    m = "2_2"; c = cal[m]; k = int(c["laser"][-1]) - 1
    I = nA(d[f"{m}_V"], dark); T = I / I[:, [k]]
    ax = axs[0]
    ax.plot(V, T[:, nets.index(c["cross_pd"])], "o", ms=3, color=BLUE, label=f"{c['cross_pd']} (light goes here when CROSS)")
    ax.plot(V, T[:, nets.index(c["bar_pd"])], "s", ms=3, color=ORANGE, label=f"{c['bar_pd']} (light goes here when BAR)")
    for key, col, lab in (("V_cross", GREEN, "CROSS"), ("V_50", GREY, "50:50"), ("V_bar", RED, "BAR")):
        ax.axvline(c[key], color=col, ls=":"); ax.text(c[key], 0.52, lab, color=col, fontsize=8, ha="center")
    ax.set(xlabel="Heater voltage on MZI 2_2 (V)", ylabel="Detector / laser input", title="Good MZI (2_2): light see-saws between two outputs", ylim=(0, 0.56))
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax = axs[1]
    for f, col, lab in (("183039", RED, "run 1: jumps near 1 V, then stuck"), ("184324", ORANGE, "run 2: same"), ("221707", BLUE, "run 4: normal")):
        e = np.load(RUNS / f"heater_sweep_20261019_{f}.npz"); I = nA(e["1_5_V"], e["dark"]); T = I / I[:, [3]]
        ax.plot(e["heater_v"], T[:, list(e["pd_nets"]).index("PD_1_5")], "o-", ms=2.5, lw=1, color=col, label=lab)
    ax.axvspan(0.9, 1.5, color="0.92")
    ax.set(xlabel="Heater voltage on MZI 1_5 (V)", ylabel="PD_1_5 / laser input", title="Faulty MZI (1_5): behaves differently run to run")
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, "f05_heaters")


# ---------------------------------------------------------------- 6. readout artifact
def readout():
    old = [33.6, 17.8, 14.2, 33.7, 33.1]; new = [0.0, 0.0, 0.1, 0.1, 0.0]       # readout_check_20261019_221422 (lasers 1-5)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(5)
    ax.bar(x - 0.2, old, 0.4, color=RED, label="before: fast reading (pydaq group_vin)")
    ax.bar(x + 0.2, new, 0.4, color=BLUE, label="after: 100 µs settling per channel")
    for i in x:
        ax.text(i - 0.2, old[i] + 0.8, f"{old[i]:.0f}%", ha="center", fontsize=8); ax.text(i + 0.2, new[i] + 0.8, f"{new[i]:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x, [f"laser {k}" for k in range(1, 6)])
    ax.set(ylabel="Fake signal on the next detector read\n(% of the bright detector)", title="Readout artifact: the next channel 'inherited' about a third of the previous one")
    ax.set_ylim(0, 40)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save(fig, "f06_readout")


# ---------------------------------------------------------------- 7. laser interaction
def interaction():
    a = np.load(RUNS / "laser_interaction_20261019_225404.npz"); b = np.load(RUNS / "laser_interaction_20261019_224416.npz")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, d, title in ((axs[0], a, "9.1-9.9 V (equal-power voltages)"), (axs[1], b, "all lasers at 10 V")):
        M = d["settled_pct"]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-40, vmax=40)
        for (r, c), v in np.ndenumerate(M):
            if np.isfinite(v):
                ax.text(c, r, f"{v:+.0f}", ha="center", va="center", fontsize=8)
        L = [f"laser {k}" for k in d["lasers"]]
        ax.set_xticks(range(len(L)), L, rotation=45, fontsize=8); ax.set_yticks(range(len(L)), L, fontsize=8)
        ax.set(xlabel="laser switched on", ylabel="laser whose power is measured", title=title)
    fig.colorbar(im, ax=axs, label="power change (%)", shrink=0.8)
    save(fig, "f07_interaction")


# ---------------------------------------------------------------- 8. routing
def routing():
    d = np.load(RUNS / "route_check_20261019_230840.npz"); R = json.loads((SIM / "routes_rows2-6.json").read_text())
    nets = list(d["pd_nets"]); cols = ["PD_1_5", "PD_2_5", "PD_3_5", "PD_4_5", "PD_5_5", "PD_6_5"]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, s in zip(axs, "ABCD"):
        I = d[f"{s}_I_nA"]; inp = np.array([I[k, nets.index(p)] for k, p in enumerate(R["input_pd"])])
        T = I[:, [nets.index(c) for c in cols]] / inp[:, None]
        ax.imshow(T, cmap="Greys", vmin=0, vmax=0.6)
        for (r, c), v in np.ndenumerate(T):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.3 else "black")
        for src, ds in R["settings"][s]["routes"].items():
            for x in ds:
                ax.add_patch(plt.Rectangle((cols.index(R["receiver_pd"][x]) - 0.5, int(src) - 0.5), 1, 1, fill=False, ec=RED, lw=2))
        routes = "; ".join(f"C{int(a) + 1}→" + "+".join(f"C{x + 1}" for x in b) for a, b in R["settings"][s]["routes"].items())
        ax.set_title(f"Setting {s}: {routes}", fontsize=9)
        ax.set_xticks(range(6), cols, rotation=60, fontsize=7.5)
    axs[0].set_yticks(range(5), [f"C{k + 1} ({l})" for k, l in enumerate(R["lasers"])], fontsize=8)
    fig.suptitle("Routing measured on the chip (lasers at 10 V): fraction of each laser's light at each output; red = intended", fontsize=10)
    save(fig, "f08_routing")


# ---------------------------------------------------------------- 9. switching
def switching():
    a = np.load(RUNS / "switch_timing_20261019_231107.npz"); b = np.load(RUNS / "switch_timing_20261019_231635.npz"); rate = 250e3
    def avg(Y):
        out = []
        for y in Y:
            lo, hi = np.median(y[:5000]), np.median(y[-5000:]); n = (y - lo) / (hi - lo)
            k = np.argmax(n > 0.5); out.append(n[k - 25:k + 50])
        return np.mean(out, 0)
    t = (np.arange(-25, 50)) * 4.0
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.5))
    ax = axs[0]
    ax.plot(t, avg(b["laser_laser_on"]), color=GREY, lw=2, label="laser switched on (detector speed check)")
    ax.plot(t, avg(a["2_2_A->C"]), color=RED, lw=2, label="heater 2_2 BAR→CROSS (route A→C)")
    ax.plot(t, avg(a["4_2_A->D"]), color=BLUE, lw=2, label="heater 4_2 BAR→CROSS (route A→D)")
    ax.axhline(0.9, color="0.8", lw=0.8); ax.axhline(0.1, color="0.8", lw=0.8)
    ax.set(xlabel="Time around the edge (µs)", ylabel="Light change (0 = before, 1 = after)", title="Route switching: ~32-36 µs (10-90%), detector ~20 µs")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax = axs[1]
    for y in a["2_2_A->C"]:
        lo, hi = np.median(y[:5000]), np.median(y[-5000:])
        ax.plot((np.arange(len(y)) / rate - 0.02) * 1e3, (y - lo) / (hi - lo), lw=0.6, alpha=0.6)
    ax.set(xlim=(-2, 8), xlabel="Time after the heater command (ms)", ylabel="Light change", title="20 switches: light moves ~1 ms after the command\n(the delay is our Python→USB→Pico path)")
    save(fig, "f09_switching")


# ---------------------------------------------------------------- 10. replay excerpt
def replay():
    d = np.load(RUNS / "spike_replay_20261019_235342.npz"); y, nets = d["y"], list(d["nets"]); sender, det = d["sender"], d["detected"]
    rp = json.loads((SIM / "replay_excerpt.json").read_text())
    T0, slot, rate = 0.5, 0.015, 800.0
    t = np.arange(y.shape[1]) / rate
    k0, k1 = 120, 165
    m = (t >= T0 + k0 * slot) & (t < T0 + k1 * slot)
    dark = np.median(y[:, t < 0.4], axis=1)
    fig, axs = plt.subplots(3, 1, figsize=(11, 5.4), sharex=True)
    for ax, j, lab in ((axs[0], 2, "C3 receiver (PD_4_5)"), (axs[1], 3, "C4 receiver (PD_5_5)"), (axs[2], 4, "C5 receiver (PD_6_5)")):
        ax.plot((t[m] - T0) / slot, (y[j, m] - dark[j]) / TIA * 1e9, color="#333", lw=0.9)
        for k in range(k0, k1):
            if sender[k, j] >= 0:
                ax.axvspan(k, k + 0.4, color={0: RED, 1: BLUE, 2: GREEN, 3: ORANGE}[sender[k, j]], alpha=0.18)
            if not det[k, j] and sender[k, j] >= 0:
                ax.text(k + 0.2, 0, "miss", color=RED, fontsize=7)
        ax.set_ylabel("nA", fontsize=8); ax.set_title(lab, fontsize=9, loc="left")
    for k in range(k0, k1):
        if k % 5 == 0:
            w = rp["windows"][k // 5]["setting"]
            axs[0].text(k + 0.1, axs[0].get_ylim()[1] * 0.92, w, fontsize=8, color=GREY)
            for ax in axs:
                ax.axvline(k, color=GREY, lw=0.6, ls=":")
    axs[2].set_xlabel("SNN time step (each 2 ms step replayed as a 15 ms slot); grey letters = router setting, changed every 5 steps")
    fig.suptitle("Spike replay excerpt: shaded = a spike was sent to this receiver (colour = sending core: C1 red, C2 blue, C3 green, C4 orange)", fontsize=9.5)
    save(fig, "f10_replay")


# ---------------------------------------------------------------- 11. bus delay
def bus():
    d = np.load(RUNS / "spike_replay_20261019_235342.npz"); sender, det = d["sender"], d["detected"]
    rp = json.loads((SIM / "replay_excerpt.json").read_text())
    P = np.array([s["packets"] for s in rp["steps"]], float); total = P.sum((1, 2))
    opt = np.array([sum(P[k, sender[k, j], j] for j in range(5) if sender[k, j] >= 0 and det[k, j]) for k in range(len(total))])
    lam = total.mean()
    def p99(arr, cap):
        x, q = 0.0, []
        for a in arr:
            x = max(0.0, x + a - cap); q.append(x)
        return np.percentile(np.array(q) / cap * 2, 99), np.mean(np.array(q) / cap * 2)
    f = np.array([0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
    bo = np.array([p99(total, x * lam) for x in f]); wl = np.array([p99(total - opt, x * lam) for x in f])
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axs[0]
    ax.plot(f, bo[:, 0], "o-", color=GREY, label="bus only"); ax.plot(f, wl[:, 0], "o-", color=RED, label="bus + optical lanes")
    ax.set(yscale="symlog", xlabel="Bus capacity (× average traffic)", ylabel="99th-percentile delay (ms)", title="Worst-case bus waiting time (measured delivery)")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    k = slice(3000, 3600)
    ax.plot(np.arange(600) * 2, total[k] / 2, color=GREY, lw=0.8, label="all traffic (bus only)")
    ax.plot(np.arange(600) * 2, (total - opt)[k] / 2, color=RED, lw=0.8, label="left on the bus with lanes")
    ax.axhline(lam / 2, color="k", lw=0.8, ls=":"); ax.text(1190, lam / 2 + 3, "average", ha="right", fontsize=8)
    ax.set(xlabel="SNN time (ms)", ylabel="packets per ms", title="Traffic is bursty (peaks 4-5× the average)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "f11_bus")


# ---------------------------------------------------------------- 12. scaling
def scaling():
    r = json.loads((SIM / "scaling_results.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for net, col, lab in (("fc", BLUE, "fully connected net (90.9%)"), ("local", RED, "locally connected net (88.2%)")):
        Ns = sorted(int(n) for n in r[net])
        axs[0].plot(Ns, [100 * r[net][str(n)]["offload_dynamic"] for n in Ns], "o-", color=col, label=f"{lab}, dynamic")
        axs[0].plot(Ns, [100 * r[net][str(n)]["offload_fixed"] for n in Ns], "o--", color=col, alpha=0.45, label=f"{lab}, fixed")
        axs[1].plot(Ns, [r[net][str(n)]["delay_bus_only_1.5x"][1] for n in Ns], "o:", color=col, alpha=0.6, label=f"{net}: bus only")
        axs[1].plot(Ns, [r[net][str(n)]["delay_with_lanes_1.5x"][1] for n in Ns], "o-", color=col, label=f"{net}: with lanes")
    axs[0].set(xlabel="Number of cores", ylabel="Spike packets carried by light (%)", title="Share of traffic on light vs number of cores")
    axs[1].set(xlabel="Number of cores", ylabel="99th-percentile bus delay (ms)", yscale="symlog", title="Bus delay with capacity = 1.5× average traffic")
    for ax in axs:
        ax.legend(fontsize=7.5, frameon=False)
    save(fig, "f12_scaling")


# ---------------------------------------------------------------- 13. PD_6_5
def pdcheck():
    on = [1486.3, 1536.0, 1544.4, 1551.0, 1522.4]; off = [1.3, 2.6, 3.2, 3.7, 4.6]            # pd_check_20261020_004855
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    ax = axs[0]
    x = np.arange(5)
    ax.bar(x - 0.2, on, 0.4, color=RED, label="laser 5 on"); ax.bar(x + 0.2, off, 0.4, color=GREY, label="laser 5 off")
    ax.set(xticks=x, xticklabels=[f"cycle {i + 1}" for i in x], ylabel="PD_6_5 (nA)", title="Setting A: PD_6_5 follows laser 5 every time")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    ax.bar([0, 1], [1528, 9.4], color=[RED, GREY], width=0.35, label="PD_6_5")
    ax.bar([0.4, 1.4], [29, 1648], color=[GREY, BLUE], width=0.35, label="PD_3_5")
    ax.set(xticks=[0.2, 1.2], xticklabels=["setting A\n(route → row 6)", "setting B\n(route → row 3)"], ylabel="nA, laser 5 on",
           title="Re-routing moves the light from PD_6_5 to PD_3_5")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "f13_pd65")


if __name__ == "__main__":
    for f in (concept, setup, mesh, lasers, heaters, readout, interaction, routing, switching, replay, bus, scaling, pdcheck):
        f()


# ---------------------------------------------------------------- 14. network -> cores -> chip
def partition():
    import sys
    sys.path.insert(0, str(SIM))
    import scaling
    import os
    here = os.getcwd(); os.chdir(SIM)                  # scaling.py reads its data from sim/
    try:
        P = scaling.traffic("fc", 2)[:100 * 150]       # first 100 test digits = the digits replayed on the chip
    finally:
        os.chdir(here)
    rate = P.sum(0) / (P.shape[0] * 2.0)               # packets per ms of network time
    sp = np.load(SIM / "spikes_test.npz")
    ev = sp["inp"][:20].reshape(20, -1, 2, 34, 34).sum((1, 2))          # all events per pixel, first 20 digits
    img = ev[int(np.argmax(ev.reshape(20, -1).sum(1)))].astype(float)   # the busiest one: clearest shape
    fig = plt.figure(figsize=(12.5, 5.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 12.5); ax.set_ylim(0, 5.6)
    core_col = {1: RED, 2: BLUE, 3: GREEN, 4: ORANGE, 5: "#7a4fa3"}
    # inputs: the image, split into top / bottom halves
    iax = fig.add_axes([0.02, 0.18, 0.2, 0.62]); iax.imshow(img, cmap="Greys", vmin=0, vmax=np.percentile(img, 97))
    iax.axhline(16.5, color="k", lw=2); iax.set_xticks([]); iax.set_yticks([])
    for sp_ in iax.spines.values():
        sp_.set_visible(False)
    iax.add_patch(plt.Rectangle((-0.5, -0.5), 34, 17, fill=False, ec=core_col[1], lw=4))
    iax.add_patch(plt.Rectangle((-0.5, 16.5), 34, 17, fill=False, ec=core_col[2], lw=4))
    ax.text(1.37, 5.0, "Input layer: 2,312 inputs\n(34×34 pixels × 2 event polarities)", ha="center", fontsize=9)
    ax.text(2.85, 3.75, "C1\ntop half\n1,156 inputs", color=core_col[1], fontsize=9, weight="bold", va="center")
    ax.text(2.85, 1.85, "C2\nbottom half\n1,156 inputs", color=core_col[2], fontsize=9, weight="bold", va="center")
    # hidden-1: 128 neurons, two cores of 64
    def neurons(x0, y0, n, cols, col, dx=0.11, dy=0.11):
        for i in range(n):
            ax.add_patch(plt.Circle((x0 + (i % cols) * dx, y0 - (i // cols) * dy), 0.035, color=col))
    box(ax, 4.55, 3.0, 1.3, 1.35, "", ec=core_col[3]); neurons(4.66, 4.2, 64, 11, core_col[3])
    box(ax, 4.55, 1.15, 1.3, 1.35, "", ec=core_col[4]); neurons(4.66, 2.35, 64, 11, core_col[4])
    ax.text(5.2, 4.55, "Hidden layer 1: 128 neurons", ha="center", fontsize=9)
    ax.text(5.2, 2.85, "C3: neurons 1–64", ha="center", fontsize=8.5, color=core_col[3], weight="bold")
    ax.text(5.2, 1.0, "C4: neurons 65–128", ha="center", fontsize=8.5, color=core_col[4], weight="bold")
    # hidden-2 + outputs in one core
    box(ax, 7.0, 1.6, 1.5, 2.2, "", ec=core_col[5]); neurons(7.12, 3.6, 64, 11, core_col[5]); neurons(7.12, 2.15, 10, 10, "#333")
    ax.text(7.75, 4.0, "C5", ha="center", fontsize=9, color=core_col[5], weight="bold")
    ax.text(7.75, 1.4, "hidden layer 2 (64)\n+ outputs (10 digits)", ha="center", va="top", fontsize=8)
    # links with measured traffic
    def link(x0, y0, x1, y1, r, label_xy):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6 + 0.35 * r, color="#444"))
        ax.text(*label_xy, f"{r:.1f}/ms", fontsize=7.5, color="#333", ha="center",
                bbox=dict(fc="white", ec="none", pad=0.5))
    link(3.9, 3.9, 4.5, 3.75, rate[0, 2], (4.2, 4.08)); link(3.9, 3.7, 4.5, 1.95, rate[0, 3], (3.8, 3.05))
    link(3.9, 1.95, 4.5, 3.55, rate[1, 2], (3.8, 2.45)); link(3.9, 1.75, 4.5, 1.75, rate[1, 3], (4.2, 1.55))
    link(5.9, 3.6, 6.95, 3.0, rate[2, 4], (6.4, 3.5)); link(5.9, 1.85, 6.95, 2.35, rate[3, 4], (6.4, 1.85))
    ax.text(3.0, 0.45, "Arrows: average spike packets per ms of network time; line width ∝ traffic.\n"
            "Every input pixel connects to all 128 hidden-1 neurons, so C1 and C2 each\n"
            "send to both C3 and C4. Hidden-2 → outputs stays inside C5 (no packets).",
            fontsize=7.8, color=GREY, va="center")
    # chip mapping table
    ax.text(10.6, 5.05, "On the chip (rows 2–6)", ha="center", fontsize=10, weight="bold")
    rows = [("C1", "laser_2", "row 2", "PD_1_5*"), ("C2", "laser_3", "row 3", "PD_3_5"), ("C3", "laser_4", "row 4", "PD_4_5"),
            ("C4", "laser_5", "row 5", "PD_5_5"), ("C5", "laser_6", "row 6", "PD_6_5")]
    ax.text(9.05, 4.65, "core    sends with    row    receives on", fontsize=8.5, family="monospace", color=GREY)
    for i, (c, l, r, pd) in enumerate(rows):
        ax.text(9.05, 4.3 - i * 0.33, f"{c:<7} {l:<13} {r:<6} {pd}", fontsize=8.5, family="monospace",
                color=core_col[i + 1])
    ax.text(9.05, 2.45, "Optical links the router can make:", fontsize=8.8, weight="bold")
    ax.text(9.05, 2.15, "• C1 or C2 → C3 + C4 (split 50:50)\n   chosen by heater 2_2 (CROSS: C1, BAR: C2)", fontsize=8.3, va="top")
    ax.text(9.05, 1.45, "• C3 or C4 → C5\n   chosen by heater 4_2 (CROSS: C3, BAR: C4)", fontsize=8.3, va="top")
    ax.text(9.05, 0.85, "Each receiver hears one sender at a time;\nthe other sender's packets use the bus.\n* row 2 exits on PD_1_5 (MZI 1_5 left at 0 V)",
            fontsize=7.8, va="top", color=GREY)
    save(fig, "f14_partition")
