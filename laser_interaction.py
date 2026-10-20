'''Does switching one laser on change another laser's power?

    route_check (2026-10-19) found that with two lasers on, the hidden-core laser (4 or 5)
    got 9-24% brighter and the input-core laser (2 or 3) 3-8% dimmer, on their own input
    detectors, i.e. before the chip. For spikes that would make one core's spike height
    depend on whether another core fires.

    For each laser i (held on), every other laser j is switched on and off a few times while
    laser i's input detector is recorded about 10 times a second. Reported per pair:
      - fast change: laser i's power in the first 0.1-0.6 s after j turns on, vs j off
        (an immediate change points to electrical coupling: shared driver, supply, ground)
      - settled change: in the last second of j's on-time (a slower change points to heat)
    Laser i's light never reaches another laser's input detector, so any change is real.

    Results go to runs/laser_interaction_<timestamp>.{npz,png,txt}.

    Usage (from the repo root on the lab PC):
        python laser_interaction.py                 # equal-power voltages, about 7 min
        python laser_interaction.py --laser-v 10    # all lasers at 10 V, to compare
'''
import argparse
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep

import check_setup as cs
from check_setup import INPUT_PDS, LASER_NETS, PD_NETS, netlist, record, report, set_lasers, to_nA
from route_check import laser_voltages

import matplotlib.pyplot as plt
import numpy as np

SETTLE_S = 8.0       # laser i alone before the first toggle
ON_S = OFF_S = 3.0   # each half of a toggle cycle
CYCLES = 3
PERIOD_S = 0.1       # recording period
FAST = (0.1, 0.6)    # window after a switch-on counted as the fast response (s)
OK_PCT = 2.0         # changes below this count as no interaction


def toggle_run(i: int, j: int, dark: np.ndarray, v_j: float) -> tuple[np.ndarray, np.ndarray, list[float]]:
    '''Toggles laser j while laser i stays on; returns (times, laser i input nA, switch-on times).'''
    pd = PD_NETS.index(INPUT_PDS[i])
    t, y, ons = [], [], []
    start = monotonic()
    for _ in range(CYCLES):
        for state, dur in ((v_j, ON_S), (0.0, OFF_S)):
            netlist[LASER_NETS[j]].vout(state)
            if state:
                ons.append(monotonic() - start)
            t_end = monotonic() + dur
            while monotonic() < t_end:
                t.append(monotonic() - start)
                y.append(to_nA(cs.read_pds()[pd], dark[pd]))
                sleep(max(0.0, PERIOD_S - 0.03))
    return np.array(t), np.array(y), ons


def changes(t: np.ndarray, y: np.ndarray, ons: list[float]) -> tuple[float, float]:
    '''Fast and settled change of laser i (percent) when laser j is on, vs the off level before.'''
    fast, settled = [], []
    for t_on in ons:
        off = y[(t >= t_on - 1.0) & (t < t_on)]
        if not off.size:  # first cycle: no off period before it within this run
            continue
        base = off.mean()
        fast.append(y[(t >= t_on + FAST[0]) & (t < t_on + FAST[1])].mean() / base - 1)
        settled.append(y[(t >= t_on + ON_S - 1.0) & (t < t_on + ON_S)].mean() / base - 1)
    return 100 * float(np.mean(fast)), 100 * float(np.mean(settled))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lasers", type=int, nargs="+", default=[2, 3, 4, 5, 6], help="laser numbers to test")
    parser.add_argument("--laser-v", type=float, default=None, help="one voltage for all (default: equal-power voltages)")
    args = parser.parse_args()
    idx = [k - 1 for k in args.lasers]
    volts = {net: args.laser_v for net in LASER_NETS} if args.laser_v else laser_voltages()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(__file__).resolve().parent / "runs" / f"laser_interaction_{stamp}"
    out.parent.mkdir(exist_ok=True)
    fast = np.full((len(idx), len(idx)), np.nan)
    settled = np.full((len(idx), len(idx)), np.nan)
    data: dict[str, np.ndarray] = {"lasers": np.array(args.lasers),
                                   "volts": np.array([volts[LASER_NETS[k]] for k in idx])}

    cs.set_laser_range()
    with netlist:
        try:
            set_lasers(0.0)
            for net in cs.HEATER_NETS:
                netlist[net].vout(0.0)
            sleep(cs.OFF_WAIT_S)
            _, v_dark = record(3.0, 0.2)
            dark = v_dark.mean(axis=0)
            report(f"laser_interaction {stamp}: lasers {args.lasers} at " +
                   ", ".join(f"{volts[LASER_NETS[k]]:.2f} V" for k in idx) +
                   f"; each other laser toggled {CYCLES}x ({ON_S:.0f} s on / {OFF_S:.0f} s off)")
            for a, i in enumerate(idx):
                netlist[LASER_NETS[i]].vout(volts[LASER_NETS[i]])
                sleep(SETTLE_S)
                for b, j in enumerate(idx):
                    if j == i:
                        continue
                    t, y, ons = toggle_run(i, j, dark, volts[LASER_NETS[j]])
                    fast[a, b], settled[a, b] = changes(t, y, ons)
                    data[f"t_{i + 1}_{j + 1}"], data[f"y_{i + 1}_{j + 1}"] = t, y
                netlist[LASER_NETS[i]].vout(0.0)
                row = ", ".join(f"laser_{j + 1} {fast[a, b]:+.1f}/{settled[a, b]:+.1f}%"
                                for b, j in enumerate(idx) if j != i)
                report(f"  laser_{i + 1} power when another laser turns on (fast/settled): {row}")
                sleep(2.0)
            worst = np.nanmax(np.abs(settled))
            report(f"Largest settled change {worst:.1f}% -> " +
                   ("no meaningful interaction" if worst < OK_PCT else "lasers affect each other"))
        except KeyboardInterrupt:
            report("Interrupted; saving what was measured.")
        finally:
            set_lasers(0.0)
            data.update(fast_pct=fast, settled_pct=settled)
            np.savez_compressed(out.with_suffix(".npz"), **data)
            out.with_suffix(".txt").write_text("\n".join(cs.summary) + "\n")
            plot(args.lasers, fast, settled, data, out.with_suffix(".png"))
            report(f"Saved {out.name}.npz/.txt/.png in {out.parent}")


def plot(lasers: list[int], fast: np.ndarray, settled: np.ndarray, data: dict, path: Path) -> None:
    '''Two change matrices plus the time traces of one example pair.'''
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    fig.suptitle(path.stem)
    lim = max(5.0, np.nanmax(np.abs(settled)) if np.isfinite(settled).any() else 5.0)
    for ax, M, title in ((axs[0], fast, "fast (0.1-0.6 s after on)"), (axs[1], settled, "settled (last 1 s on)")):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for (r, c), v in np.ndenumerate(M):
            if np.isfinite(v):
                ax.text(c, r, f"{v:+.1f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(lasers)), [f"laser_{k}" for k in lasers], rotation=45)
        ax.set_yticks(range(len(lasers)), [f"laser_{k}" for k in lasers])
        ax.set_xlabel("laser switched on/off")
        ax.set_ylabel("laser measured (held on)")
        ax.set_title(f"Power change, % - {title}", fontsize=10)
    fig.colorbar(im, ax=axs[1], label="%")
    if np.isfinite(settled).any():
        r, c = np.unravel_index(np.nanargmax(np.abs(settled)), settled.shape)
        key = f"{lasers[r]}_{lasers[c]}"
        if f"t_{key}" in data:
            axs[2].plot(data[f"t_{key}"], data[f"y_{key}"], lw=1)
            axs[2].set_title(f"laser_{lasers[r]} input while laser_{lasers[c]} toggles (largest change)", fontsize=10)
            axs[2].set_xlabel("Time (s)")
            axs[2].set_ylabel("Photocurrent (nA)")
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
