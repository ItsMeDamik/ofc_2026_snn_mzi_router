'''Operational check of the 6x6 MZI setup: steps 0-4 of the measurement plan.

    0. Connections: the NI, laser (USB-3114) and Pico boards are found.
    1. Dark baseline: lasers off, heaters at 0 V; detector levels and noise.
    2. One laser at a time: which input row each laser feeds and how strongly.
    3. Lasers added one at a time: whether turning more lasers on dims the others.
    4. Stability: all lasers on; warm-up time, drift and sudden jumps (mode hops).

    The detector voltage drops when light arrives, so photocurrent is
    (dark voltage - voltage) / 220 kOhm, using the step 1 dark level.

    Results go to runs/check_setup_<timestamp>.{npz,png,txt}. The .txt is a
    short pass/fail summary meant to be committed and pushed back.

    Usage (from the repo root on the lab PC):
        python check_setup.py            # full run, about 25 min
        python check_setup.py --quick    # shorter settle/stability times, about 6 min
'''
import argparse
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep

import matplotlib
matplotlib.use("Agg")  # only save figures; never block on a window
import matplotlib.pyplot as plt
import numpy as np
from mcculw import ul
from mcculw.enums import BoardInfo, InfoType

from pydaq.logger import log
from board_config_6x6 import netlist

TRANSIMPEDANCE = 220e3  # ohms, same value the MVM_Experiments scripts use
N = 6
LASER_NETS = [f"laser_{k}" for k in range(1, N + 1)]
INPUT_PDS = [f"PD_{r}_0" for r in range(1, N + 1)]
OUTPUT_PDS = [f"PD_{r}_5" for r in range(1, N + 1)]
INTERNAL_PDS = ["PD_1_2", "PD_6_1", "PD_1_3", "PD_6_3"]
PD_NETS = INPUT_PDS + OUTPUT_PDS + INTERNAL_PDS  # all on the NI board, so one group read
HEATER_NETS = [net for net in netlist.pins_dict if net[:1].isdigit()]  # e.g. "1_3_i"

LIT_NA = 5.0         # a detector counts as lit above this (dark noise is ~0.2-0.3 nA)
LIT_FRACTION = 0.5   # ...and only if at least this fraction of the brightest input; weaker ones are leaks
OFF_WAIT_S = 5.0     # wait after switching lasers off before the next measurement
STEP3_TOL = 0.10     # step 3: each input within +-10% of the sum of the step 2 single-laser readings
DRIFT_TOL = 0.02     # step 4: drift after warm-up below 2%
JUMP_TOL = 0.03      # step 4: a change above 3% between consecutive 5 s means counts as a jump

summary: list[str] = []


def report(line: str) -> None:
    '''Prints a line, writes it to the log and keeps it for the summary file.'''
    print(line)
    log.info(line)
    summary.append(line)


def set_lasers(volts) -> None:
    '''Sets every laser control voltage; a scalar sets all lasers to that value.'''
    volts = np.broadcast_to(np.asarray(volts, dtype=float), (N,))
    for net, v in zip(LASER_NETS, volts):
        netlist[net].vout(float(v))


def set_laser_range() -> None:
    '''Pins every laser channel's USB-3114 output range to the range vout is called
        with. The range is a per-channel device setting that pydaq does not set
        (board_config_6x6 does the same when run directly).
    '''
    laser_board = netlist.board_dict["lasers"]
    for pin in laser_board.pins.values():
        ul.set_config(InfoType.BOARDINFO, laser_board.board_num, pin.chnl,
                      BoardInfo.DACRANGE, laser_board.ao_range)


def record(duration_s: float, period_s: float) -> tuple[np.ndarray, np.ndarray]:
    '''Reads every photodetector once per period for duration_s seconds.

        Returns (times in s from the start, voltages of shape (samples, len(PD_NETS))).
    '''
    times, volts = [], []
    start = monotonic()
    while (now := monotonic() - start) < duration_s:
        times.append(now)
        volts.append(netlist.group_vin(PD_NETS))
        sleep(max(0.0, period_s - (monotonic() - start - now)))
    return np.array(times), np.array(volts)


def to_nA(volts: np.ndarray, dark: np.ndarray) -> np.ndarray:
    '''Converts detector voltages to photocurrent in nA relative to the dark level.'''
    return (dark - volts) / TRANSIMPEDANCE * 1e9


def moving_mean(x: np.ndarray, window: int) -> np.ndarray:
    '''Moving average over `window` samples (shorter output, no edge padding).'''
    window = max(1, min(window, len(x)))
    return np.convolve(x, np.ones(window) / window, mode="valid")


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--laser-v", type=float, default=7.0, help="laser control voltage for steps 2-4 (0-10 V)")
    parser.add_argument("--quick", action="store_true", help="10 s dark, 10 s settle, 2 min stability")
    args = parser.parse_args()
    dark_s, settle_s, stability_s = (10, 10, 120) if args.quick else (30, 30, 600)
    if not 0.0 <= args.laser_v <= 10.0:
        raise ValueError(f"--laser-v must be in [0, 10] V, got {args.laser_v}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = Path(__file__).resolve().parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    out = runs_dir / f"check_setup_{stamp}"
    data: dict[str, np.ndarray] = {"pd_nets": np.array(PD_NETS), "laser_v": np.array(args.laser_v)}

    set_laser_range()

    with netlist:
        try:
            # --- Step 0: connections ---
            report(f"check_setup {stamp}, laser voltage {args.laser_v} V, "
                   f"settle {settle_s} s, stability {stability_s} s")
            boards = sorted(netlist.board_dict)
            report(f"[0] Boards in netlist: {boards}. {len(HEATER_NETS)} heater nets -> PASS")

            # --- Step 1: dark baseline ---
            set_lasers(0.0)
            for net in HEATER_NETS:
                netlist[net].vout(0.0)
            sleep(OFF_WAIT_S)
            _, v_dark = record(dark_s, 0.2)
            dark = v_dark.mean(axis=0)
            noise_nA = v_dark.std(axis=0) / TRANSIMPEDANCE * 1e9
            data.update(step1_V=v_dark, dark=dark, noise_nA=noise_nA)
            in_range = (dark > 0.05) & (dark < 1.25)  # NI range is 0-1.3 V
            ok1 = bool(in_range.all() and (noise_nA < 1.0).all())
            report(f"[1] Dark level {dark.min():.3f}-{dark.max():.3f} V, noise {noise_nA.min():.2f}-"
                   f"{noise_nA.max():.2f} nA -> {verdict(ok1)}")
            for net, d, n, ok in zip(PD_NETS, dark, noise_nA, in_range):
                if not ok or n >= 1.0:
                    report(f"    {net}: dark {d:.3f} V, noise {n:.2f} nA <- check this channel")

            # --- Step 2: one laser at a time ---
            I_alone = np.full((N, len(PD_NETS)), np.nan)
            laser_row = np.full(N, -1)
            for k in range(N):
                one_hot = np.zeros(N)
                one_hot[k] = args.laser_v
                set_lasers(one_hot)
                t, v = record(settle_s, 1.0)
                set_lasers(0.0)
                I = to_nA(v[-3:].mean(axis=0), dark)
                I_alone[k] = I
                data[f"step2_laser{k + 1}_t"], data[f"step2_laser{k + 1}_V"] = t, v
                # The median input is an unlit row, so it estimates the offset shared by all channels
                above = I[:N] - np.median(I[:N])
                lit = np.flatnonzero((above > LIT_NA) & (above >= LIT_FRACTION * above.max()))
                leaks = np.flatnonzero((above > LIT_NA) & (above < LIT_FRACTION * above.max()))
                if lit.size:
                    row = int(np.argmax(I[:N]))
                    laser_row[k] = row
                    series = to_nA(v[:, row], dark[row])
                    droop = 100 * (series[-1] / series[0] - 1) if series[0] > 0 else np.nan
                    detail = (f"brightest {INPUT_PDS[row]} {I[row]:.1f} nA, change during settle {droop:+.1f}%, "
                              f"outputs total {I[N:2 * N].sum():.1f} nA")
                    if leaks.size:
                        detail += ", leaks into " + ", ".join(f"{INPUT_PDS[i]} ({I[i] / I[row]:.0%})" for i in leaks)
                else:
                    detail = f"strongest input only {I[:N].max():.1f} nA"
                lit_names = ", ".join(INPUT_PDS[i] for i in lit) or "nothing"
                report(f"[2] {LASER_NETS[k]}: lights {lit_names}; {detail} -> {verdict(lit.size == 1)}")
                sleep(OFF_WAIT_S)
            data.update(step2_I_nA=I_alone, laser_row=laser_row)
            rows = laser_row[laser_row >= 0]
            if len(set(rows.tolist())) < len(rows):
                report("[2] WARNING: two lasers light the same input row")

            # --- Step 3: lasers added one at a time ---
            I_cumulative = np.full((N, len(PD_NETS)), np.nan)
            n_stable = 0  # most lasers on at once without any of them collapsing
            for n in range(1, N + 1):
                set_lasers([args.laser_v] * n + [0.0] * (N - n))
                _, v = record(settle_s, 1.0)
                I = to_nA(v[-3:].mean(axis=0), dark)
                I_cumulative[n - 1] = I
                # Neighbouring lasers leak onto each other's input detectors, so compare each
                # laser's row with the sum of what lasers 1..n gave there one at a time
                ratios = np.array([I[laser_row[j]] / I_alone[:n, laser_row[j]].sum() if laser_row[j] >= 0 else np.nan
                                   for j in range(n)])
                measured = ratios[~np.isnan(ratios)]
                result = verdict(bool(np.all(np.abs(measured - 1) <= STEP3_TOL))) if measured.size else "n/a"
                text = ", ".join("n/a" if np.isnan(r) else f"{r:.2f}" for r in ratios)
                collapse = " COLLAPSE" if np.nanmin(np.append(ratios, np.inf)) < 0.5 else ""
                if not collapse and n_stable == n - 1:
                    n_stable = n
                report(f"[3] lasers 1-{n} on: each row vs sum of single lasers = [{text}], "
                       f"inputs total {I[:N].sum():.0f} nA -> {result}{collapse}")
            data["step3_I_nA"] = I_cumulative

            # --- Step 4: stability with as many lasers on as step 3 allowed ---
            if n_stable < N:
                report(f"[4] Using lasers 1-{n_stable} only: adding laser {n_stable + 1} collapsed the others")
            set_lasers([args.laser_v] * n_stable + [0.0] * (N - n_stable))
            if n_stable < N:
                sleep(settle_s)
            t, v = record(stability_s, 1.0)
            data.update(step4_t=t, step4_V=v)
            data["step4_n_lasers"] = np.array(n_stable)
            watch = sorted({int(laser_row[j]) for j in range(n_stable) if laser_row[j] >= 0}) or list(range(N))
            for row in watch:
                I = moving_mean(to_nA(v[:, row], dark[row]), 10)
                final = I[-30:].mean()
                if final <= LIT_NA:
                    report(f"[4] {INPUT_PDS[row]}: only {final:.0f} nA with lasers on -> FAIL")
                    continue
                outside = np.flatnonzero(np.abs(I / final - 1) > DRIFT_TOL)
                settled_idx = outside[-1] + 1 if outside.size else 0
                warmup_s = t[min(settled_idx, len(t) - 1)]
                after = I[settled_idx:] if settled_idx < len(I) else I[-30:]
                drift = (after.max() - after.min()) / after.mean()
                five_s = moving_mean(to_nA(v[:, row], dark[row]), 5)[::5]
                jumps = int(np.sum(np.abs(np.diff(five_s)) / five_s[:-1] > JUMP_TOL))
                ok = drift < DRIFT_TOL and jumps == 0 and settled_idx < len(I)
                report(f"[4] {INPUT_PDS[row]}: {final:.0f} nA, settles within 2% after {warmup_s:.0f} s, "
                       f"drift after {100 * drift:.1f}%, jumps {jumps} -> {verdict(ok)}")
        except KeyboardInterrupt:
            report("Interrupted; saving what was measured.")
        finally:
            set_lasers(0.0)  # lasers off even if a step fails
            np.savez_compressed(out.with_suffix(".npz"), **data)
            out.with_suffix(".txt").write_text("\n".join(summary) + "\n")
            plot(data, out.with_suffix(".png"))
            report(f"Saved {out.name}.npz/.txt/.png in {runs_dir}")


def plot(data: dict[str, np.ndarray], path: Path) -> None:
    '''Saves a one-page overview of whichever steps completed.'''
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fig.suptitle(path.stem)
    x = np.arange(len(PD_NETS))

    if "noise_nA" in data:
        ax = axs[0, 0]
        ax.bar(x, data["noise_nA"])
        ax.set_xticks(x, PD_NETS, rotation=90)
        ax.set_ylabel("Noise (nA, std)")
        ax.set_title(f"[1] Dark noise; dark level {data['dark'].min():.3f}-{data['dark'].max():.3f} V")

    if "step2_I_nA" in data:
        ax = axs[0, 1]
        I = data["step2_I_nA"]
        im = ax.imshow(I, aspect="auto", cmap="viridis", vmin=0)
        ax.set_xticks(x, PD_NETS, rotation=90)
        ax.set_yticks(range(N), LASER_NETS)
        for (r, c), val in np.ndenumerate(I):
            ax.text(c, r, f"{val:.0f}", ha="center", va="center", fontsize=7, color="w")
        fig.colorbar(im, ax=ax, label="Photocurrent (nA)")
        ax.set_title("[2] One laser at a time (rows) vs detector (columns)")

    if "step3_I_nA" in data and "laser_row" in data:
        ax = axs[1, 0]
        I3, I2, rows = data["step3_I_nA"], data["step2_I_nA"], data["laser_row"]
        for j in range(N):
            if rows[j] < 0:
                continue
            n_on = np.arange(j + 1, N + 1)
            expected = np.array([I2[:n, rows[j]].sum() for n in n_on])
            ax.plot(n_on, I3[n_on - 1, rows[j]] / expected, "o-", label=f"{LASER_NETS[j]} ({INPUT_PDS[rows[j]]})")
        ax.axhspan(1 - STEP3_TOL, 1 + STEP3_TOL, color="0.9")
        ax.set_xlabel("Number of lasers on")
        ax.set_ylabel("Measured / sum of single lasers")
        ax.set_title("[3] Does adding lasers dim the others?")
        ax.legend(fontsize=8)

    if "step4_V" in data:
        ax = axs[1, 1]
        t, v, dark = data["step4_t"], data["step4_V"], data["dark"]
        for row in range(N):
            I = to_nA(v[:, row], dark[row])
            if I[-30:].mean() > LIT_NA:
                ax.plot(t / 60, I / I[-30:].mean(), lw=1, label=INPUT_PDS[row])
        ax.axhspan(1 - DRIFT_TOL, 1 + DRIFT_TOL, color="0.9")
        ax.set_xlabel("Time with lasers on (min)")
        ax.set_ylabel("Input power / final value")
        ax.set_title(f"[4] Laser stability, lasers 1-{int(data.get('step4_n_lasers', N))} on")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)

    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
