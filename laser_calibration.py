'''Laser calibration for spiking: power curve and pulse test, one laser at a time.

    A. Power curve: each laser is stepped from 0 to 10 V in short pulses (low duty
       cycle, so it does not heat up) while its own input detector PD_k_0 is read.
       Gives each laser's threshold, slope, off level, and the voltage at which
       all six lasers reach the same power.
    B. Pulse test: each laser is pulsed between --v-off and --v-on at several
       rates. Spike heights must not depend on how often the laser fires
       (heating), and the level between pulses must return to the off level.

    Laser k feeds row k (measured with check_setup.py on 2026-10-19).
    Results go to runs/laser_calibration_<timestamp>.{npz,png,txt}.

    Usage (from the repo root on the lab PC):
        python laser_calibration.py                        # about 7 min
        python laser_calibration.py --v-on 10 --v-off 7    # pulse from a bias level instead of 0 V
'''
import argparse
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep

import check_setup as cs
from check_setup import INPUT_PDS, LASER_NETS, N, PD_NETS, netlist, record, report, set_lasers, to_nA, verdict

import matplotlib.pyplot as plt  # after check_setup, which selects the file-only backend
import numpy as np

# Coarse below 6 V, fine where the lasers turn on (between 7 and 10 V on 2026-10-19)
CURVE_V = np.round(np.concatenate([np.arange(0.0, 6.0, 1.0), np.arange(6.0, 10.01, 0.25)]), 2)
CURVE_ON_S = 0.3        # each curve point: laser on this long, then read...
CURVE_OFF_S = 1.2       # ...then off this long (20% duty cycle)
PULSE_ON_S = 0.05       # requested pulse length; lengthened if one detector read takes longer
PULSE_PERIODS_S = [2.0, 0.5, 0.2, 0.1]  # isolated, 2 Hz, 5 Hz, 10 Hz
TRAIN_S = 5.0           # length of each pulse train
REST_S = 5.0            # laser at v_off between trains
PULSE_TOL = 0.05        # pulse heights within 5% (vs isolated, within a train, pulse to pulse)
HEADROOM_V = 0.1        # warn when a detector voltage gets this close to 0 V (end of its range)


def set_one(k: int, volts: float) -> None:
    '''Sets laser k (0-based) to volts, leaving the others unchanged.'''
    netlist[LASER_NETS[k]].vout(float(volts))


def power_curves(dark: np.ndarray) -> np.ndarray:
    '''Steps each laser alone through CURVE_V in short pulses.

        Returns photocurrent (nA) of shape (N lasers, len(CURVE_V), len(PD_NETS)).
    '''
    volts = np.full((N, len(CURVE_V), len(PD_NETS)), np.nan)
    for k in range(N):
        for i, v in enumerate(CURVE_V):
            set_one(k, v)
            sleep(CURVE_ON_S)
            volts[k, i] = netlist.group_vin(PD_NETS)
            set_one(k, 0.0)
            sleep(CURVE_OFF_S)
    if np.nanmin(volts) < HEADROOM_V:
        report(f"[A] WARNING: a detector reached {np.nanmin(volts):.2f} V, near the end of its range")
    return to_nA(volts, dark)


def analyze_curve(own: np.ndarray, noise_nA: float) -> dict[str, float]:
    '''Threshold, slope and on/off ratio from one laser's curve on its own row (nA vs CURVE_V).'''
    top = own[-1]
    above = own >= 0.3 * top  # the roughly linear part above threshold
    if top > cs.LIT_NA and above.sum() >= 2:
        slope, intercept = np.polyfit(CURVE_V[above], own[above], 1)
        threshold = -intercept / slope
    else:
        slope = threshold = np.nan
    on_off_dB = 10 * np.log10(top / max(own[0], noise_nA)) if top > 0 else np.nan
    return dict(off=own[0], at7=float(np.interp(7.0, CURVE_V, own)), top=top,
                slope=slope, threshold=threshold, on_off_dB=on_off_dB)


def measure_read_time() -> float:
    '''Median time of one group read of every detector, in seconds.'''
    times = []
    for _ in range(7):
        start = monotonic()
        netlist.group_vin(PD_NETS)
        times.append(monotonic() - start)
    return float(np.median(times))


def pulse_train(k: int, v_on: float, v_off: float, period_s: float, on_s: float,
                read_s: float) -> tuple[np.ndarray, np.ndarray]:
    '''Pulses laser k for TRAIN_S seconds, reading its own row at the end of each
        pulse and just before the next one.

        Returns (detector voltage at each pulse top, detector voltage before each pulse).
    '''
    tops, bases = [], []
    start = monotonic()
    for p in range(max(1, int(TRAIN_S / period_s))):
        t_pulse = start + p * period_s
        sleep(max(0.0, t_pulse - read_s - monotonic()))
        bases.append(netlist.group_vin(PD_NETS)[k])
        sleep(max(0.0, t_pulse - monotonic()))
        set_one(k, v_on)
        sleep(max(0.0, on_s - read_s))  # the read fills the end of the pulse
        tops.append(netlist.group_vin(PD_NETS)[k])
        set_one(k, v_off)
    return np.array(tops), np.array(bases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v-on", type=float, default=10.0, help="pulse-test on level (V)")
    parser.add_argument("--v-off", type=float, default=0.0, help="pulse-test off/bias level (V)")
    parser.add_argument("--skip-curve", action="store_true", help="only run the pulse test")
    args = parser.parse_args()
    if not 0.0 <= args.v_off < args.v_on <= 10.0:
        raise ValueError(f"Need 0 <= v_off < v_on <= 10 V, got {args.v_off}, {args.v_on}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = Path(__file__).resolve().parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    out = runs_dir / f"laser_calibration_{stamp}"
    data: dict[str, np.ndarray] = {"pd_nets": np.array(PD_NETS), "curve_v": CURVE_V,
                                   "v_on": np.array(args.v_on), "v_off": np.array(args.v_off)}

    cs.set_laser_range()
    with netlist:
        try:
            report(f"laser_calibration {stamp}, pulse test {args.v_off} -> {args.v_on} V")
            set_lasers(0.0)
            for net in cs.HEATER_NETS:
                netlist[net].vout(0.0)
            sleep(cs.OFF_WAIT_S)
            _, v_dark = record(5.0, 0.2)
            dark = v_dark.mean(axis=0)
            noise_nA = v_dark.std(axis=0) / cs.TRANSIMPEDANCE * 1e9
            data.update(dark=dark, noise_nA=noise_nA)

            # --- A: power curves ---
            if not args.skip_curve:
                I = power_curves(dark)
                data["curve_I_nA"] = I
                fits = [analyze_curve(I[k, :, k], noise_nA[k]) for k in range(N)]
                tops = np.array([f["top"] for f in fits])
                target = tops[tops > cs.LIT_NA].min() if np.any(tops > cs.LIT_NA) else np.nan
                v_equal = np.full(N, np.nan)
                for k, f in enumerate(fits):
                    own = np.maximum.accumulate(I[k, :, k])  # monotonic for interpolation
                    if f["top"] > cs.LIT_NA:
                        v_equal[k] = np.interp(target, own, CURVE_V)
                    report(f"[A] {LASER_NETS[k]} ({INPUT_PDS[k]}): 0 V {f['off']:.1f} nA, 7 V {f['at7']:.0f} nA, "
                           f"10 V {f['top']:.0f} nA; threshold ~{f['threshold']:.1f} V, "
                           f"slope above it {f['slope']:.0f} nA/V; on/off (0 vs 10 V) {f['on_off_dB']:.0f} dB; "
                           f"voltage for equal power ({target:.0f} nA) {v_equal[k]:.2f} V")
                data.update(threshold_V=np.array([f["threshold"] for f in fits]),
                            slope_nA_per_V=np.array([f["slope"] for f in fits]), v_equal=v_equal)

            # --- B: pulse test ---
            read_s = measure_read_time()
            on_s = max(PULSE_ON_S, 1.5 * read_s)
            periods = [p for p in PULSE_PERIODS_S if p >= on_s + 2 * read_s]
            report(f"[B] One detector read takes {1e3 * read_s:.0f} ms; pulses are {1e3 * on_s:.0f} ms; "
                   f"periods {periods} s")
            heights = np.full((N, len(periods), int(TRAIN_S / min(periods)) + 1), np.nan)
            base_rise = np.full((N, len(periods)), np.nan)
            for k in range(N):
                set_one(k, args.v_off)
                sleep(REST_S)
                rel, burst, spread = [], [], []
                for j, period in enumerate(periods):
                    _, v_rest = record(1.0, 0.2)
                    rest = to_nA(v_rest[:, k].mean(), dark[k])
                    tops, bases = pulse_train(k, args.v_on, args.v_off, period, on_s, read_s)
                    h = to_nA(tops, dark[k]) - to_nA(bases, dark[k])
                    heights[k, j, :len(h)] = h
                    base_rise[k, j] = to_nA(bases, dark[k]).mean() - rest
                    rel.append(h.mean() / np.nanmean(heights[k, 0]))
                    burst.append(h[-3:].mean() / h[:3].mean())
                    spread.append(h.std() / h.mean())
                    sleep(REST_S)
                set_one(k, 0.0)
                ok = all(abs(r - 1) <= PULSE_TOL for r in rel + burst) and max(spread) <= PULSE_TOL
                rates = ", ".join(f"{1 / p:g} Hz {r:.2f}" for p, r in zip(periods[1:], rel[1:]))
                report(f"[B] {LASER_NETS[k]}: isolated pulse {np.nanmean(heights[k, 0]):.0f} nA; vs isolated: {rates}; "
                       f"last/first in train {min(burst):.2f}-{max(burst):.2f}; pulse-to-pulse spread up to "
                       f"{100 * max(spread):.1f}%; level between pulses rises up to {np.nanmax(base_rise[k]):.1f} nA "
                       f"-> {verdict(ok)}")
            data.update(pulse_periods_s=np.array(periods), pulse_on_s=np.array(on_s),
                        pulse_heights_nA=heights, pulse_base_rise_nA=base_rise)
        except KeyboardInterrupt:
            report("Interrupted; saving what was measured.")
        finally:
            set_lasers(0.0)
            np.savez_compressed(out.with_suffix(".npz"), **data)
            out.with_suffix(".txt").write_text("\n".join(cs.summary) + "\n")
            plot(data, out.with_suffix(".png"))
            report(f"Saved {out.name}.npz/.txt/.png in {runs_dir}")


def plot(data: dict[str, np.ndarray], path: Path) -> None:
    '''Saves a one-page overview of whichever parts completed.'''
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fig.suptitle(path.stem)
    if "curve_I_nA" in data:
        I = data["curve_I_nA"]
        for k in range(N):
            for ax in axs[0]:
                ax.plot(data["curve_v"], I[k, :, k], "o-", ms=3, label=f"{LASER_NETS[k]} ({INPUT_PDS[k]})")
        axs[0, 0].set_title("[A] Power curve (own input detector)")
        axs[0, 1].set_yscale("log")
        axs[0, 1].set_ylim(bottom=0.1)
        axs[0, 1].set_title("[A] Power curve, log scale (threshold and off level)")
        for ax in axs[0]:
            ax.set_xlabel("Laser control voltage (V)")
            ax.set_ylabel("Photocurrent (nA)")
            ax.legend(fontsize=8)
    if "pulse_heights_nA" in data:
        h, periods = data["pulse_heights_nA"], data["pulse_periods_s"]
        ax = axs[1, 0]
        for k in range(N):
            iso = np.nanmean(h[k, 0])
            ax.plot(h[k, -1] / iso, ".-", label=LASER_NETS[k])
        ax.axhspan(1 - PULSE_TOL, 1 + PULSE_TOL, color="0.9")
        ax.set_xlabel("Pulse number")
        ax.set_ylabel("Height / isolated pulse height")
        ax.set_title(f"[B] Pulse heights at the fastest rate ({1 / periods[-1]:g} Hz)")
        ax.legend(fontsize=8)
        ax = axs[1, 1]
        width = 0.8 / N
        for k in range(N):
            rel = np.nanmean(h[k], axis=1) / np.nanmean(h[k, 0])
            ax.bar(np.arange(len(periods)) + k * width, rel, width, label=LASER_NETS[k])
        ax.set_xticks(np.arange(len(periods)) + 0.4 - width / 2, [f"{1 / p:g} Hz" for p in periods])
        ax.set_ylim(0.8, 1.1)
        ax.axhspan(1 - PULSE_TOL, 1 + PULSE_TOL, color="0.9", zorder=0)
        ax.set_ylabel("Mean height / isolated")
        ax.set_title("[B] Does pulse height depend on the spike rate?")
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
