'''Step 6: heater calibration, one MZI at a time (voltage -> phase -> BAR/CROSS/50:50).

    Each MZI "r_c" couples rows r and r+1 in column c; its "_i" heater sets the
    split between them. The heater phase grows with heater power, so
        theta(V) = theta0 + alpha * V**2
    and the light reaching a detector downstream follows
        P(V) = offset + amp * cos(theta(V)).
    For every MZI the script lights one laser whose light passes through it,
    sweeps the heater (points spaced evenly in V**2, i.e. in phase), divides every
    detector by that laser's input detector (removes laser drift) and fits P(V).

    Which laser and detectors to use comes from the mesh at 0 V being close to
    all-CROSS (check_setup at 10 V, 2026-10-19): light zig-zags one row per column.
    The "cross" detector is where the laser's light ends up when this MZI is CROSS,
    the "bar" detector where it ends up when this MZI alone is BAR.

    Results go to runs/heater_sweep_<timestamp>.{npz,png,txt,json}; the .json
    holds V_cross / V_bar / V_50 per MZI for programming routes later.

    Usage (from the repo root on the lab PC):
        python heater_sweep.py                       # all 13 MZIs, about 4 min
        python heater_sweep.py --mzi 1_1 5_1         # only these MZIs
        python heater_sweep.py --v-max 5             # wider sweep, if the heaters allow it
'''
import argparse
import json
from datetime import datetime
from pathlib import Path
from time import sleep

import check_setup as cs
from check_setup import INPUT_PDS, LASER_NETS, N, PD_NETS, netlist, record, report, set_lasers, to_nA, verdict

import matplotlib.pyplot as plt
import numpy as np

# MZI "r_c" couples rows r and r+1 in column c (5 columns, 13 MZIs)
MZIS = ["1_1", "3_1", "5_1", "2_2", "4_2", "1_3", "3_3", "5_3", "2_4", "4_4", "1_5", "3_5", "5_5"]
COLUMNS = 5
N_POINTS = 41
WARMUP_S = 20.0      # laser settle time after switching on (check_setup: up to ~17 s)
RETURN_S = 1.0       # heater back at 0 V for this long before the next MZI
MIN_R2 = 0.9         # fit quality needed to trust the calibration
MIN_SWING = 0.05     # peak-to-peak change on the cross detector, as a fraction of the input


def path(row: int, bar: frozenset[str] = frozenset()) -> list[int]:
    '''Rows (1-based) the light from input `row` is on after each column, with every
        MZI CROSS except those named in `bar`.
    '''
    rows = []
    for c in range(1, COLUMNS + 1):
        for top in (row, row - 1):  # the MZI containing this row has it as top or bottom
            name = f"{top}_{c}"
            if name in MZIS:
                if name not in bar:
                    row = top + 1 if row == top else top
                break
        rows.append(row)
    return rows


def plan(mzi: str) -> tuple[int, int, int]:
    '''Picks a laser (0-based) whose all-CROSS path enters this MZI, and the rows
        (1-based) where its light leaves the mesh with the MZI CROSS and BAR.
    '''
    top, col = (int(x) for x in mzi.split("_"))
    for k in range(N):
        entering = k + 1 if col == 1 else path(k + 1)[col - 2]
        if entering in (top, top + 1):
            return k, path(k + 1)[-1], path(k + 1, frozenset({mzi}))[-1]
    raise ValueError(f"No laser reaches MZI {mzi}")


def fit_cos_v2(v: np.ndarray, y: np.ndarray, v_max: float) -> dict[str, float]:
    '''Fits y = offset + amp*cos(alpha*V^2 + theta0) with amp >= 0.

        alpha is found by grid search (0.1 to 6 periods across the sweep); for each
        alpha the rest is a linear least-squares fit.
    '''
    x = v ** 2
    best = (np.inf, np.nan, np.zeros(3))
    for alpha in np.linspace(0.2, 12, 600) * np.pi / v_max ** 2:
        X = np.column_stack([np.ones_like(x), np.cos(alpha * x), np.sin(alpha * x)])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        residual = np.sum((X @ coef - y) ** 2)
        if residual < best[0]:
            best = (residual, alpha, coef)
    residual, alpha, (offset, c, s) = best
    total = np.sum((y - y.mean()) ** 2)
    return dict(alpha=alpha, theta0=float(np.arctan2(-s, c)), offset=offset, amp=float(np.hypot(c, s)),
                r2=1 - residual / total if total > 0 else 0.0)


def voltage_for_phase(fit: dict[str, float], target: float, v_max: float) -> tuple[float, float]:
    '''Lowest V in [0, v_max] where alpha*V^2 + theta0 = target (mod 2 pi). If no V in
        range reaches it exactly, the V that comes closest.

        Returns (voltage, remaining phase error in rad; 0 when reached exactly).
    '''
    m = np.arange(-3, 20)
    v2 = (target + 2 * np.pi * m - fit["theta0"]) / fit["alpha"]
    v = np.sqrt(v2[(v2 >= 0) & (v2 <= v_max ** 2)])
    if v.size:
        return float(v.min()), 0.0
    fine = np.linspace(0, v_max, 2001)
    error = np.angle(np.exp(1j * (fit["alpha"] * fine ** 2 + fit["theta0"] - target)))
    best = int(np.argmin(np.abs(error)))
    return float(fine[best]), float(abs(error[best]))


def fmt_v(v: float, error: float) -> str:
    '''Voltage text; "~" marks a closest-reachable voltage, with how far off it is.'''
    return f"{v:.2f} V" if error == 0 else f"~{v:.2f} V ({np.degrees(error):.0f} deg short)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mzi", nargs="+", default=MZIS, choices=MZIS, help="MZIs to sweep (default: all)")
    parser.add_argument("--v-max", type=float, default=4.2, help="highest heater voltage (4.2 V used on this chip so far)")
    parser.add_argument("--laser-v", type=float, default=10.0, help="laser control voltage (lasers are dim below ~8 V)")
    parser.add_argument("--ps-delay", type=float, default=0.05, help="heater settle time per step (s)")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = Path(__file__).resolve().parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    out = runs_dir / f"heater_sweep_{stamp}"
    volts = args.v_max * np.sqrt(np.linspace(0, 1, N_POINTS))  # evenly spaced in phase
    data: dict[str, np.ndarray] = {"pd_nets": np.array(PD_NETS), "heater_v": volts, "mzis": np.array(args.mzi)}
    calibration: dict[str, dict] = {}

    # Group MZIs by laser so each laser warms up once
    plans = {mzi: plan(mzi) for mzi in args.mzi}
    order = sorted(args.mzi, key=lambda m: plans[m][0])

    cs.set_laser_range()
    with netlist:
        try:
            report(f"heater_sweep {stamp}, heaters 0-{args.v_max} V in {N_POINTS} steps, laser {args.laser_v} V, "
                   f"settle {1e3 * args.ps_delay:.0f} ms per step")
            set_lasers(0.0)
            for net in cs.HEATER_NETS:
                netlist[net].vout(0.0)
            sleep(cs.OFF_WAIT_S)
            _, v_dark = record(5.0, 0.2)
            dark = v_dark.mean(axis=0)
            data["dark"] = dark

            current_laser = None
            for mzi in order:
                k, cross_row, bar_row = plans[mzi]
                if k != current_laser:
                    set_lasers(0.0)
                    netlist[LASER_NETS[k]].vout(args.laser_v)
                    sleep(WARMUP_S)
                    current_laser = k
                heater = f"{mzi}_i"
                v = np.full((N_POINTS, len(PD_NETS)), np.nan)
                for i, hv in enumerate(volts):
                    netlist[heater].vout(float(hv))
                    sleep(args.ps_delay)
                    v[i] = netlist.group_vin(PD_NETS)
                netlist[heater].vout(0.0)
                sleep(RETURN_S)
                back = netlist.group_vin(PD_NETS)  # back at 0 V: does the MZI return to its start?

                I = to_nA(v, dark)
                T = I / I[:, [k]]  # every detector relative to this laser's input
                T_back = to_nA(back, dark) / to_nA(back[k], dark[k])
                data[f"{mzi}_V"] = v
                cross_pd, bar_pd = PD_NETS.index(f"PD_{cross_row}_5"), PD_NETS.index(f"PD_{bar_row}_5")
                fit = fit_cos_v2(volts, T[:, cross_pd], args.v_max)
                swing = T[:, cross_pd].max() - T[:, cross_pd].min()
                span = fit["alpha"] * args.v_max ** 2 / np.pi  # phase covered, in units of pi
                v_cross, err_cross = voltage_for_phase(fit, 0.0, args.v_max)  # cross detector at its maximum
                v_bar, err_bar = voltage_for_phase(fit, np.pi, args.v_max)    # ...at its minimum
                v_50, err_50 = min(voltage_for_phase(fit, np.pi / 2, args.v_max),
                                   voltage_for_phase(fit, 3 * np.pi / 2, args.v_max), key=lambda x: (x[1], x[0]))
                hi, lo = fit["offset"] + fit["amp"], fit["offset"] - fit["amp"]
                floor = 1e-3 * hi  # cap the extinction estimate at 30 dB
                extinction_dB = 10 * np.log10(hi / max(lo, floor)) if hi > 0 else np.nan
                drift = T_back[cross_pd] / T[0, cross_pd] - 1
                ok = fit["r2"] >= MIN_R2 and swing >= MIN_SWING
                calibration[mzi] = dict(heater=heater, laser=LASER_NETS[k], cross_pd=PD_NETS[cross_pd],
                                        bar_pd=PD_NETS[bar_pd], alpha_rad_per_V2=fit["alpha"], theta0_rad=fit["theta0"],
                                        V_pi=float(np.sqrt(np.pi / fit["alpha"])), V_cross=v_cross, V_bar=v_bar,
                                        V_50=v_50, V_cross_error_rad=err_cross, V_bar_error_rad=err_bar,
                                        V_50_error_rad=err_50, extinction_dB=extinction_dB, r2=fit["r2"], ok=bool(ok))
                report(f"[6] {mzi} ({LASER_NETS[k]}; CROSS->{PD_NETS[cross_pd]}, BAR->{PD_NETS[bar_pd]}): "
                       f"swing {100 * swing:.0f}% of input, fit R2 {fit['r2']:.2f}, phase span {span:.1f} pi, "
                       f"V_pi {calibration[mzi]['V_pi']:.2f} V; CROSS {fmt_v(v_cross, err_cross)}, "
                       f"50:50 {fmt_v(v_50, err_50)}, BAR {fmt_v(v_bar, err_bar)}; extinction {extinction_dB:.0f} dB; "
                       f"back at 0 V {100 * drift:+.1f}% "
                       f"-> {verdict(ok)}")
        except KeyboardInterrupt:
            report("Interrupted; saving what was measured.")
        finally:
            set_lasers(0.0)
            for net in cs.HEATER_NETS:
                netlist[net].vout(0.0)
            np.savez_compressed(out.with_suffix(".npz"), **data)
            out.with_suffix(".json").write_text(json.dumps(calibration, indent=2))
            out.with_suffix(".txt").write_text("\n".join(cs.summary) + "\n")
            plot(data, calibration, out.with_suffix(".png"))
            report(f"Saved {out.name}.npz/.json/.txt/.png in {runs_dir}")


def plot(data: dict[str, np.ndarray], calibration: dict[str, dict], path_: Path) -> None:
    '''One panel per swept MZI: cross and bar detectors (relative to the input) vs heater voltage.'''
    mzis = [m for m in data["mzis"] if f"{m}_V" in data]
    if not mzis:
        return
    cols = 4
    rows = int(np.ceil(len(mzis) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), layout="constrained", squeeze=False)
    fig.suptitle(path_.stem)
    volts, dark = data["heater_v"], data["dark"]
    v_max = volts.max()
    for ax, mzi in zip(axs.flat, mzis):
        cal = calibration[mzi]
        k = LASER_NETS.index(cal["laser"])
        I = to_nA(data[f"{mzi}_V"], dark)
        T = I / I[:, [k]]
        for name, style in ((cal["cross_pd"], "o"), (cal["bar_pd"], "s")):
            ax.plot(volts, T[:, PD_NETS.index(name)], style, ms=3, label=name)
        fine = np.linspace(0, v_max, 300)
        a, t0 = cal["alpha_rad_per_V2"], cal["theta0_rad"]
        fit = fit_cos_v2(volts, T[:, PD_NETS.index(cal["cross_pd"])], v_max)
        ax.plot(fine, fit["offset"] + fit["amp"] * np.cos(a * fine ** 2 + t0), "-", lw=1, color="C0")
        for key, color in (("V_cross", "C2"), ("V_50", "C1"), ("V_bar", "C3")):
            if np.isfinite(cal[key]):
                ax.axvline(cal[key], color=color, ls=":", lw=1)
        ax.set_title(f"{mzi} ({cal['laser']}) R2 {cal['r2']:.2f}", fontsize=9)
        ax.set_xlabel("Heater voltage (V)")
        ax.set_ylabel("Detector / input")
        ax.legend(fontsize=7)
    for ax in list(axs.flat)[len(mzis):]:
        ax.set_visible(False)
    fig.savefig(path_, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
