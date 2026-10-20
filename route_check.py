'''Program routing settings on the chip and measure where each laser's light goes.

    For each setting in sim/routes_rows2-6.json (A-D: an input core multicasts to C3+C4
    while a hidden core sends to C5):
      1. Set every heater: routing MZIs to BAR / CROSS / 50:50, edge MZIs (1_1, 1_3)
         to BAR, 1_5 left at 0 V (faulty). Voltages come from the heater_sweep fits.
      2. Dark reading with all lasers off (heaters on: they can shift detector offsets).
      3. Each core's laser alone: every detector divided by that laser's input detector
         gives its transmission row T[sender, detector].
      4. The setting's active lasers together: checks that their light simply adds (after
         scaling for any change in each laser's power, which is reported separately).

    Reported per setting:
      - signal-to-crosstalk at each receiver: light from its intended sender vs from the
        other active sender, at equal input power, on the same detector (detector gains
        cancel). Target: 15-20 dB or more.
      - multicast split C3:C4 (includes any gain difference between the two detectors)
      - on-target share: fraction of a sender's output-detector light on its receivers
      - row 1 leak: light on the row-1 detectors PD_1_2, PD_1_3 (should stay dark)

    Results go to runs/route_check_<timestamp>.{npz,png,txt,json}.

    Usage (from the repo root on the lab PC):
        python route_check.py                  # settings A-D, about 4 min
        python route_check.py --settings A B   # only these
'''
import argparse
import json
from datetime import datetime
from pathlib import Path
from time import sleep

import check_setup as cs
from check_setup import LASER_NETS, PD_NETS, netlist, record, report, set_lasers, to_nA, verdict
from heater_sweep import voltage_for_phase

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_PDS = [f"PD_{r}_5" for r in range(1, 7)]
ROW1_PDS = ["PD_1_2", "PD_1_3"]
HEATER_SETTLE_S = 1.0     # after setting all heaters (thermal crosstalk settles too)
LASER_SETTLE_S = 3.0      # after switching a laser on
READ_S = 1.0              # averaging time per reading
SXR_TARGET_DB = 15.0
SUM_TOL = 0.05            # both lasers on vs sum of singles
PHASE = {"X": [0.0], "B": [np.pi], "S": [np.pi / 2, 3 * np.pi / 2]}   # cross-detector phase per state


def load_calibration(paths: list[Path]) -> dict[str, dict]:
    '''Merges heater_sweep .json files (later files win) into one fit per MZI.'''
    cal = {}
    for path in sorted(paths):
        for mzi, fit in json.loads(path.read_text()).items():
            if fit.get("ok", True) or mzi not in cal:
                cal[mzi] = dict(fit, source=path.name)
    return cal


def heater_voltage(fit: dict, state: str, v_max: float) -> tuple[float, float]:
    '''Voltage for BAR/CROSS/SPLIT from a heater_sweep fit, and the phase it misses by (rad).'''
    f = {"alpha": fit["alpha_rad_per_V2"], "theta0": fit["theta0_rad"]}
    return min((voltage_for_phase(f, target, v_max) for target in PHASE[state]), key=lambda x: (x[1], x[0]))


def laser_voltages() -> dict[str, float]:
    '''Equal-power laser voltages from the latest laser_calibration run (10 V if missing).'''
    runs = sorted(ROOT.glob("runs/laser_calibration_*.npz"))
    if runs:
        with np.load(runs[-1]) as d:
            if "v_equal" in d:
                return {net: float(v) if np.isfinite(v) else 10.0 for net, v in zip(LASER_NETS, d["v_equal"])}
    return {net: 10.0 for net in LASER_NETS}


def read_nA(dark: np.ndarray) -> np.ndarray:
    '''Average photocurrent (nA) of every detector over READ_S.'''
    _, v = record(READ_S, 0.1)
    return to_nA(v.mean(axis=0), dark)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--routes", type=Path, default=ROOT / "sim" / "routes_rows2-6.json")
    parser.add_argument("--cal", type=Path, nargs="+", default=None,
                        help="heater_sweep .json files (default: all in runs/, later ones win)")
    parser.add_argument("--settings", nargs="+", default=None, help="settings to run (default: all)")
    parser.add_argument("--v-max", type=float, default=4.2, help="heater voltage clamp")
    parser.add_argument("--laser-v", type=float, default=None, help="one laser voltage for all (default: equal-power voltages)")
    args = parser.parse_args()

    routes = json.loads(args.routes.read_text())
    rows, mzis = routes["core_row"], routes["mzis"]
    lasers, inputs, receivers = routes["lasers"], routes["input_pd"], routes["receiver_pd"]
    names = [f"C{k + 1}" for k in range(len(rows))]
    settings = {k: v for k, v in routes["settings"].items() if args.settings is None or k in args.settings}
    cal = load_calibration(args.cal or list(ROOT.glob("runs/heater_sweep_*.json")))
    missing = [m for m in mzis + routes["fixed_bar"] if m not in cal]
    if missing:
        raise SystemExit(f"No heater calibration for {missing}; run heater_sweep.py for them first")
    laser_v = {net: args.laser_v for net in LASER_NETS} if args.laser_v else laser_voltages()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "runs" / f"route_check_{stamp}"
    out.parent.mkdir(exist_ok=True)
    data: dict[str, np.ndarray] = {"pd_nets": np.array(PD_NETS), "cores": np.array(names)}
    results: dict[str, dict] = {}

    cs.set_laser_range()
    with netlist:
        try:
            report(f"route_check {stamp}: cores {names} on rows {rows}, receivers {receivers}; "
                   f"lasers {', '.join(f'{n} {laser_v[n]:.2f} V' for n in lasers)}")
            for name, setting in settings.items():
                states = setting["states"]
                plan = {m: ("B", *heater_voltage(cal[m], "B", args.v_max)) for m in routes["fixed_bar"]}
                plan.update({m: (s, *heater_voltage(cal[m], s, args.v_max)) for m, s in zip(mzis, states)})
                set_lasers(0.0)
                for net in cs.HEATER_NETS:
                    netlist[net].vout(0.0)
                for m, (s, v, _) in plan.items():
                    netlist[f"{m}_i"].vout(float(np.clip(v, 0.0, args.v_max)))
                sleep(HEATER_SETTLE_S)
                active = {int(k): d for k, d in setting["routes"].items()}
                desc = "; ".join(f"{names[s]} -> {'+'.join(names[x] for x in d)}" for s, d in active.items())
                short = [f"{m} {s}" + (f" ({np.degrees(e):.0f} deg short)" if e > 0 else "")
                         for m, (s, v, e) in plan.items() if s != "B"]
                report(f"[{name}] {desc} | not BAR: {', '.join(short)}")

                _, v_dark = record(2.0, 0.2)
                dark = v_dark.mean(axis=0)
                noise = v_dark.std(axis=0) / cs.TRANSIMPEDANCE * 1e9          # nA
                I = np.zeros((len(rows), len(PD_NETS)))           # one laser at a time
                for k, net in enumerate(lasers):
                    netlist[net].vout(laser_v[net])
                    sleep(LASER_SETTLE_S)
                    I[k] = read_nA(dark)
                    netlist[net].vout(0.0)
                    sleep(1.0)
                for s in active:                                   # active lasers together
                    netlist[lasers[s]].vout(laser_v[lasers[s]])
                sleep(LASER_SETTLE_S)
                I_all = read_nA(dark)
                set_lasers(0.0)

                inp = np.array([I[k, PD_NETS.index(p)] for k, p in enumerate(inputs)])
                T = I / inp[:, None]                               # transmission relative to each input
                rx = [PD_NETS.index(p) for p in receivers]
                res = {"heaters": {m: {"state": s, "V": v, "phase_short_deg": np.degrees(e)} for m, (s, v, e) in plan.items()}}
                # signal-to-crosstalk at each intended receiver, at equal input power; crosstalk
                # below 3x the detector noise only gives a lower bound
                sxr, bound = {}, {}
                for s, dests in active.items():
                    others = [o for o in active if o != s]
                    for dst in dests:
                        sig, xt = T[s, rx[dst]], sum(T[o, rx[dst]] for o in others)
                        floor = 3 * noise[rx[dst]] / np.mean([inp[o] for o in others]) if others else 0
                        bound[names[dst]] = xt < floor
                        sxr[names[dst]] = 10 * np.log10(sig / max(xt, floor)) if max(xt, floor) > 0 else np.inf
                # multicast split and on-target share per sender
                outs = [PD_NETS.index(p) for p in OUTPUT_PDS]
                lines = []
                for s, dests in active.items():
                    got = T[s, [rx[x] for x in dests]]
                    share = got.sum() / T[s, outs].sum()
                    part = f"{names[s]}: on-target {100 * share:.0f}%"
                    if len(dests) == 2:
                        part += f", split {names[dests[0]]}:{names[dests[1]]} = {got[0] / got[1]:.2f} ({10 * np.log10(got[0] / got[1]):+.1f} dB)"
                    lines.append(part)
                    res[names[s]] = {"on_target": share, "T_receivers": dict(zip([names[x] for x in dests], got.tolist()))}
                row1 = max(T[k, PD_NETS.index(p)] / T[k, outs].sum() for k in range(len(rows)) for p in ROW1_PDS)
                # lasers together: scale each single-laser row to that laser's input power while
                # together (laser powers shift when both are on), then the light should simply add
                p_in = [PD_NETS.index(inputs[a]) for a in active]
                expected = sum(T[a] * I_all[i] for a, i in zip(active, p_in))
                sum_err = max(abs(I_all[r] / expected[r] - 1) for r in rx if expected[r] > cs.LIT_NA)
                power_shift = {names[a]: I_all[i] / I[a, i] - 1 for a, i in zip(active, p_in)}
                ok = min(sxr.values()) >= SXR_TARGET_DB and sum_err <= SUM_TOL
                report(f"[{name}]   {' | '.join(lines)}")
                report(f"[{name}]   signal/crosstalk: " + ", ".join(f"{k} {'>=' if bound[k] else ''}{v:.1f} dB" for k, v in sxr.items()) +
                       f"; row-1 leak up to {100 * row1:.1f}%; lasers together vs sum {100 * sum_err:.1f}% -> {verdict(ok)}")
                report(f"[{name}]   laser power with both on vs alone: " +
                       ", ".join(f"{k} {100 * v:+.0f}%" for k, v in power_shift.items()))
                res.update(sxr_dB=sxr, sxr_is_lower_bound=bound, row1_leak=row1, sum_error=sum_err,
                           laser_power_shift=power_shift, ok=bool(ok))
                results[name] = res
                data[f"{name}_I_nA"], data[f"{name}_I_all_nA"], data[f"{name}_dark"] = I, I_all, dark
        except KeyboardInterrupt:
            report("Interrupted; saving what was measured.")
        finally:
            set_lasers(0.0)
            for net in cs.HEATER_NETS:
                netlist[net].vout(0.0)
            np.savez_compressed(out.with_suffix(".npz"), **data)
            out.with_suffix(".json").write_text(json.dumps(results, indent=2, default=float))
            out.with_suffix(".txt").write_text("\n".join(cs.summary) + "\n")
            plot(data, settings, routes, out.with_suffix(".png"))
            report(f"Saved {out.name}.npz/.json/.txt/.png in {out.parent}")


def plot(data: dict, settings: dict, routes: dict, path: Path) -> None:
    '''One heatmap per setting: rows = lasers (cores), columns = output and row-1 detectors.'''
    done = [k for k in settings if f"{k}_I_nA" in data]
    if not done:
        return
    cols = OUTPUT_PDS + ROW1_PDS
    names = [f"C{k + 1}" for k in range(len(routes["core_row"]))]
    fig, axs = plt.subplots(1, len(done), figsize=(5.2 * len(done), 4.2), layout="constrained", squeeze=False)
    fig.suptitle(path.stem)
    for ax, name in zip(axs[0], done):
        I = data[f"{name}_I_nA"]
        inp = np.array([I[k, PD_NETS.index(p)] for k, p in enumerate(routes["input_pd"])])
        T = I[:, [PD_NETS.index(c) for c in cols]] / inp[:, None]
        im = ax.imshow(T, cmap="viridis", vmin=0, aspect="auto")
        for (r, c), v in np.ndenumerate(T):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7, color="w")
        for s, dests in settings[name]["routes"].items():
            for x in dests:
                c = cols.index(routes["receiver_pd"][x])
                ax.add_patch(plt.Rectangle((c - .5, int(s) - .5), 1, 1, fill=False, ec="C3", lw=2))
        ax.set_xticks(range(len(cols)), cols, rotation=90)
        ax.set_yticks(range(len(names)), [f"{n} ({l})" for n, l in zip(names, routes["lasers"])])
        ax.set_title(f"{name}: detector / input (red = intended)", fontsize=10)
    fig.colorbar(im, ax=axs[0, -1], label="detector / input")
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
