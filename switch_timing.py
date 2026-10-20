'''How fast does the router switch routes?

    One heater is switched while one detector is recorded at 250,000 samples/s (4 us apart):
      - "2_2": setting A <-> C. Only C1's laser (row 2) is on. With 2_2 at BAR (A) its light
        leaves on row 2 (PD_1_5); at CROSS (C) it goes to C3 and C4, so PD_4_5 lights up.
      - "4_2": setting A <-> D. Only C3's laser (row 4) is on. With 4_2 at CROSS (D) its light
        goes to C5 (PD_6_5); at BAR (A) it leaves elsewhere.
    Each switch is repeated in both directions (heating and cooling differ). From each trace:
      - command delay: from sending the heater command until the light starts to change
        (our control path: Python -> USB serial -> Pico -> AD5370 DAC); the timing of the
        command against the recording is known only to about a millisecond
      - rise time (10-90%): how fast the light moves once the heater has its new voltage
        (the heater's own thermal response, expected ~10-50 us)

    Results go to runs/switch_timing_<timestamp>.{npz,png,txt}.

    Usage (from the repo root on the lab PC; about 2 min):
        python switch_timing.py
        python switch_timing.py --heaters 2_2 --repeats 30
'''
import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep

import check_setup as cs
from check_setup import LASER_NETS, netlist, report, set_lasers
from route_check import heater_voltage, load_calibration

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RATE = 250e3          # samples/s, one channel (no channel switching, so no carry-over)
WINDOW_S = 0.10       # recording length per switch
PRE_S = 0.02          # command sent this long after the recording starts
LASER_V = 10.0        # lasers are independent at 10 V (laser_interaction, 2026-10-19)
# heater -> (settings it switches between, core whose laser is on, detector watched)
TESTS = {"2_2": (("A", "C"), 0, "PD_4_5"),
         "4_2": (("A", "D"), 2, "PD_6_5")}


def open_recorder(pd: str):
    '''NI task recording one detector at RATE for WINDOW_S (finite, started by the caller).'''
    import nidaqmx
    from nidaqmx.constants import AcquisitionType
    ni = netlist.board_dict["NI"]
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(f"{ni.board_num}/ai{netlist.pins_dict[pd].chnl}",
                                         min_val=ni.min_val, max_val=ni.max_val,
                                         terminal_config=ni.terminal_config)
    task.timing.cfg_samp_clk_timing(rate=RATE, sample_mode=AcquisitionType.FINITE,
                                    samps_per_chan=int(RATE * WINDOW_S))
    return task


def edges(y: np.ndarray, t: np.ndarray, t_cmd: float) -> dict[str, float]:
    '''Levels before/after, 10% and 90% crossing times (s) and derived delay and rise time.'''
    before, after = np.median(y[t < t_cmd]), np.median(y[t > t[-1] - 0.02])
    step = after - before
    if abs(step) < 5 * np.std(y[t < t_cmd]):
        return dict(step=step, delay=np.nan, rise=np.nan)
    frac = (y - before) / step
    k10 = np.argmax((frac >= 0.1) & (t > t_cmd))
    k90 = np.argmax((frac >= 0.9) & (t > t_cmd))
    return dict(step=step, delay=t[k10] - t_cmd, rise=t[k90] - t[k10])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heaters", nargs="+", default=list(TESTS), choices=list(TESTS))
    parser.add_argument("--repeats", type=int, default=20, help="switches per direction")
    parser.add_argument("--routes", type=Path, default=ROOT / "sim" / "routes_rows2-6.json")
    parser.add_argument("--v-max", type=float, default=4.2)
    args = parser.parse_args()

    routes = json.loads(args.routes.read_text())
    cal = load_calibration(list(ROOT.glob("runs/heater_sweep_*.json")))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "runs" / f"switch_timing_{stamp}"
    out.parent.mkdir(exist_ok=True)
    data: dict[str, np.ndarray] = {"rate": np.array(RATE)}

    def apply(setting: str) -> dict[str, float]:
        '''Sets every heater for a setting; returns the voltages.'''
        volts = {m: heater_voltage(cal[m], "B", args.v_max)[0] for m in routes["fixed_bar"]}
        volts.update({m: heater_voltage(cal[m], s, args.v_max)[0]
                      for m, s in zip(routes["mzis"], routes["settings"][setting]["states"])})
        for m, v in volts.items():
            netlist[f"{m}_i"].vout(float(np.clip(v, 0.0, args.v_max)))
        return volts

    cs.set_laser_range()
    with netlist:
        try:
            report(f"switch_timing {stamp}: {RATE / 1e3:.0f} kS/s, {1e3 * WINDOW_S:.0f} ms per switch, "
                   f"{args.repeats} switches per direction, lasers at {LASER_V} V")
            for heater in args.heaters:
                (s_from, s_to), core, pd = TESTS[heater]
                laser = routes["lasers"][core]
                set_lasers(0.0)
                v_to = apply(s_to)[heater]
                v_from = apply(s_from)[heater]                               # start in s_from
                netlist[laser].vout(LASER_V)
                sleep(5.0)                                                   # laser and heaters settle
                task = open_recorder(pd)
                cmd_times = []
                results = {f"{s_from}->{s_to}": [], f"{s_to}->{s_from}": []}
                traces = {k: [] for k in results}
                for _ in range(args.repeats):
                    for label, v_new in ((f"{s_from}->{s_to}", v_to), (f"{s_to}->{s_from}", v_from)):
                        task.start()
                        t0 = perf_counter()
                        sleep(PRE_S)
                        t_cmd = perf_counter()
                        netlist[f"{heater}_i"].vout(float(v_new))
                        cmd_times.append(perf_counter() - t_cmd)
                        y = -np.array(task.read(number_of_samples_per_channel=int(RATE * WINDOW_S)))  # light lowers V
                        task.stop()
                        t = np.arange(len(y)) / RATE
                        results[label].append(edges(y, t, t_cmd - t0))
                        traces[label].append(y)
                        sleep(0.2)
                task.close()
                netlist[laser].vout(0.0)
                for label, res in results.items():
                    delay = np.array([r["delay"] for r in res]) * 1e3
                    rise = np.array([r["rise"] for r in res]) * 1e6
                    ok = np.isfinite(rise)
                    report(f"[{heater}] {label} ({pd}, {laser}): rise 10-90% {np.nanmedian(rise):.0f} us "
                           f"(range {np.nanmin(rise):.0f}-{np.nanmax(rise):.0f}); command delay "
                           f"{np.nanmedian(delay):.2f} ms (range {np.nanmin(delay):.2f}-{np.nanmax(delay):.2f}); "
                           f"{ok.sum()}/{len(res)} clean steps")
                    data[f"{heater}_{label}"] = np.array(traces[label])
                report(f"[{heater}] heater command call takes {1e3 * np.median(cmd_times):.2f} ms "
                       f"(Python -> Pico round trip)")
        except KeyboardInterrupt:
            report("Interrupted; saving what was measured.")
        finally:
            set_lasers(0.0)
            for net in cs.HEATER_NETS:
                netlist[net].vout(0.0)
            np.savez_compressed(out.with_suffix(".npz"), **data)
            out.with_suffix(".txt").write_text("\n".join(cs.summary) + "\n")
            plot(data, out.with_suffix(".png"))
            report(f"Saved {out.name}.npz/.txt/.png in {out.parent}")


def plot(data: dict, path: Path) -> None:
    '''Every recorded switch, overlaid per heater and direction (normalized 0-1).'''
    keys = [k for k in data if k != "rate"]
    if not keys:
        return
    fig, axs = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 3.6), layout="constrained", squeeze=False)
    fig.suptitle(path.stem)
    for ax, k in zip(axs[0], keys):
        for y in data[k]:
            n = int(PRE_S * RATE)
            lo, hi = np.median(y[:n]), np.median(y[-int(0.02 * RATE):])
            t_ms = (np.arange(len(y)) / RATE - PRE_S) * 1e3
            ax.plot(t_ms, (y - lo) / (hi - lo) if hi != lo else y * 0, lw=0.6, alpha=0.6)
        ax.set_xlim(-2, 20)
        ax.set_title(k.replace("_", " ", 1), fontsize=10)
        ax.set_xlabel("Time after heater command (ms)")
        ax.set_ylabel("Light change (0 = before, 1 = after)")
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
