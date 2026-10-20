'''Replay the SNN's spike traffic through the chip and check what arrives.

    Reads sim/replay_excerpt.json (sim/export_replay.py): per 2 ms SNN step, which cores fired;
    per 10 ms controller window, the router setting (A-D) and which links ride on light.
    Each SNN step becomes one time slot (time-scaled; USB laser drive is the limit):
        [0, PULSE)      every active sender that fired is pulsed (laser 0 -> 10 V)
        [PULSE, SLOT)   quiet; heater changes for the next window are made here
    Five receivers and the five laser inputs are recorded continuously (100 us settling per
    channel, so channels do not carry over).

    Afterwards, per receiver: threshold at half its smallest spike level (one level per
    sender it hears), a slot counts as "spike detected" if at least MIN_SAMPLES consecutive
    samples are above threshold (rejects the single-sample glitches seen when lasers switch).
    Compared with the schedule:
      - hits / missed spikes on every optical link
      - false spikes, split into "another core was firing" (crosstalk) and "nothing was firing"
      - spike packets delivered by light vs planned
      - laser inputs: did each laser actually pulse when commanded (drive path check)

    Results go to runs/spike_replay_<timestamp>.{npz,png,txt,json}.

    Usage (from the repo root on the lab PC; 10 digits = 1500 slots, about 25 s):
        python spike_replay.py
        python spike_replay.py --slot-ms 20 --pulse-ms 8      # slower, more margin
'''
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep

import check_setup as cs
from check_setup import netlist, report, set_lasers, verdict
from route_check import heater_voltage, load_calibration

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
LASER_V = 10.0
SCAN_RATE = 800.0          # scans/s for 10 channels at 100 us each
CONVERT_RATE = 10e3
MIN_SAMPLES = 2            # consecutive samples above threshold for a detection
LATE_MS = 2.0              # commands executed later than this count as overruns
FAULT_MS = 3.0             # a slot with any command this late is a software timing fault, reported apart


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", type=Path, default=ROOT / "sim" / "replay_excerpt.json")
    parser.add_argument("--routes", type=Path, default=ROOT / "sim" / "routes_rows2-6.json")
    parser.add_argument("--slot-ms", type=float, default=15.0)
    parser.add_argument("--pulse-ms", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=None, help="replay only the first steps")
    args = parser.parse_args()
    import nidaqmx
    from nidaqmx.constants import AcquisitionType
    if sys.platform == "win32":                 # fewer stalls of the command loop
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)  # HIGH_PRIORITY_CLASS

    rp = json.loads(args.replay.read_text())
    routes = json.loads(args.routes.read_text())
    cal = load_calibration(list(ROOT.glob("runs/heater_sweep_*.json")))
    lasers, receivers, inputs = routes["lasers"], routes["receiver_pd"], routes["input_pd"]
    ncore = len(lasers)
    steps = rp["steps"][:args.max_steps] if args.max_steps else rp["steps"]
    wsteps = rp["window_steps"]
    windows = rp["windows"][:int(np.ceil(len(steps) / wsteps))]
    slot, pulse = args.slot_ms / 1e3, args.pulse_ms / 1e3

    # heater voltages per setting; only 2_2 and 4_2 change between A-D
    def volts(setting: str) -> dict[str, float]:
        v = {m: heater_voltage(cal[m], "B", 4.2)[0] for m in routes["fixed_bar"]}
        v.update({m: heater_voltage(cal[m], s, 4.2)[0] for m, s in zip(routes["mzis"], routes["settings"][setting]["states"])})
        return {m: float(np.clip(x, 0.0, 4.2)) for m, x in v.items()}
    vset = {s: volts(s) for s in "ABCD"}

    # event list: (time, kind, target, value); slot k starts at T0 + k * slot
    T0 = 0.5
    events = []
    for k, st in enumerate(steps):
        w = windows[k // wsteps]
        t_k = T0 + k * slot
        for s in w["links"]:
            if st["fired"][int(s)]:
                events.append((t_k, 1, lasers[int(s)], LASER_V, k))
                events.append((t_k + pulse, 0, lasers[int(s)], 0.0, k))
        if k % wsteps == wsteps - 1 and k // wsteps + 1 < len(windows):     # next window's heaters
            nxt, cur = windows[k // wsteps + 1]["setting"], w["setting"]
            for m in vset[nxt]:
                if vset[nxt][m] != vset[cur][m]:
                    events.append((t_k + pulse + 0.001, 2, f"{m}_i", vset[nxt][m], k))
    events.sort(key=lambda e: (e[0], e[1]))
    duration = T0 + len(steps) * slot + 0.5
    n = int(duration * SCAN_RATE)
    nets = receivers + inputs

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "runs" / f"spike_replay_{stamp}"
    out.parent.mkdir(exist_ok=True)
    cs.set_laser_range()
    with netlist:
        try:
            set_lasers(0.0)
            for m, v in vset[windows[0]["setting"]].items():
                netlist[f"{m}_i"].vout(v)
            sleep(2.0)
            ni = netlist.board_dict["NI"]
            task = nidaqmx.Task()
            for net in nets:
                task.ai_channels.add_ai_voltage_chan(f"{ni.board_num}/ai{netlist.pins_dict[net].chnl}",
                                                     min_val=ni.min_val, max_val=ni.max_val,
                                                     terminal_config=ni.terminal_config)
            task.timing.cfg_samp_clk_timing(rate=SCAN_RATE, sample_mode=AcquisitionType.FINITE, samps_per_chan=n)
            task.timing.ai_conv_rate = CONVERT_RATE
            report(f"spike_replay {stamp}: digits {rp['digits']} ({len(steps)} SNN steps of {rp['dt_ms']} ms), "
                   f"slot {args.slot_ms:.0f} ms, pulse {args.pulse_ms:.0f} ms, {len(events)} commands, {duration:.1f} s")
            task.start()
            t0 = perf_counter()
            late = []
            for t_ev, kind, target, v, _ in events:
                sleep(max(0.0, t_ev - (perf_counter() - t0)))
                late.append(perf_counter() - t0 - t_ev)
                netlist[target].vout(v)
            y = -np.array(task.read(number_of_samples_per_channel=n, timeout=duration + 10))   # light lowers V
            task.close()
            set_lasers(0.0)
            late = np.array(late) * 1e3
            report(f"  command timing: median {np.median(late):.2f} ms late, max {late.max():.1f} ms, "
                   f"{(late > LATE_MS).sum()} of {len(late)} more than {LATE_MS:.0f} ms late")
            # slots touched by a late command (this slot's pulse, or a heater change just before it)
            faulted = np.zeros(len(steps), bool)
            for (_, kind, _, _, k), lt in zip(events, late):
                if lt > FAULT_MS:
                    faulted[k] = True
                    if kind == 2 and k + 1 < len(steps):
                        faulted[k + 1] = True
            res = analyze(y, nets, steps, windows, wsteps, rp, routes, T0, slot, pulse, faulted)
            np.savez_compressed(out.with_suffix(".npz"), y=y, nets=np.array(nets), late_ms=late,
                                **{k: np.array(v) for k, v in res.pop("arrays").items()})
            out.with_suffix(".json").write_text(json.dumps(res, indent=2, default=float))
        except KeyboardInterrupt:
            report("Interrupted.")
        finally:
            set_lasers(0.0)
            for net in cs.HEATER_NETS:
                netlist[net].vout(0.0)
            out.with_suffix(".txt").write_text("\n".join(cs.summary) + "\n")
            report(f"Saved {out.name} files in {out.parent}")


def runs_above(x: np.ndarray, thr: float) -> int:
    '''Longest run of consecutive samples above thr.'''
    best = cur = 0
    for v in x > thr:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def analyze(y, nets, steps, windows, wsteps, rp, routes, T0, slot, pulse, faulted) -> dict:
    '''Per-slot detection on every receiver and laser input, compared with the schedule.'''
    t = np.arange(y.shape[1]) / SCAN_RATE
    nslot, ncore = len(steps), len(routes["lasers"])
    dark = np.median(y[:, t < T0 - 0.1], axis=1)
    sl = lambda k, a, b: (t >= T0 + k * slot + a) & (t < T0 + k * slot + b)
    # expected[k, j] = sender core lighting receiver j in slot k (or -1)
    sender = np.full((nslot, ncore), -1)
    fired = np.array([s["fired"] for s in steps])
    for k in range(nslot):
        for s, ds in windows[k // wsteps]["links"].items():
            if fired[k, int(s)]:
                for j in ds:
                    sender[k, j] = int(s)
    # per receiver: spike level for each sender it hears, threshold at half the smallest
    det = np.zeros((nslot, ncore), bool)
    thr = {}
    lines = []
    for j in range(ncore):
        yj = y[j] - dark[j]
        levels = {s: np.median(np.concatenate([yj[sl(k, 0.003, pulse)] for k in np.flatnonzero(sender[:, j] == s)]))
                  for s in set(sender[:, j]) - {-1}}
        thr[j] = 0.5 * min(levels.values()) if levels else None
        for k in range(nslot):
            window = yj[sl(k, 0.0, slot)]
            det[k, j] = thr[j] is not None and runs_above(window, thr[j]) >= MIN_SAMPLES
        exp = sender[:, j] >= 0
        other = np.array([any(fired[k, int(s)] for s, ds in windows[k // wsteps]["links"].items() if j not in ds) for k in range(nslot)])
        dj = det[:, j]
        hits, miss = int((dj & exp).sum()), int((~dj & exp).sum())
        f_x, f_0 = int((dj & ~exp & other).sum()), int((dj & ~exp & ~other).sum())
        lvl = ", ".join(f"from C{s + 1} {1e9 * v / cs.TRANSIMPEDANCE:.0f} nA" for s, v in sorted(levels.items()))
        if exp.any():
            report(f"  C{j + 1} receiver {routes['receiver_pd'][j]} ({lvl}): {hits}/{exp.sum()} spikes, "
                   f"{miss} missed, false {f_x} while another core fired + {f_0} otherwise -> {verdict(miss + f_x + f_0 == 0)}")
        else:
            quiet = np.max([np.max(yj[sl(k, 0, slot)]) for k in range(nslot)]) * 1e9 / cs.TRANSIMPEDANCE
            report(f"  C{j + 1} receiver {routes['receiver_pd'][j]} (no optical traffic to it): largest signal {quiet:.0f} nA")
        lines.append(dict(receiver=routes["receiver_pd"][j], expected=int(exp.sum()), hits=hits, missed=miss,
                          false_crosstalk=f_x, false_other=f_0))
    # laser inputs: did each laser pulse exactly when commanded?
    drive_err = 0
    for c in range(ncore):
        yc = y[ncore + c] - dark[ncore + c]
        cmd = np.array([any(int(s) == c and fired[k, c] for s in windows[k // wsteps]["links"]) for k in range(nslot)])
        if not cmd.any():
            continue
        lvl = np.median(np.concatenate([yc[sl(k, 0.003, pulse)] for k in np.flatnonzero(cmd)]))
        seen = np.array([runs_above(yc[sl(k, 0, slot)], 0.5 * lvl) >= MIN_SAMPLES for k in range(nslot)])
        drive_err += int((seen != cmd).sum())
    report(f"  laser drive check: {drive_err} slots where a laser input disagreed with the command")
    # packets
    P = np.array([s["packets"] for s in steps])            # [step, src, dst]
    planned = sum(P[k, sender[k, j], j] for k in range(nslot) for j in range(ncore) if sender[k, j] >= 0)
    delivered = sum(P[k, sender[k, j], j] for k in range(nslot) for j in range(ncore) if sender[k, j] >= 0 and det[k, j])
    total = int(P.sum())
    errs = sum(l["missed"] + l["false_crosstalk"] + l["false_other"] for l in lines)
    exp_total = sum(l["expected"] for l in lines)
    report(f"  spike packets: {total} in total, {planned} planned by light ({100 * planned / total:.1f}%), "
           f"{delivered} delivered ({100 * delivered / max(planned, 1):.2f}% of planned)")
    report(f"  overall: {exp_total} optical spike slots, {errs} errors -> {verdict(errs == 0)}")
    # split errors by whether the software kept time in that slot
    exp = sender >= 0
    err = (exp & ~det) | (~exp & det)
    err[:, [j for j in range(ncore) if not exp[:, j].any()]] = False        # receivers with no traffic
    on_time = ~faulted
    report(f"  timing faults: {faulted.sum()} slots had a command more than {FAULT_MS:.0f} ms late; "
           f"errors in on-time slots {int(err[on_time].sum())} of {int(exp[on_time].sum())} optical spikes, "
           f"in faulted slots {int(err[faulted].sum())} of {int(exp[faulted].sum())}")
    return {"receivers": lines, "timing_fault_slots": int(faulted.sum()),
            "errors_on_time": int(err[on_time].sum()), "spikes_on_time": int(exp[on_time].sum()),
            "packets_total": total, "packets_planned": int(planned),
            "packets_delivered": int(delivered), "drive_errors": drive_err,
            "arrays": {"sender": sender, "detected": det, "fired": fired, "faulted": faulted}}


if __name__ == "__main__":
    main()
