'''Can we record spikes continuously while the lasers pulse? (groundwork for spike replay)

    Setting A is programmed (C2 -> C3 + C4, C4 -> C5). C2's laser (laser_3) and C4's laser
    (laser_5) then pulse independently, with pulses of 5, 10, 20 and 50 ms at random gaps
    (sometimes overlapping), while the NI board records 10 detectors continuously on its own
    clock (each channel waits 100 us before conversion, so channels do not carry over).

    Afterwards each detector is thresholded halfway between its off and on levels and the
    detected pulses are matched to the commanded ones:
      - missed / false pulses per detector and pulse length
      - edge delay (command -> detected rising edge) and its jitter
      - pulse-length error (detected - commanded)
    Inputs PD_3_0 / PD_5_0 show the lasers themselves; receivers PD_4_5, PD_5_5 (C3, C4) should
    see only laser_3's pulses and PD_6_5 (C5) only laser_5's.

    Results go to runs/stream_check_<timestamp>.{npz,png,txt}.

    Usage (from the repo root on the lab PC; about 1 min):
        python stream_check.py
'''
import argparse
import json
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
SCAN_RATE = 800.0          # scans/s: 10 channels x 100 us conversion = 1 ms per scan max
CONVERT_RATE = 10e3        # conversions/s: 100 us per channel to settle
WIDTHS_MS = [5, 10, 20, 50]
PULSES_PER_WIDTH = 10
GAP_MS = (30, 90)          # random gap between a laser's pulses
MATCH_MS = 15.0            # a detected edge within this long after a command counts as that pulse
# detector -> laser whose pulses it should see in setting A
CHANNELS = {"PD_3_0": "laser_3", "PD_5_0": "laser_5",             # laser inputs
            "PD_4_5": "laser_3", "PD_5_5": "laser_3",             # C3, C4 receivers (C2 multicast)
            "PD_6_5": "laser_5",                                   # C5 receiver (from C4)
            "PD_2_0": None, "PD_4_0": None, "PD_6_0": None,         # dark inputs
            "PD_1_5": None, "PD_3_5": None}                         # unused receivers


def schedule(rng: np.random.Generator) -> list[tuple[float, str, float]]:
    '''Pulse list (start s, laser, length s): each laser independently, all widths shuffled.'''
    out = []
    for laser in ("laser_3", "laser_5"):
        t = 0.3
        widths = rng.permutation(np.repeat(WIDTHS_MS, PULSES_PER_WIDTH))
        for w in widths:
            out.append((t, laser, w / 1e3))
            t += w / 1e3 + rng.uniform(*GAP_MS) / 1e3
    return sorted(out)


def detect(y: np.ndarray, t: np.ndarray, thr: float) -> list[tuple[float, float]]:
    '''(start, length) of every interval where y is above thr.'''
    on = (y > thr).astype(int)
    d = np.diff(np.r_[0, on, 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(t[s], (e - s) / SCAN_RATE) for s, e in zip(starts, ends)]


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    import nidaqmx
    from nidaqmx.constants import AcquisitionType

    routes = json.loads((ROOT / "sim" / "routes_rows2-6.json").read_text())
    cal = load_calibration(list(ROOT.glob("runs/heater_sweep_*.json")))
    pulses = schedule(np.random.default_rng(1))
    duration = max(t + w for t, _, w in pulses) + 0.5
    n = int(duration * SCAN_RATE)
    nets = list(CHANNELS)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "runs" / f"stream_check_{stamp}"
    out.parent.mkdir(exist_ok=True)

    cs.set_laser_range()
    with netlist:
        try:
            set_lasers(0.0)
            volts = {m: heater_voltage(cal[m], "B", 4.2)[0] for m in routes["fixed_bar"]}
            volts.update({m: heater_voltage(cal[m], s, 4.2)[0] for m, s in zip(routes["mzis"], routes["settings"]["A"]["states"])})
            for m, v in volts.items():
                netlist[f"{m}_i"].vout(float(np.clip(v, 0.0, 4.2)))
            sleep(2.0)
            ni = netlist.board_dict["NI"]
            task = nidaqmx.Task()
            for net in nets:
                task.ai_channels.add_ai_voltage_chan(f"{ni.board_num}/ai{netlist.pins_dict[net].chnl}",
                                                     min_val=ni.min_val, max_val=ni.max_val,
                                                     terminal_config=ni.terminal_config)
            task.timing.cfg_samp_clk_timing(rate=SCAN_RATE, sample_mode=AcquisitionType.FINITE, samps_per_chan=n)
            task.timing.ai_conv_rate = CONVERT_RATE
            report(f"stream_check {stamp}: setting A, {len(pulses)} pulses ({WIDTHS_MS} ms) on laser_3 and laser_5, "
                   f"{len(nets)} detectors at {SCAN_RATE:.0f} scans/s for {duration:.1f} s")
            task.start()
            t0 = perf_counter()
            cmd = []                                                  # (laser, on time, off time) actually sent
            events = sorted([(t, l, LASER_V, w) for t, l, w in pulses] + [(t + w, l, 0.0, w) for t, l, w in pulses])
            pending = {}
            for t_ev, laser, v, w in events:
                sleep(max(0.0, t_ev - (perf_counter() - t0)))
                t_sent = perf_counter() - t0
                netlist[laser].vout(v)
                if v:
                    pending[laser] = (t_sent, w)
                else:
                    on, w_cmd = pending.pop(laser)
                    cmd.append((laser, on, t_sent, w_cmd))
            y = -np.array(task.read(number_of_samples_per_channel=n, timeout=duration + 10))   # light lowers V
            task.close()
            set_lasers(0.0)
            t = np.arange(n) / SCAN_RATE

            data = {"t": t, "nets": np.array(nets), "y": y, "cmd_on": np.array([c[1] for c in cmd]),
                    "cmd_off": np.array([c[2] for c in cmd]), "cmd_laser": np.array([c[0] for c in cmd]),
                    "cmd_width": np.array([c[3] for c in cmd])}
            dark = np.median(y[:, t < 0.25], axis=1)
            for k, net in enumerate(nets):
                laser = CHANNELS[net]
                mine = [c for c in cmd if c[0] == laser]
                if not mine:          # should stay dark: report the largest excursion
                    peak = np.max(y[k] - dark[k])
                    report(f"  {net} (no laser routed here): largest signal {1e9 * peak / cs.TRANSIMPEDANCE:.0f} nA above dark")
                    continue
                on_mask = np.zeros(n, bool)
                for _, a, b, _ in mine:
                    on_mask |= (t > a + 0.003) & (t < b)
                level = np.median(y[k, on_mask])
                thr = dark[k] + 0.5 * (level - dark[k])
                found = detect(y[k], t, thr)
                used, rows = set(), []
                for w_ms in WIDTHS_MS:
                    hit, delays, werr = 0, [], []
                    for _, a, b, w in mine:
                        if round(w * 1e3) != w_ms:
                            continue
                        match = [i for i, (s, L) in enumerate(found) if a - 0.002 <= s <= a + MATCH_MS / 1e3 and i not in used]
                        if match:
                            i = match[0]; used.add(i); hit += 1
                            delays.append(found[i][0] - a); werr.append(found[i][1] - (b - a))
                    total = sum(1 for c in mine if round(c[3] * 1e3) == w_ms)
                    rows.append(f"{w_ms} ms {hit}/{total}" + (f" (delay {1e3 * np.median(delays):.1f}±{1e3 * np.std(delays):.1f} ms, "
                                                           f"length {1e3 * np.median(werr):+.1f} ms)" if delays else ""))
                false = len(found) - len(used)
                ok = false == 0 and all(r.split()[2].split("/")[0] == r.split()[2].split("/")[1] for r in rows)
                report(f"  {net} ({laser}, on {1e9 * (level - dark[k]) / cs.TRANSIMPEDANCE:.0f} nA): "
                       f"{'; '.join(rows)}; false {false} -> {verdict(ok)}")
            np.savez_compressed(out.with_suffix(".npz"), **data)
            plot(data, out.with_suffix(".png"))
        except KeyboardInterrupt:
            report("Interrupted.")
        finally:
            set_lasers(0.0)
            for net in cs.HEATER_NETS:
                netlist[net].vout(0.0)
            out.with_suffix(".txt").write_text("\n".join(cs.summary) + "\n")
            report(f"Saved {out.name} files in {out.parent}")


def plot(data: dict, path: Path) -> None:
    '''Recorded traces (offset per detector) with the commanded pulses shaded.'''
    t, y, nets = data["t"], data["y"], list(data["nets"])
    fig, ax = plt.subplots(figsize=(15, 6), layout="constrained")
    for k, net in enumerate(nets):
        yk = (y[k] - np.median(y[k, t < 0.25])) * 1e9 / cs.TRANSIMPEDANCE
        ax.plot(t, yk / max(yk.max(), 1) + 1.3 * k, lw=0.7)
        ax.text(t[-1] + 0.05, 1.3 * k, net, va="bottom", fontsize=8)
    for laser, a, b in zip(data["cmd_laser"], data["cmd_on"], data["cmd_off"]):
        ax.axvspan(a, b, color="C3" if laser == "laser_3" else "C0", alpha=0.08)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title(f"{path.stem}: red = laser_3 commanded on, blue = laser_5")
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
