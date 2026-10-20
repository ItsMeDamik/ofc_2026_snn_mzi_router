'''Check that one output detector (default PD_6_5, row 6) really sees light routed to it.

    Three tests a broken or mislabeled detector cannot pass:
      1. On/off: in setting A, laser_5 (core C4) is routed to row 6. Switching laser_5 on and
         off five times must switch PD_6_5 between bright and dark.
      2. Same path: PD_6_3 sits on row 6 mid-chip, on the same light path, so it must light
         up together with PD_6_5.
      3. Re-routing: in setting B the router sends laser_5's light to PD_3_5 instead, so
         PD_6_5 must go dark while PD_3_5 lights up, with the laser still on.

    Usage (from the repo root on the lab PC; about 40 s):
        python pd_check.py
'''
import json
from datetime import datetime
from pathlib import Path
from time import sleep

import check_setup as cs
from check_setup import PD_NETS, netlist, record, report, set_lasers, to_nA, verdict
from route_check import heater_voltage, load_calibration

import numpy as np

ROOT = Path(__file__).resolve().parent
LASER, LASER_V = "laser_5", 10.0
WATCH = ["PD_6_5", "PD_6_3", "PD_3_5", "PD_5_0"]     # target, same-path, re-route target, laser input


def apply(routes: dict, cal: dict, setting: str) -> None:
    '''Sets every heater for a routing setting (A-D).'''
    v = {m: heater_voltage(cal[m], "B", 4.2)[0] for m in routes["fixed_bar"]}
    v.update({m: heater_voltage(cal[m], s, 4.2)[0] for m, s in zip(routes["mzis"], routes["settings"][setting]["states"])})
    for m, x in v.items():
        netlist[f"{m}_i"].vout(float(np.clip(x, 0.0, 4.2)))
    sleep(1.0)


def reading(dark: np.ndarray) -> dict[str, float]:
    '''Photocurrent (nA) of the watched detectors, averaged over 1 s.'''
    _, v = record(1.0, 0.1)
    I = to_nA(v.mean(axis=0), dark)
    return {p: float(I[PD_NETS.index(p)]) for p in WATCH}


def main() -> None:
    routes = json.loads((ROOT / "sim" / "routes_rows2-6.json").read_text())
    cal = load_calibration(list(ROOT.glob("runs/heater_sweep_*.json")))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "runs" / f"pd_check_{stamp}.txt"
    cs.set_laser_range()
    with netlist:
        try:
            set_lasers(0.0)
            apply(routes, cal, "A")
            _, v = record(2.0, 0.2)
            dark = v.mean(axis=0)
            report(f"pd_check {stamp}: {LASER} at {LASER_V} V; watching {WATCH}")
            on, off = [], []
            for i in range(5):                                       # test 1 and 2
                netlist[LASER].vout(LASER_V); sleep(1.0); on.append(reading(dark))
                netlist[LASER].vout(0.0); sleep(1.0); off.append(reading(dark))
                report(f"  setting A, cycle {i + 1}: laser on  " + ", ".join(f"{p} {on[-1][p]:7.1f}" for p in WATCH) + " nA")
                report(f"                       laser off " + ", ".join(f"{p} {off[-1][p]:7.1f}" for p in WATCH) + " nA")
            lit = np.mean([r["PD_6_5"] for r in on]); dk = np.mean([r["PD_6_5"] for r in off])
            mid = np.mean([r["PD_6_3"] for r in on])
            ok1 = lit > 100 * max(abs(dk), cs.LIT_NA / 5)
            report(f"[1] PD_6_5 on {lit:.0f} nA vs off {dk:.1f} nA -> {verdict(ok1)}")
            report(f"[2] PD_6_3 (same row-6 path) on {mid:.0f} nA -> {verdict(mid > cs.LIT_NA)}")
            apply(routes, cal, "B")                                  # test 3
            netlist[LASER].vout(LASER_V); sleep(1.0)
            b = reading(dark)
            netlist[LASER].vout(0.0)
            ok3 = b["PD_6_5"] < 0.05 * lit and b["PD_3_5"] > 0.3 * lit
            report(f"[3] setting B, laser on: PD_6_5 {b['PD_6_5']:.1f} nA ({100 * b['PD_6_5'] / lit:.1f}% of A), "
                   f"PD_3_5 {b['PD_3_5']:.0f} nA (light moved there) -> {verdict(ok3)}")
        finally:
            set_lasers(0.0)
            for net in cs.HEATER_NETS:
                netlist[net].vout(0.0)
            out.write_text("\n".join(cs.summary) + "\n")
            report(f"Saved {out.name}")


if __name__ == "__main__":
    main()
