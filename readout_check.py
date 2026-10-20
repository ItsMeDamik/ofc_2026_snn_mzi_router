'''Checks the detector readout for channel-to-channel carry-over ("ghosting").

    The NI board reads its channels one after another through one shared input. With
    pydaq's group_vin, a channel read right after a bright one showed about a third of that
    channel's level (route_check, 2026-10-19). check_setup.read_pds() slows the board's
    conversion clock so each channel has 100 us to settle. This script lights each laser alone and compares
    both readers on the channel scanned right after that laser's input detector, which
    should be dark.

    Usage (from the repo root on the lab PC; about 40 s):
        python readout_check.py
'''
from datetime import datetime
from pathlib import Path
from time import sleep

import check_setup as cs
from check_setup import INPUT_PDS, LASER_NETS, PD_NETS, netlist, report, set_lasers, to_nA, verdict

import numpy as np

LASER_V = 10.0
GOOD = 0.02          # carry-over below 2% of the bright channel counts as fixed


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(__file__).resolve().parent / "runs" / f"readout_check_{stamp}.txt"
    out.parent.mkdir(exist_ok=True)
    cs.set_laser_range()
    with netlist:
        try:
            set_lasers(0.0)
            for net in cs.HEATER_NETS:
                netlist[net].vout(0.0)
            sleep(cs.OFF_WAIT_S)
            dark_new = np.mean([cs.read_pds() for _ in range(10)], axis=0)   # set up the new reader first
            dark_old = np.mean([netlist.group_vin(PD_NETS) for _ in range(10)], axis=0)
            report(f"readout_check {stamp}: each laser alone at {LASER_V} V; carry-over onto the next-scanned channel")
            for k, net in enumerate(LASER_NETS):
                own = PD_NETS.index(INPUT_PDS[k])
                nxt = own + 1                                   # scanned right after the lit input
                netlist[net].vout(LASER_V)
                sleep(3.0)
                old = to_nA(np.mean([netlist.group_vin(PD_NETS) for _ in range(10)], axis=0), dark_old)
                new = to_nA(np.mean([cs.read_pds() for _ in range(10)], axis=0), dark_new)
                netlist[net].vout(0.0)
                sleep(1.0)
                r_old, r_new = old[nxt] / old[own], new[nxt] / new[own]
                report(f"  {net}: {PD_NETS[own]} {new[own]:.0f} nA -> next {PD_NETS[nxt]}: "
                       f"group_vin {100 * r_old:.1f}%, read_pds ({cs.READ_MODE}) {100 * r_new:.1f}% -> {verdict(abs(r_new) < GOOD)}")
        finally:
            set_lasers(0.0)
            out.write_text("\n".join(cs.summary) + "\n")
            report(f"Saved {out.name}")


if __name__ == "__main__":
    main()
