'''Board configuration and netlist definition for a 6x6 AIM MZI mesh
    from our 2020 tapeout.
'''
import pydaq as daq
from pydaq import mcc, ser
from pydaq import nidaq as ni

ni_boards =[
    ni.USB_6210("NI", 33703915,
        ni.AIPIN("PD_1_0", 0), 
        ni.AIPIN("PD_2_0", 6), 
        ni.AIPIN("PD_3_0", 12),
        ni.AIPIN("PD_4_0", 1),
        ni.AIPIN("PD_5_0", 2),
        ni.AIPIN("PD_6_0", 3),

        ni.AIPIN("PD_1_5", 5), 
        ni.AIPIN("PD_2_5", 11),
        ni.AIPIN("PD_3_5", 4),
        ni.AIPIN("PD_4_5", 7),
        ni.AIPIN("PD_5_5", 8),
        ni.AIPIN("PD_6_5", 9),      
        
        ni.AIPIN("PD_1_2", 10),
        ni.AIPIN("PD_6_1", 13),
        ni.AIPIN("PD_1_3", 14),
        ni.AIPIN("PD_6_3", 15),
        min_val=0.0,
        max_val=1.3,
    ),
]

mcc_boards = [
    mcc.USB_3114("lasers", "1FD2F3C",
        mcc.AOPIN("laser_1", 0),
        mcc.AOPIN("laser_2", 1),
        mcc.AOPIN("laser_3", 2),
        mcc.AOPIN("laser_4", 3),
        mcc.AOPIN("laser_5", 4),
        mcc.AOPIN("laser_6", 5),
    ),
]

ser_boards = [
    ser.EVAL_AD5370("DAC", 0, 17, # 40-channel DAC with buffer board with csPin=17
        ser.AOPIN("3_5_o", 1), # NEW
        ser.AOPIN("3_5_i", 3),
        ser.AOPIN("1_5_i", 5),
        ser.AOPIN("2_4_o", 7), # NEW
        ser.AOPIN("2_4_i", 9),
        ser.AOPIN("3_3_o", 11), # NEW
        ser.AOPIN("3_3_i", 13),
        ser.AOPIN("1_3_o", 15),
        ser.AOPIN("1_3_i", 17),
        ser.AOPIN("2_2_o", 19), # NEW
        ser.AOPIN("2_2_i", 21),
        ser.AOPIN("3_1_i", 23),
        ser.AOPIN("1_1_i", 25),

        ser.AOPIN("5_1_i", 0),
        ser.AOPIN("4_2_i", 2),
        ser.AOPIN("4_2_o", 4), # NEW (mislabeled as 5_2_o on board?)
        ser.AOPIN("5_3_i", 6),
        ser.AOPIN("4_4_i", 8),
        ser.AOPIN("4_4_o", 10), # NEW (mislabeled as 5_4_o on board?)
        ser.AOPIN("5_5_i", 12),
        
        
        # New channels available if needed (currently unused):
        ser.AOPIN("2_1_i", 27),
        ser.AOPIN("3_2_i", 29),
        ser.AOPIN("2_3_i", 31),
        ser.AOPIN("3_4_i", 33),
        ser.AOPIN("2_5_i", 35),
        ser.AOPIN("1_5_o", 37),
        
        ser.AOPIN("4_1_i", 14),
        ser.AOPIN("5_2_i", 16),
        ser.AOPIN("4_3_i", 18),
        ser.AOPIN("5_4_i", 20),
        ser.AOPIN("4_5_i", 22),
    ),
]

pico = ser.BoardManager("PICO-001", *ser_boards)

# Create a netlist with the NI and MCC boards
ni.config_detected_devices(ni_boards, verbose=False)
mcc.config_detected_devices(mcc_boards, verbose=False)
ser.config_detected_devices([pico], verbose=False)
netlist = daq.Netlist(*ni_boards, *mcc_boards, pico)

if __name__ == "__main__":
    from collections import deque
    from time import monotonic

    import matplotlib.pyplot as plt
    import numpy as np
    from mcculw import ul
    from mcculw.enums import BoardInfo, InfoType

    # --- 1. Laser power: one USB-3114 output per laser, 0 V = off, 10 V = max ---
    laser_board = netlist.board_dict["lasers"]
    laser_nets = laser_board.nets  # ["laser_1", ..., "laser_6"]
    LASER_V_MIN, LASER_V_MAX = 0.0, 10.0  # full scale of the UNI10VOLTS range

    # The USB-3114 output range is a per-channel device setting (BIDACRANGE) and
    # must match the range v_out is called with, so pin every laser channel to it.
    for pin in laser_board.pins.values():
        ul.set_config(InfoType.BOARDINFO, laser_board.board_num, pin.chnl,
                      BoardInfo.DACRANGE, laser_board.ao_range)

    def set_all_lasers(volts):
        volts = np.broadcast_to(np.asarray(volts, dtype=float), (len(laser_nets),))
        if volts.min() < LASER_V_MIN or volts.max() > LASER_V_MAX:
            raise ValueError(f"Laser voltages must be in [{LASER_V_MIN}, {LASER_V_MAX}] V, got {volts}")
        for net, v in zip(laser_nets, volts):
            netlist[net].vout(v)

    # --- 2. Live plot of every photodetector until the window is closed or Ctrl+C ---
    # group_vin can only read nets that share a board, so read one group per board.
    pd_groups: dict[str, list[str]] = {}
    for net, pin in netlist.pins_dict.items():
        if pin.direction == daq.PIN_DIRECTION.INPUT:
            pd_groups.setdefault(pin.board.board_name, []).append(net)
    pd_nets = [net for nets in pd_groups.values() for net in nets]

    WINDOW_S = 10.0  # seconds of history on screen

    # Panel (r-1, c) shows PD_<r>_<c>, so the grid mirrors the 6x6 mesh and
    # detectors that aren't wired to an ADC leave their cell empty.
    plt.ion()
    fig, axes = plt.subplots(6, 6, sharex=True, sharey=True,
                             figsize=(13.2, 8.4), layout="constrained")
    pd_axes = [axes[int(net.split("_")[1]) - 1, int(net.split("_")[2])] for net in pd_nets]
    lines = [ax.plot([], [], lw=1.2)[0] for ax in pd_axes]
    for ax, net in zip(pd_axes, pd_nets):
        ax.set_title(net, fontsize=9)
        ax.grid(alpha=0.3)
    for ax in set(axes.flat) - set(pd_axes):
        ax.set_visible(False)

    def set_ylim_to(lo, hi):
        pd_axes[0].set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))

    # min_val/max_val only pick the NI hardware range (±5 V here); readings
    # outside them are not clipped, so the view grows to keep them on screen.
    v_lo, v_hi = ni_boards[0].min_val, ni_boards[0].max_val
    set_ylim_to(v_lo, v_hi)
    pd_axes[0].set_xlim(-WINDOW_S, 0)
    fig.supxlabel("Time (s)")
    fig.supylabel("Photodetector voltage (V)")
    plt.show(block=False)
    fig.canvas.draw()
    fig.set_layout_engine("none")  # lay out once: re-solving it every frame triples the draw time

    times, readings = deque(), deque()
    with netlist:
        try:
            # set_all_lasers(LASER_V_MAX)              # all lasers at max
            # set_all_lasers(0.0)                    # all lasers off
            # set_all_lasers(2.5)                    # all lasers at any value in [0, 10] V
            set_all_lasers([0, 10, 0, 0, 0, 0])    # one value per laser, laser_1 first

            print("Live photodetector plot: close the window or press Ctrl+C to stop.")
            while plt.fignum_exists(fig.number):
                now = monotonic()
                times.append(now)
                readings.append(np.concatenate([netlist.group_vin(nets) for nets in pd_groups.values()]))
                while times[0] < now - WINDOW_S:
                    times.popleft()
                    readings.popleft()

                t = np.array(times) - now
                v = np.array(readings)
                set_ylim_to(min(v_lo, v.min()), max(v_hi, v.max()))
                for i, (line, ax) in enumerate(zip(lines, pd_axes)):
                    line.set_data(t, v[:, i])
                    ax.set_title(f"{pd_nets[i]}  {v[-1, i]:.3f} V", fontsize=9)
                fig.canvas.draw()
                fig.canvas.flush_events()
        except KeyboardInterrupt:
            print("Interrupted.")
        finally:
            set_all_lasers(0.0)  # lasers off even if resetting another board fails on exit