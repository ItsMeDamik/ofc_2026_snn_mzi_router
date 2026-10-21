'''Live replay of the SNN's spike traffic through the chip: live router controller, one laser
    pulse per spike.

    Differences from spike_replay.py (kept unchanged as the reference method):
      1. Live controller. No precomputed schedule: while the run proceeds, the packet counts
         [src, dst] of the 5 steps of the current 10 ms window are accumulated from the SNN
         spike record (sim/spikes_test.npz; the electronic side knows what it sends). At each
         window boundary the previous-window rule of sim/export_replay.py picks the setting
         that would have carried the most packets in the window just finished, and the 2_2 /
         4_2 heater commands are sent in a dark gap before the next window. A link that would
         carry nothing holds its heater and stays dark. The first window has no history and
         starts from setting A (both links active).
      2. One pulse per spike. In every step, each active optical link sends one pulse per
         spiking neuron of its sending core, in ascending neuron index, so pulse i of a step
         maps to a known neuron. A multicast input spike is one pulse seen by C3 and C4. The
         two links pulse in parallel (slot i carries pulse i of both links):
             step length = max pulses on any link x (on + off), plus a heater gap at window
             boundaries.
         Laser commands either update several USB-3114 channels in one call (a_out_scan,
         "scan" mode) or are serialized v_out calls ("serial" mode, link l delayed by l x the
         measured call time, on/off stretched so the calls fit). The mode is chosen by a probe
         at start-up (--laser-mode auto) and reported.
         A command started more than --slip-ms late shifts the rest of the schedule (the
         shift is logged; lateness is always recorded against the original time).
      3. Readout. Only the three active receivers (C3 PD_4_5, C4 PD_5_5, C5 PD_6_5) are
         streamed, 100 us settling per conversion (3 channels -> ~3.3 kS/s each).
         --monitors interleaves the laser input detectors PD_2_0..PD_5_0 (scan = C3 C4 C5 M1
         C3 C4 C5 M2 ...): receivers evenly spaced at 2.5 kS/s, each monitor at 625 S/s.
         Calibration blocks (dark + each sender alone, 30 ms each, in settings B and A) run at
         the start, every --recal-every digits and at the end; their edges also align the DAQ
         clock to the PC clock (offset + drift fit) and their levels set the thresholds.
      4. Analysis (replay_analysis.analyze_live): detection per expected pulse (>= 2
         consecutive samples above half the smallest spike level), hits / misses / false
         detections, timing faults (> 3 ms late commands, reported apart but always included
         in the overall numbers), per-neuron delivery mask -> SNN rerun with exactly the
         undelivered spikes removed, bus model with undelivered packets returned to the bus,
         and a check of the live controller against export_replay.py's schedule.

    --dry-run: no hardware. Laser / heater calls are emulated (call time, occasional stalls),
    the receiver signals are synthesized from the command log (levels from the 2026-10-19
    run, DAQ clock offset and drift, laser drift, injected dropped pulses), then analysed by
    the same code. Checks timing and the pulse-to-neuron mapping.

    Results: runs/spike_replay_live_<timestamp>.{npz,json,txt}

    Usage (repo root; sim/spikes_test.npz must be present: it is not in git):
        python spike_replay_live.py --dry-run --digits 2
        python spike_replay_live.py --digits 1
        python spike_replay_live.py --digits 10
        python spike_replay_live.py --digits 100                 # ~10-20 min
        python spike_replay_live.py --digits 10 --on-ms 2 --off-ms 2 --monitors
'''
import argparse
import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sim"))
from export_replay import SETTING, WINDOW_BINS, layer_mapping     # noqa: E402
from compile_routes import feasible_assignments                   # noqa: E402

LASER_V = 10.0
HEATER_VMAX = 4.2
TRANSIMPEDANCE = 220e3
CONVERT_RATE = 10e3        # one conversion per 100 us: every channel settles before it is read
STEPS_PER_DIGIT = 150
N_CORE = 5
SENDERS = (0, 1, 2, 3)     # C1..C4 send optically; C5 only receives
DESTS = {0: (2, 3), 1: (2, 3), 2: (4,), 3: (4,)}
LINK_MZI = ("2_2", "4_2")  # input link (C1/C2 -> C3+C4), hidden link (C3/C4 -> C5)
CAL_ON_S = 0.030           # calibration: dark / laser-on / laser-off interval
CAL_SETTLE_S = 0.020       # after a heater change in a calibration block
K_PULSE, K_HEATER, K_CAL = 0, 1, 2     # command kinds in the log

summary: list[str] = []


def report(line: str) -> None:
    print(line, flush=True)
    summary.append(line)


def stats_ms(x) -> dict:
    x = np.asarray(x, float) * 1e3
    return {"n": int(x.size), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "p99": float(np.percentile(x, 99)), "max": float(x.max())} if x.size else {"n": 0}


class Clock:
    '''PC time in s since the start of the run; waits sleep, then spin for the last ~1.5 ms.'''

    def __init__(self):
        self.t0 = perf_counter()

    def now(self) -> float:
        return perf_counter() - self.t0

    def wait(self, t: float) -> None:
        while (dt := t - self.now()) > 0:
            if dt > 0.002:
                sleep(dt - 0.0015)


# ----------------------------------------------------------------------------- hardware
class Hardware:
    '''USB-3114 lasers, AD5370 heaters (Pico) and the NI USB-6210 stream.'''
    sim = False

    def __init__(self, routes: dict):
        import check_setup as cs
        from route_check import heater_voltage, load_calibration
        self.cs, self.netlist = cs, cs.netlist
        cal = load_calibration(list(ROOT.glob("runs/heater_sweep_*.json")))
        self.vset = heater_settings(routes, lambda m, s: heater_voltage(cal[m], s, HEATER_VMAX)[0])
        self.laser_nets = routes["lasers"][:4]
        self.chan = [int(self.netlist.pins_dict[n].chnl) for n in self.laser_nets]
        self.mode = "serial"
        self._scan_bufs = None
        self._task = None

    def __enter__(self):
        self.cs.set_laser_range()
        self.netlist.__enter__()
        if sys.platform == "win32":                 # fewer stalls of the command loop, 1 ms timer
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
            ctypes.windll.winmm.timeBeginPeriod(1)
        return self

    def __exit__(self, *exc):
        try:
            self.all_off()
            for net in self.cs.HEATER_NETS:
                self.netlist[net].vout(0.0)
        finally:
            if self._scan_bufs:
                from mcculw import ul
                for mh in self._scan_bufs.values():
                    ul.win_buf_free(mh)
            self.netlist.__exit__(*exc)

    def all_off(self) -> None:
        self.cs.set_lasers(0.0)

    # lasers -------------------------------------------------------------
    def v_out(self, core: int, v: float) -> None:
        assert 0.0 <= v <= LASER_V
        self.netlist[self.laser_nets[core]].vout(v)

    def _make_scan(self, option) -> None:
        '''One 16-bit buffer per on/off pattern of C1..C4, covering channels lo..hi.'''
        from ctypes import POINTER, c_ushort, cast
        from mcculw import ul
        b = self.netlist.board_dict["lasers"]
        lo, hi = min(self.chan), max(self.chan)
        self._scan_args = (b.board_num, lo, hi, hi - lo + 1, 100, b.ao_range)
        self._scan_opt = option
        self._scan_bufs = {}
        for pat in range(16):
            mh = ul.win_buf_alloc(hi - lo + 1)
            arr = cast(mh, POINTER(c_ushort))
            for ch in range(lo, hi + 1):
                on = ch in self.chan and pat >> self.chan.index(ch) & 1
                arr[ch - lo] = ul.from_eng_units(b.board_num, b.ao_range, LASER_V if on else 0.0)
            self._scan_bufs[pat] = mh

    def scan_out(self, pattern: int) -> None:
        from mcculw import ul
        bn, lo, hi, n, rate, rng = self._scan_args
        ul.a_out_scan(bn, lo, hi, n, rate, rng, self._scan_bufs[pattern], self._scan_opt)

    # heaters -------------------------------------------------------------
    def heater(self, mzi: str, v: float) -> None:
        self.netlist[f"{mzi}_i"].vout(float(np.clip(v, 0.0, HEATER_VMAX)))

    # probe ----------------------------------------------------------------
    def _read_inputs(self, nets: list[str]) -> np.ndarray:
        '''Mean voltage of a few laser input detectors (short NI task, closed afterwards).'''
        import nidaqmx
        from nidaqmx.constants import AcquisitionType
        ni = self.netlist.board_dict["NI"]
        with nidaqmx.Task() as task:
            for net in nets:
                task.ai_channels.add_ai_voltage_chan(f"{ni.board_num}/ai{self.netlist.pins_dict[net].chnl}",
                                                     min_val=ni.min_val, max_val=ni.max_val,
                                                     terminal_config=ni.terminal_config)
            task.timing.cfg_samp_clk_timing(rate=CONVERT_RATE / len(nets), sample_mode=AcquisitionType.FINITE,
                                            samps_per_chan=200)
            task.timing.ai_conv_rate = CONVERT_RATE
            return np.array(task.read(number_of_samples_per_channel=200)).mean(axis=1)

    def probe(self, routes: dict, mode: str, n: int = 200) -> dict:
        '''Times v_out and a_out_scan; checks optically that one scan call turns on 2 lasers.'''
        from mcculw import ul
        from mcculw.enums import ScanOptions
        res = {}
        t = []
        for _ in range(n):
            a = perf_counter(); self.v_out(0, 0.0); t.append(perf_counter() - a)
        res["v_out_call"] = stats_ms(t)
        opts = [("SIMULTANEOUS", getattr(ScanOptions, "SIMULTANEOUS", None)), ("FOREGROUND", ScanOptions.FOREGROUND)]
        for name, opt in opts:
            if opt is None:
                continue
            try:
                self._make_scan(opt)
                t = []
                for _ in range(n):
                    a = perf_counter(); self.scan_out(0); t.append(perf_counter() - a)
                res["scan_call"], res["scan_option"] = stats_ms(t), name
                break
            except Exception as err:                     # ULError: not supported with this option
                res[f"scan_error_{name}"] = str(err)
                self._scan_bufs = None
        if res.get("scan_option"):
            mons = [routes["input_pd"][0], routes["input_pd"][2]]           # C1, C3 laser inputs
            dark = self._read_inputs(mons)
            self.scan_out(0b0101); sleep(0.5)                               # C1 + C3 in one call
            both = (dark - self._read_inputs(mons)) / TRANSIMPEDANCE * 1e9
            self.scan_out(0); sleep(0.5)
            single = []
            for core in (0, 2):
                self.v_out(core, LASER_V); sleep(0.5)
                single.append((dark - self._read_inputs(mons)) / TRANSIMPEDANCE * 1e9)
                self.v_out(core, 0.0); sleep(0.5)
            ratio = [float(both[0] / single[0][0]), float(both[1] / single[1][1])]
            res["scan_optical_nA"] = {"both_one_call": both.tolist(), "C1_alone": single[0].tolist(),
                                      "C3_alone": single[1].tolist(), "ratio_vs_alone": ratio}
            res["scan_works"] = bool(min(ratio) > 0.8)
        t = []
        for _ in range(20):                         # same voltage again: no optical change
            a = perf_counter(); self.heater("2_2", self.vset["A"]["2_2"]); t.append(perf_counter() - a)
        res["heater_call"] = stats_ms(t)
        if mode == "auto":
            mode = "scan" if res.get("scan_works") else "serial"
        if mode == "scan" and not self._scan_bufs:
            raise SystemExit(f"--laser-mode scan, but a_out_scan failed: {res}")
        self.mode = mode
        res["mode"] = mode
        return res

    # readout ----------------------------------------------------------------
    def start_daq(self, nets: list[str], clock: Clock) -> dict:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType
        from nidaqmx.stream_readers import AnalogMultiChannelReader
        ni = self.netlist.board_dict["NI"]
        task = nidaqmx.Task()
        for i, net in enumerate(nets):
            task.ai_channels.add_ai_voltage_chan(f"{ni.board_num}/ai{self.netlist.pins_dict[net].chnl}",
                                                 name_to_assign_to_channel=f"{net}_{i}",
                                                 min_val=ni.min_val, max_val=ni.max_val,
                                                 terminal_config=ni.terminal_config)
        rate = CONVERT_RATE / len(nets)
        task.timing.cfg_samp_clk_timing(rate=rate, sample_mode=AcquisitionType.CONTINUOUS,
                                        samps_per_chan=int(rate * 120))
        task.timing.ai_conv_rate = CONVERT_RATE
        task.in_stream.input_buf_size = int(rate * 120)                 # 2 min of headroom
        reader = AnalogMultiChannelReader(task.in_stream)
        self._task, self._chunks, self._stop = task, [], threading.Event()

        def pump():
            while True:
                stopping = self._stop.is_set()
                avail = task.in_stream.avail_samp_per_chan
                if avail:
                    buf = np.empty((len(nets), avail))
                    reader.read_many_sample(buf, number_of_samples_per_channel=avail, timeout=10.0)
                    self._chunks.append(buf)
                if stopping:
                    return
                sleep(0.2)
        task.start()
        t_start = clock.now()
        self._thread = threading.Thread(target=pump, daemon=True)
        self._thread.start()
        return {"scan_rate": float(task.timing.samp_clk_rate), "conv_rate": float(task.timing.ai_conv_rate),
                "daq_start_pc": t_start}

    def stop_daq(self, log) -> tuple[np.ndarray, dict]:
        self._stop.set()
        self._thread.join()
        self._task.close()
        return np.concatenate(self._chunks, axis=1), {}


def heater_settings(routes: dict, volt) -> dict[str, dict[str, float]]:
    '''{setting: {mzi: V}} for every heater; only 2_2 and 4_2 may differ between A-D.'''
    out = {}
    for s in "ABCD":
        v = {m: volt(m, "B") for m in routes["fixed_bar"]}
        v.update({m: volt(m, st) for m, st in zip(routes["mzis"], routes["settings"][s]["states"])})
        out[s] = {m: float(np.clip(x, 0.0, HEATER_VMAX)) for m, x in v.items()}
    for m in out["A"]:
        if m not in LINK_MZI:
            assert len({out[s][m] for s in "ABCD"}) == 1, f"{m} differs between settings"
    return out


class SimHardware:
    '''Dry run: emulated call times; receiver signals synthesized afterwards from the log.'''
    sim = True
    LEVEL_NA = {(0, 2): 616, (1, 2): 623, (0, 3): 678, (1, 3): 687, (2, 4): 1386, (3, 4): 1569}   # 2026-10-19
    LEAK = 0.01            # light from the sender a heater does not select
    DARK_V, NOISE_NA = 0.55, 0.3
    PPM, OFFSET_S = 40e-6, 0.0037

    def __init__(self, routes: dict, args):
        placeholder = {"B": 0.0, "X": 1.6, "S": 0.8}       # dry run: no heater calibration needed
        self.vset = heater_settings(routes, lambda m, s: placeholder[s])
        self.mode = "serial"
        self.a = args
        self.rng = np.random.default_rng(args.sim_seed)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def all_off(self) -> None:
        pass

    def _cost(self, base_ms: float) -> None:
        r = self.rng.random()
        ms = base_ms * (1 + 0.1 * self.rng.standard_normal())
        if r < self.a.sim_stall:
            ms += self.rng.uniform(1.0, 5.0)
        elif r < self.a.sim_stall + 0.001:
            ms += self.rng.uniform(10.0, 60.0)
        end = perf_counter() + max(ms, 0.05) / 1e3
        while perf_counter() < end:
            pass

    def v_out(self, core: int, v: float) -> None:
        self._cost(self.a.sim_cmd_ms)

    def scan_out(self, pattern: int) -> None:
        self._cost(self.a.sim_cmd_ms)

    def heater(self, mzi: str, v: float) -> None:
        self._cost(3.0)

    def probe(self, routes: dict, mode: str, n: int = 0) -> dict:
        self.mode = "scan" if mode in ("auto", "scan") else "serial"
        return {"mode": self.mode, "note": "dry run: no probe, call times emulated "
                f"({self.a.sim_cmd_ms} ms, stalls {self.a.sim_stall:.0%})"}

    def start_daq(self, nets: list[str], clock: Clock) -> dict:
        self.nets = nets
        rate = CONVERT_RATE / len(nets)
        return {"scan_rate": rate, "conv_rate": CONVERT_RATE, "daq_start_pc": clock.now()}

    def stop_daq(self, log) -> tuple[np.ndarray, dict]:
        '''Synthesizes every scan entry from the commands actually executed.'''
        L = log.arrays()
        rate = CONVERT_RATE / len(self.nets)
        t_daq0 = self.daq_start = log.daq_info["daq_start_pc"]
        dur = L["cmd_t1"].max() - t_daq0 + 0.05
        nscan = int(dur * rate)
        pc = lambda t_daq: t_daq0 + self.OFFSET_S + t_daq / (1 + self.PPM)      # DAQ starts a bit late
        # laser state per core as step functions of PC time; injected dropped pulses
        lit = L["cmd_kind"] != K_HEATER
        on_t = {c: [] for c in SENDERS}
        tx, rx = [], []
        for i in np.flatnonzero(lit):
            for c in SENDERS:
                v = L["cmd_vals"][i, c]
                if np.isnan(v):
                    continue
                emit = v > 0
                if emit and L["cmd_kind"][i] == K_PULSE:
                    if self.rng.random() < self.a.sim_drop_tx:
                        emit = False; tx.append((i, c))
                    for j in DESTS[c]:
                        if self.rng.random() < self.a.sim_drop_rx:
                            rx.append((i, c, j))
                on_t[c].append((L["cmd_t1"][i] + 50e-6, emit, i))
        h_t = {m: [(-1.0, self.vset["A"][m])] for m in LINK_MZI}
        for i in np.flatnonzero(L["cmd_kind"] == K_HEATER):
            h_t[LINK_MZI[int(L["cmd_heater"][i])]].append((L["cmd_t1"][i] + 36e-6, L["cmd_hv"][i]))
        rx_set = {}
        for i, c, j in rx:
            rx_set.setdefault((c, j), []).append(i)
        y = np.zeros((len(self.nets), nscan))
        blips = []
        drift = lambda t: 1 + 0.03 * np.sin(2 * np.pi * t / 400.0)
        for e, net in enumerate(self.nets):
            t_pc = pc(np.arange(nscan) / rate + e / CONVERT_RATE)
            if net not in log.receivers:
                y[e] = self.DARK_V + 1e-4 * self.rng.standard_normal(nscan)
                continue
            j = 2 + log.receivers.index(net)
            I = np.zeros(nscan)
            for c in SENDERS:
                if j not in DESTS[c]:
                    continue
                ev = on_t[c]
                if not ev:
                    continue
                te = np.array([x[0] for x in ev]); st = np.array([x[1] for x in ev]); ci = np.array([x[2] for x in ev])
                k = np.searchsorted(te, t_pc, side="right") - 1
                on = (k >= 0) & st[np.maximum(k, 0)]
                for i in rx_set.get((c, j), []):              # this receiver misses this pulse
                    on &= ci[np.maximum(k, 0)] != i
                m = LINK_MZI[0] if c < 2 else LINK_MZI[1]
                ht = np.array([x[0] for x in h_t[m]]); hv = np.array([x[1] for x in h_t[m]])
                hcur = hv[np.searchsorted(ht, t_pc, side="right") - 1]
                want = self.vset["B" if c in (0, 2) else "A"][m]  # heater value that selects c
                gain = np.where(np.isclose(hcur, want), 1.0, self.LEAK)
                I += on * gain * self.LEVEL_NA[(c, j)] * drift(t_pc)
            nb = self.rng.poisson(self.a.sim_blips * (t_pc[-1] - t_pc[0]))      # stray light (false detections)
            for tb in self.rng.uniform(t_pc[0], t_pc[-1], nb):
                I += 1000.0 * ((t_pc >= tb) & (t_pc < tb + 1e-3))
                blips.append((tb, j))
            I += self.NOISE_NA * self.rng.standard_normal(nscan)
            y[e] = self.DARK_V - I * 1e-9 * TRANSIMPEDANCE
        return y, {"sim_tx_drop": np.array(tx, int).reshape(-1, 2), "sim_rx_drop": np.array(rx, int).reshape(-1, 3),
                   "sim_blips": np.array(blips, float).reshape(-1, 2)}


# ----------------------------------------------------------------------------- run
class CommandLog:
    '''Executes every command at its scheduled time and records when it really ran.'''

    def __init__(self, clock: Clock, hw):
        self.clock, self.hw = clock, hw
        self.rows = []                   # kind, sched, late_raw, t0, t1, window, heater, hv, vals[4]
        self.state = [0.0] * 4           # laser voltages C1..C4
        self.heaters = {}

    def _run(self, kind, sched, fn, window=-1, heater=-1, hv=np.nan, vals=None, orig=None):
        self.clock.wait(sched)
        t0 = self.clock.now()
        fn()
        t1 = self.clock.now()
        v = [np.nan] * 4 if vals is None else vals
        self.rows.append((kind, sched, t0 - (sched if orig is None else orig), t0, t1, window, heater, hv, *v))
        return len(self.rows) - 1

    def lasers(self, kind, sched, change: dict, orig=None) -> int:
        '''Sets the lasers in `change` ({core: V}); one call in scan mode, one channel in serial mode.'''
        new = list(self.state)
        for c, v in change.items():
            new[c] = v
        vals = [new[c] if c in change else np.nan for c in range(4)]
        if self.hw.mode == "scan":
            pat = sum(1 << c for c in range(4) if new[c] > 0)
            fn = lambda: self.hw.scan_out(pat)
        else:
            assert len(change) == 1
            (c, v), = change.items()
            fn = lambda: self.hw.v_out(c, v)
        self.state = new
        return self._run(kind, sched, fn, vals=vals, orig=orig)

    def heater(self, sched, mzi, v, window) -> int | None:
        if self.heaters.get(mzi) == v:
            return None
        self.heaters[mzi] = v
        return self._run(K_HEATER, sched, lambda: self.hw.heater(mzi, v), window=window,
                         heater=LINK_MZI.index(mzi), hv=v)

    def arrays(self) -> dict:
        r = np.array(self.rows, float).reshape(-1, 12)
        return {"cmd_kind": r[:, 0].astype(int), "cmd_sched": r[:, 1], "cmd_late": r[:, 2], "cmd_t0": r[:, 3],
                "cmd_t1": r[:, 4], "cmd_window": r[:, 5].astype(int), "cmd_heater": r[:, 6].astype(int),
                "cmd_hv": r[:, 7], "cmd_vals": r[:, 8:12]}


class Controller:
    '''export_replay.py's previous-window rule, applied live to the counts of the last window.'''

    def __init__(self):
        self.A, _ = feasible_assignments()
        self.Ai = np.where(self.A < 0, N_CORE, self.A)
        self.inp, self.hid = 1, 3                  # setting A

    def decide(self, W: np.ndarray) -> tuple[str, int, int]:
        '''Returns (setting, input-link sender or -1, hidden-link sender or -1).'''
        Wp = np.vstack([W.astype(float), np.zeros((1, N_CORE))])
        a = self.A[int(sum(Wp[self.Ai[:, j], j] for j in range(N_CORE)).argmax())]
        new_inp = int(a[2]) if a[2] in (0, 1) and W[a[2], 2] + W[a[2], 3] > 0 else None
        new_hid = int(a[4]) if a[4] in (2, 3) and W[a[4], 4] > 0 else None     # empty link: hold heater
        self.inp = self.inp if new_inp is None else new_inp
        self.hid = self.hid if new_hid is None else new_hid
        return SETTING[(self.inp, self.hid)], -1 if new_inp is None else new_inp, -1 if new_hid is None else new_hid


class Traffic:
    '''Per-step packet counts and per-core spiking neurons from the software spike record.'''

    def __init__(self, sp):
        self.layers = [sp["inp"], sp["h1"], sp["h2"], sp["out"]]
        self.m = layer_mapping([l.shape[2] for l in self.layers])
        self.dst = [sorted(set(self.m[L + 1].tolist())) for L in range(3)]

    def packets(self, digit: int, t: int) -> np.ndarray:
        P = np.zeros((N_CORE, N_CORE), np.int32)
        for L in range(3):
            spk = self.layers[L][digit, t]
            for c in range(N_CORE):
                cnt = int(spk[self.m[L] == c].sum())
                for j in self.dst[L]:
                    if j != c:
                        P[c, j] += cnt
        return P

    def neurons(self, digit: int, t: int, core: int) -> tuple[int, np.ndarray]:
        '''(layer, ascending neuron indices) that core `core` sends optically in this step.'''
        L = 0 if core < 2 else 1
        return L, np.flatnonzero(self.layers[L][digit, t].astype(bool) & (self.m[L] == core))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=0, help="first test digit")
    ap.add_argument("--digits", type=int, default=1)
    ap.add_argument("--on-ms", type=float, default=1.0)
    ap.add_argument("--off-ms", type=float, default=1.0)
    ap.add_argument("--gap-ms", type=float, default=8.0, help="dark gap at window boundaries (heater changes)")
    ap.add_argument("--slip-ms", type=float, default=0.5, help="shift the schedule when a pulse starts this late")
    ap.add_argument("--recal-every", type=int, default=10, help="calibration block every N digits")
    ap.add_argument("--monitors", action="store_true",
                    help="also stream the laser inputs PD_2_0..PD_5_0 (receivers 2.5 kS/s, monitors 625 S/s)")
    ap.add_argument("--laser-mode", choices=["auto", "scan", "serial"], default="auto")
    ap.add_argument("--dry-run", action="store_true", help="no hardware; simulated detections")
    ap.add_argument("--sim-cmd-ms", type=float, default=0.9, help="dry run: time per laser call")
    ap.add_argument("--sim-stall", type=float, default=0.03, help="dry run: fraction of calls stalled 1-5 ms")
    ap.add_argument("--sim-drop-tx", type=float, default=0.002, help="dry run: pulses the laser drops")
    ap.add_argument("--sim-drop-rx", type=float, default=0.001, help="dry run: pulses one receiver misses")
    ap.add_argument("--sim-blips", type=float, default=0.5, help="dry run: stray 1 ms light blips per s per receiver")
    ap.add_argument("--sim-seed", type=int, default=1)
    ap.add_argument("--no-analysis", action="store_true")
    args = ap.parse_args()

    routes = json.loads((ROOT / "sim" / "routes_rows2-6.json").read_text())
    spk_path = ROOT / "sim" / "spikes_test.npz"
    if not spk_path.exists():
        raise SystemExit(f"{spk_path} missing (not in git): copy it from the Mac or rerun sim/train_snn.py")
    sp = np.load(spk_path)
    digits = list(range(args.start, args.start + args.digits))
    sp = {k: sp[k][args.start:args.start + args.digits] for k in ("inp", "h1", "h2", "out", "labels")}
    traffic = Traffic(sp)
    ctrl = Controller()
    on, off, gap = args.on_ms / 1e3, args.off_ms / 1e3, args.gap_ms / 1e3
    receivers = routes["receiver_pd"][2:5]                         # C3, C4, C5
    # with monitors: (C3 C4 C5 M_i) for each laser input M_i, so receivers stay evenly spaced
    nets = [n for m in routes["input_pd"][:4] for n in receivers + [m]] if args.monitors else receivers

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "runs" / f"spike_replay_live_{stamp}{'_dry' if args.dry_run else ''}"
    out.parent.mkdir(exist_ok=True)
    hw = SimHardware(routes, args) if args.dry_run else Hardware(routes)
    vset = hw.vset
    clock = Clock()
    log = CommandLog(clock, hw)
    log.receivers = receivers
    slots, pulses, windows, cal, cal_dark, steps_P = [], [], [], [], [], []
    meta = {"stamp": stamp, "dry_run": args.dry_run, "args": vars(args), "digits": digits,
            "labels": sp["labels"].tolist(), "receivers": receivers, "scan_nets": nets,
            "senders": list(SENDERS), "dests": {str(k): list(v) for k, v in DESTS.items()},
            "heater_V": vset, "fault_ms": 3.0, "late_ms": 2.0}
    extra = {}
    interrupted = False

    with hw:
        try:
            probe = hw.probe(routes, args.laser_mode)
            meta["probe"] = probe
            report(f"spike_replay_live {stamp}{' (DRY RUN)' if args.dry_run else ''}: digits {digits[0]}-{digits[-1]}, "
                   f"pulse {args.on_ms:g} ms on / {args.off_ms:g} ms off, gap {args.gap_ms:g} ms, "
                   f"recalibration every {args.recal_every} digits, streaming {len(nets)} entries/scan")
            if not args.dry_run:
                report(f"  USB-3114: v_out call median {probe['v_out_call']['median']:.2f} ms (p99 "
                       f"{probe['v_out_call']['p99']:.2f}); a_out_scan "
                       + (f"{probe.get('scan_option')} median {probe['scan_call']['median']:.2f} ms, optical check "
                          f"ratio both-in-one-call / alone {probe['scan_optical_nA']['ratio_vs_alone']}"
                          if probe.get("scan_option") else f"not available: {probe}")
                       + f"; heater call median {probe['heater_call']['median']:.2f} ms")
            dt_cmd = 0.0
            if hw.mode == "serial":
                dt_cmd = probe["v_out_call"]["median"] / 1e3 if "v_out_call" in probe else args.sim_cmd_ms / 1e3
                report(f"  laser mode SERIAL: one v_out per channel ({dt_cmd * 1e3:.2f} ms each); link 2 is delayed "
                       f"by one call and on/off are stretched to >= 1.2 x calls per edge")
            else:
                report("  laser mode SCAN: both links' lasers change in one USB-3114 call")
            meta["laser_mode"], meta["dt_cmd_s"] = hw.mode, dt_cmd
            for m, v in vset["A"].items():
                hw.heater(m, v)
                log.heaters[m] = v if m in LINK_MZI else None
            hw.all_off()
            sleep(0.2 if args.dry_run else 2.0)
            daq = hw.start_daq(nets, clock)
            log.daq_info = daq
            meta.update(daq)
            sleep(0.1)

            def calibrate(block: int) -> float:
                '''Dark + each sender alone in setting B (C1, C3) and A (C2, C4).'''
                t = clock.now() + 0.005
                for cfg, senders in (("B", (0, 2)), ("A", (1, 3))):
                    for m in LINK_MZI:
                        log.heater(t, m, vset[cfg][m], -1)
                    t = max(t, clock.now()) + CAL_SETTLE_S
                    cal_dark.append((block, t, t + CAL_ON_S)); t += CAL_ON_S
                    for s in senders:
                        i_on = log.lasers(K_CAL, t, {s: LASER_V})
                        i_off = log.lasers(K_CAL, t + CAL_ON_S, {s: 0.0})
                        cal.append((block, s, t, t + CAL_ON_S, i_on, i_off))
                        t += 2 * CAL_ON_S
                clock.wait(t)
                return t

            setting, li, lh = "A", 1, 3                     # first window: no history
            counts = np.zeros((N_CORE, N_CORE), np.int64)
            t_sched = 0.0
            n_steps = args.digits * STEPS_PER_DIGIT
            t_run0 = clock.now()
            for k in range(n_steps):
                d_i, t_d = divmod(k, STEPS_PER_DIGIT)
                w = k // WINDOW_BINS
                if t_d == 0 and d_i % args.recal_every == 0:
                    t_sched = calibrate(d_i // args.recal_every)
                    if k == 0:
                        t_sched += gap
                        windows.append((w, setting, li, lh, *counts.ravel(), 0.0, 0.0, 0, t_sched - gap, t_sched))
                if k % WINDOW_BINS == 0 and k > 0:          # window boundary: decide, heaters in the dark gap
                    t_gap = max(t_sched, clock.now())
                    a = perf_counter()
                    setting, li, lh = ctrl.decide(counts)
                    dec_ms = (perf_counter() - a) * 1e3
                    clock.wait(t_gap)
                    n_h = 0
                    for m in LINK_MZI:
                        n_h += log.heater(t_gap, m, vset[setting][m], w) is not None
                    heater_ms = (clock.now() - t_gap) * 1e3
                    windows.append((w, setting, li, lh, *counts.ravel(), dec_ms, heater_ms, n_h, t_gap, t_gap + gap))
                    counts[:] = 0
                    t_sched = t_gap + gap
                links = [(s, *traffic.neurons(d_i, t_d, s)) for s in (li, lh) if s >= 0]
                npulse = max([len(n) for _, _, n in links], default=0)
                for i in range(npulse):
                    who = [(s, L, n[i]) for s, L, n in links if i < len(n)]
                    calls = 1 if hw.mode == "scan" else len(who)
                    on_e = max(on, 1.2 * calls * dt_cmd) if hw.mode == "serial" else on
                    off_e = max(off, 1.2 * calls * dt_cmd) if hw.mode == "serial" else off
                    orig = t_sched
                    if clock.now() - t_sched > args.slip_ms / 1e3:
                        t_sched = clock.now()                   # schedule slips; lateness kept vs orig
                    sl = len(slots)
                    if hw.mode == "scan":
                        c_on = log.lasers(K_PULSE, t_sched, {s: LASER_V for s, _, _ in who}, orig=orig)
                        c_off = log.lasers(K_PULSE, t_sched + on_e, {s: 0.0 for s, _, _ in who})
                        for s, L, n in who:
                            pulses.append((sl, k, s, L, n, i, c_on, c_off, 0.0))
                    else:
                        ons = [log.lasers(K_PULSE, t_sched + l * dt_cmd, {s: LASER_V}, orig=orig + l * dt_cmd)
                               for l, (s, _, _) in enumerate(who)]
                        offs = [log.lasers(K_PULSE, t_sched + l * dt_cmd + on_e, {s: 0.0})
                                for l, (s, _, _) in enumerate(who)]
                        for l, (s, L, n) in enumerate(who):
                            pulses.append((sl, k, s, L, n, i, ons[l], offs[l], l * dt_cmd))
                    slots.append((k, w, i, t_sched, orig, on_e, off_e))
                    t_sched += on_e + off_e
                P = traffic.packets(d_i, t_d)                   # the electronic side knows what it sent
                counts += P
                steps_P.append(P)
                if t_d == STEPS_PER_DIGIT - 1:
                    el = clock.now() - t_run0
                    if d_i in (0, 9) or (d_i + 1) % 10 == 0:
                        report(f"  digit {d_i + 1}/{args.digits} done, {el:.1f} s elapsed, "
                               f"~{el / (d_i + 1) * (args.digits - d_i - 1):.0f} s left")
            calibrate(args.digits // args.recal_every + (args.digits % args.recal_every > 0))
            sleep(0.05)
        except KeyboardInterrupt:
            interrupted = True
            report("Interrupted: saving what was run (the analysis uses the completed steps).")
        finally:
            hw.all_off()
            if "daq_start_pc" in meta:
                y, extra = hw.stop_daq(log)
            else:
                y = np.zeros((len(nets), 0))

    meta["interrupted"] = interrupted
    meta["steps_done"] = len(steps_P)
    A = log.arrays()
    win = np.array(windows, dtype=object).reshape(-1, 4 + 25 + 5)
    np.savez_compressed(
        out.with_suffix(".npz"), meta=json.dumps(meta, default=float), y=y.astype(np.float32), **A,
        slot_step=np.array([s[0] for s in slots], int), slot_window=np.array([s[1] for s in slots], int),
        slot_i=np.array([s[2] for s in slots], int), slot_sched=np.array([s[3] for s in slots]),
        slot_orig=np.array([s[4] for s in slots]), slot_on=np.array([s[5] for s in slots]),
        slot_off=np.array([s[6] for s in slots]),
        p_slot=np.array([p[0] for p in pulses], int), p_step=np.array([p[1] for p in pulses], int),
        p_sender=np.array([p[2] for p in pulses], int), p_layer=np.array([p[3] for p in pulses], int),
        p_neuron=np.array([p[4] for p in pulses], int), p_i=np.array([p[5] for p in pulses], int),
        p_cmd_on=np.array([p[6] for p in pulses], int), p_cmd_off=np.array([p[7] for p in pulses], int),
        p_offset=np.array([p[8] for p in pulses]),
        w_index=win[:, 0].astype(int), w_setting=win[:, 1].astype(str), w_inp=win[:, 2].astype(int),
        w_hid=win[:, 3].astype(int), w_counts=win[:, 4:29].astype(int).reshape(-1, 5, 5),
        w_decision_ms=win[:, 29].astype(float), w_heater_ms=win[:, 30].astype(float),
        w_heater_cmds=win[:, 31].astype(int), w_gap_start=win[:, 32].astype(float), w_gap_end=win[:, 33].astype(float),
        step_P=np.array(steps_P, int).reshape(-1, 5, 5),
        cal=np.array(cal, float).reshape(-1, 6), cal_dark=np.array(cal_dark, float).reshape(-1, 3), **extra)
    lat = A["cmd_late"][A["cmd_kind"] == K_PULSE] * 1e3
    if lat.size:
        report(f"  run: {len(steps_P)} steps, {len(slots)} pulse slots, {len(pulses)} laser pulses, {len(A['cmd_kind'])} "
               f"commands; pulse-command lateness median {np.median(lat):.2f} ms, {(lat > 2).sum()} > 2 ms, "
               f"{(lat > 3).sum()} > 3 ms, max {lat.max():.1f} ms; schedule slips "
               f"{int((np.array([s[3] - s[4] for s in slots]) > 0).sum())}")
    if not args.no_analysis and len(steps_P):
        from replay_analysis import analyze_live
        res, lines = analyze_live(out.with_suffix(".npz"))
        for line in lines:
            report(line)
        out.with_suffix(".json").write_text(json.dumps(res, indent=2, default=float))
    out.with_suffix(".txt").write_text("\n".join(summary) + "\n")
    print(f"Saved {out.name}.npz/.json/.txt in {out.parent}")


if __name__ == "__main__":
    main()
