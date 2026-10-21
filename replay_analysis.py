'''Offline analysis of a spike_replay run (on the Mac, no hardware needed).

    A spike_replay_live.py run (.npz with per-pulse tables) is analysed by analyze_live()
    instead: see its docstring and spike_replay_live.py.

    Recomputes from the saved files:
      1. error counts overall, and split by software timing: a slot is a "timing fault" if
         any of its commands (laser pulse, or a heater change just before it) ran more than
         FAULT_MS late. The command list is rebuilt exactly as spike_replay.py builds it and
         matched one-for-one to the recorded lateness (late_ms).
      2. network accuracy when spikes cross an optical link only in slots where the chip
         actually detected them (numpy LIF identical to the trained snnTorch network);
         ALL detections are used here, including timing-faulted slots
      3. bus delay with and without the lanes, from the measured delivery

    Inputs: runs/spike_replay_<stamp>.npz, sim/replay_excerpt.json, sim/routes_rows2-6.json,
            sim/snn_weights.npz and sim/spikes_test.npz (not in git: regenerate with
            sim/train_snn.py, which reproduces the network exactly).
    Writes runs/spike_replay_<stamp>_analysis.{json,txt}.

        python replay_analysis.py runs/spike_replay_20261019_235342.npz
'''
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FAULT_MS = 3.0
T0, SLOT_S, PULSE_S = 0.5, 0.015, 0.006        # spike_replay.py defaults used for the run
STEPS_PER_DIGIT = 150


def rebuild_events(rp: dict, routes: dict) -> list[tuple[float, int, int]]:
    '''(time, kind, slot) for every command, in the order spike_replay.py executed them.'''
    states = {s: dict(zip(routes["mzis"], routes["settings"][s]["states"])) for s in "ABCD"}
    W, wsteps, ev = rp["windows"], rp["window_steps"], []
    for k, st in enumerate(rp["steps"]):
        w, t_k = W[k // wsteps], T0 + k * SLOT_S
        for s in w["links"]:
            if st["fired"][int(s)]:
                ev += [(t_k, 1, k), (t_k + PULSE_S, 0, k)]
        if k % wsteps == wsteps - 1 and k // wsteps + 1 < len(W):
            a, b = states[w["setting"]], states[W[k // wsteps + 1]["setting"]]
            ev += [(t_k + PULSE_S + 0.001, 2, k)] * sum(a[m] != b[m] for m in a)
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev


def accuracy(sender, det, labels, n_digits) -> tuple[float, float, int]:
    '''Software accuracy vs accuracy with optical links gated by the chip's detections.'''
    w = np.load(ROOT / "sim" / "snn_weights.npz"); sp = np.load(ROOT / "sim" / "spikes_test.npz")
    Wt = [(w["fc1.weight"], w["fc1.bias"]), (w["fc2.weight"], w["fc2.bias"]), (w["fc3.weight"], w["fc3.bias"])]
    beta, thr = float(w["beta"]), float(w["threshold"])
    row = (np.arange(2312) % 1156) // 34
    in_core, h1_core = np.where(row < 17, 0, 1), np.where(np.arange(128) < 64, 2, 3)

    def run(x, gate):
        mem = [np.zeros(b.shape) for _, b in Wt]; spk = [np.zeros(b.shape) for _, b in Wt]; cnt = np.zeros(10)
        for t in range(x.shape[0]):
            xi = x[t].astype(float); cur1 = np.zeros(128)
            for c in (2, 3):
                m = h1_core == c
                cur1[m] = Wt[0][0][m] @ (xi * np.array([gate(t, s, c) for s in (0, 1)])[in_core]) + Wt[0][1][m]
            mem[0] = beta * mem[0] + cur1 - spk[0] * thr; spk[0] = (mem[0] > thr).astype(float)
            h1in = spk[0] * np.array([gate(t, s, 4) for s in (2, 3)])[h1_core - 2]
            mem[1] = beta * mem[1] + Wt[1][0] @ h1in + Wt[1][1] - spk[1] * thr; spk[1] = (mem[1] > thr).astype(float)
            mem[2] = beta * mem[2] + Wt[2][0] @ spk[1] + Wt[2][1] - spk[2] * thr; spk[2] = (mem[2] > thr).astype(float)
            cnt += spk[2]
        return int(cnt.argmax())

    soft, hw = [], []
    for i in range(n_digits):
        off = i * STEPS_PER_DIGIT
        soft.append(run(sp["inp"][i], lambda k, s, c: True))
        hw.append(run(sp["inp"][i], lambda k, s, c: bool(det[off + k, c]) if sender[off + k, c] == s else True))
    soft, hw = np.array(soft), np.array(hw)
    return float(np.mean(soft == labels)), float(np.mean(hw == labels)), int((soft != hw).sum())


def bus_delays(P, sender, det) -> dict:
    '''99th-percentile and mean waiting time (ms) of one shared bus, without / with lanes.'''
    opt = np.array([sum(P[k, sender[k, j], j] for j in range(5) if sender[k, j] >= 0 and det[k, j]) for k in range(len(P))])
    return bus_queue(P.sum((1, 2)), opt)


def bus_queue(total, opt) -> dict:
    '''Bus waiting time per 2 ms step: all packets on the bus vs only those not delivered by
        light (undelivered optical packets go back on the bus).'''
    lam = total.mean()
    def q(arr, cap):
        x, out = 0.0, np.zeros(len(arr))
        for k, a in enumerate(arr):
            x = max(0.0, x + a - cap); out[k] = x
        d = out / cap * 2.0
        return {"mean_ms": float(d.mean()), "p99_ms": float(np.percentile(d, 99))}
    return {"offload_delivered": float(opt.sum() / total.sum()), "mean_load_pkts_per_ms": float(lam / 2),
            "peak_over_mean": float(total.max() / lam),
            **{f"{f}x": {"bus_only": q(total, f * lam), "with_lanes": q(total - opt, f * lam)} for f in (0.8, 1.0, 1.2, 1.5, 2.0)}}


# ----------------------------------------------------------------------------- live replay
# analysis of spike_replay_live.py runs: one pulse per spike, live controller
LIVE_MIN_SAMPLES = 2
LIVE_DESTS = {0: (2, 3), 1: (2, 3), 2: (4,), 3: (4,)}


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    from scipy.stats import beta
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def snn_masked(x, drop_in, drop_h1):
    '''One digit through the numpy LIF; drop_in = {(t, input neuron, dest core 2|3)} and
        drop_h1 = {(t, h1 neuron)} are spikes removed (not delivered). Returns (prediction,
        h1 spikes). Same equations as accuracy() above.'''
    w = _weights()
    (W1, b1), (W2, b2), (W3, b3), beta, thr = w
    row = (np.arange(2312) % 1156) // 34
    half = np.arange(128) < 64                              # C3: h1 0-63, C4: 64-127
    mem = [np.zeros(128), np.zeros(64), np.zeros(10)]; spk = [np.zeros(128), np.zeros(64), np.zeros(10)]
    cnt, h1 = np.zeros(10), np.zeros((x.shape[0], 128), np.uint8)
    din, dh = {}, {}
    for t, n, c in drop_in:
        din.setdefault((t, c), []).append(n)
    for t, n in drop_h1:
        dh.setdefault(t, []).append(n)
    for t in range(x.shape[0]):
        xi = x[t].astype(float); cur1 = np.empty(128)
        for c, m in ((2, half), (3, ~half)):
            xc = xi.copy(); xc[din.get((t, c), [])] = 0.0
            cur1[m] = W1[m] @ xc + b1[m]
        mem[0] = beta * mem[0] + cur1 - spk[0] * thr; spk[0] = (mem[0] > thr).astype(float)
        h1[t] = spk[0]
        h1in = spk[0].copy(); h1in[dh.get(t, [])] = 0.0
        mem[1] = beta * mem[1] + W2 @ h1in + b2 - spk[1] * thr; spk[1] = (mem[1] > thr).astype(float)
        mem[2] = beta * mem[2] + W3 @ spk[1] + b3 - spk[2] * thr; spk[2] = (mem[2] > thr).astype(float)
        cnt += spk[2]
    return int(cnt.argmax()), h1


_W = None


def _weights():
    global _W
    if _W is None:
        w = np.load(ROOT / "sim" / "snn_weights.npz")
        _W = ((w["fc1.weight"], w["fc1.bias"]), (w["fc2.weight"], w["fc2.bias"]), (w["fc3.weight"], w["fc3.bias"]),
              float(w["beta"]), float(w["threshold"]))
    return _W


def export_schedule(start: int, n: int) -> dict:
    '''sim/export_replay.py's schedule for these digits, run in a temporary folder.'''
    import runpy, shutil, sys, tempfile, os
    cwd, argv = os.getcwd(), sys.argv
    sys.path.insert(0, str(ROOT / "sim"))
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(ROOT / "sim" / "spikes_test.npz", tmp)
        try:
            os.chdir(tmp)
            sys.argv = ["export_replay.py", "--start", str(start), "--samples", str(n)]
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(ROOT / "sim" / "export_replay.py"), run_name="__main__")
            return json.loads(Path(tmp, "replay_excerpt.json").read_text())
        finally:
            os.chdir(cwd); sys.argv = argv


def _edges(t, x):
    '''Rising and falling 50% crossings of the longest bright run in (t, x), interpolated.'''
    lo, hi = np.percentile(x, 10), np.percentile(x, 90)
    th = (lo + hi) / 2
    a = np.r_[False, x > th, False].astype(int)
    st, en = np.flatnonzero(np.diff(a) == 1), np.flatnonzero(np.diff(a) == -1)     # run [st, en)
    if not len(st) or hi - lo <= 0:
        return None
    r = int(np.argmax(en - st)); s, e = st[r], en[r]
    if s == 0 or e >= len(x) or e - s < 5:
        return None
    rise = t[s - 1] + (th - x[s - 1]) / (x[s] - x[s - 1]) * (t[s] - t[s - 1])
    fall = t[e - 1] + (x[e - 1] - th) / (x[e - 1] - x[e]) * (t[e] - t[e - 1])
    return rise, fall


def analyze_live(path: Path, fault_ms: float | None = None) -> tuple[dict, list[str]]:
    '''Everything from the saved file of a spike_replay_live.py run (no hardware needed).'''
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    fault_ms = meta["fault_ms"] if fault_ms is None else fault_ms
    args = meta["args"]
    on_s, off_s = args["on_ms"] / 1e3, args["off_ms"] / 1e3
    recv = meta["receivers"]                                  # nets of C3, C4, C5
    nets, rate, conv = meta["scan_nets"], meta["scan_rate"], meta["conv_rate"]
    y = d["y"].astype(float)
    lines = []

    # per receiver: sample times (DAQ clock) and light signal (light lowers the voltage)
    ts, sig = {}, {}
    for j, net in zip((2, 3, 4), recv):
        ent = [e for e, n in enumerate(nets) if n == net]
        tt = np.concatenate([np.arange(y.shape[1]) / rate + e / conv for e in ent])
        vv = np.concatenate([-y[e] for e in ent])
        o = np.argsort(tt, kind="stable"); ts[j], sig[j] = tt[o], vv[o]

    # 1. DAQ clock vs PC clock, from the calibration pulses' edges (done block by block)
    cal, cal_dark, t1 = d["cal"], d["cal_dark"], d["cmd_t1"]
    pc_pts, daq_pts = [], []
    a, b = -meta["daq_start_pc"], 1.0
    for blk in np.unique(cal[:, 0]):
        for blk_, s, son, soff, ion, ioff in cal[cal[:, 0] == blk]:
            pon, poff = t1[int(ion)], t1[int(ioff)]
            for j in LIVE_DESTS[int(s)]:
                lo_t, hi_t = a + b * pon - 0.02, a + b * poff + 0.02
                m = (ts[j] >= lo_t) & (ts[j] < hi_t)
                e = _edges(ts[j][m], sig[j][m]) if m.sum() > 20 else None
                if e:
                    pc_pts += [pon, poff]; daq_pts += list(e)
        if len(pc_pts) >= 2:
            P_ = np.array(pc_pts)
            if np.ptp(P_) > 5.0:
                b, a = np.polyfit(P_, daq_pts, 1)
            else:
                a = float(np.mean(np.array(daq_pts) - P_)); b = 1.0
    pc_pts, daq_pts = np.array(pc_pts), np.array(daq_pts)
    resid = daq_pts - (a + b * pc_pts)
    to_daq = lambda t_pc: a + b * np.asarray(t_pc)
    lines.append(f"clock alignment: {len(pc_pts)} calibration edges, DAQ = {a * 1e3:+.2f} ms + PC x (1 {(b - 1) * 1e6:+.1f} ppm), "
                 f"residual rms {1e3 * resid.std():.3f} ms, max {1e3 * np.abs(resid).max():.3f} ms")

    # 2. dark / spike levels per calibration block -> thresholds (half the smallest spike level)
    def med(j, t0, t1_):
        m = (ts[j] >= to_daq(t0)) & (ts[j] < to_daq(t1_))
        return float(np.median(sig[j][m])) if m.any() else np.nan
    blocks = np.unique(cal[:, 0]).astype(int)
    dark = {(blk, j): np.nanmean([med(j, s0, s1) for bb, s0, s1 in cal_dark if bb == blk]) for blk in blocks for j in (2, 3, 4)}
    level = {}
    for blk, s, son, soff, ion, ioff in cal:
        for j in LIVE_DESTS[int(s)]:
            level[(int(blk), int(s), j)] = med(j, son + 0.25 * CAL_S, soff - 0.1 * CAL_S) - dark[(int(blk), j)]
    thr = {(blk, j): 0.5 * min(level[(blk, s, j)] for s in (0, 1, 2, 3) if j in LIVE_DESTS[s]) for blk in blocks for j in (2, 3, 4)}
    nA = lambda v: v / TIA * 1e9
    for j in (2, 3, 4):
        lv = [f"C{s + 1} {nA(np.nanmin([level[(blk, s, j)] for blk in blocks])):.0f}-{nA(np.nanmax([level[(blk, s, j)] for blk in blocks])):.0f}"
              for s in (0, 1, 2, 3) if j in LIVE_DESTS[s]]
        lines.append(f"C{j + 1} {recv[j - 2]}: spike levels over {len(blocks)} calibrations (nA) {', '.join(lv)}; "
                     f"threshold {nA(min(thr[(blk, j)] for blk in blocks)):.0f}-{nA(max(thr[(blk, j)] for blk in blocks)):.0f} nA")

    # 3. detection per (slot, receiver), windows centred on where the light should be
    kind, late, sched = d["cmd_kind"], d["cmd_late"], d["cmd_sched"]
    is_p = kind == 0
    L = float(np.median(t1[is_p] - sched[is_p])) if is_p.any() else 0.0     # typical command latency
    nsteps = int(meta["steps_done"])
    keep = d["slot_step"] < nsteps
    S_step, S_win, S_sched = d["slot_step"][keep], d["slot_window"][keep], d["slot_sched"][keep]
    S_on, S_off = d["slot_on"][keep], d["slot_off"][keep]
    ns = len(S_step)
    pk = d["p_slot"] < ns
    p_slot, p_sender, p_step = d["p_slot"][pk], d["p_sender"][pk], d["p_step"][pk]
    p_layer, p_neuron, p_i = d["p_layer"][pk], d["p_neuron"][pk], d["p_i"][pk]
    p_on, p_off, p_offset = d["p_cmd_on"][pk], d["p_cmd_off"][pk], d["p_offset"][pk]
    S_digit = S_step // STEPS_PER_DIGIT
    S_block = S_digit // args["recal_every"]

    def run_ok(j):
        th = np.array([thr[(blk, j)] + dark[(blk, j)] for blk in blocks])
        # threshold of the block each sample belongs to (block start = its calibration's dark)
        starts = np.array([to_daq(min(c[1] for c in cal_dark if c[0] == blk)) for blk in blocks])
        above = sig[j] > th[np.clip(np.searchsorted(starts, ts[j], side="right") - 1, 0, None)]
        c = np.r_[0, np.cumsum(above)]          # ok[i]: samples i-MIN+1..i all above
        ok = np.zeros(len(above), bool)
        ok[LIVE_MIN_SAMPLES - 1:] = c[LIVE_MIN_SAMPLES:] - c[:-LIVE_MIN_SAMPLES] == LIVE_MIN_SAMPLES
        return np.r_[0, np.cumsum(ok)]

    def detect(j, cs, w0, w1):
        '''True where [w0, w1) (DAQ s) holds >= MIN consecutive samples above threshold.'''
        i0 = np.searchsorted(ts[j], w0) + LIVE_MIN_SAMPLES - 1
        i1 = np.searchsorted(ts[j], w1)
        return cs[np.maximum(i1, i0)] - cs[np.minimum(i0, i1)] > 0

    exp = np.zeros((ns, 5), bool); exp_off = np.zeros((ns, 5))        # expected, sender's offset
    lit = np.zeros((ns, 4), bool)
    lit[p_slot, p_sender] = True
    for s in range(4):
        for j in LIVE_DESTS[s]:
            m = p_sender == s
            exp[p_slot[m], j] = True; exp_off[p_slot[m], j] = p_offset[m]
    det = np.zeros((ns, 5), bool)
    for j in (2, 3, 4):
        cs = run_ok(j)
        t0 = S_sched + exp_off[:, j] + L - S_off / 2
        det[:, j] = detect(j, cs, to_daq(t0), to_daq(t0 + S_on + S_off))
    other = np.zeros((ns, 5), bool)              # a laser not sending to j is on in this slot
    for j in (2, 3, 4):
        other[:, j] = lit[:, [s for s in range(4) if j not in LIVE_DESTS[s]]].any(1)
    # dark gaps at window boundaries: lasers off, nothing should be seen
    # test only between the last slot's detection window and the next slot's (no overlap with either)
    gs, ge = d["w_gap_start"], d["w_gap_end"]
    per = on_s + off_s
    w_end = S_sched + L - S_off / 2 + S_on + S_off                   # end of each slot's window
    w_beg = S_sched + L - S_off / 2
    gb = []
    for g0, g1 in zip(gs, ge):
        i = np.searchsorted(S_sched, g0) - 1                         # last slot before the gap
        k = np.searchsorted(S_sched, g1 - 1e-9)                      # first slot after it
        lo = max(w_end[i], g0 + L - off_s / 2) if i >= 0 else g0 + L    # never before the gap (calibration)
        hi = w_beg[k] if k < ns else g1 + L
        gb += [(t0, min(t0 + per, hi)) for t0 in np.arange(lo, hi - 1e-9, per)]
    gb = np.array(gb).reshape(-1, 2)
    gap_det = {j: detect(j, run_ok(j), to_daq(gb[:, 0]), to_daq(gb[:, 1])) if len(gb) else np.zeros(0, bool) for j in (2, 3, 4)}

    # 4. timing faults: a pulse command of this slot (or the previous slot's switch-off) more than
    # fault_ms late, or this window's heater command not finished (+ switching) before the slot started
    # lateness = when the command took effect (call returned) vs its original schedule time
    lat_ms = (t1 - d["cmd_t0"] + late) * 1e3
    bad_cmd = lat_ms > fault_ms
    fault = np.zeros(ns, bool)
    np.logical_or.at(fault, p_slot, bad_cmd[p_on] | bad_cmd[p_off])
    prev_off = np.zeros(ns, bool); np.logical_or.at(prev_off, p_slot, bad_cmd[p_off])
    fault[1:] |= prev_off[:-1] & (S_step[1:] == S_step[:-1])
    hm = kind == 1
    for w in np.unique(d["cmd_window"][hm & (d["cmd_window"] >= 0)]):
        hc = np.flatnonzero(hm & (d["cmd_window"] == w))
        m = S_win == w
        fault[m] |= S_sched[m] < t1[hc].max() + 40e-6      # heaters still switching (~36 us)
    faulted_slots = int(fault.sum())

    # 5. counts
    def tally(sel):
        e, dt, oth = exp[sel][:, 2:], det[sel][:, 2:], other[sel][:, 2:]
        r = {"expected": int(e.sum()), "hits": int((e & dt).sum()), "misses": int((e & ~dt).sum()),
             "false_other_core": int((~e & dt & oth).sum()), "false_nothing": int((~e & dt & ~oth).sum())}
        return r
    allr, ontr = tally(np.ones(ns, bool)), tally(~fault)
    per_rx = {f"C{j + 1}": {"receiver": recv[j - 2], **{k: int(v) for k, v in {
        "expected": exp[:, j].sum(), "hits": (exp[:, j] & det[:, j]).sum(), "misses": (exp[:, j] & ~det[:, j]).sum(),
        "false_other_core": (~exp[:, j] & det[:, j] & other[:, j]).sum(),
        "false_nothing": (~exp[:, j] & det[:, j] & ~other[:, j]).sum(), "false_in_gaps": gap_det[j].sum()}.items()}}
        for j in (2, 3, 4)}
    gap_false = int(sum(v.sum() for v in gap_det.values()))
    for r in (allr, ontr):
        r["false_in_gaps"] = gap_false                 # gaps hold no pulse commands: never timing-faulted
        r["errors"] = r["misses"] + r["false_other_core"] + r["false_nothing"] + r["false_in_gaps"]
        r["error_rate"] = r["errors"] / max(r["expected"], 1)
        r["error_rate_95ci"] = clopper_pearson(min(r["errors"], r["expected"]), r["expected"]) if r["expected"] else None
    n_pulses = int(len(p_slot))
    res = {"run": path.name, "dry_run": meta["dry_run"], "laser_mode": meta["laser_mode"],
           "on_ms": args["on_ms"], "off_ms": args["off_ms"], "gap_ms": args["gap_ms"], "digits": meta["digits"],
           "steps": nsteps, "pulse_slots": ns, "laser_pulses_sent": n_pulses,
           "received_pulses_expected": allr["expected"], "command_latency_used_ms": L * 1e3,
           "clock_fit": {"offset_ms": a * 1e3, "drift_ppm": (b - 1) * 1e6, "edges": int(len(pc_pts)),
                         "residual_rms_ms": float(1e3 * resid.std())},
           "overall": allr, "excluding_timing_faults": ontr, "timing_fault_slots": faulted_slots,
           "receivers": per_rx}
    lines.append(f"{nsteps} SNN steps, {ns} pulse slots ({meta['laser_mode']} mode, {args['on_ms']:g}/{args['off_ms']:g} ms), "
                 f"{n_pulses} laser pulses sent, {allr['expected']} received pulses expected")
    for name, r in (("overall", allr), (f"excluding {faulted_slots} timing-fault slots", ontr)):
        ci = r["error_rate_95ci"] or (np.nan, np.nan)
        lines.append(f"{name}: {r['hits']}/{r['expected']} hits, {r['misses']} misses, false {r['false_other_core']} "
                     f"while another core fired + {r['false_nothing']} with nothing firing + {r['false_in_gaps']} in "
                     f"heater gaps -> error rate {r['error_rate']:.2e} per received pulse (95% CI {ci[0]:.1e}-{ci[1]:.1e})")
    for c, r in per_rx.items():
        lines.append(f"  {c} {r['receiver']}: {r['hits']}/{r['expected']}, {r['misses']} missed, false "
                     f"{r['false_other_core']} + {r['false_nothing']} + {r['false_in_gaps']} (gaps)")

    # 6. timing
    lp, hl = lat_ms[is_p], lat_ms[hm]
    ls = late[is_p] * 1e3
    slips = d["slot_sched"][keep] - d["slot_orig"][keep]
    res["timing"] = {"pulse_cmd_late_ms": {"median": float(np.median(lp)), "p90": float(np.percentile(lp, 90)),
                                           "p99": float(np.percentile(lp, 99)), "max": float(lp.max()),
                                           "over_2ms": int((lp > 2).sum()), "over_3ms": int((lp > 3).sum()), "n": int(lp.size)},
                     "pulse_cmd_start_late_ms": {"median": float(np.median(ls)), "p99": float(np.percentile(ls, 99)),
                                                 "over_3ms": int((ls > 3).sum())},
                     "pulse_cmd_duration_ms": float(np.median((t1 - d["cmd_t0"])[is_p]) * 1e3),
                     "heater_cmd_late_ms": {"median": float(np.median(hl)) if hl.size else None,
                                            "max": float(hl.max()) if hl.size else None, "n": int(hl.size)},
                     "schedule_slips": int((slips > 0).sum()), "slip_total_s": float(slips.sum()),
                     "wall_time_s": float(t1.max() - t1.min())}
    errs_slot = (exp & ~det) | (~exp & det)
    e_f, e_o = int(errs_slot[fault][:, 2:].sum()), int(errs_slot[~fault][:, 2:].sum())
    res["errors_in_faulted_slots"], res["errors_in_on_time_slots"] = e_f, e_o
    # misses vs how late the pulse's own on/off commands took effect
    pl_ms = np.maximum(lat_ms[p_on], lat_ms[p_off])
    edges_ms = [0, 1, 1.5, 2, 3, np.inf]
    by = []
    for lo_, hi_ in zip(edges_ms[:-1], edges_ms[1:]):
        m = (pl_ms >= lo_) & (pl_ms < hi_)
        e_n = sum(int(np.isin(p_sender[m], [s for s in range(4) if j in LIVE_DESTS[s]]).sum()) for j in (2, 3, 4))
        e_m = sum(int((np.isin(p_sender[m], [s for s in range(4) if j in LIVE_DESTS[s]]) & ~det[p_slot[m], j]).sum()) for j in (2, 3, 4))
        by.append({"late_ms": [lo_, hi_], "received_expected": e_n, "misses": e_m})
    res["misses_by_lateness"] = by
    lines.append("misses by how late the pulse's commands took effect: " + ", ".join(
        f"{b_['late_ms'][0]:g}-{b_['late_ms'][1]:g} ms {b_['misses']}/{b_['received_expected']}" for b_ in by))
    lines.append(f"timing (lateness = command took effect vs schedule): pulse commands late median {np.median(lp):.2f} ms, p99 {np.percentile(lp, 99):.2f} ms, "
                 f"max {lp.max():.1f} ms; {(lp > 2).sum()} > 2 ms, {(lp > 3).sum()} > {fault_ms:g} ms of {lp.size}; "
                 f"call {res['timing']['pulse_cmd_duration_ms']:.2f} ms; {res['timing']['schedule_slips']} schedule slips "
                 f"({slips.sum():.2f} s); wall time {res['timing']['wall_time_s']:.0f} s")
    dom = "timing faults DOMINATE the errors" if e_f > e_o else "most errors are in on-time slots"
    lines.append(f"errors in {faulted_slots} timing-fault slots: {e_f}; in on-time slots: {e_o} ({dom})" if e_f + e_o else
                 f"no slot errors; {faulted_slots} timing-fault slots")

    # 7. controller: decisions logged live vs export_replay.py's schedule
    ws, wi, wh = d["w_setting"], d["w_inp"], d["w_hid"]
    res["controller"] = {"windows": int(len(ws)), "decision_ms_max": float(d["w_decision_ms"].max()),
                         "decision_ms_median": float(np.median(d["w_decision_ms"][1:])) if len(ws) > 1 else 0.0,
                         "heater_ms_max": float(d["w_heater_ms"].max()),
                         "settings": {s: int((ws == s).sum()) for s in "ABCD"},
                         "heater_changes": int(d["w_heater_cmds"].sum())}
    try:
        ex = export_schedule(meta["digits"][0], len(meta["digits"]))
        nW = min(len(ws), len(ex["windows"]), (nsteps + 4) // 5)
        lk = lambda w: (int(next((s for s, ds in w["links"].items() if 4 not in ds), -1)),
                        int(next((s for s, ds in w["links"].items() if 4 in ds), -1)))
        mism = [w for w in range(nW) if (ws[w], int(wi[w]), int(wh[w])) != (ex["windows"][w]["setting"], *lk(ex["windows"][w]))]
        Pex = np.array([s["packets"] for s in ex["steps"][:nsteps]])
        res["controller"].update(export_mismatches=[int(w) for w in mism], export_window0=ex["windows"][0],
                                 live_window0={"setting": str(ws[0]), "inp": int(wi[0]), "hid": int(wh[0])},
                                 live_counts_equal_export=bool(np.array_equal(Pex, d["step_P"][:nsteps])))
        later = [w for w in mism if w > 0]
        lines.append(f"controller: {nW} windows, settings {res['controller']['settings']}, decision max "
                     f"{res['controller']['decision_ms_max']:.2f} ms, heater commands max {res['controller']['heater_ms_max']:.1f} ms; "
                     f"vs export_replay.py: window 0 live {ws[0]} (no history, started from A) / export "
                     f"{ex['windows'][0]['setting']}; windows 1+: {len(later)} mismatches {later[:10]}; "
                     f"live packet counts {'==' if res['controller']['live_counts_equal_export'] else '!='} export's")
    except Exception as err:                         # spikes_test.npz missing etc.: keep the rest
        lines.append(f"controller check vs export_replay.py skipped: {err}")

    # 8. pulse -> neuron mapping check against the spike record
    d0 = meta["digits"][0]
    with np.load(ROOT / "sim" / "spikes_test.npz") as f:      # load once: NpzFile decompresses per access
        n_d = (int(meta["steps_done"]) + STEPS_PER_DIGIT - 1) // STEPS_PER_DIGIT
        sp = {k: f[k][d0:d0 + n_d] for k in ("inp", "h1")}
    d0 = 0
    m0 = np.where(((np.arange(2312) % 1156) // 34) < 17, 0, 1); m1 = np.where(np.arange(128) < 64, 2, 3)
    bad_map = 0
    order = np.lexsort((p_i, p_sender, p_step))
    ps, pp, pn, pl = p_step[order], p_sender[order], p_neuron[order], p_layer[order]
    keys = np.stack([ps, pp], 1)
    bounds = np.flatnonzero(np.r_[True, (np.diff(keys, axis=0) != 0).any(1), True])
    for a_, b_ in zip(bounds[:-1], bounds[1:]):
        k, s = int(ps[a_]), int(pp[a_])
        dg, t = divmod(k, STEPS_PER_DIGIT)
        want = np.flatnonzero(sp["inp"][d0 + dg, t].astype(bool) & (m0 == s)) if s < 2 else \
            np.flatnonzero(sp["h1"][d0 + dg, t].astype(bool) & (m1 == s))
        bad_map += not (np.array_equal(pn[a_:b_], want) and (pl[a_:b_] == (s >= 2)).all())
    res["mapping_check"] = {"step_links": int(len(bounds) - 1), "mismatched": int(bad_map)}
    lines.append(f"pulse -> neuron map: {len(bounds) - 1 - bad_map}/{len(bounds) - 1} (step, link) pulse trains equal "
                 "the senders' spiking neurons in ascending order")

    # 9. delivery mask -> SNN rerun without exactly the undelivered spikes
    und = []                                                 # (pulse, dest)
    for j in (2, 3, 4):
        m = np.isin(p_sender, [s for s in range(4) if j in LIVE_DESTS[s]])
        bad = m & ~det[p_slot, j]
        und += [(int(p), j) for p in np.flatnonzero(bad)]
    und_f = [(p, j) for p, j in und if fault[p_slot[p]]]
    if meta["dry_run"] and "sim_tx_drop" in d.files:
        inj = set()
        tx, rx = d["sim_tx_drop"], d["sim_rx_drop"]
        for ci, c in tx:
            for p in np.flatnonzero((p_on == ci) & (p_sender == c)):
                inj |= {(int(p), j) for j in LIVE_DESTS[int(c)]}
        for ci, c, j in rx:
            inj |= {(int(p), int(j)) for p in np.flatnonzero((p_on == ci) & (p_sender == c))}
        got = set(und)
        res["dry_run_check"] = {"injected": len(inj), "found": len(inj & got), "missed_injections": len(inj - got),
                                "extra_misses": len(got - inj), "extra_in_faulted_slots": len({x for x in got - inj if fault[p_slot[x[0]]]})}
        bl = d["sim_blips"] if "sim_blips" in d.files else np.zeros((0, 2))
        fd = int((~exp[:, 2:] & det[:, 2:]).sum()) + gap_false
        res["dry_run_check"].update(blips=int(len(bl)), false_detections=fd)
        lines.append(f"dry run: {len(bl)} stray 1 ms blips injected over the whole recording (incl. calibration "
                     f"and on top of real pulses), {fd} false detections reported")
        lines.append(f"dry run: {len(inj)} injected lost pulses (pulse, receiver), {len(inj & got)} found as undelivered "
                     f"(same neurons), {len(inj - got)} not found, {len(got - inj)} other misses "
                     f"({res['dry_run_check']['extra_in_faulted_slots']} in timing-fault slots)")
    labels = np.array(meta["labels"])
    nd = (nsteps + STEPS_PER_DIGIT - 1) // STEPS_PER_DIGIT
    drop_in = [[] for _ in range(nd)]; drop_h1 = [[] for _ in range(nd)]
    for p, j in und:
        dg, t = divmod(int(p_step[p]), STEPS_PER_DIGIT)
        (drop_in[dg].append((t, int(p_neuron[p]), j)) if p_layer[p] == 0 else drop_h1[dg].append((t, int(p_neuron[p]))))
    soft, chip, h1_same = [], [], 0
    for dg in range(nd):
        x = sp["inp"][d0 + dg]
        s_pred, h1 = snn_masked(x, [], [])
        h1_same += np.array_equal(h1, sp["h1"][d0 + dg])
        soft.append(s_pred)
        chip.append(snn_masked(x, drop_in[dg], drop_h1[dg])[0] if drop_in[dg] or drop_h1[dg] else s_pred)
    soft, chip = np.array(soft), np.array(chip)
    res["accuracy"] = {"digits": nd, "software": float(np.mean(soft == labels[:nd])), "chip": float(np.mean(chip == labels[:nd])),
                       "predictions_changed": int((soft != chip).sum()), "undelivered_spikes": len(und),
                       "undelivered_in_faulted_slots": len(und_f), "rerun_reproduces_recorded_h1": f"{h1_same}/{nd}",
                       "note": "all undelivered spikes removed, including timing-faulted slots; false detections "
                               "map to no neuron and are not added"}
    lines.append(f"accuracy on {nd} digits: software {res['accuracy']['software']:.1%}, chip (all {len(und)} undelivered "
                 f"spikes removed, {len(und_f)} of them in timing-fault slots) {res['accuracy']['chip']:.1%}, "
                 f"{res['accuracy']['predictions_changed']} predictions changed; numpy rerun reproduces recorded h1 "
                 f"spikes in {h1_same}/{nd} digits")

    # 10. bus: undelivered optical packets go back on the bus
    P = d["step_P"][:nsteps].astype(float)
    opt = np.zeros(nsteps)
    for j in (2, 3, 4):
        m = np.isin(p_sender, [s for s in range(4) if j in LIVE_DESTS[s]]) & det[p_slot, j]
        np.add.at(opt, p_step[m], 1)
    planned = np.zeros(nsteps)
    np.add.at(planned, p_step, [len(LIVE_DESTS[int(s)]) for s in p_sender])
    res["bus"] = {"packets_total": int(P.sum()), "packets_planned_optical": int(planned.sum()),
                  "packets_delivered_optical": int(opt.sum()), **bus_queue(P.sum((1, 2)), opt)}
    q = res["bus"]["1.5x"]
    lines.append(f"bus: {int(P.sum())} packets, {int(planned.sum())} sent by light ({planned.sum() / P.sum():.1%}), "
                 f"{int(opt.sum())} delivered ({opt.sum() / max(planned.sum(), 1):.3%}); 99th-pct delay at 1.5x capacity "
                 f"{q['bus_only']['p99_ms']:.1f} ms bus only vs {q['with_lanes']['p99_ms']:.1f} ms with lanes")
    np.savez_compressed(path.with_name(path.stem + "_detections.npz"), detected=det, expected=exp, other=other,
                        fault=fault, undelivered=np.array(und, int).reshape(-1, 2))
    return res, lines


CAL_S, TIA = 0.030, 220e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="runs/spike_replay_<stamp>.npz or runs/spike_replay_live_<stamp>.npz")
    args = ap.parse_args()
    d = np.load(args.run)
    if "p_slot" in d.files:                      # spike_replay_live.py run
        res, lines = analyze_live(args.run)
        out = args.run.with_name(args.run.stem + "_analysis")
        out.with_suffix(".json").write_text(json.dumps(res, indent=2, default=float))
        out.with_suffix(".txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    late, sender, det = d["late_ms"], d["sender"], d["detected"]
    rp = json.loads((ROOT / "sim" / "replay_excerpt.json").read_text())
    routes = json.loads((ROOT / "sim" / "routes_rows2-6.json").read_text())
    ev = rebuild_events(rp, routes)
    if len(ev) != len(late):
        raise SystemExit(f"command list mismatch: rebuilt {len(ev)}, recorded {len(late)}")
    n = len(rp["steps"])
    faulted = np.zeros(n, bool)
    for (_, kind, k), lt in zip(ev, late):
        if lt > FAULT_MS:
            faulted[k] = True
            if kind == 2 and k + 1 < n:
                faulted[k + 1] = True
    exp = sender >= 0
    err = (exp & ~det) | (~exp & det)
    err[:, [j for j in range(exp.shape[1]) if not exp[:, j].any()]] = False      # receivers with no traffic
    tot_spk, tot_err = int(exp.sum()), int(err.sum())
    on_spk, on_err = int(exp[~faulted].sum()), int(err[~faulted].sum())
    err_slots = sorted({int(k) for k in np.nonzero(err)[0]})
    labels = np.array(rp["labels"])
    acc_soft, acc_hw, changed = accuracy(sender, det, labels, len(labels))
    P = np.array([s["packets"] for s in rp["steps"]], float)
    res = {"run": args.run.name, "fault_threshold_ms": FAULT_MS, "commands": len(ev),
           "overall": {"optical_spikes": tot_spk, "errors": tot_err, "error_rate": tot_err / tot_spk},
           "timing_fault_slots": int(faulted.sum()),
           "on_time": {"optical_spikes": on_spk, "errors": on_err, "error_rate": on_err / on_spk},
           "faulted": {"optical_spikes": tot_spk - on_spk, "errors": tot_err - on_err},
           "slots_with_errors": err_slots, "error_slots_not_faulted": [k for k in err_slots if not faulted[k]],
           "accuracy": {"digits": len(labels), "software": acc_soft, "with_chip_detections": acc_hw,
                        "predictions_changed": changed, "note": "uses all detections, including timing-faulted slots"},
           "bus": bus_delays(P, sender, det)}
    out = args.run.with_name(args.run.stem + "_analysis")
    out.with_suffix(".json").write_text(json.dumps(res, indent=2))
    lines = [f"{res['run']}: {len(ev)} commands rebuilt and matched to the recorded lateness",
             f"overall: {tot_err} errors in {tot_spk} optical spikes (rate {tot_err / tot_spk:.1e})",
             f"timing faults (> {FAULT_MS:.0f} ms late): {faulted.sum()} slots; on-time: {on_err} errors in {on_spk} "
             f"(rate {on_err / on_spk:.1e}); faulted: {tot_err - on_err} in {tot_spk - on_spk}",
             f"slots with errors: {err_slots}; not faulted: {res['error_slots_not_faulted']}",
             f"accuracy on {len(labels)} digits: software {acc_soft:.1%}, with chip detections {acc_hw:.1%}, "
             f"{changed} predictions changed (all detections used)",
             f"bus: {100 * res['bus']['offload_delivered']:.1f}% delivered by light; 99th-pct delay at 1.5x capacity "
             f"{res['bus']['1.5x']['bus_only']['p99_ms']:.1f} ms bus only vs {res['bus']['1.5x']['with_lanes']['p99_ms']:.1f} ms with lanes"]
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
