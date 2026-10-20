"""Export a short excerpt of the SNN's traffic as a replay schedule for the chip.

For each 2 ms SNN step the schedule holds which cores fired (sent at least one packet) and
the packet counts; for each 10 ms controller window the router setting (A-D). The controller
is the one in traffic.py: per window, the reachable setting that would have carried the most
traffic in the previous window. Its choice is mapped onto the two heaters:
  2_2 picks the input core that multicasts to C3+C4 (CROSS: C1, BAR: C2)
  4_2 picks the hidden core that sends to C5 (CROSS: C3, BAR: C4)
If a window's best setting leaves one of the two links unused, the heater keeps its previous
position and that link carries nothing optically in that window.

Writes replay_excerpt.json, read by ../spike_replay.py.

    python export_replay.py                    # first 10 test digits (1500 steps)
    python export_replay.py --start 20 --samples 5
"""
import argparse, json, numpy as np
from compile_routes import N, CORE_ROW, feasible_assignments

WINDOW_BINS = 5                                            # 10 ms controller window
SETTING = {(1, 3): "A", (0, 2): "B", (0, 3): "C", (1, 2): "D"}   # (input sender, hidden sender)


def layer_mapping(sizes):
    """5-core split used by traffic.py (MAPPING = "layer")."""
    row = (np.arange(2312) % 1156) // 34
    return [np.where(row < 17, 0, 1), np.where(np.arange(128) < 64, 2, 3),
            np.full(sizes[2], 4), np.full(sizes[3], 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=0, help="first test digit")
    ap.add_argument("--samples", type=int, default=10, help="number of digits")
    args = ap.parse_args()
    assert N == 5 and CORE_ROW == [2, 3, 4, 5, 6], "export_replay expects the 5-core rows 2-6 layout"

    d = np.load("spikes_test.npz")
    sl = slice(args.start, args.start + args.samples)
    layers = [d["inp"][sl], d["h1"][sl], d["h2"][sl], d["out"][sl]]
    sizes = [l.shape[2] for l in layers]
    S, T = layers[0].shape[:2]
    m = layer_mapping(sizes)
    tr = np.zeros((S * T, N, N), np.int32)                 # packets[step, src, dst]
    for L in range(3):
        spk = layers[L].reshape(S * T, -1)
        for c in range(N):
            cnt = spk[:, m[L] == c].sum(1).astype(np.int32)
            for j in set(m[L + 1].tolist()) - {c}:
                tr[:, c, j] += cnt

    A, _ = feasible_assignments()
    Ai = np.where(A < 0, N, A)
    nW = len(tr) // WINDOW_BINS
    W = tr[:nW * WINDOW_BINS].reshape(nW, WINDOW_BINS, N, N).sum(1)
    Wp = np.concatenate([W.astype(float), np.zeros((nW, 1, N))], 1)
    best = sum(Wp[:, Ai[:, j], j] for j in range(N)).argmax(1)
    total_best = np.concatenate([tr.sum(0).astype(float), np.zeros((1, N))], 0)
    choice = np.r_[sum(total_best[None][:, Ai[:, j], j] for j in range(N)).argmax(1), best[:-1]]   # previous window

    windows, inp, hid = [], 1, 3                            # start from A
    for w in range(nW):
        a = A[choice[w]]
        new_inp = a[2] if a[2] in (0, 1) else None          # who C3 listens to
        new_hid = a[4] if a[4] in (2, 3) else None          # who C5 listens to
        inp, hid = new_inp if new_inp is not None else inp, new_hid if new_hid is not None else hid
        links = {}
        if new_inp is not None:
            links[str(inp)] = [2, 3]
        if new_hid is not None:
            links[str(hid)] = [4]
        windows.append({"setting": SETTING[(inp, hid)], "links": links})

    steps = [{"fired": [bool(x) for x in (tr[k].sum(1) > 0)], "packets": tr[k].tolist()}
             for k in range(nW * WINDOW_BINS)]
    carried = sum(tr[w * WINDOW_BINS:(w + 1) * WINDOW_BINS, int(s), d].sum()
                  for w, win in enumerate(windows) for s, ds in win["links"].items() for d in ds)
    out = {"dt_ms": int(d["dt_ms"]), "window_steps": WINDOW_BINS, "digits": list(range(args.start, args.start + args.samples)),
           "labels": d["labels"][sl].tolist(), "core_row": CORE_ROW, "windows": windows, "steps": steps,
           "packets_total": int(tr.sum()), "packets_optical_plan": int(carried)}
    json.dump(out, open("replay_excerpt.json", "w"), separators=(",", ":"))
    counts = {s: sum(w["setting"] == s for w in windows) for s in "ABCD"}
    changes = sum(windows[i]["setting"] != windows[i - 1]["setting"] for i in range(1, nW))
    print(f"{args.samples} digits, {len(steps)} steps, {nW} windows; settings {counts}, {changes} changes; "
          f"packets {out['packets_total']}, planned by light {carried} ({100 * carried / out['packets_total']:.1f}%)")


if __name__ == "__main__":
    main()
