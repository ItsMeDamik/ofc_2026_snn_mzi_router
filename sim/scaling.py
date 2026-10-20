"""Scaling study: how much spike traffic optical lanes carry, and what that does to bus delay,
as the network is spread over more cores.

Networks (both 2312-128-64-10 LIF, trained on the same N-MNIST data):
  fc:    train_snn.py, fully connected first layer (every pixel feeds every hidden-1 neuron)
  local: train_local.py, each hidden-1 group of 8 sees one tile of a 4x4 grid (+2 px halo)
Cores, N = 2M + 1: M input cores (image tiles), M hidden-1 cores, 1 core for hidden-2 + outputs.
  M = 2: top/bottom halves (the 5-core split measured on the chip); 4: quadrants; 8: 4x2; 16: 4x4.
  fc: hidden-1 split into M equal blocks; local: each hidden-1 group joins the core of its tile.
A spike becomes one packet per other core that holds a neuron it connects to.

Controller (as on the chip): every 10 ms window, each receiver core listens to the sender that
sent it the most in the previous window (multicast = equal split). The mesh is assumed ideal
(any such pattern); measured switching (35 us) is charged as dead time on a receiver whenever
its sender changes. Bus: one shared queue; capacity is given relative to the bus-only mean load.

Optical budget, from the chip's measured margins (route_check / spike_replay, 2026-10-19):
  a single-path spike arrives at ~1550 nA after 5 MZI columns; noise 0.25 nA rms; a receiver
  needs its spike >= 2 x (10 x noise) = 5 nA; worst measured signal/crosstalk 16.8 dB at 5
  columns, needs >= 10 dB; crosstalk assumed to grow with mesh depth (10 log10(N/5)).

    python scaling.py              # writes scaling_results.json and scaling.png
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIGITS = 200                    # test digits used (30,000 steps of 2 ms)
WINDOW = 5                      # steps per controller window (10 ms)
SWITCH_MS, WINDOW_MS = 0.035, 10.0
MS = [2, 4, 8, 16]
CAPACITY = [1.0, 1.5, 2.0]      # bus capacity relative to the bus-only mean load
SPIKE_NA, NOISE_NA, NEED_NA = 1550.0, 0.25, 5.0
SXR_5_DB, SXR_NEED_DB = 16.8, 10.0
LOSS_DB = [0.1, 0.3, 0.5]       # assumed insertion loss per MZI column
EDGES = [0, 9, 17, 26, 34]


def tile_of(M):
    """Input core (0..M-1) of every pixel (index = pol*1156 + row*34 + col) for M tiles."""
    rows, cols = np.divmod(np.arange(1156), 34)
    if M == 2:
        t = (rows >= 17).astype(int)
    elif M == 4:
        t = 2 * (rows >= 17) + (cols >= 17)
    else:
        rb = np.searchsorted(EDGES, rows, side="right") - 1
        cb = np.searchsorted(EDGES, cols, side="right") - 1
        t = 2 * rb + (cols >= 17) if M == 8 else 4 * rb + cb
    return np.tile(t, 2)


def traffic(net, M):
    """packets[step, src, dst] for one network split over N = 2M + 1 cores."""
    sp = np.load("spikes_test.npz" if net == "fc" else "spikes_test_local.npz")
    inp = sp["inp"][:DIGITS].reshape(-1, 2312).astype(np.float32)
    h1 = sp["h1"][:DIGITS].reshape(-1, 128).astype(np.float32)
    N = 2 * M + 1
    pix_core = tile_of(M)
    if net == "fc":
        h1_core = M + np.arange(128) * M // 128
        conn = np.ones((2312, 128), bool)
    else:
        w = np.load("snn_local_weights.npz")
        conn = w["mask"].T > 0                                   # pixel x neuron
        centers = [(EDGES[g // 4] + EDGES[g // 4 + 1]) // 2 * 34 + (EDGES[g % 4] + EDGES[g % 4 + 1]) // 2 for g in range(16)]
        h1_core = M + np.array([pix_core[centers[n // 8]] for n in range(128)])
    reach = np.zeros((2312, N), np.float32)                       # pixel -> dst core it sends to
    for j in range(M, 2 * M):
        reach[:, j] = conn[:, h1_core == j].any(1)
    P = np.zeros((len(inp), N, N), np.float32)
    for c in range(M):
        m = pix_core == c
        P[:, c, :] = inp[:, m] @ reach[m]
    for j in range(M, 2 * M):
        P[:, j, 2 * M] = h1[:, h1_core == j].sum(1)               # hidden-1 -> hidden-2 core
    return P


def controller(P):
    """Per-step packets carried by light, for the dynamic controller and a fixed setting."""
    T, N, _ = P.shape
    nW = T // WINDOW
    W = P[:nW * WINDOW].reshape(nW, WINDOW, N, N).sum(1)
    Wd = W.copy(); Wd[:, np.arange(N), np.arange(N)] = 0
    best = Wd.argmax(1)                                           # [window, receiver] -> sender
    has = Wd.max(1) > 0
    static = Wd.sum(0).argmax(0)
    carried_dyn = np.zeros(nW * WINDOW); carried_fix = np.zeros(nW * WINDOW)
    Pw = P[:nW * WINDOW].reshape(nW, WINDOW, N, N)
    changes = 0
    for w in range(1, nW):
        for j in range(N):
            if not has[w - 1, j]:
                continue
            s = best[w - 1, j]
            loss = 1.0
            if w > 1 and has[w - 2, j] and best[w - 2, j] != s:
                loss = 1 - SWITCH_MS / WINDOW_MS; changes += 1
            carried_dyn[w * WINDOW:(w + 1) * WINDOW] += Pw[w, :, s, j] * loss
            carried_fix[w * WINDOW:(w + 1) * WINDOW] += Pw[w, :, static[j], j]
    return carried_dyn, carried_fix, changes / max(nW - 2, 1), best, has


def queue_delay(arr, cap):
    """Mean and 99th-percentile waiting time (ms) of a fluid queue, per 2 ms step."""
    x, q = 0.0, np.zeros(len(arr))
    for k, a in enumerate(arr):
        x = max(0.0, x + a - cap); q[k] = x
    d = q / cap * 2.0
    return float(d.mean()), float(np.percentile(d, 99))


def budget(N, fanout=1):
    """Loss- and crosstalk-limited feasibility for an N x N mesh (N columns), from measured margins.
        A multicast to `fanout` receivers costs 10 log10(fanout) dB on top of the extra columns."""
    margin_db = 10 * np.log10(SPIKE_NA / NEED_NA)                # spare optical budget at 5 columns
    split_db = 10 * np.log10(fanout)
    out = {"split_loss_dB": round(split_db, 1)}
    for L in LOSS_DB:
        out[f"margin_left_dB_at_{L}dB_per_column"] = round(margin_db - L * (N - 5) - split_db, 1)
    out["sxr_dB_est"] = round(SXR_5_DB - 10 * np.log10(N / 5), 1)
    return out


def main():
    results = {}
    nets = ["fc"]
    try:
        np.load("spikes_test_local.npz"); nets.append("local")
    except FileNotFoundError:
        print("spikes_test_local.npz missing: run train_local.py for the local network")
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    for net in nets:
        results[net] = {}
        acc = float(np.load("spikes_test.npz" if net == "fc" else "spikes_test_local.npz")["acc"])
        for M in MS:
            N = 2 * M + 1
            P = traffic(net, M)
            total = P.sum((1, 2))
            dyn, fix, change_rate, best, has = controller(P)
            n = len(dyn)
            tot = total[:n]
            off_dyn, off_fix = dyn.sum() / tot.sum(), fix.sum() / tot.sum()
            # largest multicast fan-out actually used in a window
            fan = max((np.bincount(best[w][has[w]]).max() for w in range(len(best)) if has[w].any()), default=1)
            lam = tot.mean()
            row = {"cores": N, "accuracy": acc, "offload_dynamic": off_dyn, "offload_fixed": off_fix,
                   "mean_load_pkts_per_ms": lam / 2, "peak_over_mean": float(tot.max() / lam),
                   "receiver_switch_rate_per_window": change_rate, "max_fanout": int(fan),
                   "budget": budget(N, fan)}
            for f in CAPACITY:
                row[f"delay_bus_only_{f}x"] = queue_delay(tot, f * lam)
                row[f"delay_with_lanes_{f}x"] = queue_delay(tot - dyn, f * lam)
            results[net][N] = row
            print(f"{net:5s} {N:2d} cores: offload {100 * off_dyn:5.1f}% (fixed {100 * off_fix:5.1f}%), "
                  f"peak/mean {row['peak_over_mean']:.1f}, max fan-out {fan}, "
                  f"delay @1.5x bus-only {row['delay_bus_only_1.5x'][0]:.2f}/{row['delay_bus_only_1.5x'][1]:.1f} ms "
                  f"vs lanes {row['delay_with_lanes_1.5x'][0]:.2f}/{row['delay_with_lanes_1.5x'][1]:.1f} ms (mean/p99); "
                  f"budget {row['budget']}")
        Ns = sorted(results[net])
        style = "-o" if net == "fc" else "-s"
        axs[0].plot(Ns, [100 * results[net][N]["offload_dynamic"] for N in Ns], style, label=f"{net}, dynamic")
        axs[0].plot(Ns, [100 * results[net][N]["offload_fixed"] for N in Ns], style, alpha=0.4, label=f"{net}, fixed")
        for f, ls in zip(CAPACITY, ["-", "--", ":"]):
            axs[1].plot(Ns, [results[net][N][f"delay_bus_only_{f}x"][1] for N in Ns], ls, color="0.5" if net == "fc" else "0.2",
                        marker="o" if net == "fc" else "s", label=f"{net} bus only, {f}x")
            axs[1].plot(Ns, [results[net][N][f"delay_with_lanes_{f}x"][1] for N in Ns], ls, color="C0" if net == "fc" else "C3",
                        marker="o" if net == "fc" else "s", label=f"{net} with lanes, {f}x")
    Ns = [2 * M + 1 for M in MS] + [65]
    for L in LOSS_DB:   # worst case: multicast to every hidden-1 core, fan-out (N - 1) / 2
        axs[2].plot(Ns, [budget(N, max(1, (N - 1) // 2))[f"margin_left_dB_at_{L}dB_per_column"] for N in Ns], "-o",
                    label=f"power margin, {L} dB/column, fan-out (N-1)/2")
    axs[2].plot(Ns, [budget(N)["sxr_dB_est"] for N in Ns], "k--", label="signal/crosstalk (est.)")
    axs[2].axhline(SXR_NEED_DB, color="k", lw=0.8); axs[2].axhline(0, color="C3", lw=0.8)
    axs[0].set(xlabel="Cores", ylabel="Spike packets carried by light (%)", title="Offload vs number of cores")
    axs[1].set(xlabel="Cores", ylabel="Bus delay, 99th percentile (ms)", yscale="symlog", title="Bus delay at capacity = 1.0/1.5/2.0 x load")
    axs[2].set(xlabel="Mesh size N (N x N, N columns)", ylabel="dB", title="Optical budget from measured margins")
    for ax in axs:
        ax.legend(fontsize=7)
    fig.savefig("scaling.png", dpi=120)
    json.dump(results, open("scaling_results.json", "w"), indent=1, default=float)
    print("wrote scaling_results.json, scaling.png")


if __name__ == "__main__":
    main()
