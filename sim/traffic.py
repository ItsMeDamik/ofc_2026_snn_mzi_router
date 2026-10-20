"""Map the trained SNN's neurons onto the cores (4 on rows 2-5 by default, see
compile_routes.CORE_ROW), extract inter-core spike traffic, and run the mesh
configuration optimizer for the real chip.

Rule for a valid mesh configuration: every destination core (mesh output / PD)
listens to at most one source core. Multicast from a source to k destinations is
an equal 1/k split. On the chip only BAR, CROSS and 50:50 are used, so the
reachable settings are those compile_routes.feasible_assignments() finds: every
pattern except a multicast to exactly 3 destinations. The optimizer picks, per
window, the reachable setting that carries the most traffic.
"""
import json, numpy as np
from compile_routes import CORE_ROW, FIXED_BAR, MZIS, N, PAIR, RECEIVER_PD, feasible_assignments

# "layer": input halves / hidden-1 halves / rest, so two input cores compete for the
#          hidden-1 cores and reconfiguration matters. 4 cores: C4 = hidden-1 second half
#          + hidden-2 + outputs. 5 cores: C5 = hidden-2 + outputs.
# "inputs_one_core": all inputs on C1 (little competition; for comparison)
MAPPING = "layer"
CHUNK = 4000                                               # windows per optimizer batch

d = np.load("spikes_test.npz")
dt = int(d["dt_ms"])
layers = [d["inp"], d["h1"], d["h2"], d["out"]]          # (Nsamp, T, n_l)
sizes = [l.shape[2] for l in layers]                      # 2312,128,64,10
Ns, T = layers[0].shape[:2]
print("acc", float(d["acc"]), "sizes", sizes, "samples", Ns, "bins/sample", T)

# ---------------- core mappings ----------------
def layer_mapping():
    m = [np.zeros(s, int) for s in sizes]
    # input index = pol*1156 + row*34 + col -> top half rows 0-16
    idx = np.arange(2312); row = (idx % 1156) // 34
    if MAPPING == "layer":
        m[0] = np.where(row < 17, 0, 1)                    # C1 top, C2 bottom
        m[1] = np.where(np.arange(128) < 64, 2, 3)         # C3, C4
        m[2][:] = N - 1                                    # last core: hidden 2 ...
        m[3][:] = N - 1                                    # ... and outputs
    elif N == 4:
        m[0][:] = 0                                        # C1: all inputs
        m[1] = np.where(np.arange(128) < 64, 1, 2)         # C2, C3
        m[2][:] = 3                                        # C4: hidden 2 + outputs
        m[3][:] = 3
    else:
        m[0][:] = 0                                        # C1: all inputs
        m[1] = np.where(np.arange(128) < 64, 1, 2)         # C2, C3
        m[2][:] = 3                                        # C4
        m[3][:] = 4                                        # C5
    return m

CORE_NAMES = {("layer", 4): ["inputs, top half", "inputs, bottom half", "hidden 1, first half",
                             "hidden 1, second half + hidden 2 + outputs"],
              ("layer", 5): ["inputs, top half", "inputs, bottom half", "hidden 1, first half",
                             "hidden 1, second half", "hidden 2 + outputs"],
              ("inputs_one_core", 4): ["all inputs", "hidden 1, first half", "hidden 1, second half",
                                       "hidden 2 + outputs"],
              ("inputs_one_core", 5): ["all inputs", "hidden 1, first half", "hidden 1, second half",
                                       "hidden 2", "outputs"]}[MAPPING, N]

def random_mapping(seed=1):
    rng = np.random.default_rng(seed)
    tot = sum(sizes)
    cores = rng.permutation(np.arange(tot) % N)
    out, o = [], 0
    for s in sizes:
        out.append(cores[o:o + s]); o += s
    return out

def traffic(mapping):
    """Return traffic[t, src, dst] packets per bin over the concatenated stream."""
    tr = np.zeros((Ns * T, N, N), np.int32)
    for L in range(3):                                     # layers that project forward
        spk = layers[L].reshape(Ns * T, -1)
        dst_cores = set(mapping[L + 1].tolist())
        for c in range(N):
            cnt = spk[:, mapping[L] == c].sum(1).astype(np.int32)  # spikes from core c
            for j in dst_cores:
                if j != c:
                    tr[:, c, j] += cnt                     # one packet per destination core
    return tr

maps = {"layer": layer_mapping(), "random": random_mapping()}
traf = {k: traffic(m) for k, m in maps.items()}

# ---------------- optimizer over reachable settings ----------------
A, STATES = feasible_assignments()
A_IDX = np.where(A < 0, N, A)                              # -1 -> padded zero row
print(f"reachable settings: {len(A)} of {N ** N} patterns (all but 3-way multicasts)")

def pad(W):
    return np.concatenate([W, np.zeros((W.shape[0], 1, N), W.dtype)], 1)

def carried_all(W):
    """W (nW, N, N) -> (nW, F): packets each reachable setting carries in each window."""
    Wp = pad(W.astype(float))
    return sum(Wp[:, A_IDX[:, j], j] for j in range(N))

def best_index(W):
    """Index of the best reachable setting for each window of W (nW, N, N)."""
    return np.concatenate([carried_all(W[i:i + CHUNK]).argmax(1) for i in range(0, len(W), CHUNK)])

def carried_by(W, idx):
    """Packets carried in each window by the chosen settings idx (nW,)."""
    Wp = pad(W.astype(float))
    return Wp[np.arange(len(W))[:, None], A_IDX[idx], np.arange(N)].sum(1)

def ideal_best(W):
    """Unrestricted mesh: each destination takes its largest sender (per window)."""
    Wc = W.astype(float).copy()
    Wc[:, np.arange(N), np.arange(N)] = 0
    return Wc.max(1).sum(1)

# check: on random traffic, how often do the chip's limits (no 1/3 split) cost anything?
rng = np.random.default_rng(0)
Tm = rng.integers(0, 100, (500, N, N)) * (rng.random((500, N, N)) < 0.5)
loss = 1 - carried_by(Tm, best_index(Tm)) / ideal_best(Tm)
print(f"random traffic: chip carries the ideal optimum in {np.mean(loss < 1e-9):.0%} of cases, "
      f"worst shortfall {loss.max():.1%}")

# ---------------- full-dataset statistics ----------------
def evaluate(tr, win_bins, tau_ms):
    nW = tr.shape[0] // win_bins
    W = tr[:nW * win_bins].reshape(nW, win_bins, N, N).sum(1)
    total = tr[:nW * win_bins].sum()
    static = best_index(tr.sum(0)[None])[0]
    best = best_index(W)
    choice = np.r_[static, best[:-1]]                       # chosen from last window
    c = carried_by(W, choice)
    changed = np.r_[False, choice[1:] != choice[:-1]]       # dead time after a change
    c[changed] *= max(0.0, 1 - tau_ms / (win_bins * dt))
    return {"static": carried_by(W, np.full(nW, static)).sum() / total,
            "adaptive": c.sum() / total,
            "oracle": carried_by(W, best).sum() / total,
            "ideal_oracle": ideal_best(W).sum() / total}

stats = {}
for k, tr in traf.items():
    stats[k] = {}
    for win_ms in [2, 10, 20, 50, 100, 300]:
        stats[k][win_ms] = evaluate(tr, win_ms // dt, tau_ms=0.1)
        print(k, win_ms, {a: round(b, 3) for a, b in stats[k][win_ms].items()})
    tot = tr.sum(0)
    print(k, "total traffic matrix (packets):\n", tot)
    s = best_index(tot[None])[0]
    print(k, "best fixed setting:", ", ".join(f"C{A[s][j] + 1}->C{j + 1}" for j in range(N) if A[s][j] >= 0),
          "| MZIs:", " ".join(f"{m}={x}" for m, x in zip(MZIS, STATES[s])))

# ---------------- export for the interactive page ----------------
SEG = 20                                                   # samples shown in page
seg_bins = SEG * T
exp = {"real": bool(d["real_nmnist"]) if "real_nmnist" in d else False, "dt_ms": dt, "bins_per_sample": T, "acc": float(d["acc"]),
       "labels": d["labels"][:SEG].tolist(), "stats": stats, "mappings": {},
       "n": N, "core_names": CORE_NAMES,
       "mesh": {"core_row": CORE_ROW, "fixed_bar": FIXED_BAR, "mzis": MZIS, "pair": [PAIR[m] for m in MZIS],
                "receiver_pd": RECEIVER_PD, "n_patterns": N ** N},
       "feasible": A.tolist(), "states": STATES}
for k, tr in traf.items():
    seg = tr[:seg_bins]
    pairs = [(i, j) for i in range(N) for j in range(N) if seg[:, i, j].sum() > 0]
    exp["mappings"][k] = {
        "pairs": pairs,
        "series": [seg[:, i, j].tolist() for i, j in pairs],
        "total_matrix": tr.sum(0).tolist(),
        "core_sizes": [[int((m == c).sum()) for m in maps[k]] for c in range(N)],
    }
# accumulated event image per sample (for display): ON - OFF counts, 34x34
inp = d["inp"][:SEG].reshape(SEG, T, 2, 34, 34).sum(1)
img = (inp[:, 0] - inp[:, 1]).astype(int)
exp["images"] = img.tolist()
# predicted labels
pred = d["out"][:SEG].sum(1).argmax(1)
exp["pred"] = pred.tolist()
json.dump(exp, open("sim_data.json", "w"), separators=(",", ":"))
print("exported", len(json.dumps(exp)) / 1e3, "kB")
