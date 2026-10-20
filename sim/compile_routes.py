"""Compile a routing setting (who listens to whom) into MZI states on the real chip, using
only BAR, CROSS and SPLIT (50/50) states, so split branches never recombine and routing is
independent of every external phase (phi) and laser phase.

Hardware (measured 2026-10-19): 6 waveguide rows, 5 columns, 13 MZIs. MZI "r_c" couples
rows r and r+1 in column c. Columns 1, 3, 5 have MZIs on rows 1-2, 3-4, 5-6; columns 2, 4
on rows 2-3, 4-5. At 0 V every MZI sits near CROSS.

CORE_ROW picks the rows used as cores (core Ck on CORE_ROW[k-1]); everything else follows:
  - routing MZIs: both rows used
  - boundary MZIs: one row used; held at BAR so light stays inside the used rows
  - 1_5 is faulty (latches between ~0.9 and 1.5 V), so it is never driven: at 0 V it sits
    near CROSS, so row 2's light leaves on PD_1_5, which becomes that core's receiver
    (~85% of the light; row 1 is dark, so nothing mixes in)
  - 1_1, 1_3, 5_5 cannot reach full CROSS within 4.2 V; ties are broken away from that
Every routing pattern is reachable except a multicast to exactly 3 receivers (needs 1/3).
"""
import itertools, json, numpy as np

CORE_ROW = [2, 3, 4, 5, 6]                       # 5 cores on rows 2-6; [2, 3, 4, 5] for the 4x4
ALL_MZIS = ["1_1", "3_1", "5_1", "2_2", "4_2", "1_3", "3_3", "5_3", "2_4", "4_4", "1_5", "3_5", "5_5"]
FAULTY = ["1_5"]                                 # held at 0 V, never driven
CROSS_OUT_OF_RANGE = ["1_1", "1_3", "5_5"]       # full CROSS needs ~5 V

N = len(CORE_ROW)
_rows = lambda m: {int(m.split("_")[0]), int(m.split("_")[0]) + 1}
MZIS = [m for m in ALL_MZIS if _rows(m) <= set(CORE_ROW)]                       # column order
FIXED_BAR = [m for m in ALL_MZIS if len(_rows(m) & set(CORE_ROW)) == 1 and m not in FAULTY]
PAIR = {m: int(m.split("_")[0]) - CORE_ROW[0] for m in MZIS}   # local port p couples p, p+1
STATE_NAMES = {"B": "BAR", "X": "CROSS", "S": "SPLIT 50/50"}
AVOID_CROSS = [m for m in MZIS if m in CROSS_OUT_OF_RANGE]
# where each core's light leaves the chip (1_5 at 0 V swaps row 2 onto row 1's detector)
RECEIVER_PD = ["PD_1_5" if (r == 2 and "1_5" in FAULTY) else f"PD_{r}_5" for r in CORE_ROW]


def power_matrix(states):
    """Incoherent power transfer P[in, out] for a string of "B"/"X"/"S" states (one per MZI)."""
    P = np.eye(N)
    for m, s in zip(MZIS, states):
        p, M = PAIR[m], np.eye(N)
        if s == "X": M[[p, p + 1]] = M[[p + 1, p]]
        if s == "S": M[p:p + 2, p:p + 2] = 0.5
        P = P @ M
    return P


def coherent_check(states, trials=50, rng=np.random.default_rng(0)):
    """Max change of required powers under random phases on every arm (should be ~0)."""
    ref = None; worst = 0
    for _ in range(trials):
        U = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, N)))
        for m, s in zip(MZIS, states):
            p, t = PAIR[m], {"B": 0.0, "X": np.pi / 2, "S": np.pi / 4}[s]
            B = np.array([[np.cos(t), 1j * np.sin(t)], [1j * np.sin(t), np.cos(t)]])
            M = np.eye(N, dtype=complex); M[p:p + 2, p:p + 2] = B
            U = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, N))) @ M @ U
        P = np.abs(U) ** 2           # P[out, in]
        if ref is None: ref = P
        worst = max(worst, np.abs(P - ref).max())
    return worst


def feasible_assignments():
    """Every assignment a[dst] = src (or -1) the chip can make, with its preferred MZI states.

    A destination listens to at most one source; a source sending to k destinations gives
    each 1/k. Returns (A of shape (F, N), list of F state strings like "BXSB...").
    Among states realizing an assignment, prefers no CROSS on AVOID_CROSS, then fewer splits.
    """
    codes = np.array(list(itertools.product(range(3), repeat=len(MZIS))))  # 0 B, 1 X, 2 S
    P = np.broadcast_to(np.eye(N), (len(codes), N, N)).copy()
    for i, m in enumerate(MZIS):
        p, M = PAIR[m], np.broadcast_to(np.eye(N), (len(codes), N, N)).copy()
        x, s = codes[:, i] == 1, codes[:, i] == 2
        M[x, p, p] = M[x, p + 1, p + 1] = 0; M[x, p, p + 1] = M[x, p + 1, p] = 1
        M[s, p:p + 2, p:p + 2] = 0.5
        P = P @ M
    quarters = np.round(P * 4).astype(np.int8)        # every reachable power is a multiple of 1/4
    penalty = (codes == 2).sum(1)
    for m in AVOID_CROSS:
        penalty = penalty + 10 * (codes[:, MZIS.index(m)] == 1)
    A, states = [], []
    for a in itertools.product(range(-1, N), repeat=N):
        a = np.array(a)
        if any(a[j] == j for j in range(N)):
            continue
        ok = np.ones(len(codes), bool)
        for s in set(a[a >= 0].tolist()):
            k = int((a == s).sum())
            if 4 % k:                                    # 1/3 split: not reachable
                ok[:] = False; break
            want = np.zeros(N, np.int8); want[a == s] = 4 // k
            ok &= (quarters[:, s, :] == want).all(1)
        if ok.any():
            best = np.flatnonzero(ok)[np.argmin(penalty[ok])]
            A.append(a); states.append("".join("BXS"[c] for c in codes[best]))
    return np.array(A), states


def compile_setting(target, feasible=None):
    """target: dict src -> list of dst (0-based cores). Returns MZI states or None."""
    A, states = feasible if feasible is not None else feasible_assignments()
    a = np.full(N, -1)
    for s, ds in target.items():
        a[ds] = s
    hit = np.flatnonzero((A == a).all(1))
    return states[hit[0]] if hit.size else None


def describe(states):
    """Hardware heater settings for a compiled setting, including the fixed MZIs."""
    out = {m: "BAR" for m in FIXED_BAR}
    out.update({m: STATE_NAMES[s] for m, s in zip(MZIS, states)})
    out.update({m: "0 V (not driven)" for m in FAULTY})
    return out


# The settings the controller uses (C1/C2 = input halves, C3/C4 = hidden-1 halves, C5 = hidden-2 +
# outputs). With 5 cores two heaters choose independently: 2_2 picks the input sender (CROSS: C1),
# 4_2 the hidden sender (CROSS: C3), so all four combinations occur.
SETTINGS = {"A": {1: [2, 3]}, "B": {0: [2, 3]}} if N == 4 else \
           {"A": {1: [2, 3], 3: [4]}, "B": {0: [2, 3], 2: [4]},
            "C": {0: [2, 3], 3: [4]}, "D": {1: [2, 3], 2: [4]}}


if __name__ == "__main__":
    feasible = feasible_assignments()
    A, states = feasible
    print(f"{N} cores on rows {CORE_ROW}: routing MZIs {MZIS}; boundary at BAR {FIXED_BAR}; "
          f"not driven {FAULTY}; receivers {RECEIVER_PD}")
    print(f"{len(A)} of {N ** N} assignments reachable (all except 3-way multicasts)")
    tests = {f"{k} ({', '.join(f'C{s+1}->' + '+'.join(f'C{d+1}' for d in ds) for s, ds in v.items())})": v
             for k, v in SETTINGS.items()}
    tests[f"permutation (C1->C2->...->C{N}->C1)"] = {i: [(i + 1) % N] for i in range(N)}
    tests[f"1->3 test (C1->C2,C3,C4)"] = {0: [1, 2, 3]}
    for name, tgt in tests.items():
        st = compile_setting(tgt, feasible)
        print("\n" + name)
        if st is None:
            print("  no bar/cross/50:50 solution; needs an unequal split (e.g. 1/3 : 2/3)"); continue
        print("  " + ", ".join(f"{m} {s}" for m, s in describe(st).items()))
        P = power_matrix(st)
        print("  power from active inputs:", {f"C{s+1}": {f"C{j+1}": round(P[s, j], 2) for j in range(N) if P[s, j] > 0} for s in tgt})
        print(f"  phase sensitivity of these powers: {coherent_check(st):.1e}")
    # Hand-off to the bench scripts: every reachable assignment with its MZI states
    out = f"routes_rows{CORE_ROW[0]}-{CORE_ROW[-1]}.json"
    json.dump({"core_row": CORE_ROW, "lasers": [f"laser_{r}" for r in CORE_ROW],
               "input_pd": [f"PD_{r}_0" for r in CORE_ROW], "receiver_pd": RECEIVER_PD,
               "fixed_bar": FIXED_BAR, "not_driven": FAULTY, "mzis": MZIS,
               "settings": {k: {"states": compile_setting(v, feasible), "routes": {str(s): d for s, d in v.items()}}
                            for k, v in SETTINGS.items()},
               "assignments": [{"a": a.tolist(), "states": s} for a, s in zip(A, states)]},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")
