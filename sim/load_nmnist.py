"""Download real N-MNIST with tonic and bin it into 2 ms frames.

Output: events.npz with events[n, 150, 2, 34, 34] (uint8, 0/1) and labels[n],
the same format the rest of the pipeline expects.
"""
import numpy as np
import tonic

DT_US = 2000           # 2 ms bins
STEPS = 150            # 300 ms per sample (N-MNIST: three ~100 ms saccades)
N_TRAIN, N_TEST = 4000, 1000   # raise these if you have RAM/time (each sample ~350 kB)


def to_frames(ev):
    b = ev["t"] // DT_US
    keep = b < STEPS
    fr = np.zeros((STEPS, 2, 34, 34), np.uint8)
    fr[b[keep], ev["p"][keep].astype(int), ev["y"][keep], ev["x"][keep]] = 1
    return fr


def load(train, n, seed):
    ds = tonic.datasets.NMNIST(save_to="./data", train=train)
    idx = np.random.default_rng(seed).permutation(len(ds))[:n]
    X = np.zeros((n, STEPS, 2, 34, 34), np.uint8)
    y = np.zeros(n, np.int64)
    for k, i in enumerate(idx):
        ev, lab = ds[int(i)]
        X[k], y[k] = to_frames(ev), lab
        if k % 1000 == 0:
            print(f"  {'train' if train else 'test'} {k}/{n}", flush=True)
    return X, y


Xtr, ytr = load(True, N_TRAIN, 0)
Xte, yte = load(False, N_TEST, 1)
events = np.concatenate([Xtr, Xte])
labels = np.concatenate([ytr, yte])
np.savez_compressed("events.npz", events=events, labels=labels, dt_ms=DT_US // 1000,
                    n_train=N_TRAIN, real_nmnist=True)
print(f"saved {events.shape}, mean rate {events.mean() * 1000 / (DT_US / 1000):.2f} Hz/pixel, "
      f"events/sample {events.reshape(len(events), -1).sum(1).mean():.0f}")
