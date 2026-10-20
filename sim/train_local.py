"""Train a locally connected variant of the SNN (2312-128-64-10, LIF) for the scaling study.

Same as train_snn.py except the first layer: the 128 hidden-1 neurons form 16 groups of 8,
one per tile of a 4x4 grid over the 34x34 image (tiles widened by a 2-pixel halo, both
event polarities). A neuron only has weights to its own tile, so input spikes travel to a few
nearby cores instead of to all of them, the kind of locality large SNNs have.

Writes spikes_test_local.npz (same format as spikes_test.npz) and snn_local_weights.npz.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, snntorch as snn
from snntorch import surrogate

EDGES = [0, 9, 17, 26, 34]                     # 4x4 tiles of the 34x34 image
HALO = 2
GROUP = 8                                      # hidden-1 neurons per tile (16 x 8 = 128)


def tile_mask() -> np.ndarray:
    """mask[n, p] = 1 if hidden-1 neuron n sees input pixel p (index = pol*1156 + row*34 + col)."""
    mask = np.zeros((128, 2312), np.float32)
    rows, cols = np.divmod(np.arange(1156), 34)
    for n in range(128):
        g = n // GROUP
        r0, r1 = EDGES[g // 4] - HALO, EDGES[g // 4 + 1] + HALO
        c0, c1 = EDGES[g % 4] - HALO, EDGES[g % 4 + 1] + HALO
        inside = (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)
        mask[n] = np.tile(inside, 2)
    return mask


if __name__ == "__main__":
    torch.manual_seed(0)
    d = np.load("events.npz")
    ev, lab = d["events"], d["labels"]
    N, T = ev.shape[:2]
    ev = ev.reshape(N, T, -1)
    ntr = int(d["n_train"]) if "n_train" in d else 4000
    Xtr, ytr, Xte, yte = ev[:ntr], lab[:ntr], ev[ntr:], lab[ntr:]
    mask = torch.from_numpy(tile_mask())

    sg = surrogate.fast_sigmoid(slope=25)
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.fc1, s.fc2, s.fc3 = nn.Linear(2312, 128), nn.Linear(128, 64), nn.Linear(64, 10)
            s.l1, s.l2, s.l3 = (snn.Leaky(beta=0.9, spike_grad=sg) for _ in range(3))
        def forward(s, x):
            m1, m2, m3 = s.l1.init_leaky(), s.l2.init_leaky(), s.l3.init_leaky()
            rec = []
            for t in range(x.shape[1]):
                s1, m1 = s.l1(F.linear(x[:, t], s.fc1.weight * mask, s.fc1.bias), m1)
                s2, m2 = s.l2(s.fc2(s1), m2)
                s3, m3 = s.l3(s.fc3(s2), m3)
                rec.append((s1, s2, s3))
            return (torch.stack(z, 1) for z in zip(*rec))

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss()
    B = 50
    for ep in range(4):
        perm = np.random.default_rng(ep).permutation(ntr)
        tot = 0
        for i in range(0, ntr, B):
            idx = perm[i:i + B]
            _, _, s3 = net(torch.from_numpy(Xtr[idx]).float())
            loss = lossf(s3.sum(1), torch.from_numpy(ytr[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f"epoch {ep}: loss {tot / ntr:.3f}", flush=True)

    net.eval()
    S1, S2, S3, correct = [], [], [], 0
    with torch.no_grad():
        for i in range(0, len(Xte), 100):
            s1, s2, s3 = net(torch.from_numpy(Xte[i:i + 100]).float())
            correct += (s3.sum(1).argmax(1).numpy() == yte[i:i + 100]).sum()
            S1.append(s1.numpy().astype(np.uint8)); S2.append(s2.numpy().astype(np.uint8))
            S3.append(s3.numpy().astype(np.uint8))
    acc = correct / len(Xte)
    print(f"test accuracy {acc:.3f}")
    state = {k: v.numpy() for k, v in net.state_dict().items()}
    state["fc1.weight"] = state["fc1.weight"] * mask.numpy()
    np.savez_compressed("snn_local_weights.npz", **state, mask=mask.numpy(), beta=0.9, threshold=1.0)
    np.savez_compressed("spikes_test_local.npz", inp=Xte, h1=np.concatenate(S1), h2=np.concatenate(S2),
                        out=np.concatenate(S3), labels=yte, acc=acc, dt_ms=int(d["dt_ms"]))
