"""Train a small feedforward SNN (2312-128-64-10, LIF) on the N-MNIST-style events,
then record every neuron's spikes on the test set for traffic extraction."""
import numpy as np, torch, torch.nn as nn, snntorch as snn
from snntorch import surrogate

torch.manual_seed(0)
d = np.load("events.npz")
ev, lab = d["events"], d["labels"]
N, T = ev.shape[:2]
ev = ev.reshape(N, T, -1)                      # (N, T, 2312)
if "n_train" in d:                              # real N-MNIST: first n_train are train
    ntr = int(d["n_train"])
else:                                           # emulated subset is sorted by class
    shuf = np.random.default_rng(42).permutation(N)
    ev, lab = ev[shuf], lab[shuf]
    ntr = 4000
Xtr, ytr, Xte, yte = ev[:ntr], lab[:ntr], ev[ntr:], lab[ntr:]

sg = surrogate.fast_sigmoid(slope=25)
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.fc1, s.fc2, s.fc3 = nn.Linear(2312, 128), nn.Linear(128, 64), nn.Linear(64, 10)
        s.l1 = snn.Leaky(beta=0.9, spike_grad=sg)
        s.l2 = snn.Leaky(beta=0.9, spike_grad=sg)
        s.l3 = snn.Leaky(beta=0.9, spike_grad=sg)
    def forward(s, x):                          # x: (B, T, 2312)
        m1, m2, m3 = s.l1.init_leaky(), s.l2.init_leaky(), s.l3.init_leaky()
        rec = []
        for t in range(x.shape[1]):
            s1, m1 = s.l1(s.fc1(x[:, t]), m1)
            s2, m2 = s.l2(s.fc2(s1), m2)
            s3, m3 = s.l3(s.fc3(s2), m3)
            rec.append((s1, s2, s3))
        s1, s2, s3 = (torch.stack(z, 1) for z in zip(*rec))
        return s1, s2, s3

net = Net()
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
lossf = nn.CrossEntropyLoss()
B = 50
for ep in range(4):
    perm = np.random.default_rng(ep).permutation(ntr)
    tot = 0
    for i in range(0, ntr, B):
        idx = perm[i:i + B]
        x = torch.from_numpy(Xtr[idx]).float()
        _, _, s3 = net(x)
        loss = lossf(s3.sum(1), torch.from_numpy(ytr[idx]))
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(idx)
    print(f"epoch {ep}: loss {tot / ntr:.3f}", flush=True)

# test + record spikes
net.eval()
S1, S2, S3, correct = [], [], [], 0
with torch.no_grad():
    for i in range(0, len(Xte), 100):
        x = torch.from_numpy(Xte[i:i + 100]).float()
        s1, s2, s3 = net(x)
        correct += (s3.sum(1).argmax(1).numpy() == yte[i:i + 100]).sum()
        S1.append(s1.numpy().astype(np.uint8)); S2.append(s2.numpy().astype(np.uint8))
        S3.append(s3.numpy().astype(np.uint8))
acc = correct / len(Xte)
print(f"test accuracy {acc:.3f}")
np.savez_compressed("spikes_test.npz", inp=Xte, h1=np.concatenate(S1),
                    h2=np.concatenate(S2), out=np.concatenate(S3), labels=yte,
                    acc=acc, dt_ms=int(d["dt_ms"]),
                    real_nmnist=bool(d["real_nmnist"]) if "real_nmnist" in d else False)
