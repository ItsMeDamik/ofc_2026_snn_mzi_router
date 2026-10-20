# Spike express lanes: SNN traffic offload through the chip's MZI mesh (5x5 on rows 2-6)

Run everything (from Terminal, on the Mac; the lab PC does not need this folder's dependencies):

    cd sim
    bash run_all.sh

It creates a local virtual environment, installs numpy/torch/snntorch/tonic,
downloads the real N-MNIST dataset into `./data`, and runs:

| Step | Script | Output |
|---|---|---|
| 1 | `load_nmnist.py` | `events.npz`: 4000 train + 1000 test samples, 2 ms bins, 150 bins (300 ms) |
| 2 | `train_snn.py` | 2312-128-64-10 LIF SNN; `spikes_test.npz` with every neuron's test spikes |
| 3 | `traffic.py` | maps neurons to cores, inter-core traffic, optimizer over the chip's reachable settings, `sim_data.json` |
| 4 | `build_page.py` | `spike_express_lanes.html`, the interactive simulation (opens in your browser) |

Time on a recent MacBook: download a few minutes, training roughly 5-10 min, the rest under a minute.
Change `N_TRAIN` / `N_TEST` in `load_nmnist.py` for a bigger run (about 350 kB RAM per sample).
The `.npz` files are not in git; `spikes_test.npz` (the trained SNN's spikes) is enough to rerun
steps 3-4 without the dataset.

## The mesh rule used by the optimizer

Each mesh output (photodiode = receiving core) listens to at most one sending core per setting.
Multicast to k cores is an equal 1/k split.

The chip (measured 2026-10-19) has 13 MZIs in 5 columns, so the 6x6 is not universal, and heater
1_5 is faulty (never driven). `CORE_ROW` in `compile_routes.py` picks the rows used as cores
(default rows 2-6, 5 cores; `[2, 3, 4, 5]` gives the 4x4). From it follow the routing MZIs,
the edge MZIs held at BAR, and each core's receiver (row 2's light leaves on `PD_1_5` because the
undriven 1_5 sits near CROSS). Using only BAR, CROSS and 50:50, every pattern except a multicast
to exactly 3 cores is reachable (2,725 of 3,125 for 5 cores). `python compile_routes.py` writes
them, plus the named settings, to `routes_rows2-6.json`, which `../route_check.py` programs on the chip.

`traffic.py` picks, per window, the reachable setting that carries the most traffic; for this
network that always equals an unrestricted mesh. With 5 cores (C1/C2 input halves, C3/C4 hidden-1
halves, C5 hidden-2 + outputs) two heaters choose independently: 2_2 picks which input core
multicasts to C3+C4 (CROSS: C1), 4_2 which hidden core sends to C5 (CROSS: C3). All four
combinations occur (settings A-D in `compile_routes.SETTINGS`).

## If tonic fails to install

`tonic` depends on `expelliarmus`, which needs a C compiler. On macOS run `xcode-select --install`
once, then rerun `bash run_all.sh`.
