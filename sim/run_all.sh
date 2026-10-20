#!/usr/bin/env bash
# One-command pipeline: real N-MNIST -> train SNN -> traffic + mesh optimizer -> interactive page.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "1/4  Downloading and binning N-MNIST (first run downloads ~1 GB)"
python load_nmnist.py
echo "2/4  Training the SNN"
python train_snn.py
echo "3/4  Extracting core-to-core traffic and running the mesh optimizer"
python traffic.py
echo "4/4  Building the interactive page"
python build_page.py
open spike_express_lanes.html 2>/dev/null || echo "Open spike_express_lanes.html in your browser."
