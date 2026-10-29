"""Model size, FLOPs and CPU latency (random weights, 224x224 input, 4 threads)."""
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fusionlib as fl
from tea_models import LABELS, MODELS, build

torch.set_num_threads(4)
R = Path("outputs/revision")
R.mkdir(parents=True, exist_ok=True)


def flops(model):
    """Multiply-adds of conv and linear layers counted with forward hooks, times 2."""
    total = []

    def conv(m, i, o):
        total.append(o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1])

    def lin(m, i, o):
        total.append(o.numel() * m.in_features)

    hooks = [m.register_forward_hook(conv) for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    hooks += [m.register_forward_hook(lin) for m in model.modules() if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        model(torch.zeros(1, 3, 224, 224))
    for h in hooks:
        h.remove()
    return 2 * sum(total) / 1e9


@torch.no_grad()
def latency(model, batch, runs=30):
    x = torch.randn(batch, 3, 224, 224)
    for _ in range(3):
        model(x)
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        t.append((time.perf_counter() - t0) * 1000 / batch)
    return float(np.median(t))


summ = json.loads(Path("outputs/summary.json").read_text())["models"]
rows = []
for m in MODELS:
    net, _ = build(m, 7, pretrained=False)
    if m == "vgg16":
        for p in net.features[:24].parameters():
            p.requires_grad = False
    net.eval()
    params = sum(p.numel() for p in net.parameters())
    rows.append({"model": LABELS[m], "parameters": params, "trainable": sum(p.numel() for p in net.parameters() if p.requires_grad),
                 "size_mb": params * 4 / 1e6, "gflops": flops(net), "cpu_ms_b1": latency(net, 1), "cpu_ms_b32": latency(net, 32, 10),
                 "mps_ms_train_log": np.mean([summ[f"{LABELS[m]} seed {s}"]["ms_per_image"] for s in (42, 7, 123)])})
    print(rows[-1], flush=True)
df = pd.DataFrame(rows)
probs = [np.random.dirichlet(np.ones(7), 1) for _ in range(3)]
t0 = time.perf_counter()
for _ in range(1000):
    fl.weighted_average(probs, np.ones(3))
fusion_ms = (time.perf_counter() - t0)
tot = df.sum(numeric_only=True)
df.loc[len(df)] = {"model": "Triple ensemble (sequential)", "parameters": tot.parameters, "trainable": tot.trainable, "size_mb": tot.size_mb,
                   "gflops": tot.gflops, "cpu_ms_b1": tot.cpu_ms_b1 + fusion_ms, "cpu_ms_b32": tot.cpu_ms_b32, "mps_ms_train_log": tot.mps_ms_train_log}
try:
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
except OSError:
    cpu = platform.processor()
df.to_csv(R / "e15_benchmark.csv", index=False)
(R / "e15_env.json").write_text(json.dumps({"cpu": cpu or platform.processor(), "threads": 4, "torch": torch.__version__,
                                            "fusion_ms_per_image": fusion_ms}, indent=1))
print(df.round(2).to_string(index=False))
print("cpu:", cpu, "fusion ms/image:", round(fusion_ms, 4))
