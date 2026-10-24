"""Remedies that need no labels from the new dataset, run for every seed:
background removal at test time, and AdaBN (VGG16 has no BatchNorm so it is unchanged).
The agreement gate is in gate.py.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from leafmask import apply
from tea_models import LABELS, MODELS, build

OUT, A = Path("outputs"), Path("outputs/analysis")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
M, S = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SEEDS = json.loads((OUT / "summary.json").read_text())["seeds"]
z = np.load(A / "external_cache.npz")
X, y = z["X"], z["y"]
masked = A / "external_masked.npy"
if not masked.exists():
    np.save(masked, np.stack([apply(x) for x in X]))
Xm = np.load(masked)
shared = sorted(set(y.tolist()))
yv = pd.read_csv(OUT / "split.csv").query("split == 'val'").label.to_numpy()


def batches(arr, bs=64):
    for i in range(0, len(arr), bs):
        xb = torch.from_numpy(arr[i:i + bs]).permute(0, 3, 1, 2).float().div(255)
        yield ((F.interpolate(xb, size=(224, 224), mode="bilinear", antialias=True, align_corners=False) - M) / S).to(DEVICE)


@torch.no_grad()
def predict(net, arr):
    net.eval()
    return torch.cat([F.softmax(net(xb).float(), 1).cpu() for xb in batches(arr)]).numpy()


@torch.no_grad()
def adabn(net, arr):
    net = copy.deepcopy(net)
    bns = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    net.eval()
    for m in bns:
        m.train()
    for xb in batches(arr):
        net(xb)
    return net, len(bns)


def cache(path, fn):
    if path.exists():
        return np.load(path)
    p = fn()
    np.save(path, p)
    return p


rows = []
for s in SEEDS:
    P = {"Baseline": {}, "Background removal (test time)": {}, "AdaBN (test time)": {}}
    w = {m: accuracy_score(yv, np.load(OUT / f"probs_{m}_seed{s}.npz")["val"].argmax(1)) for m in MODELS}
    for m in MODELS:
        net, _ = build(m, 7, pretrained=False)
        net.load_state_dict(torch.load(OUT / f"{m}_seed{s}.pt", map_location="cpu", weights_only=True))
        net.to(DEVICE)
        P["Baseline"][m] = np.load(A / f"ext_probs_{m}_seed{s}.npy")
        P["Background removal (test time)"][m] = cache(A / f"rem_cutout_{m}_seed{s}.npy", lambda: predict(net, Xm))
        P["AdaBN (test time)"][m] = cache(A / f"rem_adabn_{m}_seed{s}.npy", lambda: predict(adabn(net, X)[0], X))
        print(f"seed {s} {LABELS[m]} done", flush=True)
    tw = sum(w.values())
    for rem, pm in P.items():
        for c in [("densenet121",), tuple(MODELS)]:
            ww = sum(w[m] for m in c)
            fused = sum(w[m] / ww * pm[m] for m in c)
            rows.append({"seed": s, "remedy": rem, "config": " + ".join(LABELS[m] for m in c),
                         "acc": accuracy_score(y, fused.argmax(1)),
                         "macro_f1": f1_score(y, fused.argmax(1), labels=shared, average="macro", zero_division=0)})

df = pd.DataFrame(rows)
df.to_csv(A / "remedies_all_seeds.csv", index=False)
g = df.drop(columns="seed").groupby(["remedy", "config"])
out = {"mean": {f"{r}|{c}": v for (r, c), v in g.mean().round(4).to_dict(orient="index").items()},
       "std": {f"{r}|{c}": v for (r, c), v in g.std().fillna(0).round(4).to_dict(orient="index").items()}, "seeds": SEEDS}
(A / "remedies_summary.json").write_text(json.dumps(out, indent=1))
print((g.mean() * 100).round(2).to_string())
