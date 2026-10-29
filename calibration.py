"""ECE, temperature scaling, AUC and the 0.80 confidence threshold.

Temperature is fitted on the teaLeafBD val set and reused as is on the test set and TeaLeafDiseaseBD.
"""
import json
import os
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, roc_auc_score

from tea_models import LABELS, MODELS

OUT = Path(os.environ.get("TEA_OUT", "outputs"))
A = OUT / "analysis"
THRESH, BINS = 0.80, 15
CONFIGS = [c for r in (1, 2, 3) for c in combinations(MODELS, r)]

split = pd.read_csv(OUT / "split.csv")
y_val = split.loc[split.split == "val", "label"].to_numpy()
y_test = split.loc[split.split == "test", "label"].to_numpy()
SEEDS = json.loads((OUT / "summary.json").read_text())["seeds"]
probs = {(s, m): np.load(OUT / f"probs_{m}_seed{s}.npz") for s in SEEDS for m in MODELS}
has_ext = (A / "external_labels.npy").exists()
if has_ext:
    y_ext = np.load(A / "external_labels.npy")
    ext = {(s, m): np.load(A / f"ext_probs_{m}_seed{s}.npy") for s in SEEDS for m in MODELS}
    shared = sorted(set(y_ext.tolist()))


def name(c):
    return " + ".join(LABELS[m] for m in c)


def fuse(p, s, c):
    va = {m: accuracy_score(y_val, probs[s, m]["val"].argmax(1)) for m in c}
    w = np.array([va[m] for m in c]) / sum(va.values())
    return sum(wi * p[m] for wi, m in zip(w, c))


def scale(p, t):
    z = np.log(np.clip(p, 1e-12, 1)) / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(p, y):
    nll = lambda t: -np.log(np.clip(scale(p, t)[np.arange(len(y)), y], 1e-12, 1)).mean()
    return minimize_scalar(nll, bounds=(0.05, 10), method="bounded").x


def ece(p, y):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, BINS + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return total


def threshold(p, y):
    keep = p.max(1) >= THRESH
    return keep.mean(), (p.argmax(1)[keep] == y[keep]).mean() if keep.any() else float("nan")


def auc(p, y, labels):
    return np.mean([roc_auc_score(y == c, p[:, c]) for c in labels])


rows = []
for s in SEEDS:
    for c in CONFIGS:
        pv = fuse({m: probs[s, m]["val"] for m in MODELS}, s, c)
        pt = fuse({m: probs[s, m]["test"] for m in MODELS}, s, c)
        t = fit_temperature(pv, y_val)
        pts = scale(pt, t)
        cov, accu = threshold(pt, y_test)
        cov_s, acc_s = threshold(pts, y_test)
        r = {"seed": s, "config": name(c), "T": t, "auc": auc(pt, y_test, range(pt.shape[1])),
             "ece_raw": ece(pt, y_test), "ece_ts": ece(pts, y_test),
             "cover_raw": cov, "acc_above_raw": accu, "cover_ts": cov_s, "acc_above_ts": acc_s}
        if has_ext:
            pe = fuse({m: ext[s, m] for m in MODELS}, s, c)
            pes = scale(pe, t)
            ce, ae = threshold(pe, y_ext)
            ces, aes = threshold(pes, y_ext)
            r.update({"ext_auc": auc(pe, y_ext, shared), "ext_ece_raw": ece(pe, y_ext), "ext_ece_ts": ece(pes, y_ext),
                      "ext_cover_raw": ce, "ext_acc_above_raw": ae, "ext_cover_ts": ces, "ext_acc_above_ts": aes,
                      "ext_conf_wrong_raw": ce * (1 - ae), "ext_conf_wrong_ts": ces * (1 - aes)})
        rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(A / "calibration_all_seeds.csv", index=False)
order = [name(c) for c in CONFIGS]
g = df.drop(columns="seed").groupby("config")
summary = {"mean": g.mean().reindex(order).round(4).to_dict(orient="index"),
           "std": g.std().fillna(0).reindex(order).round(4).to_dict(orient="index"), "threshold": THRESH, "bins": BINS}
(A / "calibration_summary.json").write_text(json.dumps(summary, indent=1))
print((g.mean().reindex(order) * 1).round(3).to_string())

# reliability plot, triple ensemble, first seed
s0, tri = SEEDS[0], CONFIGS[-1]
pv = fuse({m: probs[s0, m]["val"] for m in MODELS}, s0, tri)
t0 = fit_temperature(pv, y_val)
sets = [("teaLeafBD test", fuse({m: probs[s0, m]["test"] for m in MODELS}, s0, tri), y_test)]
if has_ext:
    sets.append(("TeaLeafDiseaseBD (unseen)", fuse({m: ext[s0, m] for m in MODELS}, s0, tri), y_ext))
plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, len(sets), figsize=(3.5, 1.9), dpi=300, sharey=True)
axes = np.atleast_1d(axes)
edges = np.linspace(0, 1, 11)
for ax, (title, p, y) in zip(axes, sets):
    for pp, lab, col, mk in [(p, "uncalibrated", "#9db4c0", "o"), (scale(p, t0), f"temperature-scaled (T={t0:.2f})", "#2b5876", "s")]:
        conf, ok = pp.max(1), pp.argmax(1) == y
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 20:
                xs.append(conf[m].mean())
                ys.append(ok[m].mean())
        ax.plot(xs, ys, marker=mk, ms=2.5, lw=1, color=col, label=lab)
    ax.plot([0, 1], [0, 1], ls="--", lw=0.6, color="0.5")
    ax.axvline(THRESH, ls=":", lw=0.6, color="#c0504d")
    ax.set_title(title, fontsize=7)
    ax.set_xlabel("Confidence", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
axes[0].set_ylabel("Accuracy", fontsize=7)
axes[0].legend(fontsize=5.5, frameon=False, loc="lower right")
fig.tight_layout(pad=0.4)
fig.savefig(A / "fig_reliability.png")
print("wrote", A / "fig_reliability.png")
