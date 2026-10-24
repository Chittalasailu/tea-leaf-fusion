"""Diversity between the three models, and testing them on TeaLeafDiseaseBD without retraining.

TEA_OUT  training outputs (default outputs)
TEA_EXT  second dataset (default data/TeaLeafDiseaseBD), skipped if missing
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from tea_models import LABELS, MODELS, build

OUT = Path(os.environ.get("TEA_OUT", "outputs"))
EXT = Path(os.environ.get("TEA_EXT", "data/TeaLeafDiseaseBD"))
A = OUT / "analysis"
A.mkdir(exist_ok=True)
IMG, CACHE = 224, 256
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
CONFIGS = [c for r in (1, 2, 3) for c in combinations(MODELS, r)]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cfg_name(c):
    return " + ".join(LABELS[m] for m in c)


split = pd.read_csv(OUT / "split.csv")
CLASSES = sorted(split.cls.unique())
y_val = split.loc[split.split == "val", "label"].to_numpy()
y_test = split.loc[split.split == "test", "label"].to_numpy()
SEEDS = sorted({int(m.group(1)) for f in OUT.glob("probs_vgg16_seed*.npz") if (m := re.search(r"seed(\d+)", f.name))})
if (OUT / "summary.json").exists():  # same seed order as training
    SEEDS = [x for x in json.loads((OUT / "summary.json").read_text())["seeds"] if x in SEEDS]
probs = {(s, m): np.load(OUT / f"probs_{m}_seed{s}.npz") for s in SEEDS for m in MODELS}
weights = {s: {m: accuracy_score(y_val, probs[s, m]["val"].argmax(1)) for m in MODELS} for s in SEEDS}
print(f"seeds {SEEDS} | device {DEVICE} | classes {CLASSES}")


def fuse(p, s, c):
    w = np.array([weights[s][m] for m in c])
    w = w / w.sum()
    return sum(wi * p[m] for wi, m in zip(w, c))


# --- diversity on the teaLeafBD test set
pair_rows, subset_rows = [], []
for s in SEEDS:
    correct = {m: probs[s, m]["test"].argmax(1) == y_test for m in MODELS}
    for a, b in combinations(MODELS, 2):
        ca, cb = correct[a], correct[b]
        n11, n00 = int((ca & cb).sum()), int((~ca & ~cb).sum())
        n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        q = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10) if (n11 * n00 + n01 * n10) else 0.0
        rho = np.corrcoef(~ca, ~cb)[0, 1] if (~ca).any() and (~cb).any() else 0.0
        pair_rows.append({"seed": s, "pair": f"{LABELS[a]} / {LABELS[b]}", "disagreement": (n10 + n01) / len(ca),
                          "double_fault": n00 / len(ca), "yule_q": q, "error_correlation": rho})
    best_single = max(MODELS, key=lambda m: correct[m].mean())
    for c in CONFIGS:
        pred = fuse({m: probs[s, m]["test"] for m in MODELS}, s, c).argmax(1)
        ok = pred == y_test
        oracle = np.any([correct[m] for m in c], axis=0).mean()
        subset_rows.append({"seed": s, "config": cfg_name(c), "accuracy": ok.mean(), "oracle_accuracy": oracle,
                            "errors_fixed_vs_best_single": int((ok & ~correct[best_single]).sum()),
                            "errors_added_vs_best_single": int((~ok & correct[best_single]).sum()),
                            "best_single": LABELS[best_single]})
pairs = pd.DataFrame(pair_rows)
subsets = pd.DataFrame(subset_rows)
pairs.to_csv(A / "diversity_pairs.csv", index=False)
subsets.to_csv(A / "diversity_subsets.csv", index=False)
pairs_mean = pairs.drop(columns="seed").groupby("pair").agg(["mean", "std"]).round(4)
subsets_mean = subsets.drop(columns=["seed", "best_single"]).groupby("config").mean().reindex([cfg_name(c) for c in CONFIGS])
print(pairs_mean.to_string(), "\n", subsets_mean.round(4).to_string())

# --- cross-dataset test
KEYWORDS = [("algal", "Tea algal leaf spot"), ("brown blight", "Brown Blight"), ("gray blight", "Gray Blight"),
            ("grey blight", "Gray Blight"), ("helopelt", "Helopeltis"), ("red spider", "Red spider"),
            ("healthy", "Healthy leaf"), ("green mirid", "Green mirid bug")]


def source_class(path):
    # match folder names to teaLeafBD classes, None if there is no match
    text = " ".join(p.lower().replace("_", " ").replace("-", " ") for p in path.parts[-4:-1])
    for kw, name in KEYWORDS:
        if kw in text:
            return next(c for c in CLASSES if c.lower().endswith(name.lower()))
    return None


def load_one(p):
    im = Image.open(p)
    im.draft("RGB", (CACHE * 2, CACHE * 2))
    return np.asarray(im.convert("RGB").resize((CACHE, CACHE), Image.BILINEAR), dtype=np.uint8)


ext_summary = None
if EXT.exists():
    files = sorted(p for p in EXT.rglob("*") if p.suffix.lower() in EXTS and not p.name.startswith("."))
    folders = pd.Series([p.parent.name for p in files]).value_counts()
    print("external folders:\n" + folders.to_string())
    mapped = [(p, source_class(p)) for p in files]
    skipped = pd.Series([p.parent.name for p, c in mapped if c is None]).value_counts()
    mapped = [(p, c) for p, c in mapped if c is not None]
    cache = A / "external_cache.npz"
    if cache.exists():
        z = np.load(cache)
        X, y_ext = z["X"], z["y"]
    else:
        t = time.time()
        with ThreadPoolExecutor(os.cpu_count() or 8) as ex:
            X = np.stack(list(ex.map(load_one, [p for p, _ in mapped])))
        y_ext = np.array([CLASSES.index(c) for _, c in mapped])
        np.savez(cache, X=X, y=y_ext)
        print(f"loaded {len(X)} external images in {time.time() - t:.0f}s")
    shared = sorted(set(y_ext.tolist()))
    print("shared classes:", [CLASSES[i] for i in shared], "| skipped:", skipped.to_dict())

    @torch.no_grad()
    def predict_ext(model):
        model.eval()
        out = []
        for i in range(0, len(X), 64):
            xb = torch.from_numpy(X[i:i + 64]).permute(0, 3, 1, 2).float().div(255)
            xb = F.interpolate(xb, size=(IMG, IMG), mode="bilinear", antialias=True, align_corners=False)
            xb = ((xb - MEAN) / STD).to(DEVICE)
            out.append(F.softmax(model(xb).float(), 1).cpu())
        return torch.cat(out).numpy()

    ext_probs = {}
    for s in SEEDS:
        for m in MODELS:
            f = A / f"ext_probs_{m}_seed{s}.npy"
            if f.exists():
                ext_probs[s, m] = np.load(f)
                continue
            model, _ = build(m, len(CLASSES), pretrained=False)
            model.load_state_dict(torch.load(OUT / f"{m}_seed{s}.pt", map_location="cpu", weights_only=True))
            ext_probs[s, m] = predict_ext(model.to(DEVICE))
            np.save(f, ext_probs[s, m])
            print(f"external predictions: {LABELS[m]} seed {s}")

    rows = []
    for s in SEEDS:
        for c in CONFIGS:
            pred = fuse({m: ext_probs[s, m] for m in MODELS}, s, c).argmax(1)
            in_dom = accuracy_score(y_test, fuse({m: probs[s, m]["test"] for m in MODELS}, s, c).argmax(1))
            rows.append({"seed": s, "config": cfg_name(c), "external_accuracy": accuracy_score(y_ext, pred),
                         "external_macro_f1": f1_score(y_ext, pred, labels=shared, average="macro", zero_division=0),
                         "teaLeafBD_accuracy": in_dom})
    ext = pd.DataFrame(rows)
    ext.to_csv(A / "external_results_all_seeds.csv", index=False)
    order = [cfg_name(c) for c in CONFIGS]
    agg = ext.drop(columns="seed").groupby("config").agg(["mean", "std"]).reindex(order) * 100
    agg.round(2).to_csv(A / "external_table.csv")
    print(agg.round(2).to_string())

    s0, triple = SEEDS[0], CONFIGS[-1]
    pred = fuse({m: ext_probs[s0, m] for m in MODELS}, s0, triple).argmax(1)
    cm = confusion_matrix(y_ext, pred, labels=range(len(CLASSES)))[shared]
    per_class = pd.DataFrame({"class": [CLASSES[i] for i in shared],
                              "recall_triple": recall_score(y_ext, pred, labels=shared, average=None, zero_division=0),
                              "n": [int((y_ext == i).sum()) for i in shared]})
    per_class.to_csv(A / "external_per_class_triple.csv", index=False)

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.5, 2.9), dpi=300)
    ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.5, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(CLASSES)), [c.split(". ", 1)[-1] for c in CLASSES], rotation=45, ha="right", fontsize=5.5)
    ax.set_yticks(range(len(shared)), [CLASSES[i].split(". ", 1)[-1] for i in shared], fontsize=5.5)
    ax.set_xlabel("Predicted (teaLeafBD classes)", fontsize=6.5)
    ax.set_ylabel("True (TeaLeafDiseaseBD)", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(A / "fig_external_cm_triple.png")
    plt.close(fig)

    m_in, m_ex = agg[("teaLeafBD_accuracy", "mean")], agg[("external_accuracy", "mean")]
    sd_ex = agg[("external_accuracy", "std")].fillna(0)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)
    ax.bar(x - 0.19, m_in, 0.38, label="teaLeafBD test", color="#9db4c0")
    ax.bar(x + 0.19, m_ex, 0.38, yerr=sd_ex if len(SEEDS) > 1 else None, capsize=1.5, label="TeaLeafDiseaseBD (unseen)", color="#2b5876")
    ax.set_xticks(x, [o.replace(" + ", "+\n") for o in order], fontsize=5.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(A / "fig_external_accuracy.png")
    plt.close(fig)
    ext_summary = {"images": int(len(y_ext)), "shared_classes": [CLASSES[i] for i in shared],
                   "skipped_folders": skipped.to_dict(), "per_class_counts": per_class.set_index("class").n.to_dict()}
else:
    print(f"external dataset not found at {EXT.resolve()}; cross-dataset part skipped")

# --- report
lines = ["# Analysis numbers", "", f"Seeds: {SEEDS}", "", "## Pairwise diversity on teaLeafBD test (mean over seeds)", "",
         pairs.drop(columns="seed").groupby("pair").mean().round(4).to_markdown(), "",
         "## Fusion vs oracle on teaLeafBD test (mean over seeds)", "", subsets_mean.round(4).to_markdown(), ""]
if ext_summary:
    lines += ["## Cross-dataset (trained on teaLeafBD, tested on TeaLeafDiseaseBD, no retraining)", "",
              f"Images: {ext_summary['images']}; shared classes: {ext_summary['shared_classes']}; "
              f"skipped: {ext_summary['skipped_folders']}", "", agg.round(2).to_markdown(), "",
              "Per-class recall of the triple ensemble (first seed):", "", per_class.round(4).to_markdown(index=False), ""]
(A / "analysis_numbers.md").write_text("\n".join(lines) + "\n")
(A / "analysis_summary.json").write_text(json.dumps({"seeds": SEEDS, "external": ext_summary}, indent=2))
print("\n".join(lines))
