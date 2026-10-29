"""Reruns the released models on the teaLeafBD images (CPU):

E16  regenerate the saved val/test predictions of all 9 models and compare
E17  reverse-direction DenseNet121 on the 3,996 teaLeafBD images of the shared classes
E18  seed-42 models under the six corruptions of corruptions.py

Needs the images in data/teaLeafBD and the release weights in outputs/ and outputs_reverse/.
TEA_PARTS picks the parts (default e16,e17,e18); TEA_DEVICE picks the device for E18 (default cpu).
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
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

import fusionlib as fl
from corruptions import NAMES, conditions, corrupt
from inference import load_image, to_tensor
from tea_models import LABELS, MODELS, build

OUT, A = Path("outputs"), Path("outputs/analysis")
R = OUT / "revision"
R.mkdir(exist_ok=True)
torch.set_num_threads(os.cpu_count() or 4)
SEEDS = json.loads((OUT / "summary.json").read_text())["seeds"]
split = pd.read_csv(OUT / "split.csv")
CLASSES = sorted(split.cls.unique())
y_val = split.loc[split.split == "val", "label"].to_numpy()
y_test = split.loc[split.split == "test", "label"].to_numpy()
BLUE, LIGHT, GREY, RED = "#2b5876", "#9db4c0", "#7f8c8d", "#c0504d"
PARTS = os.environ.get("TEA_PARTS", "e16,e17,e18").split(",")
DEVICE = torch.device(os.environ.get("TEA_DEVICE", "cpu"))


def net(name, path, n=7):
    m, _ = build(name, n, pretrained=False)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


@torch.no_grad()
def run(model, imgs, bs=64, device=torch.device("cpu")):
    model = model.to(device)
    return torch.cat([F.softmax(model(to_tensor(imgs[i:i + bs]).to(device)).float(), 1).cpu()
                      for i in range(0, len(imgs), bs)]).numpy()


print("loading images")
imgs = {s: [load_image(p) for p in split.loc[split.split == s, "path"]] for s in ("val", "test")}

# ---------------------------------------------------------------- E16
rows, regen = [], {}
for s in SEEDS if "e16" in PARTS else []:
    for m in MODELS:
        model = net(m, OUT / f"{m}_seed{s}.pt")
        saved = np.load(OUT / f"probs_{m}_seed{s}.npz")
        for part in ("val", "test"):
            p = run(model, imgs[part])
            regen[s, m, part] = p
            d = np.abs(p - saved[part])
            y = y_val if part == "val" else y_test
            rows.append({"seed": s, "model": LABELS[m], "split": part, "max_abs_diff": d.max(), "mean_abs_diff": d.mean(),
                         "argmax_agreement": (p.argmax(1) == saved[part].argmax(1)).mean() * 100,
                         "acc_saved": accuracy_score(y, saved[part].argmax(1)) * 100, "acc_rerun": accuracy_score(y, p.argmax(1)) * 100})
        print("E16", s, m, flush=True)
if "e16" in PARTS:
    e16 = pd.DataFrame(rows)
    e16.to_csv(R / "e16_reproduction.csv", index=False)
fused_same = []
for s in SEEDS if "e16" in PARTS else []:
    w = fl.accuracy_weights([np.load(OUT / f"probs_{m}_seed{s}.npz")["val"] for m in MODELS], y_val)
    for c in [c for r in (2, 3) for c in combinations(range(3), r)]:
        ws = w[list(c)]
        a = fl.weighted_average([regen[s, MODELS[i], "test"] for i in c], ws).argmax(1)
        b = fl.weighted_average([np.load(OUT / f"probs_{MODELS[i]}_seed{s}.npz")["test"] for i in c], ws).argmax(1)
        fused_same.append(bool((a == b).all()))
if "e16" in PARTS:
    print(f"E16: max diff {e16.max_abs_diff.max():.2e}, mean diff <= {e16.mean_abs_diff.max():.2e}, "
          f"all argmax equal: {bool((e16.argmax_agreement == 100).all())}, fused equal: {all(fused_same)}")

# ---------------------------------------------------------------- E17
if "e17" in PARTS:
    shared = sorted(set(np.load(A / "external_labels.npy").tolist()))
    to6 = {c: i for i, c in enumerate(shared)}
    tgt = split[split.label.isin(shared)]
    X = [Image.open(p).convert("RGB").resize((256, 256), Image.BILINEAR) for p in tgt.path]  # as in reverse.py
    yt6 = np.array([to6[c] for c in tgt.label])
    p = run(net("densenet121", Path("outputs_reverse/densenet121_reverse_seed42.pt"), len(shared)), X)
    saved = json.loads(Path("outputs_reverse/reverse_summary.json").read_text())
    e17 = {"images": len(yt6), "acc_rerun": accuracy_score(yt6, p.argmax(1)) * 100, "f1_rerun": f1_score(yt6, p.argmax(1), average="macro") * 100,
           "acc_saved": saved["cross_acc"] * 100, "f1_saved": saved["cross_f1"] * 100}
    (R / "e17_reverse.json").write_text(json.dumps(e17, indent=1))
    print("E17", e17)
if "e18" not in PARTS:
    raise SystemExit

# ---------------------------------------------------------------- E18
s0 = SEEDS[0]
models = {m: net(m, OUT / f"{m}_seed{s0}.pt") for m in MODELS}
w = fl.accuracy_weights([np.load(OUT / f"probs_{m}_seed{s0}.npz")["val"] for m in MODELS], y_val)
CFG = {"CNN": ["cnn"], "DenseNet121": ["densenet121"], "VGG16": ["vgg16"], "CNN + DenseNet121": ["cnn", "densenet121"],
       "CNN + VGG16": ["cnn", "vgg16"], "DenseNet121 + VGG16": ["densenet121", "vgg16"], "Triple": MODELS}
rows, div_rows, examples = [], [], {}
for kind, lvl in [("clean", None)] + conditions():
    cur = imgs["test"] if kind == "clean" else [corrupt(im, kind, lvl, seed=i) for i, im in enumerate(imgs["test"])]
    examples[kind, lvl] = cur[0]
    pm = {m: run(models[m], cur, device=DEVICE) for m in MODELS}
    label = "Clean" if kind == "clean" else f"{NAMES[kind]}, {'moderate' if lvl == 0 else 'severe'}"
    row = {"condition": label}
    for cname, mm in CFG.items():
        p = fl.weighted_average([pm[m] for m in mm], w[[MODELS.index(m) for m in mm]])
        row[cname] = accuracy_score(y_test, p.argmax(1)) * 100
        if cname in ("Triple", "DenseNet121"):
            row[f"wrong_advice_{cname}"] = fl.selective_stats(p, y_test, fl.advise(p, 0.8))["wrong_advice"] * 100
            row[f"advised_{cname}"] = fl.advise(p, 0.8).mean() * 100
    pt = fl.weighted_average([pm[m] for m in MODELS], w).argmax(1) == y_test
    pd_ = pm["densenet121"].argmax(1) == y_test
    row["mcnemar_p_triple_vs_dn"] = fl.mcnemar(pt, pd_)[2]
    rows.append(row)
    qs = {f"{LABELS[a]} / {LABELS[b]}": fl.pair_diversity(pm[a].argmax(1), pm[b].argmax(1), y_test)["q"] for a, b in combinations(MODELS, 2)}
    div_rows.append({"condition": label, **qs, "mean_q": np.mean(list(qs.values()))})
    print("E18", label, {k: round(v, 2) for k, v in row.items() if k != "condition"}, flush=True)
e18 = pd.DataFrame(rows)
e18.to_csv(R / "e18_corruptions.csv", index=False)
pd.DataFrame(div_rows).to_csv(R / "b2_diversity.csv", index=False)
print("E18 mean over corrupted sets:", e18.iloc[1:].mean(numeric_only=True).round(2).to_dict())

fig, axes = plt.subplots(2, 7, figsize=(14, 4.6))
for j, (kind, _) in enumerate([("clean", None)] + [(k, None) for k in NAMES]):
    for i, lvl in enumerate((0, 1)):
        ax = axes[i, j]
        im = examples["clean", None] if kind == "clean" else examples[kind, lvl]
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_title("Clean" if kind == "clean" else NAMES[kind], fontsize=9)
axes[0, 0].set_ylabel("moderate")
axes[1, 0].set_ylabel("severe")
fig.tight_layout()
fig.savefig(R / "fig_corruption_examples.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4.6))
cond = list(NAMES)
x = np.arange(len(cond))
for k, (cname, col) in enumerate([("DenseNet121", GREY), ("VGG16", LIGHT), ("CNN + DenseNet121", "#d4a373"), ("Triple", BLUE)]):
    mod = [e18.loc[e18.condition == f"{NAMES[c]}, moderate", cname].item() for c in cond]
    sev = [e18.loc[e18.condition == f"{NAMES[c]}, severe", cname].item() for c in cond]
    ax.bar(x + (k - 1.5) * 0.2, mod, 0.2, color=col, alpha=0.55)
    ax.bar(x + (k - 1.5) * 0.2, sev, 0.2, color=col, label=cname)
ax.axhline(e18.loc[0, "Triple"], color=BLUE, ls=":", lw=0.9)
ax.set_xticks(x, [NAMES[c] for c in cond])
ax.set(ylabel="Accuracy (%) (light = moderate, dark = severe)", ylim=(0, 100))
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
fig.tight_layout()
fig.savefig(R / "fig_corruptions.png", dpi=300)
plt.close(fig)
