# %% [markdown]
# # Tea leaf disease classification with model fusion
#
# Trains a small CNN, DenseNet121 and VGG16 on teaLeafBD and scores all seven
# combinations (weighted soft voting) on one fixed test split.
#
# Env vars:
# - `TEA_DATA`  dataset folder, one sub-folder per class (default `data/teaLeafBD`)
# - `TEA_OUT`   where results go (default `outputs`)
# - `TEA_SEEDS` training seeds, e.g. `42,7,123` (default `42`)
# - `TEA_SMOKE` set to `1` for a quick 1-epoch test run
#
# Works on CUDA, Apple MPS or CPU. Already trained models are skipped on rerun.

# %%
import json
import math
import os
import random
import time
from contextlib import nullcontext
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2 as T

from tea_models import LABELS, MODELS, build

DATA_DIR = Path(os.environ.get("TEA_DATA", "data/teaLeafBD"))
OUT = Path(os.environ.get("TEA_OUT", "outputs"))
SEEDS = [int(s) for s in os.environ.get("TEA_SEEDS", "42").split(",")]
SMOKE = os.environ.get("TEA_SMOKE") == "1"

SPLIT_SEED = 42
SPLIT = (0.70, 0.10, 0.20)  # train / val / test
IMG, CACHE = 224, 256
BATCH = 32
EPOCHS = {"cnn": 40, "densenet121": 25, "vgg16": 25}
PATIENCE = 6
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

if SMOKE:
    EPOCHS = {m: 1 for m in MODELS}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
NUM_WORKERS = 4 if DEVICE.type == "cuda" else 0
OUT.mkdir(parents=True, exist_ok=True)
print(f"device: {DEVICE} | torch {torch.__version__} | data: {DATA_DIR.resolve()} | out: {OUT.resolve()}")
T_START = time.time()

# %% [markdown]
# ## 1. Dataset

# %%
files = sorted(p for p in DATA_DIR.rglob("*") if p.suffix.lower() in EXTS and not p.name.startswith("."))
if not files:
    raise SystemExit(f"No images found under {DATA_DIR.resolve()}")
df = pd.DataFrame({"path": [str(p) for p in files], "cls": [p.parent.name for p in files]})
if SMOKE:
    df = df.groupby("cls").head(14).reset_index(drop=True)
CLASSES = sorted(df.cls.unique())
df["label"] = df.cls.map({c: i for i, c in enumerate(CLASSES)})
counts = df.cls.value_counts().reindex(CLASSES)
counts.rename("images").to_csv(OUT / "dataset_counts.csv", index_label="class")
print(counts.to_string(), f"\nTotal: {len(df)} images, {len(CLASSES)} classes")

# %% [markdown]
# ## 2. Fixed split (saved to split.csv, reused by every run)

# %%
split_file = OUT / "split.csv"
if split_file.exists():
    df = pd.read_csv(split_file)
    CLASSES = sorted(df.cls.unique())
else:
    rest_frac = SPLIT[1] + SPLIT[2]
    tr, rest = train_test_split(df, test_size=rest_frac, stratify=df.label, random_state=SPLIT_SEED)
    va, te = train_test_split(rest, test_size=SPLIT[2] / rest_frac, stratify=rest.label, random_state=SPLIT_SEED)
    df = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")]).sort_index()
    df.to_csv(split_file, index=False)
print(pd.crosstab(df.cls, df.split)[["train", "val", "test"]].to_string())

# %% [markdown]
# ## 3. Load images into memory (256x256)

# %%
def load_all(paths):
    arr = torch.empty((len(paths), 3, CACHE, CACHE), dtype=torch.uint8)
    for i, p in enumerate(paths):
        im = Image.open(p)
        im.draft("RGB", (CACHE, CACHE))
        im = im.convert("RGB").resize((CACHE, CACHE), Image.BILINEAR)
        arr[i] = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
        if (i + 1) % 1000 == 0:
            print(f"  loaded {i + 1}/{len(paths)}")
    return arr


t0 = time.time()
X = load_all(df.path.tolist())
Y = torch.tensor(df.label.values)
idx = {s: np.flatnonzero(df.split.values == s) for s in ("train", "val", "test")}
print(f"cached {len(X)} images in {time.time() - t0:.0f}s")

train_tf = T.Compose([
    T.RandomResizedCrop(IMG, scale=(0.75, 1.0), antialias=True),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.2),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([T.Resize(IMG, antialias=True), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])


class Images(torch.utils.data.Dataset):
    def __init__(self, ids, tf):
        self.ids, self.tf = ids, tf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        j = self.ids[i]
        return self.tf(X[j]), Y[j]


def loader(split, seed=0):
    train = split == "train"
    return torch.utils.data.DataLoader(
        Images(idx[split], train_tf if train else eval_tf), batch_size=BATCH, shuffle=train,
        num_workers=NUM_WORKERS, generator=torch.Generator().manual_seed(seed), pin_memory=DEVICE.type == "cuda")


y_val, y_test = Y[idx["val"]].numpy(), Y[idx["test"]].numpy()

# %% [markdown]
# ## 4. Models

# %%
# model definitions are in tea_models.py


def make_optimizer(model, backbone):
    if backbone is None:
        return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    in_backbone = {id(p) for p in backbone.parameters()}
    return torch.optim.AdamW([
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": 1e-4},
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in in_backbone], "lr": 1e-3},
    ], weight_decay=1e-4)


# %% [markdown]
# ## 5. Training (early stopping on val loss)

# %%
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, data, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    total_loss = correct = n = 0
    amp = torch.autocast("cuda", dtype=torch.float16) if DEVICE.type == "cuda" else nullcontext()
    with torch.set_grad_enabled(train):
        for xb, yb in data:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE)
            with amp:
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


def fit(name, seed):
    set_seed(seed)
    model, backbone = build(name, len(CLASSES), pretrained=not SMOKE)
    model.to(DEVICE)
    opt = make_optimizer(model, backbone)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    train_dl, val_dl = loader("train", seed), loader("val")
    best, best_state, bad, hist = math.inf, None, 0, []
    for ep in range(1, EPOCHS[name] + 1):
        t = time.time()
        tl, ta = run_epoch(model, train_dl, opt, scaler)
        vl, va = run_epoch(model, val_dl)
        sched.step(vl)
        hist.append({"epoch": ep, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va})
        print(f"{LABELS[name]:<11} seed {seed} epoch {ep:02d}  train acc {ta:.3f} loss {tl:.3f}"
              f"  val acc {va:.3f} loss {vl:.3f}  {time.time() - t:.0f}s")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    torch.save(best_state, OUT / f"{name}_seed{seed}.pt")
    pd.DataFrame(hist).to_csv(OUT / f"history_{name}_seed{seed}.csv", index=False)
    return model


@torch.no_grad()
def predict(model, split):
    model.eval()
    out, t = [], time.time()
    for xb, _ in loader(split):
        out.append(F.softmax(model(xb.to(DEVICE)).float(), 1).cpu())
    probs = torch.cat(out).numpy()
    return probs, (time.time() - t) / len(probs) * 1000


probs, info = {}, {}
for seed in SEEDS:
    for name in MODELS:
        f = OUT / f"probs_{name}_seed{seed}.npz"
        if f.exists():
            z = np.load(f)
            probs[seed, name] = {"val": z["val"], "test": z["test"]}
            info[seed, name] = json.loads(str(z["info"]))
            print(f"loaded saved predictions: {f.name}")
            continue
        t = time.time()
        model = fit(name, seed)
        pv, _ = predict(model, "val")
        pt, ms = predict(model, "test")
        params = sum(p.numel() for p in model.parameters())
        info[seed, name] = {"train_minutes": (time.time() - t) / 60, "ms_per_image": ms, "parameters": params,
                            "epochs_run": len(pd.read_csv(OUT / f"history_{name}_seed{seed}.csv"))}
        probs[seed, name] = {"val": pv, "test": pt}
        np.savez(f, val=pv, test=pt, info=json.dumps(info[seed, name]))
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE.type == "mps":
            torch.mps.empty_cache()

# %% [markdown]
# ## 6. All seven combinations, weighted soft voting

# %%
CONFIGS = [c for r in (1, 2, 3) for c in combinations(MODELS, r)]


def cfg_name(c):
    return " + ".join(LABELS[m] for m in c)


def metrics(y, pred):
    acc = accuracy_score(y, pred)
    pw, rw, fw, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"accuracy": acc, "precision_w": pw, "recall_w": rw, "f1_w": fw,
            "precision_macro": pm, "recall_macro": rm, "f1_macro": fm}


rows, preds = [], {}
for seed in SEEDS:
    val_acc = {m: accuracy_score(y_val, probs[seed, m]["val"].argmax(1)) for m in MODELS}
    for c in CONFIGS:
        w = np.array([val_acc[m] for m in c])
        w = w / w.sum()
        fused = sum(wi * probs[seed, m]["test"] for wi, m in zip(w, c))
        equal = sum(probs[seed, m]["test"] for m in c) / len(c)
        pred = fused.argmax(1)
        preds[seed, c] = pred
        rows.append({"seed": seed, "config": cfg_name(c), "n_models": len(c),
                     "weights": ", ".join(f"{LABELS[m]} {wi:.3f}" for m, wi in zip(c, w)),
                     **metrics(y_test, pred), "accuracy_equal_weights": accuracy_score(y_test, equal.argmax(1))})
res = pd.DataFrame(rows)
res.to_csv(OUT / "results_all_seeds.csv", index=False)

order = [cfg_name(c) for c in CONFIGS]
cols = ["accuracy", "precision_w", "recall_w", "f1_w", "f1_macro", "accuracy_equal_weights"]
mean = res.groupby("config")[cols].mean().reindex(order) * 100
table3 = mean.round(2)
if len(SEEDS) > 1:
    std = res.groupby("config")[cols].std().reindex(order) * 100
    table3 = mean.round(2).astype(str) + " ± " + std.round(2).astype(str)
table3.to_csv(OUT / "table3.csv")
print(table3.to_string())

# %% [markdown]
# ## 7. Figures and per-class results (first seed)

# %%
S0 = SEEDS[0]
short = [c.replace("_", " ") for c in CLASSES]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

for c in CONFIGS:
    p, r, f1, sup = precision_recall_fscore_support(y_test, preds[S0, c], labels=range(len(CLASSES)), zero_division=0)
    pd.DataFrame({"class": CLASSES, "precision": p, "recall": r, "f1": f1, "support": sup}).to_csv(
        OUT / f"per_class_{cfg_name(c).replace(' + ', '_')}.csv", index=False)

fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=200)
for ax, c in zip(axes.flat, CONFIGS):
    cm = confusion_matrix(y_test, preds[S0, c], labels=range(len(CLASSES)))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"{cfg_name(c)}\nacc {np.trace(cm) / cm.sum() * 100:.2f}% (n={cm.sum()})", fontsize=8)
    ax.set_xticks(range(len(CLASSES)), short, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(CLASSES)), short, fontsize=6)
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("True", fontsize=7)
axes.flat[-1].axis("off")
fig.tight_layout()
fig.savefig(OUT / "fig_confusion_matrices.png")
plt.close(fig)

fig, axes = plt.subplots(len(MODELS), 2, figsize=(7, 2.2 * len(MODELS)), dpi=200)
for row, m in zip(axes, MODELS):
    h = pd.read_csv(OUT / f"history_{m}_seed{S0}.csv")
    for ax, key in zip(row, ("acc", "loss")):
        ax.plot(h.epoch, h[f"train_{key}"], label="train")
        ax.plot(h.epoch, h[f"val_{key}"], label="validation")
        ax.set_title(f"{LABELS[m]}: {'accuracy' if key == 'acc' else 'loss'}", fontsize=8)
        ax.set_xlabel("epoch", fontsize=7)
    row[0].legend(fontsize=6, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_training_curves.png")
plt.close(fig)

acc_mean = res.groupby("config").accuracy.mean().reindex(order) * 100
acc_std = res.groupby("config").accuracy.std().reindex(order).fillna(0) * 100
colors = ["#b8c4cc"] * 3 + ["#7f9cad"] * 3 + ["#2b5876"]
fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)
bars = ax.bar(range(len(order)), acc_mean, yerr=acc_std if len(SEEDS) > 1 else None, color=colors, capsize=2)
for b, a in zip(bars, acc_mean):
    ax.text(b.get_x() + b.get_width() / 2, a + 0.5, f"{a:.2f}", ha="center", fontsize=6)
abbr = {"CNN": "C", "DenseNet121": "D", "VGG16": "V"}
ax.set_xticks(range(len(order)), ["+".join(abbr[p] for p in o.split(" + ")) for o in order], fontsize=7)
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(max(0, acc_mean.min() - 10), 100)
fig.tight_layout()
fig.savefig(OUT / "fig_accuracy.png")
plt.close(fig)

duals = [c for c in CONFIGS if len(c) == 2]
best_dual = max(duals, key=lambda c: accuracy_score(y_test, preds[S0, c]))
triple = CONFIGS[-1]
f_d = precision_recall_fscore_support(y_test, preds[S0, best_dual], labels=range(len(CLASSES)), zero_division=0)[2]
f_t = precision_recall_fscore_support(y_test, preds[S0, triple], labels=range(len(CLASSES)), zero_division=0)[2]
x = np.arange(len(CLASSES))
fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)
ax.bar(x - 0.19, f_d * 100, 0.38, label=cfg_name(best_dual), color="#9db4c0")
ax.bar(x + 0.19, f_t * 100, 0.38, label="Triple ensemble", color="#2b5876")
ax.set_xticks(x, short, rotation=45, ha="right", fontsize=6)
ax.set_ylabel("F1-score (%)")
ax.legend(fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout()
fig.savefig(OUT / "fig_perclass_f1.png")
plt.close(fig)

# %% [markdown]
# ## 8. McNemar test and summary

# %%
def mcnemar(a, b):
    ca, cb = preds[S0, a] == y_test, preds[S0, b] == y_test
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"a": cfg_name(a), "b": cfg_name(b), "only_a_correct": only_a, "only_b_correct": only_b, "p_value": p}


singles = [c for c in CONFIGS if len(c) == 1]
best_single = max(singles, key=lambda c: accuracy_score(y_test, preds[S0, c]))
tests = [mcnemar(triple, best_dual), mcnemar(triple, best_single)]

summary = {
    "device": str(DEVICE), "torch": torch.__version__, "seeds": SEEDS, "smoke_test": SMOKE,
    "classes": CLASSES, "images_per_class": counts.to_dict(), "total_images": int(len(df)),
    "split_sizes": {s: int(len(v)) for s, v in idx.items()}, "split_ratio": SPLIT,
    "training": {"image_size": IMG, "batch_size": BATCH, "optimizer": "AdamW (weight decay 1e-4)",
                 "learning_rates": "CNN 1e-3; pretrained backbones 1e-4, new heads 1e-3",
                 "scheduler": "ReduceLROnPlateau(factor 0.3, patience 2)", "max_epochs": EPOCHS,
                 "early_stopping_patience": PATIENCE, "normalization": "ImageNet mean/std",
                 "augmentation": "random resized crop (0.75-1.0), h/v flip, rotation ±20°, brightness ±20%"},
    "models": {f"{LABELS[m]} seed {s}": info[s, m] for s in SEEDS for m in MODELS},
    "mcnemar": tests, "total_minutes": (time.time() - T_START) / 60,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

lines = ["# Results summary", "", f"Device: {DEVICE}, seeds: {SEEDS}, total time {summary['total_minutes']:.0f} min", "",
         "## Table II: images per class", "", counts.to_markdown(), "", f"Split (train/val/test): {summary['split_sizes']}", "",
         "## Table III: test-set results (%)", "", table3.to_markdown(), "",
         "## Fusion weights (first seed)", ""]
lines += [f"- {r.config}: {r.weights}" for r in res[res.seed == S0].itertuples()]
lines += ["", "## McNemar exact test (first seed)", ""]
lines += [f"- {t['a']} vs {t['b']}: {t['only_a_correct']} vs {t['only_b_correct']} discordant, p = {t['p_value']:.4f}" for t in tests]
lines += ["", "## Models", ""]
lines += [f"- {k}: {v['parameters'] / 1e6:.1f}M params, {v['epochs_run']} epochs, "
          f"{v['train_minutes']:.1f} min, {v['ms_per_image']:.1f} ms/image" for k, v in summary["models"].items()]
(OUT / "results_summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nAll outputs in {OUT.resolve()}")
