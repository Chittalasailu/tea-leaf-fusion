"""Other direction: train DenseNet121 on TeaLeafDiseaseBD (6 shared classes) and test on teaLeafBD.
Same preprocessing and training settings as tea_fusion.py.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2 as T

from tea_models import build

SEED, IMG, BATCH, EPOCHS, PATIENCE = 42, 224, 32, 25, 6
OUT = Path("outputs_reverse")
OUT.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

split = pd.read_csv("outputs/split.csv")
CLASSES = sorted(split.cls.unique())
z = np.load("outputs/analysis/external_cache.npz")
X_src, y7 = z["X"], z["y"]
SHARED = sorted(set(y7.tolist()))  # classes both datasets have
to6 = {c: i for i, c in enumerate(SHARED)}
y_src = np.array([to6[c] for c in y7])
names = [CLASSES[c].split(". ", 1)[-1] for c in SHARED]
print("classes:", names, "| source images:", len(y_src))

idx = np.arange(len(y_src))
tr, rest = train_test_split(idx, test_size=0.30, stratify=y_src, random_state=SEED)
va, te = train_test_split(rest, test_size=2 / 3, stratify=y_src[rest], random_state=SEED)

# all teaLeafBD images of the shared classes (this model never saw them)
tgt = split[split.label.isin(SHARED)].reset_index(drop=True)
X_tgt = np.stack([np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.BILINEAR)) for p in tgt.path])
y_tgt = np.array([to6[c] for c in tgt.label])
print("target (teaLeafBD) images:", len(y_tgt))

train_tf = T.Compose([T.RandomResizedCrop(IMG, scale=(0.75, 1.0), antialias=True), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                      T.RandomRotation(20), T.ColorJitter(brightness=0.2), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize(IMG, antialias=True), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])


class DS(torch.utils.data.Dataset):
    def __init__(self, X, y, ids, tf):
        self.X, self.y, self.ids, self.tf = X, y, ids, tf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        j = self.ids[i]
        return self.tf(torch.from_numpy(self.X[j]).permute(2, 0, 1)), int(self.y[j])


def loader(X, y, ids, train):
    return torch.utils.data.DataLoader(DS(X, y, ids, train_tf if train else eval_tf), batch_size=BATCH, shuffle=train,
                                       generator=torch.Generator().manual_seed(SEED))


torch.manual_seed(SEED)
np.random.seed(SEED)
model, backbone = build("densenet121", len(SHARED), pretrained=True)
model.to(DEVICE)
bb = {id(p) for p in backbone.parameters()}
opt = torch.optim.AdamW([{"params": list(backbone.parameters()), "lr": 1e-4},
                         {"params": [p for p in model.parameters() if id(p) not in bb], "lr": 1e-3}], weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=2)


def run(dl, train):
    model.train(train)
    tot = cor = n = 0
    with torch.set_grad_enabled(train):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            tot += loss.item() * len(yb)
            cor += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return tot / n, cor / n


@torch.no_grad()
def predict(X, y, ids):
    model.eval()
    return torch.cat([F.softmax(model(xb.to(DEVICE)), 1).cpu() for xb, _ in loader(X, y, ids, False)]).numpy()


tr_dl, va_dl = loader(X_src, y_src, tr, True), loader(X_src, y_src, va, False)
best, state, bad, t0 = math.inf, None, 0, time.time()
for ep in range(1, EPOCHS + 1):
    t = time.time()
    tl, ta = run(tr_dl, True)
    vl, vacc = run(va_dl, False)
    sched.step(vl)
    print(f"reverse DenseNet121 epoch {ep:02d} train acc {ta:.3f} val acc {vacc:.3f} loss {vl:.3f} {time.time() - t:.0f}s", flush=True)
    if vl < best - 1e-4:
        best, bad, state = vl, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= PATIENCE:
            break
model.load_state_dict(state)
torch.save(state, OUT / "densenet121_reverse_seed42.pt")

p_in = predict(X_src, y_src, te)
p_out = predict(X_tgt, y_tgt, np.arange(len(y_tgt)))
res = {"classes": names, "train_images": int(len(tr)), "source_test_images": int(len(te)), "target_images": int(len(y_tgt)),
       "epochs_run": ep, "minutes": (time.time() - t0) / 60,
       "in_domain_acc": accuracy_score(y_src[te], p_in.argmax(1)), "in_domain_f1": f1_score(y_src[te], p_in.argmax(1), average="macro"),
       "cross_acc": accuracy_score(y_tgt, p_out.argmax(1)), "cross_f1": f1_score(y_tgt, p_out.argmax(1), average="macro"),
       "cross_recall_per_class": dict(zip(names, np.round([(p_out.argmax(1)[y_tgt == k] == k).mean() for k in range(len(names))], 4).tolist()))}

# forward direction on the same 6 classes, for comparison
ext = np.load("outputs/analysis/ext_probs_densenet121_seed42.npy")[:, SHARED]
test_fwd = split[split.split == "test"].label.to_numpy()
fwd_test = np.load("outputs/probs_densenet121_seed42.npz")["test"]
keep = np.isin(test_fwd, SHARED)
res["forward_in_domain_acc_6"] = accuracy_score([to6[c] for c in test_fwd[keep]], fwd_test[keep][:, SHARED].argmax(1))
res["forward_cross_acc_6"] = accuracy_score(y_src, ext.argmax(1))
(OUT / "reverse_summary.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
