"""Agreement gate: give advice only if confidence >= 0.80 and all three models agree."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from tea_models import MODELS

OUT = Path("outputs")
A = OUT / "analysis"
THRESH = 0.80
split = pd.read_csv(OUT / "split.csv")
y_val = split.loc[split.split == "val", "label"].to_numpy()
y_test = split.loc[split.split == "test", "label"].to_numpy()
y_ext = np.load(A / "external_labels.npy")
SEEDS = json.loads((OUT / "summary.json").read_text())["seeds"]


def stats(pred, conf, keep, y):
    ok = pred == y
    return {"advised": keep.mean(), "correct_when_advised": ok[keep].mean() if keep.any() else np.nan,
            "wrong_advice": (keep & ~ok).mean(), "referred": 1 - keep.mean()}


rows = []
for s in SEEDS:
    z = {m: np.load(OUT / f"probs_{m}_seed{s}.npz") for m in MODELS}
    va = {m: accuracy_score(y_val, z[m]["val"].argmax(1)) for m in MODELS}
    w = {m: va[m] / sum(va.values()) for m in MODELS}
    best = max(MODELS, key=lambda m: va[m])
    for data, y, P in [("teaLeafBD", y_test, {m: z[m]["test"] for m in MODELS}),
                       ("unseen", y_ext, {m: np.load(A / f"ext_probs_{m}_seed{s}.npy") for m in MODELS})]:
        fused = sum(w[m] * P[m] for m in MODELS)
        pred, conf = fused.argmax(1), fused.max(1)
        agree = np.all([P[m].argmax(1) == pred for m in MODELS], axis=0)
        pb = P["densenet121"]
        for rule, pr, cf, keep in [
            ("DenseNet121, confidence rule", pb.argmax(1), pb.max(1), pb.max(1) >= THRESH),
            ("Triple, confidence rule", pred, conf, conf >= THRESH),
            ("Triple, agreement gate (proposed)", pred, conf, (conf >= THRESH) & agree),
        ]:
            rows.append({"seed": s, "data": data, "rule": rule, **stats(pr, cf, keep, y)})
        # error rate when the models agree vs disagree
        ok = pred == y
        rows.append({"seed": s, "data": data, "rule": "_disagreement", "err_when_disagree": (~ok[~agree]).mean(),
                     "err_when_agree": (~ok[agree]).mean(), "disagree_rate": (~agree).mean()})

df = pd.DataFrame(rows)
df.to_csv(A / "gate_all_seeds.csv", index=False)
g = df.drop(columns="seed").groupby(["data", "rule"])
clean = lambda frame: {f"{d}|{r}": {k: x for k, x in v.items() if pd.notna(x)} for (d, r), v in frame.round(4).to_dict(orient="index").items()}
summary = {"mean": clean(g.mean()), "std": clean(g.std()),
           "threshold": THRESH, "seeds": SEEDS}
(A / "gate_summary.json").write_text(json.dumps(summary, indent=1, default=float))
print((g.mean() * 100).round(1).to_string())
