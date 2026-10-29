"""Rebuild the published numbers from the saved predictions with independent code."""
import json
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from conftest import MODELS, OUT, SEEDS

LABEL = {"cnn": "CNN", "densenet121": "DenseNet121", "vgg16": "VGG16"}
CONFIGS = [c for r in (1, 2, 3) for c in combinations(MODELS, r)]
A = OUT / "analysis"


def fused(probs, labels, s, c, part="test", src=None):
    acc = np.array([(probs[s, m]["val"].argmax(1) == labels["val"]).mean() for m in c])
    w = acc / acc.sum()
    get = (lambda m: probs[s, m][part]) if src is None else (lambda m: src[s, m])
    return sum(wi * get(m) for wi, m in zip(w, c))


def name(c):
    return " + ".join(LABEL[m] for m in c)


@pytest.mark.parametrize("c", CONFIGS, ids=[name(c) for c in CONFIGS])
def test_main_table_accuracy(probs, labels, c):
    table = pd.read_csv(OUT / "table3.csv", index_col=0)
    acc = [(fused(probs, labels, s, c).argmax(1) == labels["test"]).mean() * 100 for s in SEEDS]
    mean, sd = (float(x) for x in table.loc[name(c), "accuracy"].split(" ± "))
    assert np.mean(acc) == pytest.approx(mean, abs=0.006) and np.std(acc, ddof=1) == pytest.approx(sd, abs=0.006)


def test_mcnemar_matches_summary(probs, labels):
    summ = json.loads((OUT / "summary.json").read_text())["mcnemar"]
    s = SEEDS[0]
    by = {name(c): fused(probs, labels, s, c).argmax(1) == labels["test"] for c in CONFIGS}
    for t in summ:
        a, b = by[t["a"]], by[t["b"]]
        n1, n2 = int((a & ~b).sum()), int((~a & b).sum())
        assert (n1, n2) == (t["only_a_correct"], t["only_b_correct"])
        assert binomtest(min(n1, n2), n1 + n2, 0.5).pvalue == pytest.approx(t["p_value"])


def test_diversity_matches_published(probs, labels):
    pub = pd.read_csv(A / "diversity_pairs.csv")
    for _, r in pub.iterrows():
        pair = [k for k, v in LABEL.items() if v in r.pair]
        ca = probs[r.seed, pair[0]]["test"].argmax(1) == labels["test"]
        cb = probs[r.seed, pair[1]]["test"].argmax(1) == labels["test"]
        assert ((ca != cb).mean()) == pytest.approx(r.disagreement, abs=1e-6)
        assert ((~ca & ~cb).mean()) == pytest.approx(r.double_fault, abs=1e-6)


def test_oracle_of_triple(probs, labels):
    o = [np.any([probs[s, m]["test"].argmax(1) == labels["test"] for m in MODELS], 0).mean() for s in SEEDS]
    assert np.mean(o) * 100 == pytest.approx(97.82, abs=0.01)


def test_all_42_cross_dataset_values(probs, labels, ext, y_ext):
    pub = pd.read_csv(A / "external_results_all_seeds.csv")
    shared = sorted(set(y_ext.tolist()))
    n = 0
    for _, r in pub.iterrows():
        c = next(c for c in CONFIGS if name(c) == r.config)
        p = fused(probs, labels, r.seed, c, src=ext).argmax(1)
        assert (p == y_ext).mean() == pytest.approx(r.external_accuracy, abs=1e-9)
        assert f1_score(y_ext, p, labels=shared, average="macro", zero_division=0) == pytest.approx(r.external_macro_f1, abs=1e-9)
        n += 2
    assert n == 42


def test_threshold_numbers(probs, labels):
    cov, sel = [], []
    for s in SEEDS:
        p = fused(probs, labels, s, tuple(MODELS))
        keep = p.max(1) >= 0.8
        cov.append(keep.mean())
        sel.append((p.argmax(1) == labels["test"])[keep].mean())
    assert np.mean(cov) * 100 == pytest.approx(75.35, abs=0.01) and np.mean(sel) * 100 == pytest.approx(99.08, abs=0.01)


def test_confident_wrong_on_unseen(probs, labels, ext, y_ext):
    def wrong(c):
        return np.mean([((p := fused(probs, labels, s, c, src=ext)).max(1) >= 0.8) & (p.argmax(1) != y_ext) for s in SEEDS]) * 100
    assert wrong(tuple(MODELS)) == pytest.approx(19.8, abs=0.05)
    assert wrong(("densenet121",)) == pytest.approx(37.9, abs=0.05)


def test_calibration_matches_published():
    pub = json.loads((A / "calibration_summary.json").read_text())["mean"]
    rev = pd.read_csv(OUT / "revision" / "e8_calibration.csv").set_index("config")
    for cfg in pub:
        assert rev.loc[cfg, "ece"] == pytest.approx(pub[cfg]["ece_raw"], abs=1e-3)
        assert rev.loc[cfg, "T"] == pytest.approx(pub[cfg]["T"], abs=1e-3)


def test_agreement_gate_matches_published(probs, labels, ext, y_ext):
    pub = json.loads((A / "gate_summary.json").read_text())["mean"]
    err = []
    for s in SEEDS:
        p = fused(probs, labels, s, tuple(MODELS), src=ext)
        agree = np.all([ext[s, m].argmax(1) == p.argmax(1) for m in MODELS], 0)
        err.append((p.argmax(1) != y_ext)[agree].mean())
    assert np.mean(err) == pytest.approx(pub["unseen|_disagreement"]["err_when_agree"], abs=1e-4)


def test_remedies_match_published(probs, labels, y_ext):
    pub = json.loads((A / "remedies_summary.json").read_text())["mean"]
    for tag, key in [("cutout", "Background removal (test time)"), ("adabn", "AdaBN (test time)")]:
        src = {(s, m): np.load(A / f"rem_{tag}_{m}_seed{s}.npy") for s in SEEDS for m in MODELS}
        acc = np.mean([(fused(probs, labels, s, tuple(MODELS), src=src).argmax(1) == y_ext).mean() for s in SEEDS])
        assert acc == pytest.approx(pub[f"{key}|CNN + DenseNet121 + VGG16"]["acc"], abs=1e-4)


def test_reverse_summary_consistent():
    r = json.loads((OUT.parent / "outputs_reverse" / "reverse_summary.json").read_text())
    assert r["target_images"] == 3996 and r["train_images"] == 7667
    assert r["cross_acc"] * 100 == pytest.approx(68.19, abs=0.01)


def test_revision_rules_table_has_ten_rules():
    assert len(pd.read_csv(OUT / "revision" / "e6_rules.csv")) == 10


def test_revision_numbers_headline():
    n = json.loads((OUT / "revision" / "numbers.json").read_text())
    assert n["all_correct"] == pytest.approx(77.56, abs=0.01) and n["all_wrong"] == pytest.approx(2.18, abs=0.01)
