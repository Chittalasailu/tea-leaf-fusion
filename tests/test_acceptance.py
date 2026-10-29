"""Criteria fixed before the reruns of Section 11.2: argmax agreement >= 99% and accuracy within one point."""
import json

import pandas as pd
import pytest

from conftest import OUT

R = OUT / "revision"
pytestmark = pytest.mark.skipif(not (R / "e16_reproduction.csv").exists(), reason="run robustness_eval.py first")


def test_rerun_argmax_agreement_at_least_99():
    assert (pd.read_csv(R / "e16_reproduction.csv").argmax_agreement >= 99).all()


def test_rerun_accuracy_within_one_point():
    d = pd.read_csv(R / "e16_reproduction.csv")
    assert ((d.acc_rerun - d.acc_saved).abs() <= 1).all()


def test_reverse_direction_within_one_point():
    r = json.loads((R / "e17_reverse.json").read_text())
    assert abs(r["acc_rerun"] - r["acc_saved"]) <= 1


def test_corruption_table_complete():
    d = pd.read_csv(R / "e18_corruptions.csv")
    assert len(d) == 13 and d.condition.iloc[0] == "Clean"


def test_clean_rerun_close_to_saved_seed_accuracy():
    d = pd.read_csv(R / "e18_corruptions.csv")
    saved = pd.read_csv(OUT / "results_all_seeds.csv")
    acc = saved[(saved.seed == 42) & (saved.config == "CNN + DenseNet121 + VGG16")].accuracy.item() * 100
    assert abs(d.Triple.iloc[0] - acc) <= 1
