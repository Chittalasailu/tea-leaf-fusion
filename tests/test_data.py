"""Integrity of the split, the saved predictions and the rebuilt labels."""
import numpy as np
import pandas as pd
import pytest

from conftest import MODELS, OUT, SEEDS

COUNTS = {"Tea algal leaf spot": 418, "Brown Blight": 508, "Gray Blight": 1013, "Helopeltis": 607,
          "Red spider": 515, "Green mirid bug": 1282, "Healthy leaf": 935}


def test_split_sizes(split):
    assert split.split.value_counts().to_dict() == {"train": 3694, "val": 528, "test": 1056}


def test_split_has_no_duplicate_images(split):
    assert split.path.is_unique


def test_split_class_counts(split):
    got = split.cls.str.split(". ", n=1).str[-1].value_counts().to_dict()
    assert got == COUNTS


def test_split_is_stratified(split):
    share = pd.crosstab(split.cls, split.split, normalize="columns")
    assert (share.max(axis=1) - share.min(axis=1)).max() < 0.01


def test_labels_match_class_folders(split):
    classes = sorted(split.cls.unique())
    assert (split.label == split.cls.map({c: i for i, c in enumerate(classes)})).all()


def test_paths_are_relative(split):
    assert not split.path.str.startswith("/").any()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("model", MODELS)
def test_saved_predictions_are_valid(probs, labels, seed, model):
    z = probs[seed, model]
    for part in ("val", "test"):
        p = z[part]
        assert p.shape == (len(labels[part]), 7)
        assert np.allclose(p.sum(1), 1, atol=1e-4) and (p >= 0).all()


def test_external_labels(y_ext):
    assert len(y_ext) == 10954
    assert np.bincount(y_ext, minlength=7).tolist() == [1799, 1957, 1949, 1879, 1588, 0, 1782]


def test_external_predictions_match_labels(ext, y_ext):
    assert all(p.shape == (len(y_ext), 7) for p in ext.values())


def test_remedy_predictions_present():
    for tag in ("cutout", "adabn"):
        for s in SEEDS:
            for m in MODELS:
                assert np.load(OUT / "analysis" / f"rem_{tag}_{m}_seed{s}.npy").shape == (10954, 7)
