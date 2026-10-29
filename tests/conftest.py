import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = ROOT / "outputs"
SEEDS = json.loads((OUT / "summary.json").read_text())["seeds"]
MODELS = ["cnn", "densenet121", "vgg16"]
HAS_IMAGES = (ROOT / "data" / "teaLeafBD").exists()
HAS_WEIGHTS = all((OUT / f"{m}_seed{s}.pt").exists() for m in MODELS for s in SEEDS)
needs_images = pytest.mark.skipif(not HAS_IMAGES, reason="teaLeafBD images not in data/")
needs_weights = pytest.mark.skipif(not HAS_WEIGHTS, reason="release weights not in outputs/")


@pytest.fixture(scope="session")
def split():
    return pd.read_csv(OUT / "split.csv")


@pytest.fixture(scope="session")
def labels(split):
    return {s: split.loc[split.split == s, "label"].to_numpy() for s in ("train", "val", "test")}


@pytest.fixture(scope="session")
def probs():
    return {(s, m): np.load(OUT / f"probs_{m}_seed{s}.npz") for s in SEEDS for m in MODELS}


@pytest.fixture(scope="session")
def ext():
    return {(s, m): np.load(OUT / "analysis" / f"ext_probs_{m}_seed{s}.npy") for s in SEEDS for m in MODELS}


@pytest.fixture(scope="session")
def y_ext():
    return np.load(OUT / "analysis" / "external_labels.npy")


def rand_probs(n=50, k=7, seed=0):
    return np.random.default_rng(seed).dirichlet(np.ones(k), n)
