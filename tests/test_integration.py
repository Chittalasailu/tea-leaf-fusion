"""Rerun the original scripts in a temporary copy of the repository and compare with the committed files."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import ROOT, needs_images

SCRIPTS = ["tea_models.py", "fusionlib.py", "calibration.py", "gate.py", "tea_fusion.py", "revision.py"]


@pytest.fixture(scope="module")
def copy(tmp_path_factory):
    d = tmp_path_factory.mktemp("repo")
    for f in SCRIPTS:
        shutil.copy(ROOT / f, d / f)
    shutil.copytree(ROOT / "outputs", d / "outputs", ignore=shutil.ignore_patterns("*.pt", "revision"))
    (d / "outputs" / "revision").mkdir()
    shutil.copy(ROOT / "outputs" / "revision" / "e8_calibration.csv", d / "outputs" / "revision")
    return d


def run(cmd, cwd, env=None, timeout=1800):
    r = subprocess.run([sys.executable, *cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "MPLBACKEND": "Agg", **(env or {})})
    assert r.returncode == 0, r.stderr[-2000:]


def test_calibration_script_reproduces_summary(copy):
    run(["calibration.py"], copy)
    a = json.loads((copy / "outputs/analysis/calibration_summary.json").read_text())
    b = json.loads((ROOT / "outputs/analysis/calibration_summary.json").read_text())
    assert a == b


def test_gate_script_reproduces_summary(copy):
    run(["gate.py"], copy)
    a = json.loads((copy / "outputs/analysis/gate_summary.json").read_text())
    b = json.loads((ROOT / "outputs/analysis/gate_summary.json").read_text())
    assert a == b


def test_revision_script_reproduces_tables(copy):
    run(["revision.py"], copy)
    for f in ["e1_main_results.csv", "e2_significance.csv", "e6_rules.csv", "e9_advisory.csv", "e10_cross_dataset.csv"]:
        a = pd.read_csv(copy / "outputs/revision" / f).drop(columns=["cost_ms"], errors="ignore")
        b = pd.read_csv(ROOT / "outputs/revision" / f).drop(columns=["cost_ms"], errors="ignore")
        pd.testing.assert_frame_equal(a, b)


@needs_images
def test_training_script_rebuilds_main_table_byte_identical(copy):
    os.symlink(ROOT / "data", copy / "data")
    run(["tea_fusion.py"], copy, {"TEA_SEEDS": "42,7,123"})
    assert (copy / "outputs/table3.csv").read_bytes() == (ROOT / "outputs/table3.csv").read_bytes()


def test_one_epoch_training_on_synthetic_images(tmp_path):
    data = tmp_path / "data"
    rng = np.random.default_rng(0)
    for k in range(7):
        (data / f"{k + 1}. class{k}").mkdir(parents=True)
        for i in range(14):
            a = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
            a[..., k % 3] = 40 * k
            Image.fromarray(a).save(data / f"{k + 1}. class{k}" / f"{i}.jpg")
    for f in ["tea_models.py", "tea_fusion.py"]:
        shutil.copy(ROOT / f, tmp_path / f)
    run(["tea_fusion.py"], tmp_path, {"TEA_SMOKE": "1", "TEA_DATA": str(data), "TEA_OUT": str(tmp_path / "out"),
                                      "CUDA_VISIBLE_DEVICES": "", "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
    assert len(pd.read_csv(tmp_path / "out" / "table3.csv")) == 7
