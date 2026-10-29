"""Corruption functions used in the robustness experiment."""
import numpy as np
import pytest
from PIL import Image

from corruptions import conditions, corrupt


@pytest.fixture(scope="module")
def img():
    rng = np.random.default_rng(0)
    a = (rng.random((256, 256, 3)) * 120 + 60).astype(np.uint8)
    a[64:192, 64:192, 1] = 220
    return Image.fromarray(a)


@pytest.mark.parametrize("kind,level", conditions(), ids=[f"{k}-{l}" for k, l in conditions()])
def test_corruption_keeps_size_and_mode(img, kind, level):
    out = corrupt(img, kind, level, seed=1)
    assert out.size == img.size and out.mode == "RGB"


def test_twelve_conditions():
    assert len(conditions()) == 12


def test_noise_is_seeded(img):
    a = np.asarray(corrupt(img, "gaussian_noise", 0, seed=5))
    b = np.asarray(corrupt(img, "gaussian_noise", 0, seed=5))
    c = np.asarray(corrupt(img, "gaussian_noise", 0, seed=6))
    assert (a == b).all() and not (a == c).all()


def test_darkening_reduces_brightness(img):
    base = np.asarray(img).mean()
    assert np.asarray(corrupt(img, "darken", 1)).mean() < np.asarray(corrupt(img, "darken", 0)).mean() < base


def test_low_contrast_reduces_spread(img):
    assert np.asarray(corrupt(img, "low_contrast", 1)).std() < np.asarray(img).std()


def test_blur_removes_detail(img):
    d = lambda a: np.abs(np.diff(np.asarray(a, float), axis=0)).mean()
    assert d(corrupt(img, "gaussian_blur", 1)) < d(corrupt(img, "gaussian_blur", 0)) < d(img)


def test_unknown_corruption_rejected(img):
    with pytest.raises(ValueError):
        corrupt(img, "fog", 0)
    with pytest.raises(ValueError):
        corrupt(img, "jpeg", 2)


def test_batched_preprocessing_equals_single_image(img):
    from inference import to_tensor
    imgs = [img, corrupt(img, "jpeg", 1), corrupt(img, "darken", 0)]
    batch = to_tensor(imgs)
    for i, im in enumerate(imgs):
        assert (to_tensor([im])[0] == batch[i]).all()
