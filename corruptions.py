"""Six image corruptions at two levels, applied to the 256x256 RGB images before the eval transform."""
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

LEVELS = {
    "gaussian_noise": (0.05, 0.10),   # std on a 0-1 scale
    "gaussian_blur": (1.5, 3.0),      # radius in pixels
    "jpeg": (20, 8),                  # quality
    "darken": (0.6, 0.35),            # brightness factor
    "low_contrast": (0.6, 0.35),      # contrast factor
    "low_resolution": (96, 48),       # downsample size, then back up
}
NAMES = {"gaussian_noise": "Gaussian noise", "gaussian_blur": "Gaussian blur", "jpeg": "JPEG compression",
         "darken": "Darkening", "low_contrast": "Low contrast", "low_resolution": "Low resolution"}


def corrupt(img, kind, level, seed=0):
    """img: PIL RGB image. level: 0 = moderate, 1 = severe. seed only matters for noise."""
    if kind not in LEVELS or level not in (0, 1):
        raise ValueError(f"unknown corruption {kind} / {level}")
    v = LEVELS[kind][level]
    if kind == "gaussian_noise":
        a = np.asarray(img, dtype=np.float64) / 255
        a = np.clip(a + np.random.default_rng(seed).normal(0, v, a.shape), 0, 1)
        return Image.fromarray((a * 255).round().astype(np.uint8))
    if kind == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(v))
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=v)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if kind == "darken":
        return ImageEnhance.Brightness(img).enhance(v)
    if kind == "low_contrast":
        return ImageEnhance.Contrast(img).enhance(v)
    size = img.size
    return img.resize((v, v), Image.BILINEAR).resize(size, Image.BILINEAR)


def conditions():
    return [(k, lvl) for k in LEVELS for lvl in (0, 1)]
