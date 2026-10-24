"""Cut out the leaf and put it on a plain grey background."""
import numpy as np
from scipy import ndimage

FILL = np.array([124, 116, 104], dtype=np.uint8)  # ImageNet mean, ~0 after normalising


def leaf_mask(img, d_min=0.22, s_min=0.10):
    # img is HxWx3 uint8. background colour comes from the border, leaf = biggest
    # region that differs from it and isn't grey (shadows on white paper are grey)
    x = img.astype(np.float32) / 255.0
    border = np.concatenate([x[:8].reshape(-1, 3), x[-8:].reshape(-1, 3), x[:, :8].reshape(-1, 3), x[:, -8:].reshape(-1, 3)])
    bg = np.median(border, 0)
    dist = np.sqrt(((x - bg) ** 2).sum(2))
    mx, mn = x.max(2), x.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    m = (dist > d_min) & ((sat > s_min) | (np.abs(mx - bg.max()) > 0.35))
    m = ndimage.binary_opening(m, iterations=2)
    lab, n = ndimage.label(m)
    if n == 0:
        return np.ones(img.shape[:2], bool)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    m = ndimage.binary_closing(m, iterations=4)
    m = ndimage.binary_fill_holes(m)
    if m.mean() < 0.02:  # mask failed, keep whole image
        return np.ones(img.shape[:2], bool)
    return m


def apply(img):
    out = img.copy()
    out[~leaf_mask(img)] = FILL
    return out

