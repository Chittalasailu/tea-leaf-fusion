"""One sample image per class from both datasets."""
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SRC, EXT, OUT = Path("data/teaLeafBD"), Path("data/TeaLeafDiseaseBD"), Path(sys.argv[1])
EXT_NAMES = {"Tea algal leaf spot": "Algal leaf spot", "Brown Blight": "Brown Blight", "Gray Blight": "Gray Blight",
             "Helopeltis": "Helopelties", "Red spider": "Red Spider", "Green mirid bug": None, "Healthy leaf": "Healthy"}
rng = random.Random(3)
classes = sorted(p.name for p in SRC.iterdir() if p.is_dir())

plt.rcParams.update({"font.family": "serif", "font.size": 7})
fig, axes = plt.subplots(2, len(classes), figsize=(7.16, 2.35), dpi=300)
for j, folder in enumerate(classes):
    name = folder.split(". ", 1)[-1]
    for i, (root, sub) in enumerate([(SRC, folder), (EXT, EXT_NAMES[name])]):
        ax = axes[i, j]
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if sub is None:
            ax.text(0.5, 0.5, "not in\ndataset", ha="center", va="center", fontsize=6, color="0.4", transform=ax.transAxes)
            continue
        f = rng.choice(sorted((root / sub).glob("*.jpg")))
        im = Image.open(f)
        im.draft("RGB", (300, 300))
        ax.imshow(im.convert("RGB").resize((300, 300)))
    axes[0, j].set_title(name, fontsize=6.5)
axes[0, 0].set_ylabel("teaLeafBD", fontsize=7)
axes[1, 0].set_ylabel("TeaLeafDiseaseBD", fontsize=7)
fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.3)
fig.savefig(OUT)
print("wrote", OUT)
