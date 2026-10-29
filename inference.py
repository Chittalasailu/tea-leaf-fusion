"""Single-image prediction with the three models and the confidence-gated advice rule (Algorithm 3).

    python inference.py leaf.jpg --threshold 0.8
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2 as T

import fusionlib as fl
from tea_models import LABELS, MODELS, build

CACHE, IMG = 256, 224
# same as eval_tf in tea_fusion.py
EVAL_TF = T.Compose([T.Resize(IMG, antialias=True), T.ToDtype(torch.float32, scale=True),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_image(path_or_img):
    """Same decoding as training: draft-mode JPEG, RGB, bilinear resize to 256x256."""
    im = Image.open(path_or_img) if not isinstance(path_or_img, Image.Image) else path_or_img
    im.draft("RGB", (CACHE, CACHE))
    return im.convert("RGB").resize((CACHE, CACHE), Image.BILINEAR)


def to_tensor(imgs):
    """List of 256x256 PIL images (or uint8 arrays) -> normalised 224x224 batch."""
    return torch.stack([EVAL_TF(torch.from_numpy(np.asarray(i, dtype=np.uint8).copy()).permute(2, 0, 1)) for i in imgs])


class Ensemble:
    def __init__(self, out_dir="outputs", seed=42, device="cpu"):
        out_dir = Path(out_dir)
        split = pd.read_csv(out_dir / "split.csv")
        self.classes = [c.split(". ", 1)[-1] for c in sorted(split.cls.unique())]
        y_val = split.loc[split.split == "val", "label"].to_numpy()
        val = [np.load(out_dir / f"probs_{m}_seed{seed}.npz")["val"] for m in MODELS]
        self.weights = fl.accuracy_weights(val, y_val)
        self.device = torch.device(device)
        self.models = {}
        for m in MODELS:
            net, _ = build(m, len(self.classes), pretrained=False)
            net.load_state_dict(torch.load(out_dir / f"{m}_seed{seed}.pt", map_location="cpu", weights_only=True))
            self.models[m] = net.to(self.device).eval()

    @torch.no_grad()
    def member_probs(self, imgs):
        x = to_tensor(imgs).to(self.device)
        return {m: F.softmax(net(x).float(), 1).cpu().numpy() for m, net in self.models.items()}

    def predict(self, img, threshold=0.8, require_agreement=True):
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        probs = self.member_probs([load_image(img)])
        fused = fl.weighted_average([probs[m] for m in MODELS], self.weights)[0]
        cls = int(fused.argmax())
        labels = {LABELS[m]: self.classes[int(probs[m][0].argmax())] for m in MODELS}
        agree = all(int(probs[m][0].argmax()) == cls for m in MODELS)
        ok = fused[cls] >= threshold and (agree or not require_agreement)
        return {"class": self.classes[cls], "confidence": float(fused[cls]), "decision": "advise" if ok else "refer",
                "members": labels, "agree": agree, "probabilities": dict(zip(self.classes, fused.round(4).tolist()))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--no-agreement", action="store_true")
    ap.add_argument("--outputs", default="outputs")
    a = ap.parse_args()
    print(json.dumps(Ensemble(a.outputs).predict(a.image, a.threshold, not a.no_agreement), indent=2))
