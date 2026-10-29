"""Architectures, preprocessing and the single-image inference module."""
import numpy as np
import pytest
import torch
from PIL import Image

from conftest import needs_weights
from tea_models import build

PARAMS = {"cnn": 1_241_767, "densenet121": 7_218_055, "vgg16": 14_718_279}


@pytest.mark.parametrize("name", list(PARAMS))
def test_output_shape(name):
    m, _ = build(name, 7, pretrained=False)
    with torch.no_grad():
        assert m.eval()(torch.zeros(2, 3, 224, 224)).shape == (2, 7)


@pytest.mark.parametrize("name", list(PARAMS))
def test_parameter_count(name):
    m, _ = build(name, 7, pretrained=False)
    assert sum(p.numel() for p in m.parameters()) == PARAMS[name]


def test_vgg16_blocks_1_to_4_frozen():
    m, _ = build("vgg16", 7, pretrained=False)
    frozen = [p.requires_grad for p in m.features[:24].parameters()]
    trainable = [p.requires_grad for p in m.features[24:].parameters()]
    assert not any(frozen) and all(trainable)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 7_083_015


def test_cnn_has_no_backbone():
    assert build("cnn", 7, pretrained=False)[1] is None


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build("resnet50", 7, pretrained=False)


def test_inference_preprocessing_equals_training_eval_transform():
    from torchvision.transforms import v2 as T

    from inference import to_tensor
    eval_tf = T.Compose([T.Resize(224, antialias=True), T.ToDtype(torch.float32, scale=True),
                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = (np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8)
    a = to_tensor([img])[0]
    b = eval_tf(torch.from_numpy(img).permute(2, 0, 1))
    assert torch.allclose(a, b)


def test_load_image_size_and_mode(tmp_path):
    from inference import load_image
    f = tmp_path / "x.jpg"
    Image.new("RGBA", (640, 480), (10, 200, 30, 255)).convert("RGB").save(f)
    im = load_image(f)
    assert im.size == (256, 256) and im.mode == "RGB"


def test_grayscale_image_is_converted(tmp_path):
    from inference import load_image
    f = tmp_path / "g.png"
    Image.new("L", (300, 300), 128).save(f)
    assert load_image(f).mode == "RGB"


@needs_weights
def test_predict_output_fields():
    from inference import Ensemble
    r = Ensemble().predict(Image.new("RGB", (400, 400), (40, 120, 40)))
    assert set(r) == {"class", "confidence", "decision", "members", "agree", "probabilities"}
    assert np.isclose(sum(r["probabilities"].values()), 1, atol=1e-3)


@needs_weights
def test_threshold_one_always_refers():
    from inference import Ensemble
    assert Ensemble().predict(Image.new("RGB", (256, 256), (0, 0, 0)), threshold=1.0)["decision"] == "refer"


@needs_weights
def test_invalid_threshold_rejected():
    from inference import Ensemble
    with pytest.raises(ValueError):
        Ensemble().predict(Image.new("RGB", (256, 256)), threshold=1.5)
