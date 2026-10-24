"""The three models: small CNN, DenseNet121, VGG16."""
import torch.nn as nn
from torchvision import models

MODELS = ["cnn", "densenet121", "vgg16"]
LABELS = {"cnn": "CNN", "densenet121": "DenseNet121", "vgg16": "VGG16"}


class SmallCNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        layers, c = [], 3
        for w in (32, 64, 128, 256):
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 256), nn.ReLU(inplace=True),
                                  nn.Dropout(0.5), nn.Linear(256, n))

    def forward(self, x):
        return self.head(self.features(x))


def build(name, n, pretrained):
    # returns (model, backbone), backbone is trained with a lower lr
    if name == "cnn":
        return SmallCNN(n), None
    if name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        m.classifier = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, n))
        return m, m.features
    if name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        for p in m.features[:24].parameters():  # freeze blocks 1-4, train block 5
            p.requires_grad = False
        m.avgpool = nn.AdaptiveAvgPool2d(1)
        m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, n))
        return m, m.features
    raise ValueError(name)
