"""IResNet-50 + embedder ArcFace : backbone PARTAGÉ entre generator/ (perte d'identité),
fidelity/ (cosinus synth<->réel) et recognition/ (modèle effectivement fine-tuné).

Une seule définition d'architecture évite toute dérive entre "ce qui verrouille
l'identité pendant la génération" et "ce qui est réellement évalué" (recette B1 :
iresnet50 + poids arcface_ms1mv3). Architecture standard InsightFace (arcface_torch) :
IBasicBlock (BN-Conv-BN-PReLU-Conv-BN) + tête FC-BN -> embedding 512-d.
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_SIZE = 512
INPUT_SIZE = 112  # résolution attendue par ArcFace (alignement 5 points, cf. data/pairs.py)


class IBasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    """layers=(3,4,14,3) -> IResNet-50 (recette B1 : iresnet50/arcface_ms1mv3)."""

    def __init__(self, layers: tuple[int, int, int, int] = (3, 4, 14, 3)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1])
        self.layer3 = self._make_layer(256, layers[2])
        self.layer4 = self._make_layer(512, layers[3])
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.fc = nn.Linear(512 * 7 * 7, EMBEDDING_SIZE)  # 112 / 2**4 = 7
        self.features = nn.BatchNorm1d(EMBEDDING_SIZE, eps=1e-5)

    def _make_layer(self, planes: int, blocks: int) -> nn.Sequential:
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes, 1, 2, bias=False),
            nn.BatchNorm2d(planes, eps=1e-5),
        )
        layers = [IBasicBlock(self.inplanes, planes, stride=2, downsample=downsample)]
        self.inplanes = planes
        layers += [IBasicBlock(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.features(x)


def iresnet50() -> IResNet:
    return IResNet(layers=(3, 4, 14, 3))


def preprocess_for_arcface(images: torch.Tensor) -> torch.Tensor:
    """images: (B,3,H,W) en [0,1] -> resize 112x112 + normalisation [-1,1] (ArcFace)."""
    if images.shape[-2:] != (INPUT_SIZE, INPUT_SIZE):
        images = F.interpolate(images, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
    return (images - 0.5) / 0.5


class ArcFaceEmbedder(nn.Module):
    """ArcFace IResNet-50 GELÉ : embeddings 512-d L2-normalisés.

    Frozen en POIDS (requires_grad_(False)) mais pas en graphe : le gradient doit
    pouvoir remonter jusqu'aux pixels d'entrée pour la perte d'identité (cf.
    generator/identity_loss.py). Ne jamais appeler sous torch.no_grad() côté train.
    """

    def __init__(self, weights_path: str | Path | None = None):
        super().__init__()
        self.net = iresnet50()
        if weights_path is not None and Path(weights_path).exists():
            state = torch.load(weights_path, map_location="cpu")
            self.net.load_state_dict(state)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = preprocess_for_arcface(images)
        return F.normalize(self.net(x), dim=-1)


def load_image_as_tensor(path: str | Path, size: int = INPUT_SIZE) -> torch.Tensor:
    """Charge une image disque -> tenseur (3,size,size) en [0,1]. Partagé par
    generator/fidelity/recognition pour éviter les redondances de chargement."""
    import torchvision.transforms.functional as TF
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size))
    return TF.to_tensor(img)


def load_arcface_embedder(cfg: dict) -> ArcFaceEmbedder:
    """Charge l'embedder ArcFace partagé. Poids attendus sous cfg['paths']['arcface_weights']
    (à déposer manuellement : checkpoint iresnet50/arcface_ms1mv3, cf. recognition.*)."""
    weights_path = cfg["paths"].get("arcface_weights")
    embedder = ArcFaceEmbedder(weights_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return embedder.to(device)
