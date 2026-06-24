"""Tête ArcFace (cosine softmax à marge additive) pour le fine-tuning.

Standard ArcFace (Deng et al.) : logits = s * cos(theta + m) sur la classe cible,
s * cos(theta) ailleurs. Uniquement utilisée pendant train() ; l'embedding nu
(iresnet50) sert seul à l'évaluation (rank-1) et aux pertes/cosinus d'identité.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    def __init__(self, embedding_size: int, num_classes: int, scale: float = 64.0, margin: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.cos_m, self.sin_m = math.cos(margin), math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=0.0, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        onehot = torch.zeros_like(cosine)
        onehot.scatter_(1, labels.view(-1, 1), 1)
        return (onehot * phi + (1.0 - onehot) * cosine) * self.scale
