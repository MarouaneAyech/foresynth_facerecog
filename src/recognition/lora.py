"""LoRA pour IResNet-50 (recette du chapitre 3 : D:\\projects\\bari, papier paper_VF.pdf).

Réf. Hu et al. 2022 (LoRA) ; Ding et al. 2024 (LoRA-C, extension aux convolutions).
A = conv 1x1 (in -> r), B = conv k x k d'origine (r -> out), B initialisée à zéro ;
W_eff = W_figé + (B @ A) * alpha/r. Classique (alpha/r), PAS rsLoRA (réservé au générateur).
`peft` ne cible pas cette architecture : implémentation manuelle, identique à bari.
"""
from __future__ import annotations
import math

import torch.nn as nn


class LoRAConv2d(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, r: int, alpha: int):
        super().__init__()
        self.conv = conv_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        for p in self.conv.parameters():
            p.requires_grad = False

        k = conv_layer.kernel_size[0]
        self.lora_A = nn.Conv2d(conv_layer.in_channels, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(r, conv_layer.out_channels, kernel_size=k,
                                stride=conv_layer.stride, padding=conv_layer.padding, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.conv(x) + self.lora_B(self.lora_A(x)) * self.scaling


class LoRALinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.linear = linear_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        for p in self.linear.parameters():
            p.requires_grad = False

        self.lora_A = nn.Linear(linear_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, linear_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scaling


def inject_lora(model: nn.Module, r: int, alpha: int,
                target_layers: tuple[str, ...] = ("layer3", "layer4"),
                include_fc: bool = True) -> tuple[int, int]:
    """Remplace les Conv2d 3x3 des `target_layers` (et optionnellement `fc`) par des
    adaptateurs LoRA. Retourne (n_convs, n_fc). Appeler APRÈS le chargement des poids
    pré-entraînés et APRÈS requires_grad_(False) sur tout le modèle."""
    targets = []
    for name, module in model.named_modules():
        if not any(name.startswith(t) for t in target_layers):
            continue
        for child_name, child in module.named_children():
            if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3):
                targets.append((module, child_name, child))
    for parent, child_name, conv in targets:
        setattr(parent, child_name, LoRAConv2d(conv, r=r, alpha=alpha))

    n_fc = 0
    if include_fc and isinstance(getattr(model, "fc", None), nn.Linear):
        model.fc = LoRALinear(model.fc, r=r, alpha=alpha)
        n_fc = 1
    return len(targets), n_fc


def freeze_all_batchnorm(model: nn.Module) -> int:
    """Met tous les BatchNorm en eval() : requires_grad=False ne fige PAS
    running_mean/running_var, seul .eval() le fait (papier, §3.3 : -36 points sinon).
    À rappeler après tout appel à model.train()."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            n += 1
    return n
