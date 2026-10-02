"""Tests CPU de la LoRA de reconnaissance (sans données ni GPU)."""
import torch

from src.config import load_config
from src.recognition.lora import freeze_all_batchnorm, inject_lora
from src.recognition.naming import recognition_tag
from src.utils.arcface_backbone import iresnet50


def _lora_net(r=32):
    net = iresnet50()
    net.requires_grad_(False)
    inject_lora(net, r=r, alpha=2 * r, target_layers=("layer3", "layer4"), include_fc=True)
    return net


def test_lora_r32_trainable_params_match_paper():
    # papier ch.3, Table 1 : 4.14M (tete ArcFace 100 classes incluse, ~0.05M) ;
    # backbone seul : 4,08M calcule analytiquement (conv layer3/4 + fc, r*(in+9*out))
    net = _lora_net(32)
    n = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert abs(n / 1e6 - 4.14) < 0.08, n


def test_lora_starts_at_pretrained_function():
    base = iresnet50().eval()
    net = iresnet50().eval()
    net.load_state_dict(base.state_dict())
    net.requires_grad_(False)
    inject_lora(net, r=8, alpha=16)
    net.eval()
    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        assert torch.allclose(base(x), net(x), atol=1e-5)  # B = 0 -> meme fonction


def test_frozen_batchnorm_statistics_do_not_drift():
    net = _lora_net(8)
    net.train()
    freeze_all_batchnorm(net)
    before = {k: v.clone() for k, v in net.state_dict().items() if "running" in k}
    net(torch.randn(4, 3, 112, 112))
    after = net.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_recognition_tags_do_not_collide():
    cfg = load_config("configs/visible_d1.yaml")
    assert cfg["recognition"]["mechanism"] == "lora"
    assert recognition_tag(cfg, "real", 42) == "recognition_lora_r32_real_seed42"
    cfg["recognition"]["synthetic_ratio"] = 0.2
    assert recognition_tag(cfg, "mixed", 42) == "recognition_lora_r32_mixed_ratio020_seed42"
    cfg["recognition"]["mechanism"] = "full_finetune"
    assert recognition_tag(cfg, "mixed", 42) == "recognition_mixed_seed42"
