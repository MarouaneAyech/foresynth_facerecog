import pytest

from src.config import apply_overrides, load_config


def test_overrides_apply_typed_values():
    cfg = apply_overrides(load_config("configs/visible_d1.yaml"), [
        "recognition.synthetic_ratio=0.2", "seeds=[7, 42]", "recognition.lora.rank=16",
        "generator.sample.filter_during_generation=true",
    ])
    assert cfg["recognition"]["synthetic_ratio"] == 0.2
    assert cfg["seeds"] == [7, 42]
    assert cfg["recognition"]["lora"]["rank"] == 16
    assert cfg["generator"]["sample"]["filter_during_generation"] is True


def test_overrides_reject_unknown_keys():
    cfg = load_config("configs/visible_d1.yaml")
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["recognition.nope=1"])
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["nope.sub=1"])
