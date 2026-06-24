"""Smoke tests : valident le câblage (config, fabrique, gate, dispatch) sans GPU ni données."""
import subprocess, sys
from src.config import load_config
from src.generator.api import build_generator
from src.fidelity.gate import decide
from experiments.run import DISPATCH, STAGES


def test_config_resolves_variables():
    cfg = load_config("configs/visible_d1.yaml")
    assert cfg["modality"] == "visible" and cfg["distance"] == "d1"
    # héritage depuis base.yaml
    assert cfg["recognition"]["scope"] == "layer3_4"
    # résolution ${...}
    assert cfg["paths"]["synth_dataset"].endswith("visible_d1")
    assert "${" not in cfg["paths"]["synth_dataset"]


def test_ir_template_switches_modality():
    cfg = load_config("configs/ir_d2.yaml")
    assert cfg["modality"] == "ir" and cfg["distance"] == "d2"
    assert cfg["eval"]["terrains"] == [["ir", "d2"]]


def test_generator_factory_instantiates():
    cfg = load_config("configs/visible_d1.yaml")
    gen = build_generator(cfg)
    assert hasattr(gen, "fit") and hasattr(gen, "sample")


def test_gate_logic():
    cfg = load_config("configs/visible_d1.yaml")
    assert decide(10.0, 0.9, cfg).passed
    assert not decide(999.0, 0.9, cfg).passed
    assert not decide(10.0, 0.0, cfg).passed


def test_all_stages_registered():
    assert set(DISPATCH) == set(STAGES)


def test_smoke_stage_runs_end_to_end():
    r = subprocess.run(
        [sys.executable, "-m", "experiments.run", "--config", "configs/visible_d1.yaml", "--stage", "smoke"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "SMOKE OK" in r.stdout
