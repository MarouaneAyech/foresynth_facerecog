"""Orchestrateur unique : un 'stage' lançable seul (idéal Colab + comptes multiples).

Usage:
  python -m experiments.run --config configs/visible_d1.yaml --stage smoke
  python -m experiments.run --config configs/visible_d1.yaml --stage partition
  ... check_faces | train_generator | generate | fidelity | train_recognition | evaluate
"""
from __future__ import annotations
import argparse
from src.config import load_config
from src.utils.logging import get_logger
from src.utils.paths import ensure_dirs
from src.utils.seed import set_seed

log = get_logger()
STAGES = ["smoke", "partition", "check_faces", "train_generator", "generate",
          "fidelity", "train_recognition", "evaluate"]


def stage_smoke(cfg: dict) -> None:
    """Valide le câblage sans GPU ni données."""
    from src.generator.api import build_generator
    from src.fidelity.gate import decide
    assert cfg["modality"] in ("visible", "ir")
    assert cfg["distance"] in ("d1", "d2", "d3")
    gen = build_generator(cfg)             # fabrique OK (instanciation seule)
    res = decide(fid=10.0, cos=0.9, cfg=cfg)
    assert res.passed, "Logique de gate cassée"
    log.info("SMOKE OK | modality=%s distance=%s adapter=%s | gate(%.1f,%.2f)->%s",
             cfg["modality"], cfg["distance"], cfg["generator"]["adapter"],
             res.fid, res.cos, res.passed)


def stage_partition(cfg: dict) -> None:
    from src.data import partition; partition.run(cfg)


def stage_check_faces(cfg: dict) -> None:
    """Diagnostic AVANT entraînement : détectabilité des mugshots (Bloc A et B),
    rapide (insightface seul) -- évite de découvrir un échec au milieu d'un run
    de plusieurs heures (cf. incident identité 058, cadrage trop serré)."""
    from src.generator.face_detect import check_faces
    for block in ("A", "B"):
        check_faces(cfg, block)


def stage_train_generator(cfg: dict) -> None:
    from src.generator.api import build_generator
    from src.data.pairs import list_pairs
    build_generator(cfg).fit(list_pairs(cfg, block="A"))


def stage_generate(cfg: dict) -> None:
    from pathlib import Path
    from src.generator.api import build_generator
    from src.data.pairs import list_pairs
    gen = None
    k = cfg["generator"]["samples_per_identity"]
    synth_root = Path(cfg["paths"]["synth_dataset"])
    # list_pairs renvoie 7 paires par identité (une par caméra), même mugshot_path à
    # chaque fois : dédupliquer, sinon sample() est appelé 7x par identité pour rien
    # (écrase chaque fois les mêmes fichiers de sortie -> 7x plus lent que nécessaire).
    mugshot_by_identity = {p.identity: p.mugshot_path for p in list_pairs(cfg, block="B")}
    for identity, mugshot_path in mugshot_by_identity.items():
        # Reprenable : une coupure Colab en cours de route (ex. génération interrompue
        # à 15/20 images pour une identité) n'oblige pas à tout refaire -- relancer le
        # même stage saute les identités déjà complètes et termine/refait les autres.
        existing = len(list((synth_root / identity).glob("*.png"))) if (synth_root / identity).is_dir() else 0
        if existing >= k:
            log.info("Identité %s : déjà %d/%d images, ignorée", identity, existing, k)
            continue
        if gen is None:
            gen = build_generator(cfg)
        gen.sample(mugshot_path, k=k)
        log.info("Identité %s : %d échantillons générés", identity, k)


def stage_fidelity(cfg: dict) -> None:
    from src.fidelity import fid, embedding, gate
    f = fid.compute_fid(cfg)
    c = embedding.mean_identity_cosine(cfg)
    r = gate.decide(f, c, cfg)
    log.info("FIDELITY %s | %s", "PASS" if r.passed else "FAIL", r.reason)


def stage_train_recognition(cfg: dict) -> None:
    from src.recognition.finetune import train
    for cond in cfg["recognition"]["conditions"]:
        for seed in cfg["seeds"]:
            path = train(cfg, condition=cond, seed=seed)
            log.info("Entraînement %s/seed=%d terminé -> %s", cond, seed, path)


def stage_evaluate(cfg: dict) -> None:
    import statistics
    from pathlib import Path
    from src.recognition.eval import evaluate
    from src.utils.checkpoint import latest_checkpoint

    # Baseline SANS fine-tuning : le backbone pre-entraine (arcface_ms1mv3) tel quel,
    # comme reference pour juger si un quelconque fine-tuning (real/synthetic/mixed)
    # apporte vraiment un gain -- pas de seed/checkpoint, deterministe, toujours evalue.
    baseline_weights = cfg["paths"].get("arcface_weights")
    if baseline_weights and Path(baseline_weights).exists():
        rank1_baseline = evaluate(cfg, weights_path=baseline_weights)["visible_d1"]
        log.info("RANK-1 baseline (sans fine-tuning) : %.4f", rank1_baseline)
    else:
        log.info("Baseline sans fine-tuning ignoree : paths.arcface_weights introuvable (%s)", baseline_weights)

    for condition in cfg["recognition"]["conditions"]:
        rank1s = []
        for seed in cfg["seeds"]:
            tag = f"recognition_{condition}_seed{seed}"
            ckpt = latest_checkpoint(cfg["paths"]["checkpoints"], tag)
            if ckpt is None:
                log.info("Pas de checkpoint pour %s (seed=%d) : 'train_recognition' doit tourner avant.",
                          condition, seed)
                continue
            rank1s.append(evaluate(cfg, weights_path=str(ckpt))["visible_d1"])
        if rank1s:
            mean = statistics.mean(rank1s)
            std = statistics.pstdev(rank1s) if len(rank1s) > 1 else 0.0
            log.info("RANK-1 %s : %.4f ± %.4f (n=%d seeds)", condition, mean, std, len(rank1s))


DISPATCH = {
    "smoke": stage_smoke, "partition": stage_partition, "check_faces": stage_check_faces,
    "train_generator": stage_train_generator, "generate": stage_generate,
    "fidelity": stage_fidelity, "train_recognition": stage_train_recognition,
    "evaluate": stage_evaluate,
}


def main() -> None:
    from src.utils.logging import attach_file_handler

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=STAGES)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    ensure_dirs(cfg)
    # Copie persistante sur Drive (cf. CLAUDE.md), en plus de la console -- la sortie
    # de cellule Colab seule se perd si la session coupe sans sauvegarde du notebook.
    attach_file_handler(log, cfg["paths"]["checkpoints"],
                         f"log_{args.stage}_{cfg['modality']}_{cfg['distance']}.txt")
    log.info("STAGE=%s CONFIG=%s (%s/%s)", args.stage, args.config, cfg["modality"], cfg["distance"])
    DISPATCH[args.stage](cfg)


if __name__ == "__main__":
    main()
