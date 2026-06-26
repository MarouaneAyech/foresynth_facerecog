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
    from src.generator.api import build_generator
    from src.data.pairs import list_pairs
    gen = build_generator(cfg)
    k = cfg["generator"]["samples_per_identity"]
    # list_pairs renvoie 7 paires par identité (une par caméra), même mugshot_path à
    # chaque fois : dédupliquer, sinon sample() est appelé 7x par identité pour rien
    # (écrase chaque fois les mêmes fichiers de sortie -> 7x plus lent que nécessaire).
    mugshot_by_identity = {p.identity: p.mugshot_path for p in list_pairs(cfg, block="B")}
    for identity, mugshot_path in mugshot_by_identity.items():
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
    from src.recognition.eval import evaluate
    from src.utils.checkpoint import latest_checkpoint

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=STAGES)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    ensure_dirs(cfg)
    log.info("STAGE=%s CONFIG=%s (%s/%s)", args.stage, args.config, cfg["modality"], cfg["distance"])
    DISPATCH[args.stage](cfg)


if __name__ == "__main__":
    main()
