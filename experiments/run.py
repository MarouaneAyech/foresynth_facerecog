"""Orchestrateur unique : un 'stage' lançable seul (idéal Colab + comptes multiples).

Usage:
  python -m experiments.run --config configs/visible_d1.yaml --stage smoke
  python -m experiments.run --config configs/visible_d1.yaml --stage partition
  ... check_faces | train_generator | generate | filter_synthetic | fidelity |
      train_recognition | evaluate
"""
from __future__ import annotations
import argparse
from src.config import apply_overrides, load_config
from src.utils.logging import get_logger
from src.utils.paths import ensure_dirs
from src.utils.seed import set_seed

log = get_logger()
STAGES = ["smoke", "partition", "check_faces", "train_generator", "generate",
          "ir_postprocess", "filter_synthetic", "fidelity", "train_recognition", "evaluate"]


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


def stage_ir_postprocess(cfg: dict) -> None:
    """Post-traitement déterministe des images synthétiques IR (niveaux de gris +
    flou + bruit capteur). Modifie les PNG in-place dans synth_dataset.
    Ordre : après 'generate', avant 'fidelity'. Réservé à modality=ir."""
    from src.generator.ir_postprocess import ir_postprocess_dataset
    ir_postprocess_dataset(cfg)


def stage_filter_synthetic(cfg: dict) -> None:
    """Optionnel, entre 'generate' et 'train_recognition'/'fidelity' : retire du
    pool d'entraînement les échantillons synthétiques dont l'identité n'est pas
    assez préservée (cosinus ArcFace par image -- le FID, lui, n'est pas
    définissable par image, cf. src/fidelity/filter.py)."""
    from src.fidelity.filter import filter_synthetic
    filter_synthetic(cfg)


def stage_fidelity(cfg: dict) -> None:
    from src.fidelity import fid, embedding, gate
    f = fid.compute_fid(cfg)
    c = embedding.mean_identity_cosine(cfg)
    r = gate.decide(f, c, cfg)
    log.info("FIDELITY %s | %s", "PASS" if r.passed else "FAIL", r.reason)
    # Distribution par image (moyenne, écart-type, % sous seuil) -- ce sont les
    # chiffres attendus par le tableau de fidélité de l'article, que le gate
    # go/no-go seul (mean_identity_cosine, agrégée par identité) ne donne pas.
    dist = embedding.identity_cosine_distribution(cfg)
    log.info("FIDELITY STATS | FID=%.2f | cosine=%.4f +/- %.4f (n=%d) | %.1f%% sous le seuil %.2f",
              f, dist.mean, dist.std, dist.n, dist.pct_below, cfg["fidelity"]["filter_cos_min"])
    # Baseline reel-vs-reel (leave-one-out, Bloc B) : calibre ce qu'un cosinus "normal"
    # represente dans ce domaine deja degrade, independamment du generateur -- a comparer
    # directement a la ligne FIDELITY STATS ci-dessus avant de conclure a un defaut d'identite.
    baseline = embedding.real_identity_cosine_baseline(cfg)
    log.info("FIDELITY BASELINE (reel vs reel, leave-one-out) | cosine=%.4f +/- %.4f (n=%d) | %.1f%% sous le seuil %.2f",
              baseline.mean, baseline.std, baseline.n, baseline.pct_below, cfg["fidelity"]["filter_cos_min"])
    # Decision finale : le synthetique est-il coherent avec le plancher reel-vs-reel,
    # plutot que juge contre un seuil (fidelity.cos_min) jamais calibre empiriquement.
    comparison = gate.compare_to_baseline(dist, baseline)
    log.info("FIDELITY VERDICT | %s", comparison.reason)


def stage_train_recognition(cfg: dict) -> None:
    from src.recognition.finetune import train
    for cond in cfg["recognition"]["conditions"]:
        for seed in cfg["seeds"]:
            path = train(cfg, condition=cond, seed=seed)
            log.info("Entraînement %s/seed=%d terminé -> %s", cond, seed, path)


def _append_result(cfg: dict, condition: str, seed: int, rank1: float, ckpt) -> None:
    """Une ligne par (checkpoint evalue) dans outputs/results_recognition.csv (Drive) :
    alimente directement le tableau de dosage, sans recopier les logs a la main."""
    import csv
    import datetime
    from pathlib import Path
    rec = cfg["recognition"]
    path = Path(cfg["paths"]["outputs"]) / "results_recognition.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "terrain", "mechanism", "condition", "synthetic_ratio",
                        "seed", "rank1", "checkpoint"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                    f"{cfg['modality']}_{cfg['distance']}", rec.get("mechanism", "full_finetune"),
                    condition, rec.get("synthetic_ratio") if condition == "mixed" else "",
                    seed, f"{rank1:.4f}", Path(ckpt).name])


def stage_evaluate(cfg: dict) -> None:
    import statistics
    from pathlib import Path
    from src.recognition.eval import evaluate
    from src.recognition.naming import recognition_tag
    from src.utils.checkpoint import latest_checkpoint

    # Baseline SANS fine-tuning : le backbone pre-entraine (arcface_ms1mv3) tel quel,
    # comme reference pour juger si un quelconque fine-tuning (real/synthetic/mixed)
    # apporte vraiment un gain -- pas de seed/checkpoint, deterministe, toujours evalue.
    # Clé de terrain dynamique : "visible_d1" ou "ir_d1" selon la config.
    terrain_key = f"{cfg['modality']}_{cfg['distance']}"

    baseline_weights = cfg["paths"].get("arcface_weights")
    if baseline_weights and Path(baseline_weights).exists():
        rank1_baseline = evaluate(cfg, weights_path=baseline_weights)[terrain_key]
        log.info("RANK-1 baseline (sans fine-tuning) : %.4f", rank1_baseline)
    else:
        log.info("Baseline sans fine-tuning ignoree : paths.arcface_weights introuvable (%s)", baseline_weights)

    for condition in cfg["recognition"]["conditions"]:
        rank1s = []
        for seed in cfg["seeds"]:
            tag = recognition_tag(cfg, condition, seed)
            ckpt = latest_checkpoint(cfg["paths"]["checkpoints"], tag)
            if ckpt is None:
                log.info("Pas de checkpoint pour %s (seed=%d) : 'train_recognition' doit tourner avant.",
                          condition, seed)
                continue
            rank1 = evaluate(cfg, weights_path=str(ckpt))[terrain_key]
            rank1s.append(rank1)
            _append_result(cfg, condition, seed, rank1, ckpt)
        if rank1s:
            mean = statistics.mean(rank1s)
            std = statistics.pstdev(rank1s) if len(rank1s) > 1 else 0.0
            log.info("RANK-1 %s : %.4f ± %.4f (n=%d seeds)", condition, mean, std, len(rank1s))


DISPATCH = {
    "smoke": stage_smoke, "partition": stage_partition, "check_faces": stage_check_faces,
    "train_generator": stage_train_generator, "generate": stage_generate,
    "ir_postprocess": stage_ir_postprocess, "filter_synthetic": stage_filter_synthetic,
    "fidelity": stage_fidelity, "train_recognition": stage_train_recognition,
    "evaluate": stage_evaluate,
}


def _guard_drive_mounted(cfg: dict) -> None:
    """Sans Drive monte, ensure_dirs creerait des dossiers LOCAUX sous /content/drive
    (puis plus aucun montage possible, checkpoints 'introuvables') -- incident vecu."""
    import os
    root = str(cfg["paths"]["drive_root"])
    if root.startswith("/content/drive") and os.path.isdir("/content") \
            and not os.path.ismount("/content/drive") \
            and not os.environ.get("FORENSIC_SYNTH_SKIP_MOUNT_CHECK"):
        raise SystemExit("Google Drive n'est pas monte sur /content/drive : montez-le (bon compte) "
                         "avant de lancer un stage. (Contournement : FORENSIC_SYNTH_SKIP_MOUNT_CHECK=1)")


def _record_run(cfg: dict, args) -> None:
    """Trace persistante (Drive) des choix reels de chaque lancement : config effective
    complete + ligne dans runs_log.jsonl."""
    import json
    import datetime
    from pathlib import Path
    ckpt = Path(cfg["paths"]["checkpoints"])
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    (ckpt / f"config_used_{args.stage}_{cfg['modality']}_{cfg['distance']}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    with open(Path(cfg["paths"]["outputs"]) / "runs_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": stamp, "stage": args.stage, "config": args.config,
                            "set": args.set}, ensure_ascii=False) + "\n")


def main() -> None:
    from src.utils.logging import attach_file_handler

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--set", action="append", default=[], metavar="CLE=VALEUR",
                    help="surcharge de config (ex. recognition.synthetic_ratio=0.2) ; repetable")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    set_seed(cfg.get("seed", 42))
    _guard_drive_mounted(cfg)
    ensure_dirs(cfg)
    # Copie persistante sur Drive (cf. CLAUDE.md), en plus de la console -- la sortie
    # de cellule Colab seule se perd si la session coupe sans sauvegarde du notebook.
    attach_file_handler(log, cfg["paths"]["checkpoints"],
                         f"log_{args.stage}_{cfg['modality']}_{cfg['distance']}.txt")
    log.info("STAGE=%s CONFIG=%s (%s/%s) surcharges=%s", args.stage, args.config,
              cfg["modality"], cfg["distance"], args.set or "aucune")
    _record_run(cfg, args)
    DISPATCH[args.stage](cfg)


if __name__ == "__main__":
    main()
