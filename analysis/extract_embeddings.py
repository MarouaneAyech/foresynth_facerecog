"""Extraction locale (CPU) des embeddings d'évaluation, un fichier .npz par checkpoint.

Pour chaque checkpoint de reconnaissance LoRA du terrain demandé, recharge le modèle,
embarque la galerie (mugshots du bloc C) et les probes (surveillance du bloc C) et
sauvegarde : embeddings, identités, caméra de chaque probe et matrice de scores. Tout le
reste des analyses (bootstrap, CMC, similarité genuine, par caméra) se fait ensuite sur
ces fichiers, sans modèle.

Aucune détection de visage : les images alignées sont lues dans `aligned_cache/`. Ce cache
a été produit sur Colab, où la clé de chaque entrée est le chemin Drive de l'image ; la
clé est donc reconstruite ici à partir du chemin relatif à la racine du projet.

Baseline (--baseline) : reconstruite à partir d'un checkpoint LoRA en annulant les matrices
B de chaque adaptateur. Les poids d'origine et les statistiques BatchNorm y sont intacts
(BatchNorm gelées en mode eval pendant l'entraînement), ce qui redonne exactement le
modèle pré-entraîné.

Usage (depuis la racine du projet, avec .venv-analysis) :
    .venv-analysis/Scripts/python analysis/extract_embeddings.py --terrain visible_d1 --baseline
    .venv-analysis/Scripts/python analysis/extract_embeddings.py --terrain ir_d1 --baseline \
        --ckpt-dir "checkpoints/ir d1"
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("FORENSIC_SYNTH_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.pairs import list_pairs  # noqa: E402
from src.recognition.lora import LoRALinear, LoRAConv2d, inject_lora  # noqa: E402
from src.utils.arcface_backbone import iresnet50, preprocess_for_arcface  # noqa: E402

DRIVE_ROOT = "/content/drive/MyDrive/forensic_synth"  # racine Drive où le cache a été produit
CKPT_RE = re.compile(r"^(?P<tag>recognition_.+)_step(?P<step>\d+)$")
TAG_RE = re.compile(r"_(?P<cond>real|synthetic|mixed)(?:_ratio(?P<ratio>\d{3}))?_seed(?P<seed>\d+)$")


def cache_file(image_path: str | Path, cache_dir: Path, size: int = 112) -> Path:
    rel = Path(image_path).resolve().relative_to(ROOT).as_posix()
    key = hashlib.sha1(f"{DRIVE_ROOT}/{rel}|{size}".encode()).hexdigest()
    return cache_dir / f"{key}.pt"


def load_tensors(paths: list[str], cache_dirs: list[Path]) -> torch.Tensor:
    """Cherche chaque image alignée dans les dossiers de cache, dans l'ordre."""
    tensors = []
    for p in paths:
        found = next((f for d in cache_dirs if (f := cache_file(p, d)).exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"Entrée absente du cache d'alignement : {p}\n  (clé {cache_file(p, cache_dirs[0]).name})")
        tensors.append(torch.load(found))
    return torch.stack(tensors)


@torch.no_grad()
def embed(net: torch.nn.Module, images: torch.Tensor, batch: int = 64) -> torch.Tensor:
    out = [F.normalize(net(preprocess_for_arcface(images[i:i + batch])), dim=-1)
           for i in range(0, len(images), batch)]
    return torch.cat(out)


def build_net(ckpt_path: Path, zero_lora: bool = False) -> torch.nn.Module:
    state = torch.load(ckpt_path, map_location="cpu")
    net = iresnet50()
    if state.get("mechanism") == "lora":
        inject_lora(net, r=state["lora_rank"], alpha=state["lora_alpha"],
                    target_layers=tuple(state["lora_target_layers"].split(",")),
                    include_fc=state["lora_include_fc"])
    net.load_state_dict(state["net"] if "net" in state else state)
    if zero_lora:
        for m in net.modules():
            if isinstance(m, (LoRAConv2d, LoRALinear)):
                torch.nn.init.zeros_(m.lora_B.weight)
    return net.eval()


def find_checkpoints(ckpt_dir: Path, only: list[str], include_non_lora: bool) -> dict[str, Path]:
    """Dernier checkpoint de chaque tag `recognition_*`."""
    latest: dict[str, tuple[int, Path]] = {}
    for f in sorted(ckpt_dir.glob("recognition_*_step*.ckpt")):
        m = CKPT_RE.match(f.stem)
        if not m or (not include_non_lora and not m["tag"].startswith("recognition_lora")):
            continue
        if only and not any(o in m["tag"] for o in only):
            continue
        step = int(m["step"])
        if m["tag"] not in latest or step > latest[m["tag"]][0]:
            latest[m["tag"]] = (step, f)
    return {tag: p for tag, (_, p) in sorted(latest.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrain", default="visible_d1", choices=["visible_d1", "ir_d1"])
    ap.add_argument("--only", nargs="*", default=[], help="sous-chaînes de tags à traiter")
    ap.add_argument("--baseline", action="store_true", help="évalue aussi le modèle non adapté")
    ap.add_argument("--include-non-lora", action="store_true")
    ap.add_argument("--ckpt-dir", default=None,
                    help="dossier des checkpoints (défaut : paths.checkpoints de la config)")
    ap.add_argument("--cache-dir", action="append", default=None,
                    help="dossier de cache d'alignement, répétable (défaut : paths.aligned_cache puis aligned_cache_ir)")
    ap.add_argument("--out", default=str(ROOT / "analysis" / "outputs" / "embeddings"))
    ap.add_argument("--force", action="store_true", help="recalcule les fichiers existants")
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    cfg = load_config(ROOT / "configs" / f"{args.terrain}.yaml")
    ckpt_dir = Path(args.ckpt_dir or cfg["paths"]["checkpoints"])
    cache_dirs = [Path(d) for d in (args.cache_dir or [cfg["paths"]["aligned_cache"], ROOT / "aligned_cache_ir"])]
    out_dir = Path(args.out) / args.terrain
    out_dir.mkdir(parents=True, exist_ok=True)

    # Galerie (1 mugshot par identité) et probes du bloc C, dans l'ordre de evaluate().
    gallery_paths, gallery_ids, probe_paths, probe_ids = [], [], [], []
    seen: set[str] = set()
    for p in list_pairs(cfg, block="C"):
        if p.identity not in seen:
            gallery_paths.append(p.mugshot_path)
            gallery_ids.append(p.identity)
            seen.add(p.identity)
        probe_paths.append(p.target_path)
        probe_ids.append(p.identity)
    probe_cam = [int(re.search(r"_cam(\d)_", Path(p).name).group(1)) for p in probe_paths]
    gallery_x = load_tensors(gallery_paths, cache_dirs)
    probe_x = load_tensors(probe_paths, cache_dirs)
    print(f"{args.terrain} : {len(gallery_ids)} identités en galerie, {len(probe_ids)} probes")

    checkpoints = find_checkpoints(ckpt_dir, args.only, args.include_non_lora)
    jobs: list[tuple[str, Path, bool]] = [(t, p, False) for t, p in checkpoints.items()]
    if args.baseline and checkpoints:
        jobs.insert(0, ("baseline", next(iter(checkpoints.values())), True))
    if not jobs:
        raise SystemExit(f"Aucun checkpoint de reconnaissance trouvé dans {ckpt_dir}")

    for tag, path, zero in jobs:
        out = out_dir / f"{tag}.npz"
        if out.exists() and not args.force:
            print(f"[déjà fait] {tag}")
            continue
        net = build_net(path, zero_lora=zero)
        g, q = embed(net, gallery_x), embed(net, probe_x)
        scores = (q @ g.T).numpy()
        pred = np.array(gallery_ids)[scores.argmax(1)]
        rank1 = float((pred == np.array(probe_ids)).mean())
        meta = TAG_RE.search(tag)
        np.savez_compressed(
            out, gallery_emb=g.numpy(), gallery_ids=np.array(gallery_ids), probe_emb=q.numpy(),
            probe_ids=np.array(probe_ids), probe_cam=np.array(probe_cam),
            probe_files=np.array([Path(p).name for p in probe_paths]), scores=scores,
            tag=tag, condition="baseline" if zero else (meta["cond"] if meta else ""),
            ratio=(int(meta["ratio"]) / 100 if meta and meta["ratio"] else np.nan),
            seed=(int(meta["seed"]) if meta else -1), rank1=rank1)
        print(f"[ok] {tag:62s} rank-1 = {rank1:.4f}")


if __name__ == "__main__":
    main()
