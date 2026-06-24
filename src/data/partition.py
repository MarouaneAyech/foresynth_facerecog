"""Partition anti-fuite A/B/C des 130 identités SCface.

Garanties : A, B, C disjoints. La partition est FIGÉE dans data/blocks.json (fournie,
pas régénérée) : A entraîne le générateur, B sert à le valider et produire le dataset
synthétique, C est vierge et réservé à l'évaluation finale du recognizer.
Cette répartition diffère intentionnellement de celle de l'article B1 (cf. CLAUDE.md) ;
build_partition()/save_partition() restent disponibles pour générer une partition de
secours (tests, environnements sans data/blocks.json), mais run() charge et valide
toujours la partition figée en priorité.
"""
from __future__ import annotations
import json
from pathlib import Path
from src.utils.seed import set_seed


def build_partition(all_ids: list[str], b1_test_ids: list[str], block_a_size: int,
                    block_b_size: int, seed: int = 42) -> dict[str, list[str]]:
    set_seed(seed)
    import random
    c = list(dict.fromkeys(b1_test_ids))                       # Bloc C figé
    remaining = [i for i in all_ids if i not in set(c)]
    random.shuffle(remaining)
    a = remaining[:block_a_size]
    b = remaining[block_a_size:block_a_size + block_b_size]
    parts = {"A": a, "B": b, "C": c}
    # garde-fou anti-fuite
    sa, sb, sc = map(set, (a, b, c))
    assert not (sa & sb) and not (sa & sc) and not (sb & sc), "FUITE inter-blocs !"
    return parts


def save_partition(parts: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(parts, indent=2))


def load_partition(path: str | Path) -> dict:
    """Charge un fichier de partition. Accepte le format figé du projet
    ({"blocs": {"A": [...], "B": [...], "C": [...]}, ...métadonnées}) ainsi que
    le format minimal produit par save_partition() ({"A": [...], ...})."""
    raw = json.loads(Path(path).read_text())
    return raw.get("blocs", raw)


def scan_identities(scface_root: str | Path) -> set[str]:
    """Identités présentes sur disque, déduites des mugshots (3 premiers caractères du nom)."""
    mugshot_dir = Path(scface_root) / "mugshot_frontal_cropped_all"
    if not mugshot_dir.exists():
        raise FileNotFoundError(f"Dossier mugshots introuvable : {mugshot_dir}")
    return {p.name[:3] for p in mugshot_dir.glob("*") if p.is_file()}


def validate_partition(parts: dict[str, list[str]], all_ids: set[str]) -> None:
    """Anti-fuite + cohérence avec les identités réellement présentes dans scface_root."""
    a, b, c = (parts.get(k, []) for k in ("A", "B", "C"))
    for name, block in (("A", a), ("B", b), ("C", c)):
        dupes = [i for i in set(block) if block.count(i) > 1]
        assert not dupes, f"Doublons dans le bloc {name}: {dupes}"
    sa, sb, sc = set(a), set(b), set(c)
    assert not (sa & sb), f"FUITE A∩B: {sa & sb}"
    assert not (sa & sc), f"FUITE A∩C: {sa & sc}"
    assert not (sb & sc), f"FUITE B∩C: {sb & sc}"

    partitioned = sa | sb | sc
    missing_on_disk = sorted(partitioned - all_ids)
    assert not missing_on_disk, (
        f"Identités du partition.json absentes de scface_root: {missing_on_disk}")
    unpartitioned = sorted(all_ids - partitioned)
    assert not unpartitioned, (
        f"Identités présentes dans scface_root mais absentes du partition.json: {unpartitioned}")


def run(cfg: dict) -> dict:
    """Étage 'partition'. Charge la partition figée (data/blocks.json) et la valide
    contre les identités réellement présentes sous paths.scface_root."""
    blocks_path = Path(cfg["paths"]["blocks_file"])
    if not blocks_path.exists():
        raise FileNotFoundError(
            f"Partition figée introuvable : {blocks_path}. Ce projet utilise une partition "
            "externe pré-calculée (cf. data/blocks.json versionné dans le repo), elle doit "
            "être fournie, pas régénérée à la volée.")
    parts = load_partition(blocks_path)
    all_ids = scan_identities(cfg["paths"]["scface_root"])
    validate_partition(parts, all_ids)
    return parts
