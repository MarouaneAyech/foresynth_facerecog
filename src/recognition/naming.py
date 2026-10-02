"""Tags de checkpoints de reconnaissance (sans dépendance torch : utilisable par run.py).

full_finetune garde l'ancien tag (archive de la recette B1, comparable à l'existant).
lora inclut le rang, et le ratio synthétique pour 'mixed', afin que les paliers de
dosage ne s'écrasent pas entre eux.
"""
from __future__ import annotations


def recognition_tag(cfg: dict, condition: str, seed: int) -> str:
    rec = cfg["recognition"]
    if rec.get("mechanism", "full_finetune") != "lora":
        return f"recognition_{condition}_seed{seed}"
    parts = ["recognition", f"lora_r{rec['lora']['rank']}", condition]
    if condition == "mixed":
        parts.append(f"ratio{round(100 * rec.get('synthetic_ratio', 1.0)):03d}")
    parts.append(f"seed{seed}")
    return "_".join(parts)
