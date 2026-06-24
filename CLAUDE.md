# CLAUDE.md — Contexte projet (lu automatiquement par Claude Code)

## But du projet
Générer des images de surveillance **synthétiques** à partir de mugshots HR, pour
entraîner/adapter un modèle de reconnaissance faciale forensique (IResNet-50 /
ArcFace) sans dépendre de vraies captures de surveillance (rares + sensibles).
Base de données : **SCface** (mugshots HR ; surveillance visible et IR ; 3 distances).

## Principe d'architecture NON NÉGOCIABLE
Tout est piloté par le couple `(modality, distance)`.
- `modality ∈ {visible, ir}` ; `distance ∈ {d1, d2, d3}` (d1=4.20m, d2=2.60m, d3=1.00m).
- AUCUN code ne doit coder « visible » en dur. Passer à l'IR = changer la config, pas le code.
- La logique vit dans `src/` (modules importables, testables, reprenables).
  Les notebooks (`notebooks/`) ne font QUE lancer sur Colab.

## Périmètre de l'itération ACTUELLE
N'implémente QUE le chemin **visible / d1**. Laisse des stubs `NotImplementedError`
+ commentaires `# TODO(claude)` pour IR et pour d2/d3. Ne code pas l'IR maintenant,
mais ne casse jamais l'interface qui la rendra possible.

## Pipeline (ordre de construction)
1. `data/partition.py` — charge et VALIDE la partition A/B/C **figée** (`data/blocks.json`,
   versionné dans le repo, jamais régénérée à la volée). Rôles : **A** (50 id) entraîne
   le générateur (LoRA) ; **B** (50 id) valide le générateur et sert à produire le dataset
   synthétique ; **C** (30 id, vierge) évalue le recognizer final. Disjonction stricte
   A∩B∩C=∅. Cette répartition est **intentionnellement différente** de celle de l'article
   B1 (le rôle de C est analogue — éval finale vierge — mais les identités ne sont pas
   garanties identiques à celles du test set B1 original).
2. `data/pairs.py` — charge les paires (mugshot, cible) pour une `(modality, distance)`.
3. `generator/` — Arc2Face GELÉ (identité portée par l'embedding ArcFace) + adaptateur **LoRA** indexé par `(modality, distance)`. Perte d'identité ArcFace pour verrouiller l'identité. Amorce de dégradation haute-ordre (Real-ESRGAN) optionnelle.
4. `fidelity/` — garde-fou **AVANT toute reconnaissance** : FID + cosinus ArcFace (synth vs vrai du Bloc B) → go/no-go.
5. `recognition/` — RECETTE B1 INCHANGÉE : scope=layer3+4, AdamW lr=1e-4, ancrage 50/50 mugshot/surveillance, 3 seeds. Conditions : real / synthetic / mixed.
6. `recognition/eval.py` — rank-1 par terrain sur Bloc C.

## Garde-fous scientifiques (à respecter dans le code)
- **Identité d'abord** : la LoRA apprend la DÉGRADATION, pas l'identité (gelée par Arc2Face). Toujours ajouter la perte cosinus ArcFace.
- **La LoRA s'entraîne sur des paires RÉELLES (Bloc A)**, puis GÉNÈRE depuis le Bloc B. Ne jamais entraîner la LoRA sur du synthétique.
- **Diversité intra-classe** : générer K échantillons variés par identité (K configurable), pas 1.
- **En visible, n'évaluer que d1.** d2/d3 sont saturés (100% baseline) → ne mesurent que du bruit.
- **Crochet IR** : pour l'IR, prévoir l'option « fine-tuning complet du générateur » (le B1 prédit que LoRA seule sous-apprend le décalage spectral de haut rang).

## Contraintes d'exécution (Colab Free)
- TOUT sur Google Drive (données, checkpoints, dataset généré, logs).
- Entraînement **reprenable** : sauver/restaurer un checkpoint tous les N pas (`utils/checkpoint.py`).
- Mode **smoke test** (CPU, données factices) pour valider le câblage avant de brûler du GPU.
- Scripts autonomes : un job = un `stage` lançable seul (`experiments/run.py --stage ...`).

## Commandes
- Smoke test : `pytest -q` puis `python -m experiments.run --config configs/visible_d1.yaml --stage smoke`
- Un étage : `python -m experiments.run --config configs/visible_d1.yaml --stage <partition|train_generator|generate|fidelity|train_recognition|evaluate>`

## Style de code
Python 3.10+, type hints, docstrings courtes, pas de magie. Préfère des fonctions
pures + une config explicite. Toute valeur réglable va dans `configs/`, jamais en dur.
