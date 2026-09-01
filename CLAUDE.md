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

## Périmètre actuel (mis à jour 2026-08-31)
**visible/d1** ET **ir/d1** sont désormais implémentés de bout en bout (générateur, fidelity,
reconnaissance, évaluation) et couverts par le brouillon d'article en cours
(`paper/sivp_forensic_synthetic.tex` — dossier `paper/` **local uniquement, volontairement non
versionné**, à ne jamais ajouter à git) : visible/d1 comme résultat principal, ir/d1 comme
**cas limite** (la transformation RVB→IR est de haut rang, hors de portée d'une LoRA seule ;
compensée par un post-traitement déterministe, `generator/ir_postprocess.py` + stage
`ir_postprocess`). L'IR n'est donc plus « à faire plus tard » : ne pas partir de ce principe
dans les futures sessions.
**d2/d3 restent hors périmètre** (stubs `NotImplementedError` dans `src/data/pairs.py`) :
saturés côté visible (baseline ~100%), non prioritaires côté IR. Ne les implémente pas sans
consigne explicite, mais ne casse jamais l'interface `(modality, distance)` qui les rendra
possibles.
**Point ouvert** : le garde-fou fidelity (FID + cosinus ArcFace, §ci-dessous) n'a pas encore
été exécuté/reporté pour la rédaction — le tableau correspondant est encore à `\TBD{}` dans le
brouillon alors que les tableaux de reconnaissance sont déjà chiffrés. À lancer (`--stage
fidelity`) avant de considérer ces résultats de reconnaissance comme validés pour publication.

## Pipeline (ordre de construction)
1. `data/partition.py` — charge et VALIDE la partition A/B/C **figée** (`data/blocks.json`,
   versionné dans le repo, jamais régénérée à la volée). Rôles : **A** (50 id) entraîne
   le générateur (LoRA) ; **B** (50 id) valide le générateur et sert à produire le dataset
   synthétique ; **C** (30 id, vierge) évalue le recognizer final. Disjonction stricte
   A∩B∩C=∅. Cette répartition est **intentionnellement différente** de celle de l'article
   B1 (le rôle de C est analogue — éval finale vierge — mais les identités ne sont pas
   garanties identiques à celles du test set B1 original).
2. `data/pairs.py` — charge les paires (mugshot, cible) pour une `(modality, distance)`.
   Implémenté : visible/d1, ir/d1. Stubs : d2/d3 (toute modalité).
3. `generator/` — Arc2Face GELÉ (identité portée par l'embedding ArcFace) + adaptateur **LoRA**
   (rsLoRA rang 32) indexé par `(modality, distance)`. Perte d'identité ArcFace pour verrouiller
   l'identité (actuellement pondérée à 0.0, cf. `configs/base.yaml`). Stages `train_generator`
   (fit sur Bloc A) puis `generate` (sample depuis Bloc B).
4. `ir_postprocess/` — **IR uniquement**, après `generate` et avant `fidelity`/`filter_synthetic` :
   normalisation déterministe (grayscale rec.601 + flou calibré + bruit capteur) qui complète la
   LoRA sans la remplacer (`generator/ir_postprocess.py`).
5. `fidelity/` — garde-fou **AVANT toute reconnaissance** : FID + cosinus ArcFace (synth vs vrai
   du Bloc B) → go/no-go (stage `fidelity`) ; filtre par image complémentaire (stage
   `filter_synthetic`, retire les images sous un seuil de cosinus individuel).
6. `recognition/` — RECETTE B1 INCHANGÉE : scope=layer3+4, AdamW lr=1e-4, ancrage 50/50
   mugshot/surveillance, 3 seeds. Conditions : real / synthetic / mixed (`mixed` avec
   `recognition.synthetic_ratio` variable = le levier de l'ablation dosage réel/synthétique).
7. `recognition/eval.py` — rank-1 par terrain sur Bloc C.

## Garde-fous scientifiques (à respecter dans le code)
- **Identité d'abord** : la LoRA apprend la DÉGRADATION, pas l'identité (gelée par Arc2Face). Toujours ajouter la perte cosinus ArcFace.
- **La LoRA s'entraîne sur des paires RÉELLES (Bloc A)**, puis GÉNÈRE depuis le Bloc B. Ne jamais entraîner la LoRA sur du synthétique.
- **Diversité intra-classe** : générer K échantillons variés par identité (K configurable), pas 1.
- **En visible, n'évaluer que d1.** d2/d3 sont saturés (100% baseline) → ne mesurent que du bruit.
- **IR = cas limite documenté, pas un chemin à ignorer** : le décalage spectral RVB→IR est de
  haut rang, une LoRA seule ne suffit pas (confirmé empiriquement, cf. article §"Limiting case").
  L'option `generator.adapter: full_finetune` existe dans `src/generator/api.py` pour un futur
  fine-tuning complet si nécessaire, mais n'a pas été requise pour les résultats actuels.

## Contraintes d'exécution (Colab Free)
- TOUT sur Google Drive (données, checkpoints, dataset généré, logs).
- Entraînement **reprenable** : sauver/restaurer un checkpoint tous les N pas (`utils/checkpoint.py`).
- Mode **smoke test** (CPU, données factices) pour valider le câblage avant de brûler du GPU.
- Scripts autonomes : un job = un `stage` lançable seul (`experiments/run.py --stage ...`).

## Commandes
- Smoke test : `pytest -q` puis `python -m experiments.run --config configs/visible_d1.yaml --stage smoke`
- Un étage : `python -m experiments.run --config configs/visible_d1.yaml --stage <partition|check_faces|train_generator|generate|ir_postprocess|filter_synthetic|fidelity|train_recognition|evaluate>`
  (remplacer `visible_d1.yaml` par `ir_d1.yaml` pour l'IR ; `ir_postprocess` réservé à l'IR)

## Style de code
Python 3.10+, type hints, docstrings courtes, pas de magie. Préfère des fonctions
pures + une config explicite. Toute valeur réglable va dans `configs/`, jamais en dur.
