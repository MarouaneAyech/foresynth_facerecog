# forensic-synth

Génération d'images de surveillance synthétiques (mugshot → surveillance) pour
l'adaptation d'un modèle de reconnaissance faciale forensique (IResNet-50 / ArcFace)
sur SCface. Conçu **config-driven sur `(modality, distance)`** : visible/d1 maintenant,
IR/d1-d2-d3 plus tard sans refonte.

## Démarrage (dans VSCode + Claude Code)
1. Place ce dossier comme racine de ton dépôt git.
2. Ouvre Claude Code à la racine. Il lira `CLAUDE.md` automatiquement.
3. Premier prompt suggéré (voir le bas de ce fichier).
4. Vérifie le câblage : `pip install -r requirements.txt && pytest -q`.

## Structure
```
configs/         # base.yaml + un fichier par scénario (visible_d1, ir_d1, ...)
src/data/        # partition A/B/C, chargeur de paires
src/generator/   # Arc2Face gelé + LoRA + perte d'identité + amorce dégradation
src/fidelity/    # FID + cosinus ArcFace -> go/no-go (garde-fou)
src/recognition/ # fine-tuning recette B1 + évaluation
src/utils/       # checkpoint reprenable, seed, logging, chemins Drive
experiments/run.py  # orchestrateur : --config --stage
notebooks/       # lanceurs Colab (montage Drive, clone, reprise)
tests/           # smoke test (CPU, sans données)
```

## Flux de données (important)
LoRA entraînée sur paires RÉELLES du Bloc A → GÉNÈRE depuis le Bloc B →
garde-fou de fidélité → entraînement reconnaissance (real/synth/mixed) →
évaluation sur Bloc C (= test set du B1). En visible : métrique = **d1 uniquement**.

## Workflow Colab Free + comptes multiples
- Étape lourde (entraîner LoRA + générer le dataset) : 1 bonne session GPU, résultat caché sur Drive.
- Étapes légères (fine-tunings reconnaissance, plusieurs seeds/conditions) : jobs
  INDÉPENDANTS, répartissables sur plusieurs sessions/comptes. Duplique `notebooks/colab_runner.ipynb`,
  change `STAGE`/`CONFIG`, lance. (Aucune fusion automatique de comptes — voir CLAUDE.md.)

## Premier prompt à donner à Claude Code
> Implémente UNIQUEMENT le chemin visible/d1 décrit dans CLAUDE.md, en respectant
> les interfaces des stubs de src/. Commence par data/partition.py et le smoke test,
> puis remonte le pipeline étage par étage. Laisse IR et d2/d3 en TODO.
