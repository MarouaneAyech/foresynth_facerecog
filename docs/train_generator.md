# Stage `train_generator` — documentation détaillée

> Implémentation : [src/generator/base_arc2face.py](../src/generator/base_arc2face.py) (classe `Arc2FaceGenerator`).
> Déclenché par : `python -m experiments.run --config configs/visible_d1.yaml --stage train_generator`
> (appelle `stage_train_generator()` dans [experiments/run.py](../experiments/run.py), qui appelle `Arc2FaceGenerator.fit(pairs_bloc_A)`).

## 1. Objectif

Entraîner un adaptateur **LoRA** sur le UNet d'**Arc2Face** (un modèle de diffusion figé,
conditionné par l'identité) pour qu'il apprenne la **dégradation visuelle** propre à la
surveillance `(modality, distance)` courante — sans jamais désapprendre l'identité du
visage. L'identité est verrouillée par construction : elle vient d'un embedding ArcFace
figé (conditionnement), renforcé par une perte de cosinus ArcFace (pénalité).

Entraîné sur les **350 paires réelles du Bloc A** (50 identités × 7 caméras, distance d1).

## 2. Modèles et bibliothèques impliqués

| Rôle | Modèle / Checkpoint | Bibliothèque (version figée) | Gelé ? | Emplacement attendu |
|---|---|---|---|---|
| Backbone de diffusion | UNet Arc2Face (`diffusion_pytorch_model.safetensors`) | `diffusers==0.29.2` (`UNet2DConditionModel`) | Oui, sauf LoRA | `paths.arc2face_models_dir` (Drive) ou Hub `FoivosPar/Arc2Face` |
| Encodeur de conditionnement | `CLIPTextModelWrapper` (`pytorch_model.bin`) | `transformers==4.36.0` | Oui (`requires_grad_(False)`) | idem, sous-dossier `encoder/` |
| VAE + scheduler | Stable Diffusion v1.5 | `diffusers==0.29.2` (`StableDiffusionPipeline`, `DPMSolverMultistepScheduler`) | VAE gelé | Hub `stable-diffusion-v1-5/stable-diffusion-v1-5` (téléchargé à chaque session, ~330 Mo) |
| Adaptateur entraîné | LoRA (rang 16, cibles `to_q/to_k/to_v/to_out.0`) | `peft==0.7.1` (`LoraConfig`, `unet.add_adapter`) | Non — **seul élément entraîné** |
| Extraction embedding ID (conditionnement) | pack `antelopev2` (5 fichiers `.onnx`) | `insightface` + `onnxruntime` | Gelé (inférence ONNX) | `paths.insightface_models_dir` (Drive), lié par symlink vers `~/.insightface/models/antelopev2` |
| Perte d'identité | IResNet-50 / ArcFace MS1MV3 (`arcface_ms1mv3_r50.pth`) | implémentation maison (`src/utils/arcface_backbone.py`, `torch` pur) | Gelé en poids, différentiable en entrée | `paths.arcface_weights` (Drive) |
| Projection ID → prompt | fonction `project_face_embs` du paquet `arc2face` | dépôt `github.com/foivospar/Arc2Face` (cloné, **pas pip-installable**, ajouté au `PYTHONPATH`) | — | `/content/Arc2Face_lib` (Colab, cf. notebook) |

**Pourquoi ces versions précises de `diffusers`/`transformers`/`peft` ?** `CLIPTextModelWrapper`
dépend de la structure interne de `CLIPTextModel`, qui change entre versions de
`transformers` — un mauvais alignement de version casse silencieusement le chargement
du checkpoint `encoder/` (cf. §6, incident du 2026-06-25). Ces versions sont **exactement**
celles du `requirements.txt` officiel d'Arc2Face.

## 3. Déroulé détaillé de `_ensure_loaded()` (chargement, une seule fois)

1. Détecte le device (`cuda` si disponible, sinon `cpu`) et le dtype (`float16` sur GPU,
   `float32` sur CPU).
2. Charge `encoder` (`CLIPTextModelWrapper.from_pretrained`) et `unet`
   (`UNet2DConditionModel.from_pretrained`) — depuis Drive si présent
   (`paths.arc2face_models_dir`), sinon depuis le Hub HuggingFace (repli portable, plus lent).
3. Construit le `StableDiffusionPipeline` complet en y injectant cet `encoder`/`unet` à la
   place de ceux de Stable Diffusion v1.5 ; remplace le scheduler par
   `DPMSolverMultistepScheduler` ; désactive le `safety_checker` (explicite, voir avertissement
   affiché au chargement — comportement voulu, pas une erreur) ; gèle VAE et encodeur de texte.
4. **`_setup_adapter(unet)`** : gèle tout le UNet, puis injecte la LoRA (`peft.LoraConfig`)
   sur les couches d'attention listées dans `generator.lora.target_modules`. Seuls ces
   paramètres (~3,19M sur la trace observée) restent entraînables.
5. **`_link_insightface_pack`** : crée un symlink `~/.insightface/models/antelopev2` →
   le dossier Drive, pour qu'`insightface` trouve le pack sans le dupliquer ni le
   retélécharger (non auto-téléchargeable par `insightface` lui-même).
6. Charge `FaceAnalysis(name="antelopev2")` (détection + landmarks + embedding ArcFace)
   et l'embedder ArcFace maison (`load_arcface_embedder`, pour la perte d'identité).

## 4. Déroulé détaillé de `fit()` (boucle d'entraînement)

Répété jusqu'à `generator.train.max_steps` (4000 par défaut), **reprenable** via
checkpoint (`paths.checkpoints`, toutes les `ckpt_every` pas) :

### a) Constitution du batch
`generator.train.batch_size` (4) paires tirées en cycle parmi les 350 du Bloc A.

### b) Embeddings d'identité (conditionnement)
- `_id_embedding(mugshot_path)` : détection de visage + embedding 512-d via `antelopev2`
  (le plus grand visage détecté), normalisé L2. **Mis en cache par identité**
  (`ref_embeddings`) — calculé une seule fois, pas à chaque pas.
- `_project_for_conditioning(id_emb)` : projette l'embedding ArcFace dans l'espace des
  tokens CLIP via `project_face_embs` (paquet `arc2face`), pour obtenir `prompt_embeds`
  (le "prompt" conditionnant le UNet). **Contournement appliqué** : la fonction d'origine
  contient un bug qui plante dès que plus d'une identité est traitée dans le même appel
  (cf. §6) — on l'appelle donc identité par identité (N=1) puis on concatène.

### c) Cible de diffusion
- Image de surveillance réelle chargée et redimensionnée à 512×512 (`_load_image_tensor`).
- Si `generator.degrade_prior=true` : dégradation classique supplémentaire
  (`high_order_degrade` — flou/resize/bruit/JPEG ×2, sévérité selon `distance`), comme
  augmentation (pas de cache : nouvelle réalisation aléatoire à chaque tirage).
- Encodée en latents via le VAE (`vae.encode(...).latent_dist.sample() * vae_scale`),
  sous `torch.no_grad()` (le VAE est gelé, pas besoin de gradient ici).

### d) Perte de diffusion (apprentissage de la dégradation)
- Bruit gaussien ajouté aux latents à un pas de temps aléatoire
  (`scheduler.add_noise`).
- Le UNet (LoRA active) prédit ce bruit à partir des latents bruités + `prompt_embeds`.
- `diffusion_loss = MSE(bruit_prédit, bruit_réel)`.

### e) Perte d'identité (verrou anti-dérive)
- Estimation en un pas du latent débruité `x0_pred` (formule DDPM standard à partir de
  `noise_pred` et `scheduler.alphas_cumprod`).
- Décodage en pixels via le VAE (`vae.decode`).
- `identity_cosine_loss(image_décodée, id_emb, arcface_embedder)` = `1 - cos(...)`
  (cf. [src/generator/identity_loss.py](../src/generator/identity_loss.py)).

### f) Mise à jour
```
loss = diffusion_loss + generator.identity_loss_weight * identity_loss
```
`AdamW` (`generator.train.lr`), backward, step — **seuls les poids LoRA reçoivent un
gradient** (le reste du graphe est gelé).

### g) Checkpoint
Toutes les `ckpt_every` pas : sauvegarde `unet.state_dict()` (LoRA incluse) +
`optimizer.state_dict()` sur Drive (`paths.checkpoints`). `resume_step` détecte et reprend
automatiquement au redémarrage.

## 5. Paramètres de configuration impliqués (`configs/base.yaml`)

| Clé | Rôle |
|---|---|
| `generator.backbone`, `generator.adapter` | Sélection de la fabrique (`build_generator`) |
| `generator.pretrained.*` | Chemins/repos des modèles (cf. tableau §2) |
| `generator.lora.{rank,alpha,target_modules}` | Hyperparamètres LoRA |
| `generator.identity_loss_weight` | Poids de la perte d'identité dans la perte totale |
| `generator.degrade_prior` | Active la dégradation classique en augmentation |
| `generator.train.{lr,max_steps,batch_size,ckpt_every,log_every}` | Boucle d'entraînement |
| `paths.arcface_weights`, `paths.arc2face_models_dir`, `paths.insightface_models_dir` | Assets pré-entraînés (Drive) |

## 6. Incidents déjà rencontrés et corrigés (historique pour investigation)

| Date | Symptôme | Cause | Correction |
|---|---|---|---|
| 2026-06-25 | `LOAD REPORT` : toutes les clés de `encoder/` en `MISSING`/`UNEXPECTED` | `transformers` non figé → version trop récente (4.57), structure interne de `CLIPTextModel` changée | Figé `diffusers==0.29.2`, `transformers==4.36.0`, `peft==0.7.1` dans `requirements.txt` |
| 2026-06-25 | `IndexError` dans `project_face_embs` : masque `[1,77]` vs tenseur `[4,77,768]` | Bug du paquet `arc2face` (`utils.py`) : utilise `input_ids` (taille 1) au lieu de sa version répétée comme masque — invisible en usage normal (génération, N=1), exposé par notre entraînement par batch (N>1) | Contournement dans `_project_for_conditioning` : appel par identité (N=1) puis concaténation |

## 7. Limites connues (non bloquantes, à garder en tête)

- **Pas d'alignement 5-points** sur les images cibles (`_load_image_tensor` fait un simple
  *resize*, pas un warp par similarité ArcFace). Risque modéré ici car SCface fournit déjà
  des images pré-cadrées par identité/caméra.
- **`x0_pred` est une estimation en un seul pas** (pas un débruitage complet) — approximation
  standard dans la littérature pour calculer une perte d'identité sans coût d'inférence complet.
- **Pas de cache des images cibles chargées** : les 350 images du Bloc A sont relues/redimensionnées
  depuis le disque (Drive) à chaque tirage, plutôt que mises en cache en mémoire — optimisation
  identifiée mais pas encore implémentée.
