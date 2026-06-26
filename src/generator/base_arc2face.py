"""Arc2Face GELÉ (identité = embedding ArcFace) + adaptateur LoRA pour la dégradation.

Division du travail : identité préservée par le conditionnement gelé (embedding
ArcFace extrait par insightface/antelopev2, projeté dans l'espace CLIP par Arc2Face
lui-même) ; la LoRA n'apprend QUE la signature de dégradation (modality, distance).

Tous les imports lourds (torch, diffusers, peft, insightface) sont différés (lazy) :
ce module doit rester importable sans GPU ni poids pour le smoke test (cf.
experiments.run.stage_smoke), seuls fit()/sample() les chargent réellement.

Dépendances externes attendues (cf. requirements.txt) :
- poids Arc2Face (HF Hub, repo cfg['generator']['pretrained']['arc2face_repo'])
- insightface (paquet whl) + pack de détection/embedding 'antelopev2' téléchargé
- le paquet `arc2face` (CLIPTextModelWrapper, project_face_embs), à installer depuis
  https://github.com/foivospar/Arc2Face (pas sur PyPI), cf. requirements.txt.
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

from src.data.pairs import FacePair
from src.utils.logging import get_logger

log = get_logger()


class Arc2FaceGenerator:
    def __init__(self, cfg: dict, adapter: str = "lora"):
        self.cfg = cfg
        self.adapter = adapter
        # Chargement réel différé : voir _ensure_loaded(). Garde l'instanciation
        # utilisable en CPU/smoke test sans torch/diffusers/insightface installés.
        self._pipeline = None
        self._face_app = None          # insightface.app.FaceAnalysis (extraction ID)
        self._identity_embedder = None  # src.utils.arcface_backbone.ArcFaceEmbedder (perte)
        self._trainable_params = None
        self._device = None

    # ------------------------------------------------------------------ chargement
    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        import torch
        from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
        from arc2face import CLIPTextModelWrapper  # paquet du repo Arc2Face, pas PyPI
        from src.generator.face_detect import load_face_app
        from src.utils.arcface_backbone import load_arcface_embedder

        pretrained = self.cfg["generator"]["pretrained"]
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        # Préfère les poids déjà déposés (Drive, pas de retéléchargement entre sessions) ;
        # sinon retombe sur le Hub HuggingFace (portable, plus lent).
        local_models_dir = Path(self.cfg["paths"].get("arc2face_models_dir", ""))
        arc2face_source = str(local_models_dir) if local_models_dir.is_dir() else pretrained["arc2face_repo"]

        encoder = CLIPTextModelWrapper.from_pretrained(
            arc2face_source, subfolder="encoder", torch_dtype=dtype)
        unet = UNet2DConditionModel.from_pretrained(
            arc2face_source, subfolder="arc2face", torch_dtype=dtype)

        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained["base_sd"], text_encoder=encoder, unet=unet,
            torch_dtype=dtype, safety_checker=None)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(self._device)
        # NE PAS muter pipeline.scheduler.alphas_cumprod ici : c'est un objet PARTAGÉ
        # entre fit() (calcul bas niveau, a besoin d'un tenseur GPU/dtype cohérent) et
        # l'appel haut niveau self._pipeline(...) dans sample() (scheduler.set_timesteps
        # convertit alphas_cumprod en numpy -> exige du CPU). Le déplacer ici cassait
        # sample(). fit() fait sa propre copie locale castée (cf. plus bas).
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        # Le VAE original de Stable Diffusion 1.5 est numériquement instable en
        # float16 (NaN/overflow connus dans son bloc d'attention) -> artefacts
        # néon/postérisés observés en génération, indépendants de guidance_scale
        # (confirmé : identiques à 1.0 et 3.0). Reste seul en float32 ; UNet/texte
        # restent float16. fit()/sample() castent explicitement aux frontières.
        pipeline.vae.to(dtype=torch.float32)
        # Économie mémoire (entraînement = UNet forward+backward ET VAE decode+ArcFace
        # avec gradients actifs pour la perte d'identité, simultanément en mémoire
        # jusqu'à loss.backward() -> OOM observé même sur un T4 15 Go à batch_size=4,
        # puis ré-observé à batch_size=2 après passage du VAE en float32 (~2x plus
        # gourmand que float16) -> tiling en plus du slicing, batch_size réduit à 1.
        pipeline.unet.enable_gradient_checkpointing()
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()

        self._trainable_params = self._setup_adapter(pipeline.unet)
        self._load_latest_lora(pipeline.unet)

        self._face_app = load_face_app(self.cfg)

        self._identity_embedder = load_arcface_embedder(self.cfg)
        self._pipeline = pipeline
        log.info("Arc2FaceGenerator chargé (device=%s, adapter=%s, params entraînables=%d)",
                 self._device, self.adapter, sum(p.numel() for p in self._trainable_params))

    def _load_latest_lora(self, unet) -> None:
        """Charge la LoRA déjà entraînée pour ce (modality, distance), si un checkpoint
        existe. Appelé pour fit() (reprise) ET sample() — sans ça, sample() générerait
        avec une LoRA fraîche/aléatoire, jamais celle entraînée par train_generator
        (processus/session séparés -> les poids ne survivent pas autrement)."""
        from peft import set_peft_model_state_dict
        from src.utils.checkpoint import latest_checkpoint, load_checkpoint

        tag = f"generator_{self.cfg['modality']}_{self.cfg['distance']}"
        ckpt_path = latest_checkpoint(self.cfg["paths"]["checkpoints"], tag)
        if ckpt_path is not None:
            set_peft_model_state_dict(unet, load_checkpoint(ckpt_path)["lora"])
            log.info("LoRA chargée depuis %s", ckpt_path)

    def _setup_adapter(self, unet) -> list:
        """LoRA (visible/d1, cette itération) ou full_finetune (réservé IR, cf. CLAUDE.md)."""
        unet.requires_grad_(False)
        if self.adapter == "lora":
            from peft import LoraConfig
            lora_cfg = self.cfg["generator"]["lora"]
            unet.add_adapter(LoraConfig(
                r=lora_cfg["rank"], lora_alpha=lora_cfg["alpha"],
                target_modules=lora_cfg["target_modules"]))
            trainable = [p for p in unet.parameters() if p.requires_grad]
            # La LoRA est créée au dtype de la base (float16 sur GPU, cf. _ensure_loaded).
            # Entraîner l'optimiseur (AdamW) directement en float16 sans GradScaler est
            # instable (sous-débordement des moments d'Adam, NaN après quelques pas).
            # Pattern officiel HF (examples/text_to_image/train_text_to_image_lora.py) :
            # upcast les poids ENTRAÎNABLES en float32, la base gelée reste float16 ;
            # peft gère nativement ce mélange de dtypes dans le forward (vérifié).
            for p in trainable:
                p.data = p.data.float()
            return trainable
        if self.adapter == "full_finetune":
            # TODO(claude): chemin IR (cf. CLAUDE.md "Crochet IR") — non exercé en visible/d1.
            unet.requires_grad_(True)
            return list(unet.parameters())
        raise ValueError(f"Adaptateur générateur inconnu : {self.adapter}")

    # ------------------------------------------------------------------- embeddings
    def _id_embedding(self, image_path: str):
        """Embedding ArcFace (antelopev2) du visage le plus grand détecté, normalisé."""
        import torch
        from src.generator.face_detect import detect_largest_face

        face = detect_largest_face(self._face_app, image_path)
        if face is None:
            raise ValueError(f"Aucun visage détecté par insightface : {image_path}")
        emb = torch.tensor(face.embedding, dtype=torch.float32, device=self._device)[None]
        return emb / emb.norm(dim=1, keepdim=True)

    def _project_for_conditioning(self, id_emb):
        import torch
        from arc2face import project_face_embs
        # Contournement d'un bug d'Arc2Face (arc2face/utils.py) : project_face_embs
        # indexe token_embs (taille N) avec le masque `input_ids==arcface_token_id`
        # (toujours taille 1, jamais répété), ce qui plante dès que N>1 (plusieurs
        # identités différentes dans un même batch, notre cas en entraînement).
        # Leur propre usage (génération) n'appelle jamais la fonction avec N>1, donc
        # le bug n'apparaît jamais chez eux. Contournement : appeler par identité (N=1).
        dtype = self._pipeline.unet.dtype
        embs = [project_face_embs(self._pipeline, id_emb[i:i + 1].to(dtype))
                for i in range(id_emb.shape[0])]
        return torch.cat(embs, dim=0)

    def _load_image_tensor(self, path: str, size: int):
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image
        img = Image.open(path).convert("RGB").resize((size, size))
        return TF.to_tensor(img).to(self._device)  # (3,H,W) en [0,1]

    # ----------------------------------------------------------------------- fit
    def fit(self, pairs: Sequence[FacePair]) -> None:
        """Entraîne la LoRA sur des paires RÉELLES (Bloc A), reprenable (checkpoint)."""
        import torch
        import torch.nn.functional as F
        from src.generator.identity_loss import identity_cosine_loss
        from src.generator.degrade_prior import high_order_degrade
        from src.utils.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint, resume_step

        if not pairs:
            raise ValueError("fit() appelé avec une liste de paires vide (Bloc A).")
        self._ensure_loaded()

        train_cfg = self.cfg["generator"]["train"]
        id_weight = self.cfg["generator"]["identity_loss_weight"]
        degrade_prior = self.cfg["generator"]["degrade_prior"]
        distance = self.cfg["distance"]
        ckpt_dir = self.cfg["paths"]["checkpoints"]
        tag = f"generator_{self.cfg['modality']}_{distance}"

        from peft import get_peft_model_state_dict

        optimizer = torch.optim.AdamW(self._trainable_params, lr=train_cfg["lr"])
        step = resume_step(ckpt_dir, tag)
        ckpt_path = latest_checkpoint(ckpt_dir, tag)
        if ckpt_path is not None:
            # LoRA déjà chargée par _ensure_loaded() -> _load_latest_lora() ; il ne
            # reste que l'état de l'optimiseur (moments AdamW) à restaurer ici.
            optimizer.load_state_dict(load_checkpoint(ckpt_path)["optimizer"])
            log.info("Reprise entraînement générateur depuis step=%d (%s)", step, ckpt_path)

        # Cache des embeddings ArcFace mugshot (référence identité), validé UNE FOIS
        # pour toutes les identités avant la boucle : si la détection de visage échoue
        # pour une identité (image atypique), elle est exclue proprement (log clair)
        # plutôt que de planter tout l'entraînement au pas où elle est tirée (souvent
        # après une longue progression déjà accomplie).
        ref_embeddings: dict[str, torch.Tensor] = {}
        skipped_identities: set[str] = set()
        mugshot_by_identity = {p.identity: p.mugshot_path for p in pairs}
        for identity, mugshot_path in mugshot_by_identity.items():
            try:
                ref_embeddings[identity] = self._id_embedding(mugshot_path)
            except ValueError as e:
                log.warning("Identité %s exclue du Bloc A (visage non détecté) : %s", identity, e)
                skipped_identities.add(identity)
        if skipped_identities:
            pairs = [p for p in pairs if p.identity not in skipped_identities]
        if not pairs:
            raise RuntimeError("Aucune paire valide après exclusion des échecs de détection de visage.")
        log.info("Bloc A : %d paires valides, %d identité(s) exclue(s) : %s",
                  len(pairs), len(skipped_identities), sorted(skipped_identities) or "aucune")

        unet, vae, scheduler = self._pipeline.unet, self._pipeline.vae, self._pipeline.scheduler
        vae_scale = vae.config.scaling_factor
        # Copie locale (device+dtype) d'alphas_cumprod : scheduler.alphas_cumprod lui-même
        # reste sur CPU/float32 (attendu par sample(), cf. _ensure_loaded). Sans ce cast,
        # l'indexer avec un tenseur GPU (timesteps) plante, et le multiplier par noise_pred
        # (float16) promeut implicitement vers float32 et casse vae.decode.
        alphas_cumprod = scheduler.alphas_cumprod.to(self._device, dtype=unet.dtype)

        i = 0
        n = len(pairs)
        grad_norm_max = 0.0  # diagnostic : pic de norme de gradient sur tout le run (cf. instabilite suspectee)
        while step < train_cfg["max_steps"]:
            batch = [pairs[(i + j) % n] for j in range(train_cfg["batch_size"])]
            i = (i + train_cfg["batch_size"]) % n

            id_embs, target_imgs = [], []
            for p in batch:
                id_embs.append(ref_embeddings[p.identity])
                target = self._load_image_tensor(p.target_path, size=512)
                if degrade_prior:
                    target = high_order_degrade(target, distance)
                target_imgs.append(target)

            id_emb = torch.cat(id_embs, dim=0)
            prompt_embeds = self._project_for_conditioning(id_emb)
            # VAE en float32 (cf. _ensure_loaded) : encoder en son dtype natif, puis
            # rebasculer en float16 pour le UNet (qui lui reste en float16).
            target_batch = torch.stack(target_imgs).to(vae.dtype)

            with torch.no_grad():
                latents = (vae.encode(target_batch * 2 - 1).latent_dist.sample() * vae_scale).to(unet.dtype)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],),
                                       device=self._device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds).sample
            diffusion_loss = F.mse_loss(noise_pred.float(), noise.float())

            x0_pred = (noisy_latents - alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1) * noise_pred
                       ) / (1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
            decoded = (vae.decode((x0_pred / vae_scale).to(vae.dtype)).sample / 2 + 0.5).clamp(0, 1).float()
            # Pondère par alphas_cumprod[timesteps] (proche de 1 = faible bruit, x0_pred
            # fiable ; proche de 0 = fort bruit, x0_pred quasi inexploitable). Sans ça,
            # identity_loss restait plate ~1.0 tout l'entraînement (gradient utile noyé
            # par les tirages à fort bruit, majoritaires sur l'échantillonnage uniforme).
            id_loss = identity_cosine_loss(decoded, id_emb.float(), self._identity_embedder,
                                            weights=alphas_cumprod[timesteps])

            loss = diffusion_loss + id_weight * id_loss
            optimizer.zero_grad()
            loss.backward()
            # DIAGNOSTIC (pas encore un correctif) : norme totale du gradient, sans
            # écrêtage réel (max_norm=inf -> ne modifie jamais les gradients). Sert à
            # confirmer ou écarter l'hypothèse d'instabilité (pics de gradient) avant
            # d'ajouter un vrai clipping. Cf. incident fidelity FAIL à tous les
            # checkpoints testés (2026-06-26).
            grad_norm = torch.nn.utils.clip_grad_norm_(self._trainable_params, max_norm=float("inf")).item()
            grad_norm_max = max(grad_norm_max, grad_norm)
            optimizer.step()
            step += 1

            if step % train_cfg["log_every"] == 0 or step == train_cfg["max_steps"]:
                log.info("step=%d diffusion=%.4f identity=%.4f total=%.4f grad_norm=%.4f grad_norm_max=%.4f",
                         step, diffusion_loss.item(), id_loss.item(), loss.item(), grad_norm, grad_norm_max)
            if step % train_cfg["ckpt_every"] == 0 or step == train_cfg["max_steps"]:
                # Seuls les poids LoRA (quelques Mo) sont sauvegardés, pas tout le UNet
                # gelé (~860M paramètres, ~1.7 Go, jamais modifiés) : la base se recharge
                # à l'identique à chaque session (Drive/Hub), inutile de la dupliquer à
                # chaque checkpoint. Évite de saturer l'espace Drive (incident du 2026-06-25).
                save_checkpoint({"lora": get_peft_model_state_dict(unet), "optimizer": optimizer.state_dict()},
                                 ckpt_dir, tag, step)

    # --------------------------------------------------------------------- sample
    def sample(self, mugshot_path: str, k: int) -> list[str]:
        """Génère k échantillons (diversité intra-classe) ; sauve sous paths.synth_dataset."""
        import torch
        from PIL import Image

        self._ensure_loaded()
        identity = Path(mugshot_path).stem.split("_")[0]
        id_emb = self._id_embedding(mugshot_path)
        prompt_embeds = self._project_for_conditioning(id_emb).repeat(k, 1, 1)

        out_dir = Path(self.cfg["paths"]["synth_dataset"]) / identity
        out_dir.mkdir(parents=True, exist_ok=True)

        sample_cfg = self.cfg["generator"]["sample"]
        paths = []
        with torch.no_grad():
            for idx in range(k):
                generator = torch.Generator(device=self._device).manual_seed(idx)
                # output_type="latent" : on décode nous-mêmes (cf. _ensure_loaded, VAE en
                # float32 pour éviter les artefacts néon/NaN connus du VAE SD1.5 en
                # float16 ; l'appel intégré du pipeline ne cast pas vers le dtype du VAE).
                latents = self._pipeline(
                    prompt_embeds=prompt_embeds[idx:idx + 1],
                    num_inference_steps=sample_cfg["num_inference_steps"],
                    guidance_scale=sample_cfg["guidance_scale"],
                    generator=generator, output_type="latent").images
                vae = self._pipeline.vae
                decoded = vae.decode((latents / vae.config.scaling_factor).to(vae.dtype)).sample
                image = self._pipeline.image_processor.postprocess(decoded, output_type="pil")[0]
                path = out_dir / f"{identity}_{idx:03d}.png"
                image.save(path)
                paths.append(str(path))
        return paths
