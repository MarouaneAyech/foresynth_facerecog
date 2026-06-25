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
        from insightface.app import FaceAnalysis
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
        # Le scheduler n'est pas un nn.Module : pipeline.to(device) ne déplace pas
        # alphas_cumprod, indexé directement par fit() avec un tenseur GPU (timesteps).
        pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(self._device)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)

        self._trainable_params = self._setup_adapter(pipeline.unet)

        self._link_insightface_pack(pretrained["insightface_pack"])
        self._face_app = FaceAnalysis(
            name=pretrained["insightface_pack"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device == "cuda" else ["CPUExecutionProvider"])
        self._face_app.prepare(ctx_id=0 if self._device == "cuda" else -1, det_size=(640, 640))

        self._identity_embedder = load_arcface_embedder(self.cfg)
        self._pipeline = pipeline
        log.info("Arc2FaceGenerator chargé (device=%s, adapter=%s, params entraînables=%d)",
                 self._device, self.adapter, sum(p.numel() for p in self._trainable_params))

    def _link_insightface_pack(self, pack_name: str) -> None:
        """Lie (symlink) un pack insightface déjà déposé (Drive) vers l'emplacement par
        défaut attendu par FaceAnalysis (~/.insightface/models/<pack>), sans dupliquer
        les fichiers. Sans effet si déjà présent ou si rien n'est déposé (fallback
        normal d'insightface, qui échouera pour antelopev2 — non auto-téléchargeable)."""
        local_pack_dir = Path(self.cfg["paths"].get("insightface_models_dir", "")) / pack_name
        target = Path.home() / ".insightface" / "models" / pack_name
        if target.exists() or not local_pack_dir.is_dir():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(local_pack_dir, target_is_directory=True)
        log.info("Pack insightface '%s' lié depuis %s -> %s", pack_name, local_pack_dir, target)

    def _setup_adapter(self, unet) -> list:
        """LoRA (visible/d1, cette itération) ou full_finetune (réservé IR, cf. CLAUDE.md)."""
        unet.requires_grad_(False)
        if self.adapter == "lora":
            from peft import LoraConfig
            lora_cfg = self.cfg["generator"]["lora"]
            unet.add_adapter(LoraConfig(
                r=lora_cfg["rank"], lora_alpha=lora_cfg["alpha"],
                target_modules=lora_cfg["target_modules"]))
            return [p for p in unet.parameters() if p.requires_grad]
        if self.adapter == "full_finetune":
            # TODO(claude): chemin IR (cf. CLAUDE.md "Crochet IR") — non exercé en visible/d1.
            unet.requires_grad_(True)
            return list(unet.parameters())
        raise ValueError(f"Adaptateur générateur inconnu : {self.adapter}")

    # ------------------------------------------------------------------- embeddings
    def _id_embedding(self, image_path: str):
        """Embedding ArcFace (antelopev2) du visage le plus grand détecté, normalisé."""
        import numpy as np
        import torch
        from PIL import Image

        img = np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1]  # RGB->BGR (insightface)
        faces = self._face_app.get(img)
        if not faces:
            raise ValueError(f"Aucun visage détecté par insightface : {image_path}")
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
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

        optimizer = torch.optim.AdamW(self._trainable_params, lr=train_cfg["lr"])
        step = resume_step(ckpt_dir, tag)
        ckpt_path = latest_checkpoint(ckpt_dir, tag)
        if ckpt_path is not None:
            state = load_checkpoint(ckpt_path)
            self._pipeline.unet.load_state_dict(state["unet"], strict=False)
            optimizer.load_state_dict(state["optimizer"])
            log.info("Reprise entraînement générateur depuis step=%d (%s)", step, ckpt_path)

        # cache des embeddings ArcFace mugshot (référence identité), 1 appel/identité.
        ref_embeddings: dict[str, torch.Tensor] = {}

        unet, vae, scheduler = self._pipeline.unet, self._pipeline.vae, self._pipeline.scheduler
        vae_scale = vae.config.scaling_factor

        i = 0
        n = len(pairs)
        while step < train_cfg["max_steps"]:
            batch = [pairs[(i + j) % n] for j in range(train_cfg["batch_size"])]
            i = (i + train_cfg["batch_size"]) % n

            id_embs, target_imgs = [], []
            for p in batch:
                if p.identity not in ref_embeddings:
                    ref_embeddings[p.identity] = self._id_embedding(p.mugshot_path)
                id_embs.append(ref_embeddings[p.identity])
                target = self._load_image_tensor(p.target_path, size=512)
                if degrade_prior:
                    target = high_order_degrade(target, distance)
                target_imgs.append(target)

            id_emb = torch.cat(id_embs, dim=0)
            prompt_embeds = self._project_for_conditioning(id_emb)
            target_batch = torch.stack(target_imgs).to(unet.dtype)

            with torch.no_grad():
                latents = vae.encode(target_batch * 2 - 1).latent_dist.sample() * vae_scale
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],),
                                       device=self._device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds).sample
            diffusion_loss = F.mse_loss(noise_pred.float(), noise.float())

            x0_pred = (noisy_latents - scheduler.alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1) * noise_pred
                       ) / (1 - scheduler.alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
            decoded = (vae.decode(x0_pred / vae_scale).sample / 2 + 0.5).clamp(0, 1).float()
            id_loss = identity_cosine_loss(decoded, id_emb.float(), self._identity_embedder)

            loss = diffusion_loss + id_weight * id_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % train_cfg["log_every"] == 0 or step == train_cfg["max_steps"]:
                log.info("step=%d diffusion=%.4f identity=%.4f total=%.4f",
                         step, diffusion_loss.item(), id_loss.item(), loss.item())
            if step % train_cfg["ckpt_every"] == 0 or step == train_cfg["max_steps"]:
                save_checkpoint({"unet": unet.state_dict(), "optimizer": optimizer.state_dict()},
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

        paths = []
        with torch.no_grad():
            for idx in range(k):
                generator = torch.Generator(device=self._device).manual_seed(idx)
                image = self._pipeline(
                    prompt_embeds=prompt_embeds[idx:idx + 1],
                    num_inference_steps=25, guidance_scale=3.0,
                    generator=generator).images[0]
                path = out_dir / f"{identity}_{idx:03d}.png"
                image.save(path)
                paths.append(str(path))
        return paths
