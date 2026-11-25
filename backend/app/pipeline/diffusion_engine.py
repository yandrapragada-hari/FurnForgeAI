"""
FurnForge AI - real generative engine.

Stable Diffusion 1.5 + ControlNet (Hugging Face diffusers), run locally.

* ControlNet (canny) receives the OpenCV edge map of the user's sketch and injects it
  into the SD U-Net through zero-initialised convolutions, so the generated furniture
  keeps the sketch's outline, proportions and part layout.
* Two image-to-image translation modes:
    - "controlnet": structure-conditioned translation. Denoising starts from pure
      noise; the sketch controls *shape*, the text prompt controls material/colour/style.
    - "img2img":    ControlNet + SD img2img. The sketch itself is also encoded by the
      VAE and used as the starting latent (partially noised by `strength`), so the
      result stays even closer to the drawing's composition.
* Optional LCM-LoRA (latent consistency distillation) reduces the number of denoising
  steps from ~25 to ~6, which is what makes CPU-only generation usable.

There is deliberately NO fallback: if the models are not loaded, generation fails
with a clear error instead of returning a placeholder or stock image.
"""
from __future__ import annotations

import gc
import logging
import os
import sys
import threading
import time
from typing import Callable, Optional

from PIL import Image

from app.config import (
    CONTROLNET_MODELS,
    CPU_INFERENCE_SIZE,
    DEFAULT_CONTROLNET,
    DEVICE,
    ENABLE_LCM,
    LCM_LORA,
    SD_MODEL,
)

logger = logging.getLogger("FurnForgeAI.Diffusion")

ProgressCallback = Callable[[int, int], None]


class ModelNotReadyError(RuntimeError):
    pass


class DiffusionEngine:
    def __init__(self) -> None:
        self.device = DEVICE
        self.status = "not_loaded"      # not_loaded | loading | ready | error
        self.status_detail = "Models have not been loaded yet."
        self.error: Optional[str] = None
        self.lcm_available = False
        self._lora_fused = False
        self.load_seconds: Optional[float] = None
        self._pipe = None
        self._img2img = None
        self._lock = threading.Lock()          # one generation at a time
        self._load_lock = threading.Lock()

    # ---------------------------------------------------------------- loading
    def load_async(self) -> None:
        if self.status in ("loading", "ready"):
            return
        if os.getenv("SPACES_ZERO_GPU") or "spaces" in sys.modules:
            self.load()
            return
        threading.Thread(target=self.load, name="model-loader", daemon=True).start()

    def unload(self) -> None:
        """Release CPU/GPU diffusion weights to free memory for 3D reconstruction."""
        with self._load_lock:
            with self._lock:
                self._pipe = None
                self._img2img = None
                self._quality_scheduler = None
                self._lcm_scheduler = None
                self._lora_fused = False
                self.lcm_available = False
                self.status = "not_loaded"
                self.status_detail = "Temporarily unloaded to free memory for 3D reconstruction."
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass

    def load(self) -> None:
        with self._load_lock:
            if self.status == "ready":
                return
            self.status = "loading"
            self.error = None
            t0 = time.time()
            try:
                import torch
                from diffusers import (
                    ControlNetModel,
                    LCMScheduler,
                    StableDiffusionControlNetImg2ImgPipeline,
                    StableDiffusionControlNetPipeline,
                    UniPCMultistepScheduler,
                )

                # Use float32 on GPU and CPU to eliminate the half-precision overflow (NaN -> solid 128 grey image)
                # On server GPUs with ample VRAM (e.g. ZeroGPU 95GB Blackwell), float32 provides complete numerical stability.
                dtype = torch.float32

                cn_id = CONTROLNET_MODELS[DEFAULT_CONTROLNET]
                self.status_detail = f"Downloading/loading ControlNet: {cn_id}"
                logger.info(self.status_detail)
                controlnet = self._from_pretrained(ControlNetModel, cn_id, torch_dtype=dtype)

                self.status_detail = f"Downloading/loading Stable Diffusion: {SD_MODEL}"
                logger.info(self.status_detail)
                pipe = self._from_pretrained(
                    StableDiffusionControlNetPipeline,
                    SD_MODEL,
                    controlnet=controlnet,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                pipe.set_progress_bar_config(disable=True)

                self._quality_scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                self._lcm_scheduler = LCMScheduler.from_config(pipe.scheduler.config)

                if ENABLE_LCM:
                    self.status_detail = f"Downloading/loading LCM-LoRA: {LCM_LORA}"
                    logger.info(self.status_detail)
                    try:
                        pipe.load_lora_weights(LCM_LORA, adapter_name="lcm")
                        pipe.fuse_lora()
                        self._lora_fused = True
                        self.lcm_available = True
                    except Exception as exc:  # fast mode simply becomes unavailable
                        logger.warning("LCM-LoRA unavailable: %s", exc)

                try:
                    pipe = pipe.to(self.device)
                except Exception as exc:
                    logger.warning("Could not move pipeline to %s (%s); falling back to CPU", self.device, exc)
                    self.device = "cpu"
                    pipe = pipe.to("cpu")

                # Img2img variant sharing the very same weights (no extra memory).
                try:
                    img2img = StableDiffusionControlNetImg2ImgPipeline.from_pipe(pipe)
                except Exception:
                    img2img = StableDiffusionControlNetImg2ImgPipeline(**pipe.components)
                img2img.set_progress_bar_config(disable=True)

                # Ensure VAE is strictly in float32 without slicing bugs
                for p in (pipe, img2img):
                    try:
                        if hasattr(p, "vae") and p.vae is not None:
                            p.vae.to(dtype=torch.float32)
                            if hasattr(p.vae, "disable_slicing"):
                                p.vae.disable_slicing()
                    except Exception as exc:
                        logger.debug("vae float32 note: %s", exc)

                self._pipe, self._img2img = pipe, img2img
                self.load_seconds = round(time.time() - t0, 1)
                self.status = "ready"
                self.status_detail = f"Models loaded on {self.device.upper()} in {self.load_seconds}s"
                logger.info(self.status_detail)
            except Exception as exc:
                self.status = "error"
                self.error = f"{type(exc).__name__}: {exc}"
                self.status_detail = "Model loading failed - see error."
                logger.exception("Model loading failed")

    @staticmethod
    def _from_pretrained(cls, repo_id, **kwargs):
        """Prefer safetensors with graceful fallback."""
        try:
            return cls.from_pretrained(repo_id, use_safetensors=True, **kwargs)
        except Exception as exc:
            logger.info("Direct safetensors load for %s note: %s; trying variant fallback", repo_id, exc)
            try:
                return cls.from_pretrained(repo_id, variant="fp16", use_safetensors=True, **kwargs)
            except Exception:
                return cls.from_pretrained(repo_id, **kwargs)

    # ------------------------------------------------------------- inference
    @property
    def engine_label(self) -> str:
        return f"Stable Diffusion 1.5 + ControlNet ({DEFAULT_CONTROLNET}) on {self.device.upper()}"

    def generate(
        self,
        control_image: Image.Image,
        init_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        mode: str = "controlnet",
        strength: float = 0.85,
        fast: bool = True,
        steps: int = 25,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 1.0,
        seed: int = 0,
        progress: Optional[ProgressCallback] = None,
    ) -> Image.Image:
        """Run one real diffusion generation and return the PIL image."""
        if self.status != "ready":
            raise ModelNotReadyError(f"Generative model is not ready ({self.status}). {self.status_detail}")

        import torch

        use_lcm = fast and self.lcm_available
        if use_lcm:
            # LCM-LoRA converges cleanly in 4-7 steps; higher steps slow CPU without gain.
            steps = max(4, min(int(steps), 7))
            guidance_scale = max(1.4, min(float(guidance_scale), 1.8))
        else:
            steps = max(10, min(int(steps), 50))

        # On CPU, 384x384 latent denoising (~40s) + LANCZOS 512x512 upscale is ~10x faster than 512x512.
        target_dim = CPU_INFERENCE_SIZE if (self.device == "cpu" and use_lcm) else 512
        ctrl_in = control_image.resize((target_dim, target_dim), Image.Resampling.LANCZOS) if control_image.size != (target_dim, target_dim) else control_image
        init_in = init_image.resize((target_dim, target_dim), Image.Resampling.LANCZOS) if init_image.size != (target_dim, target_dim) else init_image

        with self._lock:
            with torch.inference_mode():
                pipe = self._img2img if mode == "img2img" else self._pipe
                scheduler = self._lcm_scheduler if use_lcm else self._quality_scheduler
                self._pipe.scheduler = scheduler
                self._img2img.scheduler = scheduler
                if self.lcm_available:
                    if use_lcm and not self._lora_fused:
                        self._pipe.enable_lora()
                        self._pipe.fuse_lora()
                        self._lora_fused = True
                    elif not use_lcm and self._lora_fused:
                        self._pipe.unfuse_lora()
                        self._pipe.disable_lora()
                        self._lora_fused = False

                generator = torch.Generator(device="cpu").manual_seed(int(seed))

                def _on_step_end(p, step_index, _timestep, callback_kwargs):
                    if progress:
                        total = getattr(p, "num_timesteps", None) or steps
                        progress(step_index + 1, int(total))
                    return callback_kwargs

                common = dict(
                    prompt=prompt,
                    negative_prompt=negative_prompt if guidance_scale > 1.0 else None,
                    width=target_dim,
                    height=target_dim,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=float(controlnet_scale),
                    generator=generator,
                    callback_on_step_end=_on_step_end,
                )
                if mode == "img2img":
                    out = pipe(image=init_in, control_image=ctrl_in,
                               strength=float(strength), **common)
                else:
                    out = pipe(image=ctrl_in, **common)

                result_img = out.images[0]
                if result_img.size != (512, 512):
                    result_img = result_img.resize((512, 512), Image.Resampling.LANCZOS)
                return result_img
