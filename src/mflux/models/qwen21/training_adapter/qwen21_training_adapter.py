from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.training.adapters.base import TrainingAdapter
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.utils import TrainingUtil
from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
from mflux.models.qwen21.model.qwen21_scheduler import Qwen21TurboScheduler
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
from mflux.models.qwen21.weights.qwen21_lora_mapping import Qwen21LoRAMapping
from mflux.utils.version_util import VersionUtil


class Qwen21TrainingAdapter(TrainingAdapter):
    """LoRA training adapter for Qwen-Image-2.1 (base and Turbo).

    Training is plain flow matching, like Krea 2: the trainer noises a latent with
    z = (1 - sigma) * clean + sigma * noise and the target velocity is noise - clean, which is what
    the transformer predicts at timestep = sigma. predict_noise returns the transformer output unchanged.

    The transformer carries three inference-only fast paths that must stay off while gradients are
    taken, so predict_noise calls the uncached joint forward (``_forward``) directly instead of
    ``__call__``:

    - ``attn.use_fused_prologue``: the fused QK-norm + RoPE Metal kernel has no VJP.
    - ``use_text_cache``: the text-prefix K/V cache is built outside the autograd graph, so text-side
      LoRAs would receive zero gradient.
    - ``inference_projections()``: packs Q/K/V into one weight and routes around injected LoRA layers.
    """

    def __init__(self, *, model_config: ModelConfig, quantize: int | None, model_path: str | None = None):
        self._model_config = model_config
        self._qwen = QwenImage21(quantize=quantize, model_config=model_config, model_path=model_path)
        self._guidance: float | None = None
        self._fused_prologue_defaults = [
            block.attn.use_fused_prologue for block in self._qwen.transformer.transformer_blocks
        ]

    def model(self):
        return self._qwen

    def transformer(self):
        return self._qwen.transformer

    def create_config(self, training_spec: TrainingSpec, *, width: int, height: int) -> Config:
        self._guidance = training_spec.guidance
        return Config(
            model_config=self._model_config,
            num_inference_steps=training_spec.steps,
            width=width,
            height=height,
            guidance=training_spec.guidance,
            # Turbo samples its own fixed sigma curve; the base model uses linear + sigma shift.
            scheduler=Qwen21TurboScheduler.for_model(self._model_config, "linear"),
        )

    def freeze_base(self) -> None:
        self._qwen.vae.freeze()
        self._qwen.transformer.freeze()
        if getattr(self._qwen, "text_encoder", None) is not None:
            self._qwen.text_encoder.freeze()
        self._enter_training_mode()

    def set_gradient_checkpointing(self, enabled: bool) -> None:  # noqa: ARG002
        # Always on (see freeze_base): the 32-block activation graph does not fit otherwise.
        self._qwen.transformer.gradient_checkpointing = True

    def _enter_training_mode(self) -> None:
        transformer = self._qwen.transformer
        transformer.gradient_checkpointing = True
        transformer.use_text_cache = False
        transformer.clear_text_cache()
        transformer._step_fn = None
        transformer._image_step_fn = None
        for block in transformer.transformer_blocks:
            block.attn.use_fused_prologue = False

    def _enter_inference_mode(self) -> None:
        transformer = self._qwen.transformer
        transformer.gradient_checkpointing = False
        transformer.use_text_cache = True
        for block, default in zip(transformer.transformer_blocks, self._fused_prologue_defaults):
            block.attn.use_fused_prologue = default

    @contextmanager
    def _inference_mode(self) -> Iterator[None]:
        self._enter_inference_mode()
        try:
            yield
        finally:
            self._enter_training_mode()

    def encode_data(
        self,
        *,
        data_id: int,  # noqa: ARG002
        image_path: Path,
        prompt: str,
        width: int,
        height: int,
        input_image_path: Path | None = None,  # txt2img: unused  # noqa: ARG002
    ) -> tuple[mx.array, Any]:
        encoded = LatentCreator.encode_image(
            vae=self._qwen.vae,
            image_path=image_path,
            height=height,
            width=width,
            tiling_config=getattr(self._qwen, "tiling_config", None),
        )
        clean_latents = Qwen21LatentCreator.pack_latents(encoded, height, width).astype(mx.float32)

        embeds, mask = Qwen21PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=self._qwen.prompt_cache,
            tokenizer=self._qwen.tokenizers["qwen21"],
            text_encoder=self._qwen.text_encoder,
        )

        mx.eval(clean_latents, embeds, mask)
        return clean_latents, {"embeds": embeds, "mask": mask}

    def predict_noise(self, *, t: int, latents_t: mx.array, sigmas: mx.array, cond: Any, config: Config) -> mx.array:
        transformer = self._qwen.transformer
        embeds = cond["embeds"]
        mask = cond.get("mask")
        text_len = embeds.shape[1]
        rope_cos, rope_sin, attn_mask = transformer._geometry(
            text_len=text_len,
            latent_height=config.height // 16,
            latent_width=config.width // 16,
            encoder_hidden_states_mask=mask,
        )
        # causal_condition: text tokens read the t=0 row, image tokens the sampled sigma row.
        timestep = sigmas[t].reshape(1)
        timestep_rows = mx.concatenate([timestep, mx.zeros((1,), dtype=timestep.dtype)])
        velocity = transformer._forward(
            latents_t.astype(ModelConfig.precision),
            embeds,
            timestep_rows,
            rope_cos,
            rope_sin,
            attn_mask,
        )
        return velocity.astype(latents_t.dtype)

    def generate_preview_image(
        self,
        *,
        seed: int,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        image_paths: list[Path | str] | None = None,  # txt2img: unused  # noqa: ARG002
    ):
        guidance = self._guidance if self._guidance is not None else 1.0
        if Qwen21TurboScheduler.is_turbo(self._model_config):
            guidance = 1.0
        with self._inference_mode(), self._assistant_disabled():
            return self._qwen.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
            )

    def save_lora_adapter(self, *, path: Path, training_spec: TrainingSpec) -> None:
        weights: dict[str, mx.array] = {}
        for target in training_spec.lora_layers.targets:
            if target.blocks is not None:
                for b in target.blocks.get_blocks():
                    self._append_train_lora_weights(weights, target.module_path.format(block=b))
            else:
                self._append_train_lora_weights(weights, target.module_path)
        mx.save_safetensors(
            str(path),
            weights,
            metadata={"mflux_version": VersionUtil.get_mflux_version(), "model": training_spec.model},
        )

    def load_lora_adapter(self, *, path: str | Path) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=Qwen21LoRAMapping.get_mapping(),
            transformer=self._qwen.transformer,
            lora_paths=[str(path)],
            lora_scales=[1.0],
            role="train",
        )

    def load_training_adapter(self, *, path: str | Path, scale: float = 1.0) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=Qwen21LoRAMapping.get_mapping(),
            transformer=self._qwen.transformer,
            lora_paths=[str(path)],
            lora_scales=[float(scale)],
            role="assistant",
        )

    def _assistant_disabled(self):
        return TrainingUtil.assistant_disabled(self._qwen.transformer)

    def _append_train_lora_weights(self, weights: dict[str, mx.array], module_path: str) -> None:
        train_lora = TrainingUtil.get_train_lora(self._qwen.transformer, module_path)
        weights[f"transformer.{module_path}.lora_A.weight"] = mx.transpose(train_lora.lora_A)
        weights[f"transformer.{module_path}.lora_B.weight"] = mx.transpose(train_lora.lora_B)
