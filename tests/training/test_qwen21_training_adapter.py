from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.training.lora.target_injector import inject_lora_targets
from mflux.models.common.training.state.training_spec import BlockRange, LoraTargetSpec
from mflux.models.common.training.trainer import TrainingTrainer
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.training_adapter import qwen21_training_adapter as adapter_module

Qwen21TrainingAdapter = adapter_module.Qwen21TrainingAdapter

HEIGHT = WIDTH = 64  # 4x4 = 16 image tokens
TXT_TOKENS = 6


def _tiny_transformer() -> Qwen21Transformer:
    mx.random.seed(0)
    transformer = Qwen21Transformer(
        in_channels=64,
        out_channels=64,
        context_in_dim=32,
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=16,
        mlp_ratio=1,
        axes_dims_rope=(4, 6, 6),
    )
    transformer.apply(lambda w: mx.random.normal(w.shape, dtype=mx.float32) * 0.05)
    # the fused prologue is only built for the real head_dim; force it on so the adapter must turn it off
    for block in transformer.transformer_blocks:
        block.attn.use_fused_prologue = True
    return transformer


class _FakeQwenImage21(nn.Module):
    def __init__(self, *, quantize, model_config, model_path=None):  # noqa: ARG002
        super().__init__()
        self.transformer = _tiny_transformer()
        self.vae = nn.Linear(1, 1)
        self.text_encoder = nn.Linear(1, 1)
        self.prompt_cache = {}
        self.tokenizers = {}
        self.tiling_config = None
        self.last_generate_kwargs = None

    def generate_image(self, **kwargs):
        self.last_generate_kwargs = kwargs
        # inference paths must be back on while generating
        assert self.transformer.use_text_cache is True
        assert self.transformer.gradient_checkpointing is False
        assert all(b.attn.use_fused_prologue for b in self.transformer.transformer_blocks)
        return "preview-image"


def _adapter(monkeypatch, model_config=None) -> Qwen21TrainingAdapter:
    monkeypatch.setattr(adapter_module, "QwenImage21", _FakeQwenImage21)
    return Qwen21TrainingAdapter(model_config=model_config or ModelConfig.qwen_image_21(), quantize=None)


def _targets() -> list[LoraTargetSpec]:
    return [
        LoraTargetSpec(
            module_path="transformer_blocks.{block}.attn.to_q", blocks=BlockRange(start=0, end=2), rank=4, alpha=4
        ),
        LoraTargetSpec(
            module_path="transformer_blocks.{block}.img_mlp.out", blocks=BlockRange(start=0, end=2), rank=4, alpha=4
        ),
        LoraTargetSpec(module_path="modulation.layers.1", blocks=None, rank=4, alpha=4),
    ]


def _prepare_for_training(adapter: Qwen21TrainingAdapter) -> None:
    # same order as the runner/trainer: inject -> freeze_base -> unfreeze LoRA -> checkpointing
    inject_lora_targets(adapter.transformer(), _targets())
    adapter.freeze_base()
    TrainingTrainer._unfreeze_lora_layers(adapter.transformer())
    adapter.set_gradient_checkpointing(True)
    # a zero lora_B gives zero lora_A gradients by construction; perturb B so both sides get signal
    for _, child in adapter.transformer().named_modules():
        if isinstance(child, LoRALinear):
            child.lora_B = mx.random.normal(child.lora_B.shape) * 0.05
    mx.eval(adapter.model().parameters())


def _cond(padded: bool) -> dict:
    mx.random.seed(2)
    embeds = mx.random.normal((1, TXT_TOKENS, 32)).astype(ModelConfig.precision)
    mask = mx.ones((1, TXT_TOKENS), dtype=mx.int32)
    if padded:
        mask[:, -2:] = 0
    return {"embeds": embeds, "mask": mask}


def _config(adapter: Qwen21TrainingAdapter) -> Config:
    spec = SimpleNamespace(steps=8, guidance=1.0)
    return adapter.create_config(training_spec=spec, width=WIDTH, height=HEIGHT)


@pytest.mark.fast
def test_freeze_base_turns_off_inference_fast_paths(monkeypatch):
    adapter = _adapter(monkeypatch)
    transformer = adapter.transformer()
    assert all(b.attn.use_fused_prologue for b in transformer.transformer_blocks)
    _prepare_for_training(adapter)
    assert transformer.gradient_checkpointing is True
    assert transformer.use_text_cache is False
    assert not any(b.attn.use_fused_prologue for b in transformer.transformer_blocks)
    assert transformer._step_fn is None and transformer._image_step_fn is None
    # only LoRA matrices are trainable
    trainable = {k for k, _ in tree_flatten(adapter.model().trainable_parameters())}
    assert trainable
    assert all("lora_A" in k or "lora_B" in k for k in trainable)


@pytest.mark.fast
@pytest.mark.parametrize("padded", [False, True])
def test_predict_noise_calls_uncached_forward_with_sigma_rows(monkeypatch, padded):
    adapter = _adapter(monkeypatch)
    _prepare_for_training(adapter)
    transformer = adapter.transformer()
    config = _config(adapter)
    seen = {}
    real_forward = transformer._forward

    def spy(hidden, embeds, timestep_rows, rope_cos, rope_sin, attn_mask):
        seen.update(timestep_rows=timestep_rows, attn_mask=attn_mask, hidden=hidden)
        return real_forward(hidden, embeds, timestep_rows, rope_cos, rope_sin, attn_mask)

    monkeypatch.setattr(transformer, "_forward", spy)
    latents_t = mx.random.normal((1, 16, 64), dtype=mx.float32)
    sigmas = mx.array([1.0, 0.7, 0.4, 0.0])

    velocity = adapter.predict_noise(t=1, latents_t=latents_t, sigmas=sigmas, cond=_cond(padded), config=config)
    mx.eval(velocity)

    assert velocity.shape == latents_t.shape and velocity.dtype == latents_t.dtype
    assert bool(mx.all(mx.isfinite(velocity)))
    rows = seen["timestep_rows"]
    assert rows.shape == (2,)
    assert float(rows[0]) == pytest.approx(0.7, abs=1e-3) and float(rows[1]) == 0.0
    assert seen["hidden"].dtype == ModelConfig.precision
    # the padding mask is threaded through to the joint forward only when the prompt is padded
    assert (seen["attn_mask"] is not None) is padded
    assert not transformer._text_caches


@pytest.mark.fast
def test_lora_gradients_flow_and_match_without_checkpointing(monkeypatch):
    adapter = _adapter(monkeypatch)
    _prepare_for_training(adapter)
    config = _config(adapter)
    cond = _cond(padded=True)
    latents_t = mx.random.normal((1, 16, 64), dtype=mx.float32)
    sigmas = mx.array([1.0, 0.7, 0.4, 0.0])

    def loss_fn(model):  # noqa: ARG001
        v = adapter.predict_noise(t=2, latents_t=latents_t, sigmas=sigmas, cond=cond, config=config)
        return v.square().mean()

    def grads(checkpointing: bool) -> dict:
        adapter.transformer().gradient_checkpointing = checkpointing
        loss, g = nn.value_and_grad(adapter.model(), loss_fn)(adapter.model())
        mx.eval(loss, g)
        return dict(tree_flatten(g))

    checkpointed = grads(True)
    eager = grads(False)
    assert checkpointed.keys() == eager.keys()
    for key, g in checkpointed.items():
        assert "lora_" in key
        assert float(mx.abs(g).max()) > 0, f"zero gradient for {key}"
        assert mx.allclose(g, eager[key], atol=1e-4, rtol=1e-3).item(), key
    paths = {k.rsplit(".lora_", 1)[0] for k in checkpointed}
    assert any("modulation" in p for p in paths)
    assert any("attn.to_q" in p for p in paths)
    assert any("img_mlp.out" in p for p in paths)


@pytest.mark.fast
def test_preview_restores_inference_paths_then_training_paths(monkeypatch):
    adapter = _adapter(monkeypatch)
    _prepare_for_training(adapter)
    spec = SimpleNamespace(steps=8, guidance=3.0)
    adapter.create_config(training_spec=spec, width=WIDTH, height=HEIGHT)
    assert adapter.generate_preview_image(seed=1, prompt="p", width=WIDTH, height=HEIGHT, steps=4) == "preview-image"
    assert adapter.model().last_generate_kwargs["guidance"] == 3.0
    transformer = adapter.transformer()
    assert transformer.gradient_checkpointing is True
    assert transformer.use_text_cache is False
    assert not any(b.attn.use_fused_prologue for b in transformer.transformer_blocks)


@pytest.mark.fast
def test_turbo_preview_uses_turbo_schedule_without_cfg(monkeypatch):
    adapter = _adapter(monkeypatch, ModelConfig.qwen_image_21_turbo())
    _prepare_for_training(adapter)
    spec = SimpleNamespace(steps=8, guidance=3.0)
    config = adapter.create_config(training_spec=spec, width=WIDTH, height=HEIGHT)
    assert config._scheduler_str == "qwen21_turbo"
    adapter.generate_preview_image(seed=1, prompt="p", width=WIDTH, height=HEIGHT, steps=8)
    assert adapter.model().last_generate_kwargs["guidance"] == 1.0


@pytest.mark.fast
def test_save_lora_adapter_uses_transformer_namespace(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch)
    _prepare_for_training(adapter)
    spec = SimpleNamespace(model="qwen-image-2.1", lora_layers=SimpleNamespace(targets=_targets()))
    out = tmp_path / "adapter.safetensors"
    adapter.save_lora_adapter(path=out, training_spec=spec)
    weights = mx.load(str(out))
    assert "transformer.transformer_blocks.0.attn.to_q.lora_A.weight" in weights
    assert "transformer.modulation.layers.1.lora_B.weight" in weights
    assert len(weights) == 2 * (2 + 2 + 1)
