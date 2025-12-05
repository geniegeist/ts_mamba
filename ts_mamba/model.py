import math
import os
import json
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    if layer_idx not in attn_layer_idx:
        mixer_cls = partial(
            Mamba2,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
        
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

@dataclass
class MixerConfig():
    # dimension of hidden layer
    d_model: int
    # the total number of mixer blocks
    n_layer: int
    d_intermediate: int = 0
    rms_norm: bool = False
    norm_epsilon: float = 1e-6
    residual_in_fp32: bool = True
    fused_add_norm: bool = True

    # block config
    ssm_cfg: Dict[str, Any] = field(default_factory=dict)
    attn_layer_idx: List[int] = field(default_factory=list)
    attn_cfg: Dict[str, Any] = field(default_factory=dict)

    # init
    use_llm_init: bool = True
    llm_init_cfg: Dict[str, Any] = field(default_factory=dict)

    # for default sequence models
    d_input: Optional[int] = None
    d_output: Optional[int] = None

    # for token sequence models
    vocab_size: Optional[int] = None
    # pad the vocab_size so that it is multiple of pad_vocab_multiple
    # use 1 to not pad the vocab_size
    pad_vocab_multiple: int = 1
    # whether embedding and output projection layer should share the same weights
    tie_embeddings: bool = True



class MixerBackbone(nn.Module):
    """
    Shared backbone agnostic to the type of the input projection (linear, embedding, ...)
    Performs: add -> norm -> mixer pattern (attention / mlp)
    """
    def __init__(
        self,
        cfg: MixerConfig,
        input_proj: nn.Module,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        d_model = cfg.d_model
        d_intermediate = cfg.d_intermediate
        n_layer = cfg.n_layer
        ssm_cfg = cfg.ssm_cfg
        attn_layer_idx = cfg.attn_layer_idx
        attn_cfg = cfg.attn_cfg
        rms_norm = cfg.rms_norm
        norm_epsilon = cfg.norm_epsilon
        residual_in_fp32 = cfg.residual_in_fp32
        fused_add_norm = cfg.fused_add_norm
        llm_init_cfg = cfg.llm_init_cfg
        use_llm_init = cfg.use_llm_init

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.input_proj = input_proj

        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList([
            create_block(
                d_model=d_model,
                d_intermediate=d_intermediate,
                ssm_cfg=ssm_cfg,
                attn_layer_idx=attn_layer_idx,
                attn_cfg=attn_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                **factory_kwargs,
            )
            for i in range(n_layer)
        ])

        self.norm_f = (RMSNorm if rms_norm else nn.LayerNorm)(d_model, eps=norm_epsilon, **factory_kwargs)

        if use_llm_init:
            self.apply(
                partial(
                    _init_weights,
                    n_layer=n_layer,
                    **(llm_init_cfg if llm_init_cfg is not None else {}),
                    n_residuals_per_layer=(1 if d_intermediate == 0 else 2),
                )
            )

    def forward(self, x, inference_params=None, **mixer_kwargs):
        """
        Args:
            x: whatever input_proj expects:
                - LLM: (batch, seq) int
                - Timeseries: (batch, seq, d_input) float
        Returns:
            hidden_states: (batch, seq, d_model)
        """
        hidden_states = self.input_proj(x)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params, **mixer_kwargs)


        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        
        return hidden_states

class SequenceModel(nn.Module):
    def __init__(self, cfg: MixerConfig, device, dtype):
        super().__init__()
        self.cfg = cfg

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, device=None, dtype=None):
        cfg_file = os.path.join(pretrained_model_path, "config.json")
        model_file = os.path.join(pretrained_model_path, "pytorch_model.bin")

        with open(cfg_file, "r") as f:
            cfg_dict = json.load(f)

        cfg = MixerConfig(**cfg_dict)
        model = cls(cfg=cfg, device=device, dtype=dtype)
        state_dict = torch.load(model_file, map_location=device)
        load_result = model.load_state_dict(state_dict, strict=False)

        if len(load_result.missing_keys) > 0:
            print(f"Warning: Missing keys: {load_result.missing_keys}")
        if len(load_result.unexpected_keys) > 0:
            print(f"Warning: Unexpected keys: {load_result.unexpected_keys}")

        return model
        
    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        cfg_path = os.path.join(save_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(self.cfg.__dict__, f, indent=4)

class LinearSequenceModel(SequenceModel):
    """
    A sequence-to-sequence model built around a Mixer backbone.

    Architecture:
        Input → Linear(d_input → d_model)
              → MixerBackbone
              → Linear(d_model → d_output)
              → Output (batch, seq, d_output)
    """
    def __init__(
        self,
        cfg: MixerConfig,
        device=None,
        dtype=None,
    ):
        super().__init__(cfg=cfg)
        factory_kwargs = {"device": device, "dtype": dtype}

        d_model = cfg.d_model
        d_input = cfg.d_input
        d_output = cfg.d_output

        input_proj = nn.Linear(d_input, d_model, **factory_kwargs)
        self.decoder = nn.Linear(d_model, d_output, **factory_kwargs)
        self.backbone = MixerBackbone(
            cfg=cfg,
            input_proj=input_proj,
            **factory_kwargs,
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq, d_input)

        Returns:
            y_hat: (batch, seq, d_output)
        """
        x = x.to(self.backbone.input_proj.weight.dtype)
        hidden_states = self.backbone(x)
        y_hat = self.decoder(hidden_states)
        return y_hat

class EmbeddingSequenceModel(SequenceModel):
    """
    A sequence-to-sequence model built around a Mixer backbone.
    The input projection layer is an embedding layer, making this
    overall model akin to modern LLM architectures. It returns
    logits.

    Architecture:
        Input → Embedding
              → MixerBackbone
              → Linear(d_model → d_output)
              → Output (batch, seq, vocab_size)
    """
    def __init__(
        self,
        cfg: MixerConfig,
        device=None,
        dtype=None,
    ):
        super().__init__(cfg=cfg)
        factory_kwargs = {"device": device, "dtype": dtype}

        d_model = cfg.d_model
        use_llm_init = cfg.use_llm_init
        llm_init_cfg = cfg.llm_init_cfg
        n_layer = cfg.n_layer
        vocab_size = cfg.vocab_size
        pad_vocab_multiple = cfg.pad_vocab_multiple
        tie_embeddings = cfg.tie_embeddings

        if vocab_size % pad_vocab_multiple != 0:
            vocab_size += pad_vocab_multiple - (vocab_size % pad_vocab_multiple)
            self.cfg.vocab_size = vocab_size # sync back so when saving model, vocab_size is correct

        input_proj = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        self.backbone = MixerBackbone(
            cfg=cfg,
            input_proj=input_proj,
            **factory_kwargs,
        )

        if use_llm_init:
            self.apply(
                partial(
                    _init_weights,
                    n_layer=n_layer,
                    **(llm_init_cfg if llm_init_cfg is not None else {})
                )
            )

        if tie_embeddings:
            self.decoder.weight = self.backbone.input_proj.weight

    def forward(
        self,
        tokens,
        inference_params=None,
        num_last_tokens: int = 0,
        **mixer_kwargs,
    ):
        """
        Args:
            tokens: (batch, seq) of type long

        Returns:
            logits: (batch, seq, vocab_size)
        """
        hidden_states = self.backbone(tokens, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        logits = self.decoder(hidden_states)
        return logits
