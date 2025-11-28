# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial

import torch
import torch.nn as nn
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.utils.generation import GenerationMixin


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


class MixerModel(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_intermediate: int,
        d_input: int,
        n_layers: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        rms_norm: bool = False,
        initializer_cfg=None,
        norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        use_llm_init: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.encoder = nn.Linear(d_input, d_model, **factory_kwargs)
        
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
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
                for i in range(n_layers)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        if use_llm_init:
            self.apply(
                partial(
                    _init_weights,
                    n_layer=n_layers,
                    **(initializer_cfg if initializer_cfg is not None else {}),
                    n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
                )
            )

    def forward(self, x, inference_params=None, **mixer_kwargs):
        """
        Args:
            x: (batch, seq, d_input)

        Returns:
            hidden_states: (batch, seq, d_model)
        """
        x = x.to(self.encoder.weight.dtype)
        hidden_states = self.encoder(x)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
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


class MambaQuantileHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        d_input,
        d_model,
        quantiles,
        n_layer,
        d_intermediate,
        ssm_cfg,
        attn_layer_idx,
        attn_cfg,
        rms_norm,
        residual_in_fp32,
        fused_add_norm,
        initializer_cfg=None,
        use_llm_init: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        self.backbone = MixerModel(
            d_input=d_input,
            d_model=d_model,
            n_layers=n_layer,
            d_intermediate=d_intermediate,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            use_llm_init=use_llm_init, 
            **factory_kwargs,
        )
        self.decoder = nn.Linear(d_model, len(quantiles), **factory_kwargs)

        if use_llm_init:
            # Initialize weights and apply final processing
            self.apply(
                partial(
                    _init_weights,
                    n_layer=n_layer,
                    **(initializer_cfg if initializer_cfg is not None else {}),
                )
            )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, x, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        num_last_tokens: if > 0, only return the logits for the last n tokens

        Returns:
            logits: (batch, num_last_tokens, quantiles)
        """
        hidden_states = self.backbone(x, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        logits = self.decoder(hidden_states)
        return logits

class QuantileRegressionLoss(nn.Module):
    """
    Chronos-2 quantile regression loss (Eq. 4) specialized for D = 1.
    pred:   (B, H, Q)
    target: (B, H)
    mask:   (B, H) optional
    """
    
    def __init__(self, quantiles, device, dtype):
        super().__init__()
        self.register_buffer("quantiles", torch.as_tensor(quantiles, device=device, dtype=dtype))

    def forward(
        self,
        pred,          # (B, H, Q)
        target,        # (B, H)
    ):
        error = target - pred  # (B, H, Q)

        # Make quantiles broadcastable: (Q,) -> (1,1,Q)
        q = self.quantiles.view(1, 1, -1)

        # Pinball components
        loss_pos = torch.clamp(error, min=0.0)     # max(z - z_hat_q, 0)
        loss_neg = torch.clamp(-error, min=0.0)    # max(z_hat_q - z, 0)
        loss = q * loss_pos + (1 - q) * loss_neg   # (B, H, Q)
        loss = loss.sum(dim=-1).mean()
        return loss
