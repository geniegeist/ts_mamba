from functools import partial

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
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

class TimeseriesModel(nn.Module):

    def __init__(
        self,
        d_input: int,
        d_model: int = 64,
        d_intermediate: int = 0,
        n_layers: int = 4,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        rms_norm: bool = False,
        norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Encoder
        self.encoder = nn.Linear(d_input, d_model).to(device=device, dtype=dtype)

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
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layers)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        # Final output head for point prediction
        self.decoder = nn.Linear(d_model, 1).to(device=device, dtype=dtype)

    def forward(self, x):
        """
        x: (batch, seq_len, d_input)
        returns: point predictions (batch, seq_len, 1)
        """
        x = x.to(self.encoder.weight.dtype)
        x = self.encoder(x)
        residual = None

        for layer in self.layers:
            x, residual = layer(
                x, residual
            )

        residual = (x + residual) if residual is not None else x
        x = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

        y_hat = self.decoder(x)
        return y_hat  # (B, L, 1)

    def predict(self, x, last_only=False):
        """Make point predictions without sampling"""
        self.eval()
        with torch.no_grad():
            y_hat = self.forward(x)
            if last_only:
                return y_hat[:, -1]  # (B, 1)
            return y_hat  # (B, L, 1)


class LastKRMSELoss(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K

    def forward(self, preds, targets):
        # preds/targets shape: (batch, timesteps, ...)
        # Slice last K timesteps
        preds_lastK = preds[:, -self.K:]
        targets_lastK = targets[:, -self.K:]

        # Compute mean MSE over these last K steps only
        mse = torch.mean((preds_lastK - targets_lastK) ** 2)

        return torch.sqrt(mse)

class RMSELoss(nn.Module):
    """Root Mean Square Error Loss"""
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        return torch.sqrt(torch.mean((preds[:,-1] - targets[:,-1]) ** 2))
        #return torch.sqrt(torch.mean((preds - targets) ** 2))

class MAELoss(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.loss = torch.nn.L1Loss()
        self.k = k

    def forward(self, preds, targets):
        return self.loss(preds[-self.k:], targets[-self.k:])

class WeightedRMSELoss(nn.Module):
    def __init__(self, decay=0.99):
        super().__init__()
        self.decay = decay
        self._cached_L = None
        self._cached_weights = None

    def _get_weights(self, L, device):
        # compute weights only if sequence length changes
        if self._cached_L != L:
            weights = torch.pow(self.decay, torch.arange(L - 1, -1, -1, device=device))
            weights = weights / weights.sum()
            self._cached_weights = weights
            self._cached_L = L
        return self._cached_weights

    def forward(self, preds, targets):
        B, L, D = preds.shape
        device = preds.device

        weights = self._get_weights(L, device).view(1, L, 1)

        se = (preds - targets) ** 2
        weighted_mse = (se * weights).sum() / weights.sum()  # 🔑 fix here

        return torch.sqrt(weighted_mse)
