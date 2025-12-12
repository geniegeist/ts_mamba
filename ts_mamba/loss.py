import torch
import torch.nn as nn

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
