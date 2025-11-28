import torch
from tqdm.auto import tqdm

from ts_mamba.model import RMSELoss


def evaluate_llm(model, criterion, loader, device) -> dict[str, float]:
    eval_loss = 0.0
    
    # ---------- POINT METRIC ACCUMULATORS ----------
    squared_err_sum = 0.0
    abs_err_sum = 0.0
    total_count = 0
    
    # ---------- DISTRIBUTIONAL METRIC ACCUMULATORS ----------
    crps_sum = 0.0

    pinball_10 = 0.0
    pinball_50 = 0.0
    pinball_90 = 0.0

    coverage_hits = 0
    interval_count = 0
    interval_width_sum = 0.0

    # Define helper functions outside the loop or inline efficiently
    def percentile_from_cdf(cdf, p):
        # cdf: (batch, vocab)
        # p_tensor: must be (batch, 1) for row-wise searchsorted
        p_tensor = torch.full((cdf.size(0), 1), p, device=cdf.device)
        
        # Returns (batch, 1), so we squeeze to get (batch,)
        indices = torch.searchsorted(cdf, p_tensor, right=True)
        return indices.squeeze(-1).float()

    def pinball(y, q, tau):
        return torch.maximum(tau * (y - q), (1 - tau) * (q - y))

    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader))
        for batch_idx, data in pbar:
            obs, targets = data["context"], data["target"]
            obs = obs.squeeze(-1).to(device)
            targets = targets[:, -1].reshape(-1).to(device)  # (batch,)

            # model forward
            logits = model(obs, num_last_tokens=1).logits  # (batch,1,vocab)
            logits = logits.squeeze(1)                     # (batch,vocab)

            # ===== 1. Cross Entropy Loss =====
            loss = criterion(logits, targets)
            eval_loss += loss.item()

            # ===== 2. Probability distribution =====
            probs = torch.softmax(logits, dim=-1)         # (batch,vocab)
            vocab_size = probs.size(-1)
            class_vals = torch.arange(vocab_size, device=device, dtype=probs.dtype)

            # ===== 3. Expected value for RMSE/MAE =====
            expected = (probs * class_vals).sum(dim=-1)   # (batch,)
            err = expected - targets

            squared_err_sum += torch.sum(err ** 2).item()
            abs_err_sum += torch.sum(torch.abs(err)).item()
            total_count += targets.numel()

            # =========================================================
            #                     DISTRIBUTIONAL METRICS
            # =========================================================

            # ===== A. CRPS for discrete distributions =====
            # CRPS = sum_k (CDF_pred[k] - 1{y <= k})^2
            cdf_pred = torch.cumsum(probs, dim=-1)        # (batch,vocab)
            y_expanded = targets.unsqueeze(-1)            # (batch,1)
            indicator = (class_vals.unsqueeze(0) >= y_expanded).float()
            crps = torch.sum((cdf_pred - indicator)**2, dim=-1)
            crps_sum += crps.sum().item()

            # ===== B. Quantile extraction using searchsorted =====
            # Now using the corrected logic (input shape [Batch, 1])
            q10 = percentile_from_cdf(cdf_pred, 0.10)
            q50 = percentile_from_cdf(cdf_pred, 0.50)
            q90 = percentile_from_cdf(cdf_pred, 0.90)

            # ===== C. Pinball loss =====
            pinball_10 += pinball(targets, q10, 0.10).sum().item()
            pinball_50 += pinball(targets, q50, 0.50).sum().item()
            pinball_90 += pinball(targets, q90, 0.90).sum().item()

            # ===== D. Coverage (P10–P90 interval) =====
            covered = ((targets >= q10) & (targets <= q90)).float()
            coverage_hits += covered.sum().item()
            interval_count += targets.numel()

            # ===== E. Interval width =====
            interval_width_sum += (q90 - q10).sum().item()

            # ---- progress bar ----
            curr_rmse = (squared_err_sum / total_count) ** 0.5
            curr_mae  = abs_err_sum / total_count

            pbar.set_description(
                f"Val ({batch_idx+1}/{len(loader)}) | "
                f"Loss: {eval_loss/(batch_idx+1):.3f} | "
                f"RMSE: {curr_rmse:.3f} | MAE: {curr_mae:.3f}"
            )

    # ---------- FINAL METRICS ----------
    avg_loss  = eval_loss / len(loader)
    final_rmse = (squared_err_sum / total_count) ** 0.5
    final_mae  = abs_err_sum / total_count

    final_crps = crps_sum / total_count

    final_pin_10 = pinball_10 / total_count
    final_pin_50 = pinball_50 / total_count
    final_pin_90 = pinball_90 / total_count

    final_coverage = coverage_hits / interval_count
    final_interval_width = interval_width_sum / interval_count

    # ---------- RETURN EVERYTHING ----------
    return {
        # point forecasts
        "loss": avg_loss,
        "rmse": final_rmse,
        "mae": final_mae,

        # distributional metrics
        "crps": final_crps,
        "pinball_10": final_pin_10,
        "pinball_50": final_pin_50,
        "pinball_90": final_pin_90,
        "coverage_10_90": final_coverage,
        "interval_width_10_90": final_interval_width,
    }

def evaluate_llm2(model, criterion, loader, device) -> dict[str, float]:
    eval_loss = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(loader))
        for batch_idx, data in pbar:
            obs, targets = data["context"], data["target"]
            obs, targets = obs.squeeze(-1).to(device), targets[:,-1].reshape(-1).to(device)
            logits = model(obs, num_last_tokens=1).logits # (batch, 1, vocab_size)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            eval_loss += loss.item()

            pbar.set_description(
                'Val Batch Idx: (%d/%d) | Val loss: %.3f' %
                    (batch_idx, len(loader), eval_loss/(batch_idx+1))
            )

        avg_eval_loss = eval_loss / len(loader)

    return {
        "loss": avg_eval_loss,
    }

def evaluate_model(model, criterion, loader, device) -> dict[str, float]:
    eval_loss = 0
    eval_mae = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(loader))
        for batch_idx, data in pbar:
            obs, targets = data["context"], data["target"]
            obs, targets = obs.to(device), targets.to(device)
            preds = model(obs)
            mu, _ = preds
            loss = criterion(preds, targets)
            mae = torch.mean(torch.abs(mu - targets))

            eval_loss += loss.item()
            eval_mae += mae.item()

            pbar.set_description(
                'Val Batch Idx: (%d/%d) | Val loss: %.3f | MAE: %.3f' %
                    (batch_idx, len(loader), eval_loss/(batch_idx+1), eval_mae/(batch_idx+1))
            )

        avg_eval_loss = eval_loss / len(loader)
        avg_eval_mae = eval_mae / len(loader)

    return {
        "loss": avg_eval_loss,
        "mae": avg_eval_mae,
    }

def evaluate_model_rmse(model, criterion, loader, device) -> dict[str, float]:
    eval_loss = 0
    eval_rmse = 0
    eval_mae = 0
    eval_zero_mae = 0
    eval_pos_mae = 0

    rmse_crit = RMSELoss()

    with torch.no_grad():
        pbar = tqdm(enumerate(loader))
        for batch_idx, data in pbar:
            obs, targets = data["context"], data["target"]
            obs, targets = obs.to(device), targets.to(device)
            preds = model(obs)
            loss = criterion(preds, targets)
            rmse = rmse_crit(preds, targets)
            mae = torch.mean(torch.abs(preds[:,-1] - targets[:,-1]))

            # Make mask for samples whose last target value == 0
            zero_mask = (targets[:, -1, 0] == 0)
            pos_mask = (targets[:, -1, 0] > 0)

            # Select only the last timestep for preds and targets
            preds_last = preds[:, -1, 0]
            targets_last = targets[:, -1, 0]

            # Safe MAE computation
            if zero_mask.any():
                zero_mae = torch.mean(torch.abs(preds_last[zero_mask] - targets_last[zero_mask]))
            else:
                zero_mae = torch.tensor(0.0, device=preds.device)

            if pos_mask.any():
                pos_mae = torch.mean(torch.abs(preds_last[pos_mask] - targets_last[pos_mask]))
            else:
                pos_mae = torch.tensor(0.0, device=preds.device)

            eval_loss += loss.item()
            eval_rmse += rmse.item()
            eval_mae += mae.item()
            eval_zero_mae += zero_mae.item()
            eval_pos_mae += pos_mae.item()

            pbar.set_description(
                'Val Batch Idx: (%d/%d) | Loss: %.3f | RMSE: %.3f | MAE: %.3f | zero-MAE: %.3f | pos-MAE: %.3f' %
                    (batch_idx, len(loader), eval_loss/(batch_idx+1), eval_rmse/(batch_idx+1), eval_mae/(batch_idx+1), eval_zero_mae/(batch_idx+1), eval_pos_mae/(batch_idx+1))
            )

        avg_eval_loss = eval_loss / len(loader)
        avg_eval_loss_last = eval_rmse / len(loader)
        avg_eval_mae = eval_mae / len(loader)
        avg_eval_zero_mae = eval_zero_mae / len(loader)
        avg_eval_pos_mae = eval_pos_mae / len(loader)

    return {
        "loss": avg_eval_loss,
        "rmse": avg_eval_loss_last,
        "mae": avg_eval_mae,
        "zero_mae": avg_eval_zero_mae,
        "pos_mae": avg_eval_pos_mae,
    }

def evaluate_model_quantile(model, criterion, quantile_idx, loader, device) -> dict[str, float]:
    eval_loss = 0
    eval_rmse = 0
    eval_mae = 0
    eval_zero_mae = 0
    eval_pos_mae = 0
    eval_pos2_mae = 0

    rmse_crit = RMSELoss()

    with torch.no_grad():
        pbar = tqdm(enumerate(loader))
        for batch_idx, data in pbar:
            obs, targets = data["context"], data["target"]
            obs, targets = obs.to(device), targets.to(device)
            preds = model(obs)
            loss = criterion(preds, targets)

            point_preds = preds[:,:,quantile_idx].unsqueeze(-1)
            rmse = rmse_crit(point_preds, targets)
            mae = torch.mean(torch.abs(point_preds[:,-1] - targets[:,-1]))

            # Make mask for samples whose last target value == 0
            zero_mask = (targets[:, -1, 0] == 0)
            pos_mask = (targets[:, -1, 0] > 0)
            pos2_mask = (targets[:, -1, 0] > 1)

            # Select only the last timestep for preds and targets
            preds_last = point_preds[:, -1, 0]
            targets_last = targets[:, -1, 0]

            # Safe MAE computation
            if zero_mask.any():
                zero_mae = torch.mean(torch.abs(preds_last[zero_mask] - targets_last[zero_mask]))
            else:
                zero_mae = torch.tensor(0.0, device=preds.device)

            if pos_mask.any():
                pos_mae = torch.mean(torch.abs(preds_last[pos_mask] - targets_last[pos_mask]))
            else:
                pos_mae = torch.tensor(0.0, device=preds.device)

            if pos2_mask.any():
                pos2_mae = torch.mean(torch.abs(preds_last[pos2_mask] - targets_last[pos2_mask]))
            else:
                pos2_mae = torch.tensor(0.0, device=preds.device)

            eval_loss += loss.item()
            eval_rmse += rmse.item()
            eval_mae += mae.item()
            eval_zero_mae += zero_mae.item()
            eval_pos_mae += pos_mae.item()
            eval_pos2_mae += pos2_mae.item()

            pbar.set_description(
                'Val Batch Idx: (%d/%d) | Loss: %.3f | RMSE: %.3f | MAE: %.3f | zero-MAE: %.3f | pos-MAE: %.3f | >1-MAE: %.3f' %
                    (batch_idx, len(loader), eval_loss/(batch_idx+1), eval_rmse/(batch_idx+1), eval_mae/(batch_idx+1), eval_zero_mae/(batch_idx+1), eval_pos_mae/(batch_idx+1), eval_pos2_mae/(batch_idx+1))
            )

        avg_eval_loss = eval_loss / len(loader)
        avg_eval_loss_last = eval_rmse / len(loader)
        avg_eval_mae = eval_mae / len(loader)
        avg_eval_zero_mae = eval_zero_mae / len(loader)
        avg_eval_pos_mae = eval_pos_mae / len(loader)
        avg_eval_pos2_mae = eval_pos2_mae / len(loader)

    return {
        "loss": avg_eval_loss,
        "rmse": avg_eval_loss_last,
        "mae": avg_eval_mae,
        "zero_mae": avg_eval_zero_mae,
        "pos_mae": avg_eval_pos_mae,
        "pos2_mae": avg_eval_pos2_mae,
    }

