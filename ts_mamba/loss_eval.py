import torch
from tqdm.auto import tqdm

from ts_mamba.model import RMSELoss


def evaluate_llm(model, criterion, loader, device) -> dict[str, float]:
    eval_loss = 0.0
    
    # accumulators for point metrics
    squared_err_sum = 0.0
    abs_err_sum = 0.0
    total_count = 0
    
    # --- DISTRIBUTIONAL METRIC ACCUMULATORS ---
    crps_sum = 0.0
    pinball_10 = 0.0
    pinball_50 = 0.0
    pinball_90 = 0.0
    coverage_50_90_hits = 0
    interval_count = 0
    interval_width_sum = 0.0

    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader))
        for batch_idx, data in pbar:
            obs, targets = data["context"], data["target"]
            obs = obs.squeeze(-1).to(device)
            targets = targets[:, -1].reshape(-1).to(device)  # (batch,)

            logits = model(obs, num_last_tokens=1).logits     # (batch,1,vocab)
            logits = logits.squeeze(1)                        # (batch,vocab)

            # ----- LOSS -----
            loss = criterion(logits, targets)
            eval_loss += loss.item()

            # ----- PROBABILITIES -----
            probs = torch.softmax(logits, dim=-1)             # (batch,vocab)
            vocab_size = probs.size(-1)

            # ----- EXPECTED VALUE -----
            class_vals = torch.arange(vocab_size, device=device, dtype=probs.dtype)
            expected = (probs * class_vals).sum(dim=-1)       # (batch,)

            # ----- POINT ERRORS (RMSE, MAE) -----
            err = expected - targets
            squared_err_sum += torch.sum(err ** 2).item()
            abs_err_sum += torch.sum(torch.abs(err)).item()
            total_count += targets.numel()

            # ------------------------------------------------------------------
            #                      DISTRIBUTIONAL METRICS
            # ------------------------------------------------------------------

            # ==== 1. Discrete CRPS ====
            # CRPS = sum_k (CDF_pred[k] - 1{y <= k})^2
            cdf_pred = torch.cumsum(probs, dim=-1)  # (batch,vocab)
            y_expanded = targets.unsqueeze(-1)      # (batch,1)
            indicator = (class_vals.unsqueeze(0) >= y_expanded).float()  # (batch,vocab)
            crps = torch.sum((cdf_pred - indicator)**2, dim=-1)          # (batch,)
            crps_sum += crps.sum().item()

            # ==== 2. Quantiles (10th, 50th, 90th) ====
            # Discrete quantiles using CDF crossing
            def percentile_from_cdf(cdf, p):
                return torch.argmax(cdf >= p, dim=-1).float()

            q10 = percentile_from_cdf(cdf_pred, 0.10)
            q50 = percentile_from_cdf(cdf_pred, 0.50)
            q90 = percentile_from_cdf(cdf_pred, 0.90)

            # ==== 3. Pinball Loss for each quantile ====
            def pinball(y, q, tau):
                return torch.maximum(tau * (y - q), (1 - tau) * (q - y))

            pinball_10 += pinball(targets, q10, 0.10).sum().item()
            pinball_50 += pinball(targets, q50, 0.50).sum().item()
            pinball_90 += pinball(targets, q90, 0.90).sum().item()

            # ==== 4. Coverage (does actual fall in P10–P90 interval?) ====
            covered = ((targets >= q10) & (targets <= q90)).float()
            coverage_50_90_hits += covered.sum().item()
            interval_count += targets.numel()

            # ==== 5. Interval Width ====
            interval_width_sum += (q90 - q10).sum().item()

            # ---- DISPLAY ----
            curr_rmse = (squared_err_sum / total_count) ** 0.5
            curr_mae  = abs_err_sum / total_count
            
            pbar.set_description(
                f"Val {batch_idx+1}/{len(loader)} | "
                f"Loss: {eval_loss/(batch_idx+1):.3f} | "
                f"RMSE: {curr_rmse:.3f} | MAE: {curr_mae:.3f}"
            )

    # Final point metrics
    avg_loss  = eval_loss / len(loader)
    final_rmse = (squared_err_sum / total_count) ** 0.5
    final_mae  = abs_err_sum / total_count

    # Final distributional metrics
    final_crps = crps_sum / total_count
    final_pinball10 = pinball_10 / total_count
    final_pinball50 = pinball_50 / total_count
    final_pinball90 = pinball_90 / total_count
    final_coverage = coverage_50_90_hits / interval_count
    final_interval_width = interval_width_sum / interval_count

    return {
        # point metrics
        "loss": avg_loss,
        "rmse": final_rmse,
        "mae": final_mae,

        # distributional metrics
        "crps": final_crps,
        "pinball_10": final_pinball10,
        "pinball_50": final_pinball50,
        "pinball_90": final_pinball90,
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

