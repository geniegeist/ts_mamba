import torch
from tqdm.auto import tqdm

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
    eval_mae = 0
    eval_zero_mae = 0
    eval_pos_mae = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(loader))
        for batch_idx, data in pbar:
            obs, targets = data["context"], data["target"]
            obs, targets = obs.to(device), targets.to(device)
            preds = model(obs)
            loss = criterion(preds, targets)
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
            eval_mae += mae.item()
            eval_zero_mae += zero_mae.item()
            eval_pos_mae += pos_mae.item()

            pbar.set_description(
                    'Val Batch Idx: (%d/%d) | Val loss: %.3f | MAE: %.3f | zero-MAE: %.3f | pos-MAE: %.3f' %
                    (batch_idx, len(loader), eval_loss/(batch_idx+1), eval_mae/(batch_idx+1), eval_zero_mae/(batch_idx+1), eval_pos_mae/(batch_idx+1))
            )

        avg_eval_loss = eval_loss / len(loader)
        avg_eval_mae = eval_mae / len(loader)
        avg_eval_zero_mae = eval_zero_mae / len(loader)
        avg_eval_pos_mae = eval_pos_mae / len(loader)

    return {
        "loss": avg_eval_loss,
        "mae": avg_eval_mae,
        "zero_mae": avg_eval_zero_mae,
        "pos_mae": avg_eval_pos_mae,
    }

