import logging
import os
from collections import namedtuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm.auto import tqdm

from ts_mamba.common import get_dist_info, setup_default_logging


setup_default_logging()
logger = logging.getLogger(__name__)
def log0(msg):
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(msg)


def evaluate_forecast_base(model, criterion, loader, device, point_forecast_extractor=None):
    """
    Args:
        point_forecast_extractor: A function/lambda to extract the point forecast from the 
                                  the raw model ouput. Default: Identity (x -> x)
    """
    model.eval()
    is_ddp, rank, _, _ = get_dist_info()

    if point_forecast_extractor is None:
        point_forecast_extractor = lambda x: x

    # Shape (6 metrics, 2 cols (loss_sum, count))
    # Indices: 0:Crit 1:MSE 2:L1 3:L1Zero 4:L1Gt0 5:L1Gt1
    stats = torch.zeros(6, 2, device=device)

    iterator = enumerate(loader)
    if rank == 0:
        iterator = tqdm(iterator, total=len(loader), desc="Evaluating")

    with torch.no_grad():
        for batch_idx, batch in iterator:
            obs, targets = batch["context"].to(device), batch["target"].to(device)

            if targets.dim() == 2:
                targets = targets.unsqueeze(-1)

            raw_preds = model(obs) # (batch, seq, d_output)
            preds = point_forecast_extractor(raw_preds) # (batch, seq, 1)
            preds_last = preds[:, -1, 0] # (batch, )
            targets_last = targets[:, -1, 0] # (batch, )

            batch_size = preds.size(0)

            # compute overall loss
            stats[0, 0] += criterion(raw_preds, targets) * batch_size
            stats[0, 1] += batch_size

            stats[1, 0] += F.mse_loss(preds_last, targets_last, reduction="sum")
            stats[1, 1] += batch_size

            stats[2, 0] += F.l1_loss(preds_last, targets_last, reduction="sum")
            stats[2, 1] += batch_size

            # masked loss
            masks = [targets_last == 0, targets_last > 0, targets_last > 1]
            for i, mask in enumerate(masks, start=3):
                if mask.any():
                    stats[i, 0] += F.l1_loss(preds_last[mask], targets_last[mask], reduction="sum")
                    stats[i, 1] += mask.sum()
    if is_ddp:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    avgs = torch.zeros(6, device=device)
    has_data = stats[:,1] > 0
    avgs[has_data] = stats[has_data, 0] / stats[has_data, 1]

    log0(f"Val result: {avgs.cpu()}")

    ValidationOutput = namedtuple("ValidationOutput", ["criterion", "mse", "l1", "l1_zero", "l1_gt0", "l1_gt1"])
    return ValidationOutput(*avgs.tolist())


def evaluate_point_forecast_model(model, criterion, loader, device):
    def extractor(x): return x
    return evaluate_forecast_base(model=model, criterion=criterion, loader=loader, device=device, point_forecast_extractor=extractor)

def evaluate_quantile_model(model, criterion, quantile_idx, loader, device):
    def quantile_to_point(x): return x[..., quantile_idx].unsqueeze(-1)
    return evaluate_forecast_base(model=model, criterion=criterion, loader=loader, device=device, point_forecast_extractor=quantile_to_point)

def evaluate_token_model(model, criterion, loader, device):
    def logits_to_expected_value(logits):
        probs = F.softmax(logits, dim=-1) # (batch, seq, vocab_size)
        vocab_indices = torch.arange(probs.size(-1), device=logits.device)
        expected = (probs*vocab_indices).sum(dim=-1) # (batch,seq)
        return expected.unsqueeze(-1) # (batch, seq, 1)

    def loss_wrapper(preds, targets):
        # preds: (Batch, Seq, Vocab) -> (Batch*Seq, Vocab)
        # targets: (Batch, Seq, 1) -> (Batch*Seq)
        return criterion(preds.reshape(-1, preds.size(-1)), targets.reshape(-1).long())

    return evaluate_forecast_base(model=model, criterion=loss_wrapper, loader=loader, device=device, point_forecast_extractor=logits_to_expected_value)

