"""
Utilities for loading model and optim state.
"""
import logging
import os
import json
import torch
from ts_mamba.common import setup_default_logging


setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(message)

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, scheduler_data, meta_data, rank):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to dir: {model_path}")

        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=4)
        logger.info(f"Saved metadata to: {meta_path}")

    # each rank must save its own optimizer
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")

        scheduler_path = os.path.join(checkpoint_dir, f"scheduler_{step:06d}_rank{rank:d}.pt")
        torch.save(scheduler_data, scheduler_path)
        logger.info(f"Saved scheduler state to: {scheduler_path}")

def load_checkpoint(checkpoint_dir, step, device, dtype, load_optimizer=False, rank=0):
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)

    optimizer_data, scheduler_data = None, None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
        scheduler_path = os.path.join(checkpoint_dir, f"scheduler_{step:06d}_rank{rank:d}.pt")
        scheduler_data = torch.load(scheduler_path, map_location=device)

    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, scheduler_data, meta_data
