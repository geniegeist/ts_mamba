import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import AdamW
import wandb
import polars as pl
import yaml

import os
import argparse

from ts_mamba.model import TimeseriesModel, RMSELoss
from ts_mamba.common import DummyWandb
from ts_mamba.loss_eval import evaluate_model_rmse
from ts_mamba.split import temporal_train_val_split, spatiotemporal_subset
from ts_mamba.optimizer import WarmupCosineLR 
from ts_mamba.train_util import plot_forecast_vs_truth_rmse
from ts_mamba.dataset import TileTimeSeriesDataset
from ts_mamba.sampler import InfiniteRandomSampler


from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='PyTorch DeepAR-S4 Training Script')

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
parser.add_argument('--config', type=str, help='Path to YAML config file')

# ----------------------------------------------------------------------
# Optimizer
# ----------------------------------------------------------------------
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for non-S4 layers')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer (Adam)')
parser.add_argument('--warm_restart', type=int, help='Enable learning rate scheduler')
parser.add_argument('--warmup_steps', type=int, help='')
parser.add_argument('--eta_min', type=float, default=0, help='Minimum learning rate for cosine annealing scheduler')

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
parser.add_argument('--data_path', type=str, help='Path to main dataset file')
parser.add_argument('--meta_path', type=str, help='Path to dataset metadata file')

# ----------------------------------------------------------------------
# Dataloader
# ----------------------------------------------------------------------
parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader worker threads')

# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
parser.add_argument('--n_layers', type=int, default=4, help='Number of model layers')
parser.add_argument('--d_model', type=int, default=128, help='Number of parallel S4 channels')
parser.add_argument('--d_intermediate', type=int, default=0, help='')

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
parser.add_argument('--total_steps', type=int, default=20, help='Number of training epochs')
parser.add_argument('--samples_per_epoch', type=int, default=10000, help='Number of samples per epoch')
parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
parser.add_argument('--val_split_date', type=str, default='2025-01-01', help='Validation split date (YYYY-MM-DD)')

# ----------------------------------------------------------------------
# Checkpointing & Logging
# ----------------------------------------------------------------------
parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
parser.add_argument('--resume_model_only', action='store_true', help='Resume training from checkpoint')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/ckpt.pth', help='Path to checkpoint file')
parser.add_argument('--run', default="dummy", type=str, help='Run name for logging (e.g., wandb)')
parser.add_argument('--resume_from', type=str, help='run_id')

# ----------------------------------------------------------------------
# Features & Evaluation
# ----------------------------------------------------------------------
parser.add_argument('--eval_every', type=int, default=1, help='Evaluate model every N epochs')
parser.add_argument('--sample_every', type=int, default=1, help='Sample model every N epochs')
parser.add_argument('--save_every', type=int, default=100, help='Save model every N epochs')

# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
parser.add_argument('--model_tag', type=str, help='Custom tag for model output directory')

args = parser.parse_args()

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Detected device: {device}')

# Config
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

path_to_data = args.data_path if args.data_path else config["data"]["data"]
path_to_meta = args.meta_path if args.meta_path else config["data"]["meta"]

with open(path_to_meta, "r") as f:
    meta = yaml.safe_load(f)

val_split_date = args.val_split_date if args.val_split_date else config["validation"]["split_date"]
mini_val_dates = config["mini_validation"]["dates"]
mini_val_tiles = config["mini_validation"]["tiles"]
sample_dates = config["sample"]["dates"]
sample_tiles = config["sample"]["tiles"]
context_length = int((60 / config["time_bin_in_min"] * 24) * config["context_window_in_days"])
eval_every = args.eval_every if args.eval_every else config["eval_every"]
sample_every = args.sample_every if args.sample_every else config["sample_every"]
save_every = args.save_every if args.save_every else config["save_every"]

d_model = args.d_model
d_intermediate = args.d_intermediate
n_layers = args.n_layers

# wandb logging init
use_dummy_wandb = args.run == "dummy"
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="ts_mamba_rmse", 
    name=args.run, 
    id=args.resume_from,
    resume="must" if args.resume_from else None,
    config={ 
        "data": {"meta": meta}, 
        "args": args, 
        "model": { 
            "d_model": d_model, 
            "d_intermediate": d_intermediate, 
            "n_layers": n_layers,
            "context_length": context_length 
        },
        "training": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "eta_min": args.eta_min,
            "total_steps": args.total_steps,
            "val_split_date": val_split_date,
        }
    }
)

# Data
print(f'==> Preparing data..')

df = pl.read_parquet(path_to_data)
train_df, val_df = temporal_train_val_split(df, meta, val_split_date)
mini_val_df = spatiotemporal_subset(df, meta, mini_val_dates, mini_val_tiles, context_window_in_days=config["context_window_in_days"])
sample_df = spatiotemporal_subset(df, meta, sample_dates, sample_tiles, context_window_in_days=config["context_window_in_days"])

train_dataset = TileTimeSeriesDataset(train_df, meta, context_length=context_length)
val_dataset = TileTimeSeriesDataset(val_df, meta, context_length=context_length)
mini_val_dataset = TileTimeSeriesDataset(mini_val_df, meta, context_length=context_length)
sample_dataset = TileTimeSeriesDataset(sample_df, meta, context_length=context_length)


# Dataloaders
def get_train_loader():
    return DataLoader(train_dataset, batch_size=args.batch_size, sampler=InfiniteRandomSampler(train_dataset), num_workers=args.num_workers)

val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers)
mini_val_loader = DataLoader(
    mini_val_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers)
sample_loader = DataLoader(
    sample_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers)

train_loader = get_train_loader()


# Model
print('==> Building model..')

d_input = len(meta["features"])
model = TimeseriesModel(
    d_input=d_input,
    d_model=d_model,
    d_intermediate=d_intermediate,
    n_layers=n_layers,
    device=device,
)

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True

min_eval_loss = float("inf")
best_step = -1
start_step = 0
samples_so_far = 0

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint_path)
    if args.resume_model_only:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])
        start_step = checkpoint['step']
        samples_so_far = checkpoint['samples_so_far']
        min_eval_loss = checkpoint['min_val_loss']
        best_step = checkpoint['best_step']


criterion = RMSELoss()
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for param_group in optimizer.param_groups:
    param_group.setdefault("initial_lr", args.lr)

scheduler = WarmupCosineLR(optimizer, warmup_steps=args.warmup_steps, total_steps=args.total_steps, last_epoch=start_step-1)

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

if not use_dummy_wandb:
    wandb_run.watch(model, log="all")

ema_beta = 0.9 # EMA decay factor
smooth_train_loss = 0
smooth_mae = 0

print('==> Start training..')
pbar = tqdm(range(start_step, args.total_steps), desc="Training")
for step in pbar:
    batch = next(iter(train_loader))
    last_step = step == args.total_steps - 1
    samples_so_far += args.batch_size
    model.eval()
    # once in a while: evaluate model
    if last_step or step % eval_every == 0:
        val_res = evaluate_model_rmse(
            model=model,
            criterion=criterion,
            loader=mini_val_loader,
            device=device,
        )

        if val_res["loss"] < min_eval_loss:
            min_eval_loss = val_res["loss"]
            best_step = step

        wandb_run.log({
            "samples_so_far": samples_so_far,
            "val_loss": val_res["loss"],
            "val_mae": val_res["mae"],
            "val_zero_mae": val_res["zero_mae"],
            "val_pos_mae": val_res["pos_mae"],
            "min_val_loss": min_eval_loss,
            "best_step": best_step,
        })

    # once in a while: sample from model
    if sample_every > 0 and (last_step or (step % sample_every == 0)):
        plot_forecast_vs_truth_rmse(
            model=model,
            loader=sample_loader,
            device=device,
            wandb_run=wandb_run,
            epoch=step,
        )

    # Training
    wandb_run.log({ "last_lr": scheduler.get_last_lr() })
    model.train()

    obs, targets = batch["context"], batch["target"]
    obs, targets = obs.to(device), targets.to(device)

    optimizer.zero_grad()

    preds = model(obs)

    loss = criterion(preds, targets)
    loss.backward()

    optimizer.step()
    scheduler.step()

    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))

    mae = torch.mean(torch.abs(preds[:,-1] - targets[:,-1]))
    smooth_mae = ema_beta * smooth_mae + (1 - ema_beta) * mae.item() # EMA the training loss
    debiased_smooth_mae = smooth_mae / (1 - ema_beta**(step + 1))

    pbar.set_description(
        'Step: (%d/%d) | Train loss: %.6f | MAE: %.6f' %
        (step, args.total_steps, debiased_smooth_loss, debiased_smooth_mae)
    )

    if step % 10 == 0:
        wandb_run.log({
            "samples_so_far": samples_so_far,
            "train_loss": debiased_smooth_loss,
            "train_mae": debiased_smooth_mae,
            "last_lr": scheduler.get_last_lr(),
        })

    if args.save_every >= 0 and step % args.save_every == 0:
        print("Save checkpoint")

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'step': step,
            'samples_so_far': samples_so_far,
            "min_val_loss": min_eval_loss,
            "best_step": best_step,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        ckpt_path = f'./checkpoint/ckpt_{args.run}_step{step}.pth'
        torch.save(state, ckpt_path)

        if not use_dummy_wandb:
            artifact = wandb.Artifact(f'{wandb_run.id}-artifact-step{step}', type='model')
            artifact.add_file(ckpt_path)
            wandb_run.log_artifact(artifact)


wandb_run.finish()
