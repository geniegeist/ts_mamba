import glob
import os
import random
import threading

import hydra
import polars as pl
import torch
import torch.backends.cudnn as cudnn
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from config import Config
from ts_mamba.common import DummyWandb
from ts_mamba.dataset import TileTimeSeriesDataset
from ts_mamba.loss_eval import evaluate_model_rmse
from ts_mamba.model import TimeseriesModel, RMSELoss
from ts_mamba.optimizer import WarmupCosineLR 
from ts_mamba.train_util import plot_forecast_vs_truth_rmse


config_store = ConfigStore.instance()
config_store.store(name="timeseries_deep_learning_config", node=Config)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: Config):
    print(config)

    use_wandb = config.wandb is not None
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project = config.wandb.project,
            name = config.wandb.name,
            id = config.wandb.id,
            mode = config.wandb.mode,
            job_type = config.wandb.job_type,
            config=OmegaConf.to_container(config, resolve=True),
        )
    else:
        wandb_run = DummyWandb()


    # ---- Load metadata before creating datasets ----
    with open(config.dataset.train.meta, "r") as f:
        train_meta = yaml.safe_load(f)

    with open(config.dataset.validation.meta, "r") as f:
        validation_meta = yaml.safe_load(f)

    with open(config.dataset.sampling.meta, "r") as f:
        sample_meta = yaml.safe_load(f)

    context_length = 60 // train_meta["time_res"] * 24 * config.context_window_in_days


    # ---- Train shards setup ----
    print("=> Load trainining shards")
    train_shards = sorted(glob.glob(config.dataset.train.data))
    shard_index = 0
    train_df = None
    train_dataset = None
    train_loader = None
    prefetched_df = None
    prefetch_thread = None

    def _prefetch_next_shard():
        nonlocal prefetched_df, shard_index
        shard_path = train_shards[shard_index]
        print(f"=> Prefetching shard in background: {shard_path}")
        prefetched_df = pl.read_parquet(shard_path, memory_map=True)

    def load_next_shard():
        nonlocal shard_index, train_df, train_dataset, train_loader, prefetched_df, prefetch_thread

        # Reshuffle shards at start of each cycle
        if shard_index == 0:
            random.shuffle(train_shards)

        # If prefetching occurred — wait & use that data
        if prefetch_thread is not None:
            prefetch_thread.join()
            prefetch_thread = None

        shard_path = train_shards[shard_index]

        if prefetched_df is not None:
            print(f"🔄 Loading prefetched shard: {shard_path}")
            train_df = prefetched_df
            prefetched_df = None
        else:
            print(f"⚡ Loading shard synchronously: {shard_path}")
            train_df = pl.read_parquet(shard_path, memory_map=True)

        # Build dataset + loader
        train_dataset = TileTimeSeriesDataset(train_df, train_meta, context_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            persistent_workers=True
        )

        shard_index += 1
        if shard_index >= len(train_shards):
            shard_index = 0

        # Begin prefetching next shard
        prefetch_thread = threading.Thread(target=_prefetch_next_shard)
        prefetch_thread.start()

        print(f"Loaded shard: {shard_path}")

    load_next_shard()
    train_iter = iter(train_loader)


    # ---- Load validation + sampling datasets fully ----
    print("=> Load validation and sampling datasets")
    val_df = pl.read_parquet(config.dataset.validation.data)
    sample_df = pl.read_parquet(config.dataset.sampling.data)
    val_dataset = TileTimeSeriesDataset(val_df, validation_meta, context_length)
    sample_dataset = TileTimeSeriesDataset(sample_df, sample_meta, context_length)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=config.num_workers)
    sample_loader = DataLoader(sample_dataset, batch_size=64, shuffle=False, num_workers=config.num_workers)
    del val_df, sample_df


    print('==> Building model..')
    d_input = len(train_meta["features"])
    model = TimeseriesModel(
        d_input=d_input,
        d_model=config.model.d_model,
        d_intermediate=config.model.d_intermediate,
        n_layers=config.model.n_layers,
        ssm_cfg=config.model.ssm_cfg,
        attn_layer_idx=config.model.attn_layer_idx,
        attn_cfg=config.model.attn_cfg,
        norm_epsilon=config.model.norm_epsilon,
        rms_norm=config.model.rms_norm,
        device=config.device,
    )
    model = model.to(config.device)

    if config.device == 'cuda':
        cudnn.benchmark = True


    min_eval_loss = float("inf")
    best_step = -1
    start_step = 0
    samples_so_far = 0


    criterion = RMSELoss()
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = WarmupCosineLR(optimizer, warmup_steps=config.warmup_steps, total_steps=config.total_steps, last_epoch=start_step-1)

    if not use_wandb:
        wandb_run.watch(model, log="all")

    ema_beta = 0.9 # EMA decay factor
    smooth_train_loss = 0
    smooth_mae = 0


    print('==> Start training..')
    pbar = tqdm(range(start_step, config.total_steps), desc="Training")
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            print("=> Load next shard")
            load_next_shard()
            train_iter = iter(train_loader)
            batch = next(train_iter)

        last_step = step == config.total_steps - 1
        samples_so_far += config.batch_size
        model.eval()
        # once in a while: evaluate model
        if  last_step or (step % config.eval_every == 0 and not (step == 0 and not config.validate_at_start)):
            val_res = evaluate_model_rmse(
                model=model,
                criterion=criterion,
                loader=val_loader,
                device=config.device,
            )

            if val_res["loss"] < min_eval_loss:
                min_eval_loss = val_res["loss"]
                best_step = step

            wandb_run.log({
                "samples_so_far": samples_so_far,
                "val_loss": val_res["loss"],
                "val_loss_last": val_res["loss_last"],
                "val_mae": val_res["mae"],
                "val_zero_mae": val_res["zero_mae"],
                "val_pos_mae": val_res["pos_mae"],
                "min_val_loss": min_eval_loss,
                "best_step": best_step,
            })

        # once in a while: sample from model
        if config is not None and config.sample_every > 0 and (last_step or (step % config.sample_every == 0)):
            plot_forecast_vs_truth_rmse(
                model=model,
                loader=sample_loader,
                device=config.device,
                wandb_run=wandb_run,
                epoch=step,
            )

        # Training
        wandb_run.log({ "last_lr": scheduler.get_last_lr() })
        model.train()

        obs, targets = batch["context"], batch["target"]
        obs, targets = obs.to(config.device), targets.to(config.device)

        optimizer.zero_grad()

        preds = model(obs)

        loss = criterion(preds, targets)
        loss.backward()
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss.item() # EMA the training loss
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))

        mae = torch.mean(torch.abs(preds[:,-1] - targets[:,-1]))
        smooth_mae = ema_beta * smooth_mae + (1 - ema_beta) * mae.item() # EMA the training loss
        debiased_smooth_mae = smooth_mae / (1 - ema_beta**(step + 1))

        pbar.set_description(
            'Step: (%d/%d) | Train loss: %.6f | MAE: %.6f' %
            (step, config.total_steps, debiased_smooth_loss, debiased_smooth_mae)
        )

        if step % 10 == 0:
            wandb_run.log({
                "samples_so_far": samples_so_far,
                "train_loss": debiased_smooth_loss,
                "train_mae": debiased_smooth_mae,
                "last_lr": scheduler.get_last_lr(),
            })

        if config.save_every >= 0 and step % config.save_every == 0:
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
            ckpt_path = f'./checkpoint/ckpt_{config.run}_step{step}.pth'
            torch.save(state, ckpt_path)

            if not use_wandb:
                artifact = wandb.Artifact(f'{wandb_run.id}-artifact-step{step}', type='model')
                artifact.add_file(ckpt_path)
                wandb_run.log_artifact(artifact)

    wandb_run.finish()



if __name__ == "__main__":
    main()
