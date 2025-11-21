import glob
import os

import hydra
import polars as pl
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import yaml
from hydra.core.config_store import ConfigStore
from mamba_ssm.models.config_mamba import MambaConfig
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from config import Config
from ts_mamba.common import DummyWandb
from ts_mamba.dataset import TileTimeSeriesDataset
from ts_mamba.llm import MambaLMHeadModel 
from ts_mamba.loss_eval import evaluate_model_rmse, evaluate_llm
from ts_mamba.model import TimeseriesModel, WeightedRMSELoss
from ts_mamba.optimizer import WarmupCosineLR 
from ts_mamba.train_util import plot_forecast_vs_truth_rmse, plot_llm2


config_store = ConfigStore.instance()
config_store.store(name="timeseries_deep_learning_config", node=Config)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: Config):
    print(config)

    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    is_main = (not dist.is_initialized()) or dist.get_rank() == 0

    if dist.is_initialized():
        device = torch.device(f"cuda:{local_rank}")
    else: 
        device = torch.device(config.device)


    use_wandb = config.wandb is not None
    wandb_run = None
    if is_main and use_wandb:
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
    print("=> Load training shards")
    train_shards = sorted(glob.glob(config.dataset.train.data))
    shard_index = 0
    train_df = None
    train_dataset = None
    train_loader = None
    train_sampler = None

    start_step = 0

    def load_next_shard():
        nonlocal shard_index, train_df, train_dataset, train_loader, train_sampler

        shard_path = train_shards[shard_index]
        train_df = pl.read_parquet(shard_path, memory_map=True)

        # Build dataset + loader
        train_dataset = TileTimeSeriesDataset(train_df, train_meta, context_length, use_features=config.model.model != "llm")

        if dist.is_initialized():
            train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=dist.get_world_size(), rank=dist.get_rank())

            current_step = step if 'step' in locals() else start_step
            train_sampler.set_epoch(current_step)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=not dist.is_initialized(),
            num_workers=config.num_workers,
            persistent_workers=True,
            sampler=train_sampler,
        )

        shard_index += 1
        if shard_index >= len(train_shards):
            shard_index = 0

        print(f"Loaded shard: {shard_path}")

        if dist.is_initialized():
            dist.barrier()

    load_next_shard()
    train_iter = iter(train_loader)


    # ---- Load validation + sampling datasets fully ----
    print("=> Load validation and sampling datasets")
    val_df = pl.read_parquet(config.dataset.validation.data)
    sample_df = pl.read_parquet(config.dataset.sampling.data)
    val_dataset = TileTimeSeriesDataset(val_df, validation_meta, context_length, use_features=config.model.model != "llm")
    sample_dataset = TileTimeSeriesDataset(sample_df, sample_meta, context_length, use_features=config.model.model != "llm")
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=config.num_workers)
    sample_loader = DataLoader(sample_dataset, batch_size=64, shuffle=False, num_workers=config.num_workers)
    del val_df, sample_df


    print('==> Building model..')
    d_input = len(train_meta["features"])
    model = None
    if config.model.model == 'mamba':
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
            residual_in_fp32=config.model.residual_in_fp32,
            dtype=torch.bfloat16,
            device=device,
        )
    elif config.model.model == 'llm':
        model = MambaLMHeadModel(
            config=MambaConfig(
                d_model=config.model.d_model,
                d_intermediate=config.model.d_intermediate,
                n_layer=config.model.n_layers,
                vocab_size=config.model.vocab_size,
                ssm_cfg=config.model.ssm_cfg,
                attn_layer_idx=config.model.attn_layer_idx,
                attn_cfg=config.model.attn_cfg,
                rms_norm=config.model.rms_norm,
                residual_in_fp32=config.model.residual_in_fp32,
                fused_add_norm=True,
                pad_vocab_size_multiple=8,
                tie_embeddings=config.model.tie_embeddings,
            ),
            dtype=torch.bfloat16,
            device=device,
        )
        pass
    else:
        raise ValueError(f"Invalid config.model.model: {config.model.model}")

    model = model.to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if device.type == 'cuda':
        cudnn.benchmark = True


    min_eval_loss = float("inf")
    best_step = -1
    samples_so_far = 0

    criterion = None

    if config.loss == 'rmse':
        criterion = WeightedRMSELoss(decay=config.rmse_decay)
    elif config.loss == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid config.loss: {config.loss}")


    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = WarmupCosineLR(optimizer, warmup_steps=config.warmup_steps, total_steps=config.total_steps)

    if config.resume.enabled and config.resume.checkpoint_path is not None:
        if is_main:
            print('==> Resume from checkpoint..')

        ckpt = torch.load(config.resume.checkpoint_path, map_location=device)

        if isinstance(model, DDP):
            model.module.load_state_dict(ckpt['model'], strict=True)
        else:
            model.load_state_dict(ckpt['model'], strict=True)


        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])

        start_step = ckpt.get('step', 0) + 1
        samples_so_far = ckpt.get('samples_so_far', 0)
        min_eval_loss = ckpt.get('min_val_loss', min_eval_loss)
        best_step = ckpt.get('best_step', best_step)

        if is_main:
            print(f"Resumed from step {start_step} | "
                  f"min_val_loss={min_eval_loss:.4f} best_step={best_step}")

    if is_main and use_wandb:
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

        if dist.is_initialized():
            world_size = dist.get_world_size()
            samples_so_far += config.batch_size * world_size
        else:
            samples_so_far += config.batch_size

        model.eval()
        # once in a while: evaluate model
        if (last_step or (step % config.eval_every == 0 and not (step == 0 and not config.validate_at_start))):
            if is_main:
                if config.loss == 'rmse':
                    val_res = evaluate_model_rmse(
                        model=model.module if isinstance(model, DDP) else model,
                        criterion=criterion,
                        loader=val_loader,
                        device=device,
                    )

                    if val_res["loss"] < min_eval_loss:
                        min_eval_loss = val_res["loss"]
                        best_step = step

                    wandb_run.log({
                        "samples_so_far": samples_so_far,
                        "val_loss": val_res["loss"],
                        "val_rmse": val_res["rmse"],
                        "val_mae": val_res["mae"],
                        "val_zero_mae": val_res["zero_mae"],
                        "val_pos_mae": val_res["pos_mae"],
                        "min_val_loss": min_eval_loss,
                        "best_step": best_step,
                    })
                elif config.loss == 'cross_entropy':
                    val_res = evaluate_llm(
                        model=model.module if isinstance(model, DDP) else model,
                        criterion=criterion,
                        loader=val_loader,
                        device=device,
                    )

                    if val_res["loss"] < min_eval_loss:
                        min_eval_loss = val_res["loss"]
                        best_step = step

                    wandb_run.log({
                        "samples_so_far": samples_so_far,
                        "val_loss": val_res["loss"],
                        "val_mae": val_res["mae"],
                        "val_rmse": val_res["rmse"],
                        "min_val_loss": min_eval_loss,
                        "best_step": best_step,
                    })

            if dist.is_initialized():
                dist.barrier()



        # once in a while: sample from model
        if config is not None and config.sample_every > 0 and (last_step or (step % config.sample_every == 0)):
            if is_main:
                if config.loss == 'rmse':
                    plot_forecast_vs_truth_rmse(
                        model=model.module if isinstance(model, DDP) else model,
                        loader=sample_loader,
                        device=device,
                        wandb_run=wandb_run,
                        epoch=step,
                    )
                elif config.loss == 'cross_entropy':
                    plot_llm2(
                        model=model.module if isinstance(model, DDP) else model,
                        loader=sample_loader,
                        device=device,
                        wandb_run=wandb_run,
                        epoch=step
                    )

            if dist.is_initialized():
                dist.barrier()

        # Training
        if is_main:
            wandb_run.log({ "last_lr": scheduler.get_last_lr() })
        model.train()

        obs, targets = batch["context"], batch["target"]
        if config.loss == 'cross_entropy':
            obs, targets = obs.squeeze(-1), targets.squeeze(-1)
            targets = targets[:, -config.train.num_last_tokens:].reshape(-1)
        obs, targets = obs.to(device), targets.to(device)

        optimizer.zero_grad()

        if config.model.model == "mamba":
            preds = model(obs)
        elif config.model.model == "llm":
            preds = model(obs, num_last_tokens=config.train.num_last_tokens)
            preds = preds.logits.reshape(-1, preds.logits.size(-1))

        loss = criterion(preds, targets)
        loss.backward()
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss.item() # EMA the training loss
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))

        if config.loss == 'rmse':
            mae = torch.mean(torch.abs(preds[:,-1] - targets[:,-1]))
            smooth_mae = ema_beta * smooth_mae + (1 - ema_beta) * mae.item() # EMA the training loss
            debiased_smooth_mae = smooth_mae / (1 - ema_beta**(step + 1))

            pbar.set_description(
                'Step: (%d/%d) | Train loss: %.6f | MAE: %.6f' %
                (step, config.total_steps, debiased_smooth_loss, debiased_smooth_mae)
            )

            if is_main and step % 10 == 0:
                wandb_run.log({
                    "samples_so_far": samples_so_far,
                    "train_loss": debiased_smooth_loss,
                    "train_mae": debiased_smooth_mae,
                    "last_lr": scheduler.get_last_lr(),
                })
        elif config.loss == 'cross_entropy':
            pbar.set_description(
                'Step: (%d/%d) | Train loss: %.6f' %
                (step, config.total_steps, debiased_smooth_loss)
            )

            if is_main and step % 10 == 0:
                wandb_run.log({
                    "samples_so_far": samples_so_far,
                    "train_loss": debiased_smooth_loss,
                    "last_lr": scheduler.get_last_lr(),
                })


        if is_main and config.save_every >= 0 and step % config.save_every == 0:
            print("Save checkpoint")

            state = {
                'model': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': step,
                'samples_so_far': samples_so_far,
                "min_val_loss": min_eval_loss,
                "best_step": best_step,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            ckpt_path = f'./checkpoint/ckpt_{config.wandb.name}_step{step}.pth'
            torch.save(state, ckpt_path)

            if is_main and use_wandb:
                artifact = wandb.Artifact(f'{wandb_run.id}-artifact-step{step}', type='model')
                artifact.add_file(ckpt_path)
                wandb_run.log_artifact(artifact)

    if is_main:
        wandb_run.finish()



if __name__ == "__main__":
    main()
