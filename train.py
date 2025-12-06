import contextlib
import glob
import os
import random

import hydra.utils
import torch
import torch.distributed as dist
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm.auto import tqdm

import wandb
from config import Config
from ts_mamba.checkpoint_manager import load_checkpoint, save_checkpoint
from ts_mamba.common import DummyWandb, print_banner, print0, compute_init, compute_cleanup
from ts_mamba.dataloader import TrainShardLoader, get_timeseries_dataloader
from ts_mamba.loss_eval import evaluate_point_forecast_model, evaluate_quantile_model, evaluate_token_model
from ts_mamba.model import MixerConfig, LinearSequenceModel, EmbeddingSequenceModel
from ts_mamba.optimizer import WarmupCosineLR 
from ts_mamba.quantile_model import QuantileRegressionLoss
from ts_mamba.train_util import plot_forecast_vs_truth_rmse, plot_llm2, plot_quantile


config_store = ConfigStore.instance()
config_store.store(name="ts_mamba_config", node=Config)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: Config):
    print0(OmegaConf.to_yaml(config, resolve=True))

    # Compute DDP init
    device_type = config.device
    is_ddp, ddp_rank, ddp_local_rank, world_size, device = compute_init(device_type=device_type)
    main_process = ddp_rank == 0
    get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

    # wandb init
    use_dummy_wandb = (config.wandb is None) or not main_process
    wandb_config = OmegaConf.to_container(config.wandb, resolve=True) if config.wandb else {}
    if "name" not in wandb_config:
        wandb_config["name"] = config.model_tag
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
        **wandb_config,
        config=OmegaConf.to_container(config, resolve=True)
    )

    # load metadata about training data
    with open(config.dataset.train.meta, "r") as f:
        train_meta = yaml.safe_load(f)

    # optimizer / train / data hyperparameters
    timesteps_per_day = 60 // train_meta.get("time_res")*24
    context_length =  timesteps_per_day*config.dataset.context_window_in_days
    tokens_per_fwdbwd = config.train.device_batch_size*context_length # tokens per forward & backward pass for a single rank
    world_tokens_per_fwdbwd = tokens_per_fwdbwd*world_size # total tokens for all ranks
    assert config.train.total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size={config.train.total_batch_size} is not divisible by world_tokens_per_fwdbwd={world_tokens_per_fwdbwd}"
    grad_accum_steps = config.train.total_batch_size // world_tokens_per_fwdbwd
    print0(f"Tokens / microbatch / rank: {config.train.device_batch_size} x {context_length} = {tokens_per_fwdbwd:,}")
    print0(f"Tokens / microbatch: {world_tokens_per_fwdbwd:,}")
    print0(f"Total batch size {config.train.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")


    # ---- Init model -----
    print0("Building model...")
    d_input = len(train_meta["features"])
    model_config = MixerConfig(
        d_model=config.model.d_model,
        n_layer=config.model.n_layer,
        d_intermediate=config.model.d_intermediate,
        rms_norm=config.model.rms_norm,
        norm_epsilon=config.model.norm_epsilon,
        residual_in_fp32=config.model.residual_in_fp32,
        fused_add_norm=config.model.fused_add_norm,
        ssm_cfg=config.model.ssm_cfg,
        attn_layer_idx=config.model.attn_layer_idx,
        attn_cfg=config.model.attn_cfg,
        use_llm_init=config.model.use_llm_init,
        #llm_init_cfg=config.model.llm_init_cfg,
        d_output=config.model.d_output,
        d_input=d_input,
        vocab_size=config.model.vocab_size,
        pad_vocab_multiple=config.model.pad_vocab_multiple,
        tie_embeddings=config.model.tie_embeddings,
    )
    if config.model.architecture == "simple":
        model_cls = LinearSequenceModel
    elif config.model.architecture == "token":
        model_cls = EmbeddingSequenceModel
    else:
        raise ValueError(f"Invalid config.model.architecture {config.model.architecture}")

    dtype = torch.bfloat16
    model = model_cls(cfg=model_config, device=device, dtype=dtype)
    model = model.to(device=device, dtype=dtype)
    num_params = sum(p.numel() for p in model.parameters())
    print0(f"Total parameters: {num_params:,}.")

    if is_ddp:
        model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)

    # resume from checkpoint
    base_dir = hydra.utils.get_original_cwd()
    output_dirname = f"{config.model_tag}_{config.model.architecture}_n{config.model.n_layer}"
    checkpoint_dir = os.path.join(base_dir, "checkpoint", output_dirname) if config.checkpoint_dir is None else config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    resuming = config.resume_from_step is not None and config.resume_from_step > -1
    if resuming:
        print0(f"Resuming checkpoint from step {config.resume_from_step}")
        model_data, optimizer_data, scheduler_data, meta_data = load_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=config.resume_from_step,
            device=device,
            dtype=dtype,
            load_optimizer=config.load_optimizer,
            rank=ddp_rank,
        )
        (model.module if isinstance(model, DDP) else model).load_state_dict(model_data, strict=True, assign=True)
        del model_data

    total_tokens = config.train.total_batch_size * config.train.num_iterations
    print0(f"Total number of training tokens: {total_tokens:,}")
    print0(f"Tokens : Params ratio: {config.train.total_batch_size * config.train.num_iterations / num_params:.2f}") # Chinchilla is ~20

    # -----------------------------------------------------------------------------
    # Initialize the Optimizer 

    # TODO: Think about MUON optimizer for weight matrices (see nanochat implementation)
    warmup_steps = int(config.train.num_iterations * config.train.warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = WarmupCosineLR(optimizer, warmup_steps=warmup_steps, total_steps=config.train.num_iterations, min_lr=config.train.lr*config.train.final_lr_frac)

    if resuming and config.load_optimizer:
        optimizer.load_state_dict(optimizer_data)
        scheduler.load_state_dict(scheduler_data)

    # -----------------------------------------------------------------------------
    # Train shards setup
    seed = torch.zeros(1, dtype=torch.long, device=device)
    if is_ddp:
        if ddp_rank == 0:
            seed.fill_(torch.randint(0, 10_000_000, (1,)).item())
        dist.broadcast(seed, src=0)
        random.seed(seed.item())
    else:
        random.seed(42)

    train_shards = sorted(glob.glob(config.dataset.train.data))
    random.shuffle(train_shards)

    shard_loader = TrainShardLoader(
        shards=train_shards,
        meta=train_meta,
        context_length=context_length,
        stride=config.dataset.train.stride,
        use_covariates=config.dataset.use_covariates,
        batch_size=config.train.device_batch_size,
        num_workers=config.num_workers,
        rank=ddp_rank,
        world_size=world_size,
        shuffle_shards=True,
    )
    train_loader = shard_loader.load_next()
    train_iter = iter(train_loader)

    val_loader = get_timeseries_dataloader(
        config.dataset.validate.data,
        meta_path=config.dataset.validate.meta,
        context_length=context_length,
        use_covariates=config.dataset.use_covariates,
        batch_size=config.validate.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        distributed=True,
    )
    sample_loader = get_timeseries_dataloader(
        config.dataset.sample.data,
        meta_path=config.dataset.sample.meta,
        context_length=context_length,
        use_covariates=config.dataset.use_covariates,
        batch_size=config.sample.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        distributed=False,
    )


    # -----------------------------------------------------------------------------
    # Loop state
    ema_beta = 0.9 # EMA decay factor
    if not resuming:
        step = 0
        best_step = 0
        min_val_loss = float("inf")
        smooth_train_loss = 0
    else:
        step = meta_data["step"]
        best_step = meta_data["best_step"]
        loop_state = meta_data["loop_state"]
        min_val_loss = loop_state["min_val_loss"]
        smooth_train_loss = loop_state["smooth_train_loss"]


    # -----------------------------------------------------------------------------
    # Loss
    if config.train.loss.name == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif config.train.loss.name == "quantile":
        criterion = QuantileRegressionLoss(quantiles=config.train.loss.quantiles, device=device, dtype=dtype)
    elif config.train.loss.name == "mse":
        criterion = torch.nn.MSELoss()
    elif config.train.loss.name == "l1":
        criterion = torch.nn.L1Loss()
    else:
        raise ValueError(f"Invalid config.loss: {config.train.loss.name}")

    # -----------------------------------------------------------------------------
    # Training loop
    pbar = tqdm(range(step, config.train.num_iterations), desc="Training")

    def load_batch():
        nonlocal train_loader, train_iter
        try:
            batch = next(train_iter)
        except StopIteration:
            print("=> Load next shard")
            train_loader = shard_loader.load_next()
            train_iter = iter(train_loader)
            batch = next(train_iter)
        return batch

    for step in pbar:
        last_step = step == config.train.num_iterations - 1

        # once in a while: evaluate model
        if last_step or step % config.validate.validate_every == 0:
            model.eval()

            eval_kwargs = dict(model=model, criterion=criterion, loader=val_loader, device=device)
            if config.train.loss.name in ("l1", "mse"):
                val_res = evaluate_point_forecast_model(**eval_kwargs)
            elif config.train.loss.name == "quantile":
                val_res = evaluate_quantile_model(**eval_kwargs, quantile_idx=config.validate.quantile_point_forecast_idx)
            elif config.train.loss.name == "cross_entropy":
                val_res = evaluate_token_model(**eval_kwargs)
            else:
                raise ValueError(f"Invalid config.train.loss.name {config.train.loss.name}")

            val_res = val_res._asdict()

            if val_res["criterion"] < min_val_loss:
                min_val_loss = val_res["criterion"]
                best_step = step
            if main_process:
                wandb_run.log({
                    **val_res,
                    "best_step": best_step,
                    "min_val_loss": min_val_loss
                })

            if dist.is_initialized():
                dist.barrier()


        # once in a while: sample from model
        if main_process and (last_step or step % config.sample.sample_every == 0):
            model.eval()
            if config.train.loss.name in ('l1', "mse"):
                plot_forecast_vs_truth_rmse(
                    model=model.module if isinstance(model, DDP) else model,
                    loader=sample_loader,
                    device=device,
                    wandb_run=wandb_run,
                    epoch=step,
                )
            elif config.train.loss.name == 'cross_entropy':
                plot_llm2(
                    model=model.module if isinstance(model, DDP) else model,
                    loader=sample_loader,
                    device=device,
                    wandb_run=wandb_run,
                    epoch=step
                )
            elif config.train.loss.name == 'quantile':
                plot_quantile(
                    model=model.module if isinstance(model, DDP) else model,
                    loader=sample_loader,
                    device=device,
                    wandb_run=wandb_run,
                    epoch=step,
                    q50_idx=config.sample.quantile_point_forecast_idx,
                    q10_idx=config.sample.quantile_10_idx,
                    q90_idx=config.sample.quantile_90_idx,
                )


        # once in a while: save
        if last_step or (step > 0 and step != config.resume_from_step and config.save_every is not None and config.save_every > 0 and step % config.save_every == 0):
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                model_data=model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                optimizer_data=optimizer.state_dict(),
                meta_data={
                    "step": step,
                    "config": OmegaConf.to_container(config, resolve=True),
                    "loop_state": {
                        "min_val_loss": min_val_loss,
                        "smooth_train_loss": smooth_train_loss,
                    }
                },
                rank=ddp_rank,
            )

        if last_step:
            break

        # -------------------------------------------------------------------------
        # single training step
        # evaluate the gradient
        model.train()
        optimizer.zero_grad()

        accumulated_loss = 0.0

        for micro_step in range(grad_accum_steps):
            do_sync = (micro_step == grad_accum_steps - 1)
            sync_context = model.no_sync() if (not is_ddp or not do_sync) else contextlib.nullcontext()

            with sync_context:
                batch = load_batch()

                obs, targets = batch["context"], batch["target"]
                if config.train.loss.name == 'cross_entropy':
                    obs, targets = obs.squeeze(-1), targets.squeeze(-1)
                obs, targets = obs.to(device), targets.to(device)

                preds = model(obs)
                if config.train.loss.name == "cross_entropy":
                    preds = preds.logits.reshape(-1, preds.logits.size(-1))
                    targets = targets.reshape(-1)

                loss = criterion(preds, targets)
                accumulated_loss += loss.item()
                loss = loss / grad_accum_steps
                loss.backward()

        if config.train.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.module.parameters() if isinstance(model, DDP) else model.parameters(), 
                config.train.grad_clip
            )
        optimizer.step()
        scheduler.step()

        # logging
        avg_step_loss = accumulated_loss / grad_accum_steps
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * avg_step_loss # EMA the training loss
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))

        if main_process:
            pbar.set_description(
                'Step: (%d/%d) | Train loss: %.6f' %
                (step, config.train.num_iterations, debiased_smooth_loss)
            )

            if step % 20 == 0:
                wandb_run.log({
                    "train_loss": debiased_smooth_loss,
                    "lr": optimizer.param_groups[0]['lr']
                })

    print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Minimum val loss: {min_val_loss:.4f}")

    if main_process:
        wandb_run.finish()

    compute_cleanup()


if __name__ == "__main__":
    print_banner()
    main()
