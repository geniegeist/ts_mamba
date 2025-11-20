import torch
from tqdm.auto import tqdm
import polars as pl
import matplotlib.pyplot as plt
import wandb

def plot_forecast_vs_truth(model, loader, device, wandb_run, epoch):
    mus, alphas, tile_ids, ts = [], [], [], []
    truths = []
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader), leave=False, desc="Sample Batch index"):
            obs, targets, tile_id, target_timestamp = data["context"], data["target"], data["tile_id"], data["target_timestamp"]
            obs = obs.to(device)
            mu, alpha = model(obs)
            mu = mu[:,-1]
            alpha = alpha[:,-1]
            t = target_timestamp[:,-1]
            truths.append(targets[:,-1])
            mus.append(mu.cpu())
            alphas.append(alpha.cpu())
            tile_ids.extend(tile_id[0])
            ts.append(t)
    cat_targets = torch.cat(truths, dim=0)
    cat_mus= torch.cat(mus, dim=0)
    cat_alphas= torch.cat(alphas, dim=0)
    cat_ts = torch.cat(ts, dim=0)

    records = []
    for (truth, mu, alpha, t, tile) in zip(cat_targets, cat_mus, cat_alphas, cat_ts, tile_ids):
        records.append([tile, t, truth, mu, alpha])
    df = pl.DataFrame(
        records,
        schema=["tile_id", "reference_time", "count", "mu", "alpha"],
        orient="row"
    ).with_columns(
        pl.col("reference_time").cast(pl.Datetime).alias("reference_time")
    ).with_columns(
        pl.col("reference_time")
        .dt.replace_time_zone("UTC")  # assume timestamps are in UTC
        .dt.convert_time_zone("America/Regina")
        .alias("reference_time_local")
    ).with_columns(
        pl.col("reference_time_local").dt.date().alias("date_local")  # extract day
    )

    # Plot and log each tile
    for tile_group, tile_df in df.group_by("tile_id"):
        tile_id = tile_group[0] if isinstance(tile_group, tuple) else tile_group

        for day_group, day_df in tile_df.group_by("date_local"):
            day = day_group[0] if isinstance(day_group, tuple) else day_group

            day_pd = day_df.sort("reference_time_local").to_pandas()

            plt.figure(figsize=(8, 5))
            plt.plot(day_pd["reference_time_local"], day_pd["count"], label="Truth", marker="o")
            plt.plot(day_pd["reference_time_local"], day_pd["mu"], label="Forecast μ", marker="x")
            plt.plot(day_pd["reference_time_local"], day_pd["alpha"], label="Alpha", linestyle="--", marker="x")
            plt.title(f"Tile {tile_id} — {day} (America/Regina)")
            plt.xlabel("Local Reference Time")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()

            log_key = f"plots/epoch{epoch}/{tile_id}/day_{day.isoformat()}"
            wandb_run.log({log_key: wandb.Image(plt)})
            plt.close()


def plot_forecast_vs_truth_rmse(model, loader, device, wandb_run, epoch):
    mus, tile_ids, ts = [], [], []
    truths = []
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader), leave=False, desc="Sample Batch index"):
            obs, targets, tile_id, target_timestamp = data["context"], data["target"], data["tile_id"], data["target_timestamp"]
            obs = obs.to(device)
            preds = model(obs)
            pred = preds[:,-1]
            t = target_timestamp[:,-1]
            truths.append(targets[:,-1])
            mus.append(pred.cpu())
            tile_ids.extend(tile_id[0])
            ts.append(t)
    cat_targets = torch.cat(truths, dim=0)
    cat_mus= torch.cat(mus, dim=0)
    cat_ts = torch.cat(ts, dim=0)

    records = []
    for (truth, mu, t, tile) in zip(cat_targets, cat_mus, cat_ts, tile_ids):
        records.append([tile, t, truth, mu])
    df = pl.DataFrame(
        records,
        schema=["tile_id", "reference_time", "count", "mu"],
        orient="row"
    ).with_columns(
        pl.col("reference_time").cast(pl.Datetime).alias("reference_time")
    ).with_columns(
        pl.col("reference_time")
        .dt.replace_time_zone("UTC")  # assume timestamps are in UTC
        .dt.convert_time_zone("America/Regina")
        .alias("reference_time_local")
    ).with_columns(
        pl.col("reference_time_local").dt.date().alias("date_local")  # extract day
    )

    # Plot and log each tile
    for tile_group, tile_df in df.group_by("tile_id"):
        tile_id = tile_group[0] if isinstance(tile_group, tuple) else tile_group

        for day_group, day_df in tile_df.group_by("date_local"):
            day = day_group[0] if isinstance(day_group, tuple) else day_group

            day_pd = day_df.sort("reference_time_local").to_pandas()

            plt.figure(figsize=(8, 5))
            plt.plot(day_pd["reference_time_local"], day_pd["count"], label="Truth", marker="o")
            plt.plot(day_pd["reference_time_local"], day_pd["mu"], label="Forecast μ", marker="x")
            plt.title(f"Tile {tile_id} — {day} (America/Regina)")
            plt.xlabel("Local Reference Time")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()

            log_key = f"plots/epoch{epoch}/{tile_id}/day_{day.isoformat()}"
            wandb_run.log({log_key: wandb.Image(plt)})
            plt.close()

def plot_llm(model, loader, device, wandb_run, epoch):
    preds_top1, preds_top2, preds_top3 = [], [], []
    probs_top1, probs_top2, probs_top3 = [], [], []

    tile_ids = []
    ts = []
    truths = []

    with torch.no_grad():
        for data in tqdm(loader, total=len(loader), leave=False, desc="Sample Batch index"):
            obs, targets, tile_id, target_timestamp = (
                data["context"],
                data["target"],
                data["tile_id"],
                data["target_timestamp"]
            )

            obs = obs.squeeze(-1).to(device)
            logits = model(obs, num_last_tokens=1).logits.squeeze(1)  # (batch, vocab)
            probs = torch.softmax(logits, dim=-1)

            # --- TOP-3 ---
            top3_vals, top3_idx = torch.topk(probs, k=3, dim=-1)

            # Extract final timestamp + target for each sample in the batch
            t = target_timestamp[:, -1].cpu().long()   # (batch,)
            y = targets[:, -1].cpu().long()            # (batch,)

            truths.append(y)
            ts.append(t)

            preds_top1.append(top3_idx[:, 0].cpu().long())
            preds_top2.append(top3_idx[:, 1].cpu().long())
            preds_top3.append(top3_idx[:, 2].cpu().long())

            probs_top1.append(top3_vals[:, 0].cpu().float())
            probs_top2.append(top3_vals[:, 1].cpu().float())
            probs_top3.append(top3_vals[:, 2].cpu().float())

            # ---- FIX tile_id handling (batch-level → sample-level) ----
            if isinstance(tile_id, torch.Tensor):
                # tile_id: shape (batch, 1) or (batch,)
                tile_id_batch = tile_id.squeeze().cpu().tolist()

            elif isinstance(tile_id, list):
                # Already a list, probably length = batch_size or = 1
                tile_id_batch = tile_id
            else:
                raise TypeError(f"Unexpected tile_id type: {type(tile_id)}")

# Make sure tile_id_batch has length = 1 or = batch_size
            if len(tile_id_batch) == 1:
                # One tile ID for whole batch → repeat for each sample
                tile_ids.extend([tile_id_batch[0]] * t.shape[0])
            elif len(tile_id_batch) == t.shape[0]:
                # Perfect, one tile_id per sample
                tile_ids.extend(tile_id_batch)
            else:
                raise ValueError(
                    f"tile_id length mismatch: len(tile_id_batch)={len(tile_id_batch)}, "
                    f"batch_size={t.shape[0]}"
                )


    # --- FLATTEN EVERYTHING ---
    flat_ts        = torch.cat(ts, dim=0)
    flat_truths    = torch.cat(truths).numpy()
    flat_top1      = torch.cat(preds_top1).numpy()
    flat_top2      = torch.cat(preds_top2).numpy()
    flat_top3      = torch.cat(preds_top3).numpy()
    flat_p1        = torch.cat(probs_top1).numpy().astype(float)
    flat_p2        = torch.cat(probs_top2).numpy().astype(float)
    flat_p3        = torch.cat(probs_top3).numpy().astype(float)

    # --- BUILD POLARS DF ---
    df = pl.DataFrame({
        "tile_id": tile_ids,
        "reference_time": flat_ts,   # epoch timestamps
        "truth": flat_truths,
        "top1": flat_top1,
        "top2": flat_top2,
        "top3": flat_top3,
        "p1": flat_p1,
        "p2": flat_p2,
        "p3": flat_p3,
    })

    # Convert timestamps
    df = (
        df.with_columns(
            pl.col("reference_time").cast(pl.Datetime).alias("reference_time")
        )
        .with_columns(
            pl.col("reference_time")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("America/Regina")
            .alias("reference_time_local")
        )
        .with_columns(
            pl.col("reference_time_local").dt.date().alias("date_local")
        )
    )

    # --- PLOT FOR EACH TILE × DAY ---
    for tile_id, tile_df in df.group_by("tile_id"):
        for date_val, day_df in tile_df.group_by("date_local"):

            # Convert Polars → Pandas for plotting
            day_pd = day_df.sort("reference_time_local").to_pandas()

            plt.figure(figsize=(10, 6))

            # TRUTH LINE
            plt.plot(
                day_pd["reference_time_local"],
                day_pd["truth"],
                label="Truth",
                marker="o"
            )

            # TOP-3 PREDICTIONS
            plt.scatter(
                day_pd["reference_time_local"],
                day_pd["top1"],
                s=80 * day_pd["p1"],
                marker="x",
                label="Top-1"
            )
            plt.scatter(
                day_pd["reference_time_local"],
                day_pd["top2"],
                s=80 * day_pd["p2"],
                marker="^",
                label="Top-2"
            )
            plt.scatter(
                day_pd["reference_time_local"],
                day_pd["top3"],
                s=80 * day_pd["p3"],
                marker="s",
                label="Top-3"
            )

            plt.title(f"Tile {tile_id} — {date_val} (America/Regina)")
            plt.xlabel("Local Reference Time")
            plt.ylabel("Class / Count")
            plt.legend()
            plt.tight_layout()

            wandb_run.log({
                f"plots/epoch{epoch}/{tile_id}/day_{date_val.isoformat()}":
                    wandb.Image(plt)
            })

            plt.close()

def plot_llm2(model, loader, device, wandb_run, epoch):
    preds, tile_ids, ts = [], [], []
    truths = []
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader), leave=False, desc="Sample Batch index"):
            obs, targets, tile_id, target_timestamp = data["context"], data["target"], data["tile_id"], data["target_timestamp"]
            obs = obs.squeeze(-1).to(device)
            logits = model(obs, num_last_tokens=1).logits.squeeze(1) # (batch, vocab)
            _, most_probable_token = torch.max(logits, dim=-1) # (batch, )

            t = target_timestamp[:,-1]
            truths.append(targets[:,-1])
            preds.append(most_probable_token.cpu())
            tile_ids.extend(tile_id[0])
            ts.append(t)
    cat_targets = torch.cat(truths, dim=0)
    cat_preds = torch.cat(preds, dim=0)
    cat_ts = torch.cat(ts, dim=0)

    records = []
    for (truth, mu, t, tile) in zip(cat_targets, cat_preds, cat_ts, tile_ids):
        records.append([tile, t, truth, mu])
    df = pl.DataFrame(
        records,
        schema=["tile_id", "reference_time", "count", "mu"],
        orient="row"
    ).with_columns(
        pl.col("reference_time").cast(pl.Datetime).alias("reference_time")
    ).with_columns(
        pl.col("reference_time")
        .dt.replace_time_zone("UTC")  # assume timestamps are in UTC
        .dt.convert_time_zone("America/Regina")
        .alias("reference_time_local")
    ).with_columns(
        pl.col("reference_time_local").dt.date().alias("date_local")  # extract day
    )

    # Plot and log each tile
    for tile_group, tile_df in df.group_by("tile_id"):
        tile_id = tile_group[0] if isinstance(tile_group, tuple) else tile_group

        for day_group, day_df in tile_df.group_by("date_local"):
            day = day_group[0] if isinstance(day_group, tuple) else day_group

            day_pd = day_df.sort("reference_time_local").to_pandas()

            plt.figure(figsize=(8, 5))
            plt.plot(day_pd["reference_time_local"], day_pd["count"], label="Truth", marker="o")
            plt.plot(day_pd["reference_time_local"], day_pd["mu"], label="Forecast μ", marker="x")
            plt.title(f"Tile {tile_id} — {day} (America/Regina)")
            plt.xlabel("Local Reference Time")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()

            log_key = f"plots/epoch{epoch}/{tile_id}/day_{day.isoformat()}"
            wandb_run.log({log_key: wandb.Image(plt)})
            plt.close()


