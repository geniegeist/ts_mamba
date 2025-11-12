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

