import torch
from tqdm.auto import tqdm
import polars as pl
import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F

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

            log_key = f"plots/epoch{epoch}_{tile_id}_day_{day.isoformat()}"
            wandb_run.log({log_key: wandb.Image(plt)})
            plt.close()

def plot_llm2(model, loader, device, wandb_run, epoch):
    preds_top3 = [] 
    expectations = [] 
    
    # New lists for quantiles
    quantiles_10 = []
    quantiles_50 = []
    quantiles_90 = []
    
    tile_ids, ts = [], []
    truths = []
    
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader), leave=False, desc="Sample Batch index"):
            
            obs, targets, tile_id, target_timestamp = data["context"], data["target"], data["tile_id"], data["target_timestamp"]
            obs = obs.squeeze(-1).to(device)
            
            # Get Logits
            logits = model(obs, num_last_tokens=1).logits.squeeze(1) # (batch, vocab)
            
            # 1. Convert logits to probabilities
            probs = F.softmax(logits, dim=-1) # (batch, vocab)
            
            # 2. Calculate Expectation: Sum(Value * Probability)
            vocab_values = torch.arange(logits.shape[-1], device=device).float()
            expected_value = torch.sum(probs * vocab_values, dim=-1) # (batch, )
            
            # 3. Get Top 3 Tokens
            _, top3_indices = torch.topk(probs, k=3, dim=-1) # (batch, 3)
            
            # --- 4. Calculate Quantiles (New Code) ---
            # Calculate CDF
            cdf = torch.cumsum(probs, dim=-1) 
            
            # Find the index where CDF crosses the threshold
            # (cdf < threshold).sum() counts how many bins are below threshold, 
            # effectively giving the index of the first bin >= threshold.
            q10 = (cdf < 0.10).sum(dim=-1)
            q50 = (cdf < 0.50).sum(dim=-1) # Median
            q90 = (cdf < 0.90).sum(dim=-1)
            
            t = target_timestamp[:,-1]
            truths.append(targets[:,-1])
            
            preds_top3.append(top3_indices.cpu())
            expectations.append(expected_value.cpu())
            
            # Append quantiles
            quantiles_10.append(q10.cpu())
            quantiles_50.append(q50.cpu())
            quantiles_90.append(q90.cpu())
            
            tile_ids.extend(tile_id[0])
            ts.append(t)

    cat_targets = torch.cat(truths, dim=0)
    cat_preds_top3 = torch.cat(preds_top3, dim=0) 
    cat_expectations = torch.cat(expectations, dim=0)
    cat_ts = torch.cat(ts, dim=0)
    
    # Cat quantiles
    cat_q10 = torch.cat(quantiles_10, dim=0)
    cat_q50 = torch.cat(quantiles_50, dim=0)
    cat_q90 = torch.cat(quantiles_90, dim=0)

    records = []
    # Zip updated with quantiles
    for (truth, top3, exp, t, tile, q10, q50, q90) in zip(
        cat_targets, cat_preds_top3, cat_expectations, cat_ts, tile_ids, cat_q10, cat_q50, cat_q90
    ):
        records.append([
            tile, 
            t, 
            truth.item(), 
            top3[0].item(), 
            top3[1].item(), 
            top3[2].item(), 
            exp.item(),
            q10.item(),
            q50.item(),
            q90.item()
        ])

    df = pl.DataFrame(
        records,
        schema=["tile_id", "reference_time", "count", "top1", "top2", "top3", "expectation", "q10", "q50", "q90"],
        orient="row"
    ).with_columns(
        pl.col("reference_time").cast(pl.Datetime).alias("reference_time")
    ).with_columns(
        pl.col("reference_time")
        .dt.replace_time_zone("UTC") 
        .dt.convert_time_zone("America/Regina")
        .alias("reference_time_local")
    ).with_columns(
        pl.col("reference_time_local").dt.date().alias("date_local")
    )

    # Plot and log each tile
    for tile_group, tile_df in df.group_by("tile_id"):
        tile_id = tile_group[0] if isinstance(tile_group, tuple) else tile_group
        for day_group, day_df in tile_df.group_by("date_local"):
            day = day_group[0] if isinstance(day_group, tuple) else day_group
            day_pd = day_df.sort("reference_time_local").to_pandas()
            plt.figure(figsize=(10, 6))
            
            # 1. Plot Truth (Continuous)
            plt.plot(day_pd["reference_time_local"], day_pd["count"], label="Truth", marker="o", color="black", linewidth=1.5, zorder=10)
            
            # 2. Plot Quantile Range (Shaded area between 10th and 90th percentile)
            plt.fill_between(
                day_pd["reference_time_local"], 
                day_pd["q10"], 
                day_pd["q90"], 
                color="darkorange", 
                alpha=0.2, 
                label="10-90th Percentile"
            )
            
            # 3. Plot Median (50th percentile)
            plt.plot(day_pd["reference_time_local"], day_pd["q50"], label="Median (q50)", linestyle=":", color="darkorange", linewidth=2)

            # 4. Plot Expectation (Continuous dashed)
            plt.plot(day_pd["reference_time_local"], day_pd["expectation"], label="Expectation", linestyle="--", color="green", linewidth=1.5, alpha=0.8)

            # 5. Plot Top 1 (Continuous solid line)
            plt.plot(day_pd["reference_time_local"], day_pd["top1"], label="Top 1", linestyle="-", marker="x", color="blue", linewidth=1, alpha=0.5)
            
            # 6. Plot Top 2 and Top 3 (Scatter)
            plt.scatter(day_pd["reference_time_local"], day_pd["top2"], label="Top 2", marker="1", color="tab:blue", s=30, alpha=0.4)
            plt.scatter(day_pd["reference_time_local"], day_pd["top3"], label="Top 3", marker="2", color="tab:blue", s=30, alpha=0.2)
            
            plt.title(f"Tile {tile_id} — {day} (America/Regina)")
            plt.xlabel("Local Reference Time")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            log_key = f"plots/epoch{epoch}_{tile_id}_day_{day.isoformat()}"
            wandb_run.log({log_key: wandb.Image(plt)})
            plt.close()



def plot_quantile(model, loader, device, wandb_run, epoch, q50_idx, q10_idx, q90_idx):
    mus, tile_ids, ts = [], [], []
    quantiles_10 = []
    quantiles_90 = []

    truths = []
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader), desc="Sample"):
            obs, targets, tile_id, target_timestamp = data["context"], data["target"], data["tile_id"], data["target_timestamp"]
            obs = obs.to(device)
            preds = model(obs)
            pred = preds[:,-1,q50_idx]
            t = target_timestamp[:,-1]
            truths.append(targets[:,-1])
            mus.append(pred.cpu())
            tile_ids.extend(tile_id[0])
            ts.append(t)


            q10 = preds[:,-1,q10_idx]
            q90 = preds[:,-1,q90_idx]
            quantiles_10.append(q10.cpu())
            quantiles_90.append(q90.cpu())

    cat_targets = torch.cat(truths, dim=0)
    cat_mus= torch.cat(mus, dim=0)
    cat_ts = torch.cat(ts, dim=0)

    cat_q10 = torch.cat(quantiles_10, dim=0)
    cat_q90 = torch.cat(quantiles_90, dim=0)


    records = []
    for (truth, mu, t, tile, q10item, q90item) in zip(cat_targets, cat_mus, cat_ts, tile_ids, cat_q10, cat_q90):
        records.append([tile, t, truth, mu, q10item, q90item])

    df = pl.DataFrame(
        records,
        schema=["tile_id", "reference_time", "count", "mu", "q10", "q90"],
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

            plt.fill_between(
                day_pd["reference_time_local"], 
                day_pd["q10"], 
                day_pd["q90"], 
                color="darkorange", 
                alpha=0.2, 
                label="10-90th Percentile"
            )

            plt.title(f"Tile {tile_id} — {day} (America/Regina)")
            plt.xlabel("Local Reference Time")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()

            log_key = f"plots/epoch{epoch}_{tile_id}_day_{day.isoformat()}"
            wandb_run.log({log_key: wandb.Image(plt)})
            plt.close()


