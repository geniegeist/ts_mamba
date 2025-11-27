import hydra
import polars as pl
import torch
import yaml
from datetime import datetime
from hydra.core.config_store import ConfigStore
from mamba_ssm.models.config_mamba import MambaConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config
from ts_mamba.dataset import TileTimeSeriesDataset
from ts_mamba.llm import MambaLMHeadModel 
from ts_mamba.model import TimeseriesModel


config_store = ConfigStore.instance()
config_store.store(name="timeseries_deep_learning_config", node=Config)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: Config):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    device = torch.device(config.device)

    with open(config.benchmark.test_meta, "r") as f:
        test_meta = yaml.safe_load(f)

    context_length = 60 // test_meta["config"]["time_res"] * 24 * config.context_window_in_days

    df_test = pl.read_parquet(config.benchmark.test_file, memory_map=True)
    test_dataset = TileTimeSeriesDataset(df_test, test_meta, context_length, use_features=config.model.model != "llm")
    test_loader = DataLoader(
        test_dataset,
        batch_size=160,
        num_workers=config.num_workers,
        persistent_workers=True,
    )


    print('==> Building model..')
    d_input = len(test_meta["features"])
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
            use_llm_init = config.model.use_llm_init,
            dtype=torch.bfloat16,
            device=device,
        )
        pass
    else:
        raise ValueError(f"Invalid config.model.model: {config.model.model}")

    model = model.to(device)
    ckpt = torch.load(config.benchmark.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)


    print('==> Start benchmark..')
    model.eval()

    preds, tile_ids, ts = [], [], []
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Benchmar")
        for batch_idx, data in pbar:
            obs, tile_id, target_timestamp = data["context"], data["tile_id"], data["target_timestamp"]
            obs = obs.squeeze(-1).to(device)

            logits = model(obs, num_last_tokens=1).logits.squeeze(1) # (batch, vocab)
            t = target_timestamp[:,-1]

            preds.append(logits.cpu())
            tile_ids.extend(tile_id[0])
            ts.append(t)

    preds = torch.cat(preds, dim=0)
    ts = torch.cat(ts, dim=0)
    
    records = []
    for (logits, t, tile) in zip(preds, ts, tile_ids):
        records.append([tile, t, logits.tolist()])

    df = pl.DataFrame(
        records,
        schema=["tile_id", "reference_time", "logits"],
        orient="row"
    ).with_columns(
        pl.col("reference_time")
        .cast(pl.Datetime)
        .round(f"{test_meta['config']['time_res']}m")
        .alias("reference_time")
    ).with_columns(
        pl.col("reference_time")
        .dt.replace_time_zone("UTC")  # assume timestamps are in UTC
        .dt.convert_time_zone(config.benchmark.time_zone)
        .alias("reference_time_local")
    )

    df.write_parquet(f"benchmark/benchmark_{ts}.parquet")
    print("Written")

if __name__ == "__main__":
    main()
