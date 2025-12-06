import logging
import random
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from ts_mamba.common import setup_default_logging
from ts_mamba.dataset import TileTimeSeriesWindowedDataset
import polars as pl

setup_default_logging()
logger = logging.getLogger(__name__)

class TrainShardLoader:
    def __init__(
        self,
        shards,
        meta: dict,
        context_length: int,
        stride: int,
        use_covariates: bool,
        batch_size: int,
        shuffle_shards: bool,
        num_workers: int, 
        rank: int,
        world_size: int,
    ):
        self.local_shards = shards[rank::world_size]
        self.shard_idx = 0
        self.rank = rank
        self.meta = meta
        self.context_length = context_length
        self.use_covariates = use_covariates
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_shards = shuffle_shards

        if self.shuffle_shards:
            random.shuffle(self.local_shards)

        logger.info(f"[Rank {rank}] Assigned {len(self.local_shards)} parquet files.")

    def load_next(self):
        shard_path = self.local_shards[self.shard_idx]
        df = pl.read_parquet(shard_path, memory_map=True)
        logger.info(f"[Rank {self.rank}] Loaded shard {shard_path}")

        dataset = TileTimeSeriesWindowedDataset(
            df=df,
            meta=self.meta,
            context_length=self.context_length,
            use_features=self.use_covariates,
            stride=self.stride,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

        self.shard_idx += 1
        if self.shard_idx >= len(self.local_shards):
            self.shard_idx = 0
            if self.shuffle_shards:
                random.shuffle(self.local_shards)

        return dataloader

def get_timeseries_dataloader(
    parquet_path: str,
    meta_path: str,
    context_length: int,
    use_covariates: bool,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    distributed: bool = False,
):
    df = pl.read_parquet(parquet_path)
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)
    dataset = TileTimeSeriesWindowedDataset(
        df,
        meta=meta,
        context_length=context_length,
        use_features=use_covariates,
    )
    if distributed:
        sampler = DistributedSampler(dataset=dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
