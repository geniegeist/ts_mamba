import polars as pl
import torch
from torch.utils.data import Dataset

from ts_mamba.common import setup_default_logging
import logging
import os

setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(message)

class TileTimeSeriesDataset(Dataset):
    def __init__(self, df: pl.DataFrame, meta: dict, context_length: int, use_features: bool):
        """
        Args:
            df: Polars DataFrame containing all data.
            meta: dict with keys like:
                {
                    "tile_column": "tile_id",
                    "time_column": "datetime",
                    "target": "target_col",
                    "features": ["feat1", "feat2", ...]
                }
            context_length: number of timesteps for the context window
            horizon: number of timesteps for the prediction window
        """
        self.meta = meta
        self.context_length = context_length
        self.horizon: int = 1

        tile_col = meta["tile_column"]
        time_col = meta["time_column"]

        df = df.with_columns(
            pl.col(time_col).dt.timestamp().alias("__timestamp__")
        )

        self.tile_groups = df.partition_by(tile_col, as_dict=True)
        self.tile_tensors = {}
        self.index = []

        for tile_id, tile_df in self.tile_groups.items():
            # Sort by time if not already sorted
            tile_df = tile_df.sort(time_col)

            # Extract feature, target, and timestamp tensors
            if use_features:
                x = torch.from_numpy(tile_df.select(meta["features"]).to_numpy()).float()
                y = torch.from_numpy(tile_df.select(meta["target"]).to_numpy().copy().squeeze()).float()
            else:
                x = torch.from_numpy(tile_df.select(meta["target"]).to_numpy().copy()).long()
                y = torch.from_numpy(tile_df.select(meta["target"]).to_numpy().copy().squeeze()).long()
            # Copy these two tensors to suppress warning of non-writeable tensors
            t = torch.from_numpy(tile_df.select("__timestamp__").to_numpy().copy().squeeze()).float()

            length = len(tile_df)
            # Skip tiles with insufficient length
            min_required = self.context_length + self.horizon
            if length < min_required:
                print(f"Skipping tile {tile_id}: only {length} timesteps, "
                      f"requires >= {min_required}")
                continue

            self.tile_tensors[tile_id] = {
                "x": x,
                "y": y,
                "t": t,
                "length": length,
            }

            # Index for shifted sequence forecasting
            max_start = length - (self.context_length + self.horizon)
            self.index.extend([(tile_id, i) for i in range(max_start + 1)])


    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
        Returns:
            context: (batch, seq, context_dim)
            target: (batch, seq, 1)
        """
        tile_id, start_idx = self.index[idx]
        tensors = self.tile_tensors[tile_id]

        c_len, h_len = self.context_length, self.horizon

        x_context = tensors["x"][start_idx:start_idx + c_len]
        y_target = tensors["y"][start_idx + h_len:start_idx + c_len + h_len].unsqueeze(-1)
        y_timestamp = tensors["t"][start_idx + h_len:start_idx + c_len + h_len].unsqueeze(-1)

        return {
            "tile_id": tile_id,
            "context": x_context,
            "target": y_target,
            "target_timestamp": y_timestamp,
        }


class TileTimeSeriesWindowedDataset(Dataset):

    def __init__(self, df: pl.DataFrame, meta: dict,
                 context_length: int, use_features: bool,
                 stride: int = 1):

        self.meta = meta
        self.context_length = context_length
        self.horizon: int = 1
        self.stride = stride

        tile_col = meta["tile_column"]
        time_col = meta["time_column"]

        df = df.with_columns(pl.col(time_col).dt.timestamp().alias("__timestamp__"))

        self.tile_groups = df.partition_by(tile_col, as_dict=True)
        self.tile_tensors = {}
        self.index = []


        # -------------------------------------------------------------
        # Helper function to split a tile into contiguous time segments
        # -------------------------------------------------------------
        def split_into_contiguous(df: pl.DataFrame, time_col: str):
            df = df.with_columns(
                (pl.col(time_col).shift(-1) - pl.col(time_col)).alias("__dt__")
            )

            step = df["__dt__"][:-1].median()

            df = df.with_columns(
                (pl.col("__dt__") != step).fill_null(True).alias("__is_gap__")
            )

            df = df.with_columns(
                pl.col("__is_gap__").cast(pl.Int32).cum_sum().alias("__segment__")
            ).drop("__dt__", "__is_gap__")

            return df.partition_by("__segment__", as_dict=True)


        # ------------ MAIN LOOP OVER TILES --------------------------
        for tile_id, tile_df in self.tile_groups.items():

            tile_df = tile_df.sort(time_col)

            # Split tile into contiguous segments
            segments = split_into_contiguous(tile_df, time_col)

            for seg_df in segments.values():

                length = len(seg_df)
                min_required = self.context_length + self.horizon

                if length < min_required:
                    continue

                if use_features:
                    x = torch.from_numpy(seg_df.select(meta["features"]).to_numpy()).float()
                    y = torch.from_numpy(seg_df.select(meta["target"]).to_numpy().copy().squeeze()).float()
                else:
                    x = torch.from_numpy(seg_df.select(meta["target"]).to_numpy().copy()).long()
                    y = torch.from_numpy(seg_df.select(meta["target"]).to_numpy().copy().squeeze()).long()

                t = torch.from_numpy(seg_df.select("__timestamp__").to_numpy().copy().squeeze()).float()

                # unique segment ID
                seg_id = f"{tile_id}__{int(seg_df['__segment__'][0])}"

                self.tile_tensors[seg_id] = {
                    "x": x,
                    "y": y,
                    "t": t,
                    "length": length,
                }

                max_start = length - (self.context_length + self.horizon)
                starts = list(range(0, max_start + 1, self.stride))
                self.index.extend([(seg_id, s) for s in starts])


    def __len__(self):
        return len(self.index)


    def __getitem__(self, idx):
        tile_seg_id, start_idx = self.index[idx]
        tensors = self.tile_tensors[tile_seg_id]

        c_len, h_len = self.context_length, self.horizon

        x_context = tensors["x"][start_idx:start_idx + c_len]
        y_target = tensors["y"][start_idx + h_len:start_idx + c_len + h_len].unsqueeze(-1)
        y_timestamp = tensors["t"][start_idx + h_len:start_idx + c_len + h_len].unsqueeze(-1)

        return {
            "tile_id": tile_seg_id,
            "context": x_context,
            "target": y_target,
            "target_timestamp": y_timestamp,
        }

