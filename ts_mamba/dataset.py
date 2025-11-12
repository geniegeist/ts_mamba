import polars as pl
import torch
from torch.utils.data import Dataset

class TileTimeSeriesDataset(Dataset):
    def __init__(self, df: pl.DataFrame, meta: dict, context_length: int):
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
        for tile_id, tile_df in self.tile_groups.items():
            # Sort by time if not already sorted
            tile_df = tile_df.sort(time_col)

            # Extract feature, target, and timestamp tensors
            x = torch.from_numpy(tile_df.select(meta["features"]).to_numpy()).float()
            # Copy these two tensors to suppress warning of non-writeable tensors
            y = torch.from_numpy(tile_df.select(meta["target"]).to_numpy().copy().squeeze()).float()
            t = torch.from_numpy(tile_df.select("__timestamp__").to_numpy().copy().squeeze()).float()

            self.tile_tensors[tile_id] = {
                "x": x,
                "y": y,
                "t": t,
                "length": len(tile_df),
            }

        self.index = []
        for tile_id, tensors in self.tile_tensors.items():
            max_start = tensors["length"] - (context_length + self.horizon)
            assert max_start > 0, f"Tile id {tile_id} has not enough context size"
            self.index.extend([(tile_id, i) for i in range(max_start)])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
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
