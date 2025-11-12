from datetime import datetime, timedelta
from functools import reduce

import polars as pl

def temporal_train_val_split(
    df: pl.DataFrame, 
    meta: dict, 
    val_start_date: str,
    val_end_date: str | None = None,
):
    """
    Split the dataset into train and validation sets based on a timestamp cutoff.
    Args:
        df: Polars DataFrame
        meta: dict with "time_column"
        val_start_date: e.g. "2022-01-01"
    """
    time_col = meta["time_column"]
    cutoff = datetime.fromisoformat(val_start_date)

    train_df = df.filter(pl.col(time_col).dt.date() < cutoff)

    if val_end_date is None:
        val_df = df.filter(pl.col(time_col).dt.date() >= cutoff)
    else:
        val_df = df.filter(
            pl.col(time_col).dt.date() >= cutoff,
            pl.col(time_col).dt.date() < datetime.fromisoformat(val_end_date)
        )

    return train_df, val_df

def spatiotemporal_subset(
    df: pl.DataFrame,
    meta: dict,
    dates: list[tuple[str, str]],
    tiles: list[str],
    context_window_in_days: int,
):
    time_col = meta["time_column"]
    tile_col = meta["tile_column"]
    date_filter = []
    for start, end in dates:
        date_filter.append(
            (pl.col(time_col).dt.date() >= (datetime.fromisoformat(start) - timedelta(days=context_window_in_days))) &
            (pl.col(time_col).dt.date() < datetime.fromisoformat(end))
        )
    return df.filter(
        reduce(lambda a,b: a | b, date_filter)
    ).filter(pl.col(tile_col).is_in(tiles))
