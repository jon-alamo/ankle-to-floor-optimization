"""
Module for smoothing ankle positions.

Applies a centered rolling mean to reduce noise in the coordinates.
"""

import pandas as pd


def smooth_positions(
    data: pd.DataFrame,
    ankle_columns: dict[str, dict[str, str]],
    window: int,
) -> pd.DataFrame:
    """
    Smooth x and y positions of each ankle using centered rolling mean.

    Args:
        data: DataFrame with ankle positions.
        ankle_columns: Mapping of ankle names to column coordinates.
        window: Window size for the rolling mean.

    Returns:
        DataFrame with smoothed positions.
    """
    result = data.copy()

    for ankle_name in ankle_columns.keys():
        x_col = f"{ankle_name}_x"
        y_col = f"{ankle_name}_y"

        result[x_col] = (
            result[x_col]
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )
        result[y_col] = (
            result[y_col]
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )

    return result
