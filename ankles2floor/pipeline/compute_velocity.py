"""
Module for computing ankle velocities.

Velocity is calculated as the position difference between consecutive frames.
"""

import pandas as pd


def compute_velocity(
    data: pd.DataFrame,
    ankle_columns: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """
    Compute velocity in x and y for each ankle.

    Velocity is defined as the position difference between consecutive frames
    (in pixels/frame).

    Args:
        data: DataFrame with ankle positions (smoothed or not).
        ankle_columns: Mapping of ankle names to column coordinates.

    Returns:
        DataFrame with additional velocity columns for each ankle.
    """
    result = data.copy()

    for ankle_name in ankle_columns.keys():
        x_col = f"{ankle_name}_x"
        y_col = f"{ankle_name}_y"
        vel_x_col = f"{ankle_name}_vel_x"
        vel_y_col = f"{ankle_name}_vel_y"

        result[vel_x_col] = result[x_col].diff()
        result[vel_y_col] = result[y_col].diff()

    return result
