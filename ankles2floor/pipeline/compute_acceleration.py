"""
Module for computing ankle accelerations.

Acceleration is calculated as the velocity difference between consecutive frames.
"""

import pandas as pd


def compute_acceleration(
    data: pd.DataFrame,
    ankle_columns: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """
    Compute acceleration in x and y for each ankle.

    Acceleration is defined as the velocity difference between consecutive frames
    (in pixels/frame²).

    Args:
        data: DataFrame with computed ankle velocities.
        ankle_columns: Mapping of ankle names to column coordinates.

    Returns:
        DataFrame with additional acceleration columns for each ankle.
    """
    result = data.copy()

    for ankle_name in ankle_columns.keys():
        vel_x_col = f"{ankle_name}_vel_x"
        vel_y_col = f"{ankle_name}_vel_y"
        acc_x_col = f"{ankle_name}_acc_x"
        acc_y_col = f"{ankle_name}_acc_y"

        result[acc_x_col] = result[vel_x_col].diff()
        result[acc_y_col] = result[vel_y_col].diff()

    return result
