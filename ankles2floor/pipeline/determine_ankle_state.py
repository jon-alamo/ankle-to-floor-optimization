"""
Module for determining whether an ankle is grounded or not.

Uses velocity and acceleration thresholds to classify ankle state.
"""

import numpy as np
import pandas as pd


def compute_distance_factor(
    data: pd.DataFrame,
    ankle_columns: dict[str, dict[str, str]],
    min_distance_factor: float,
) -> pd.DataFrame:
    """
    Compute distance factor based on y coordinate.

    Objects farther away (lower y) have a smaller factor, while closer objects
    (higher y) have factor 1.

    Args:
        data: DataFrame with ankle positions.
        ankle_columns: Mapping of ankle names to column coordinates.
        min_distance_factor: Minimum factor for the farthest position.

    Returns:
        DataFrame with distance factor column for each ankle.
    """
    result = data.copy()

    all_y_values = []
    for ankle_name in ankle_columns.keys():
        y_col = f"{ankle_name}_y"
        all_y_values.extend(result[y_col].dropna().tolist())

    y_min = min(all_y_values)
    y_max = max(all_y_values)
    y_range = y_max - y_min

    for ankle_name in ankle_columns.keys():
        y_col = f"{ankle_name}_y"
        factor_col = f"{ankle_name}_distance_factor"

        normalized_y = (result[y_col] - y_min) / y_range
        result[factor_col] = min_distance_factor + normalized_y * (1 - min_distance_factor)

    return result


def determine_ankle_state(
    data: pd.DataFrame,
    ankle_columns: dict[str, dict[str, str]],
    vel_x_threshold: float,
    vel_y_threshold: float,
    acc_x_threshold: float | None,
    acc_y_threshold: float | None,
    use_acceleration: bool,
    min_distance_factor: float,
) -> pd.DataFrame:
    """
    Determine if each ankle is grounded (1) or moving (0).

    An ankle is considered grounded if both its velocity and acceleration
    (adjusted by distance factor) are below the thresholds.

    Args:
        data: DataFrame with computed velocities and accelerations.
        ankle_columns: Mapping of ankle names to column coordinates.
        vel_x_threshold: Horizontal velocity threshold (pixels/frame).
        vel_y_threshold: Vertical velocity threshold (pixels/frame).
        acc_x_threshold: Horizontal acceleration threshold (pixels/frame²).
        acc_y_threshold: Vertical acceleration threshold (pixels/frame²).
        min_distance_factor: Minimum distance factor for perspective adjustment.

    Returns:
        DataFrame with predicted state columns for each ankle.
    """
    result = compute_distance_factor(data, ankle_columns, min_distance_factor)

    for ankle_name in ankle_columns.keys():
        vel_x_col = f"{ankle_name}_vel_x"
        vel_y_col = f"{ankle_name}_vel_y"
        if use_acceleration:
            acc_x_col = f"{ankle_name}_acc_x"
            acc_y_col = f"{ankle_name}_acc_y"
        factor_col = f"{ankle_name}_distance_factor"
        pred_col = f"{ankle_name}_pred"

        adjusted_vel_x_threshold = vel_x_threshold * result[factor_col]
        adjusted_vel_y_threshold = vel_y_threshold * result[factor_col]
        if acc_x_threshold is not None and acc_y_threshold is not None:
            adjusted_acc_x_threshold = acc_x_threshold * result[factor_col]
            adjusted_acc_y_threshold = acc_y_threshold * result[factor_col]

        vel_x_ok = np.abs(result[vel_x_col]) <= adjusted_vel_x_threshold
        vel_y_ok = np.abs(result[vel_y_col]) <= adjusted_vel_y_threshold
        if acc_x_threshold is not None and acc_y_threshold is not None:
            acc_x_ok = np.abs(result[acc_x_col]) <= adjusted_acc_x_threshold
            acc_y_ok = np.abs(result[acc_y_col]) <= adjusted_acc_y_threshold
            result[pred_col] = (vel_x_ok & vel_y_ok & acc_x_ok & acc_y_ok).astype(int)
        else:
            result[pred_col] = (vel_x_ok & vel_y_ok).astype(int)

    return result
