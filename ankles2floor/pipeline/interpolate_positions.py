"""
Module for interpolating ankle positions and computing z-coordinates.

When ankles are in contact with the floor, x/y positions are linearly interpolated
between consecutive grounded positions. Z-coordinates are computed using the floor
offset for grounded positions and a parabolic trajectory for airborne positions.
"""

import numpy as np
import pandas as pd


def interpolate_positions(
    data: pd.DataFrame,
    ankle_columns: dict[str, dict[str, str]],
    z_offset: float,
) -> pd.DataFrame:
    """
    Interpolate x/y positions for grounded ankles and compute z-coordinates.

    For each ankle:
    - X and Y positions are linearly interpolated between consecutive floor contacts
    - Z is set to z_offset when grounded, and follows a parabolic trajectory when airborne

    Args:
        data: DataFrame with ankle positions and predictions.
        ankle_columns: Mapping of ankle names to column coordinates.
        z_offset: Height value for ankles in contact with the floor.

    Returns:
        DataFrame with interpolated positions and z-coordinates for each ankle.
    """
    result = data.copy()

    for ankle_name in ankle_columns.keys():
        result = _interpolate_ankle_positions(result, ankle_name, z_offset)

    return result


def _interpolate_ankle_positions(
    data: pd.DataFrame,
    ankle_name: str,
    z_offset: float,
) -> pd.DataFrame:
    """
    Interpolate positions and compute z for a single ankle.

    Args:
        data: DataFrame with ankle data.
        ankle_name: Name of the ankle to process.
        z_offset: Height value for grounded positions.

    Returns:
        DataFrame with interpolated x, y and computed z for the ankle.
    """
    result = data.copy()
    
    pred_col = f"{ankle_name}_pred"
    x_col = f"{ankle_name}_x"
    y_col = f"{ankle_name}_y"
    z_col = f"{ankle_name}_z"

    result[z_col] = np.nan

    grounded_mask = result[pred_col] == 1
    grounded_indices = result.index[grounded_mask].tolist()

    if len(grounded_indices) == 0:
        result[z_col] = np.nan
        return result

    result = _interpolate_xy_for_grounded(result, x_col, y_col, grounded_indices)
    result = _compute_z_coordinates(result, pred_col, z_col, z_offset)

    return result


def _interpolate_xy_for_grounded(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    grounded_indices: list[int],
) -> pd.DataFrame:
    """
    Linearly interpolate x and y between consecutive grounded positions.

    Args:
        data: DataFrame with position data.
        x_col: Name of the x coordinate column.
        y_col: Name of the y coordinate column.
        grounded_indices: List of indices where ankle is grounded.

    Returns:
        DataFrame with interpolated x and y values for grounded segments.
    """
    result = data.copy()

    for i in range(len(grounded_indices) - 1):
        start_idx = grounded_indices[i]
        end_idx = grounded_indices[i + 1]

        if end_idx - start_idx <= 1:
            continue

        start_x = float(result.loc[start_idx, x_col])  # type: ignore[arg-type]
        end_x = float(result.loc[end_idx, x_col])  # type: ignore[arg-type]
        start_y = float(result.loc[start_idx, y_col])  # type: ignore[arg-type]
        end_y = float(result.loc[end_idx, y_col])  # type: ignore[arg-type]

        indices_between = range(start_idx, end_idx + 1)
        num_steps = end_idx - start_idx

        for j, idx in enumerate(indices_between):
            t = j / num_steps
            result.loc[idx, x_col] = start_x + t * (end_x - start_x)
            result.loc[idx, y_col] = start_y + t * (end_y - start_y)

    return result


def _compute_z_coordinates(
    data: pd.DataFrame,
    pred_col: str,
    z_col: str,
    z_offset: float,
) -> pd.DataFrame:
    """
    Compute z-coordinates: z_offset when grounded, parabolic trajectory when airborne.

    The parabolic trajectory is computed such that z equals z_offset at both
    the start and end of the airborne segment, with the apex determined by
    the time elapsed (using physics: z = z_offset + 0.5 * g * t * (T - t),
    where T is total airborne time).

    Args:
        data: DataFrame with ankle data.
        pred_col: Name of the prediction column.
        z_col: Name of the z coordinate column.
        z_offset: Height value for grounded positions.

    Returns:
        DataFrame with computed z-coordinates.
    """
    result = data.copy()

    grounded_mask = result[pred_col] == 1
    result.loc[grounded_mask, z_col] = z_offset

    airborne_segments = _find_airborne_segments(result, pred_col)

    for start_idx, end_idx in airborne_segments:
        result = _compute_parabolic_z(result, z_col, z_offset, start_idx, end_idx)

    return result


def _find_airborne_segments(
    data: pd.DataFrame,
    pred_col: str,
) -> list[tuple[int, int]]:
    """
    Find segments where the ankle is airborne (not grounded).

    Args:
        data: DataFrame with prediction data.
        pred_col: Name of the prediction column.

    Returns:
        List of tuples (start_idx, end_idx) for each airborne segment.
        start_idx is the last grounded index before airborne.
        end_idx is the first grounded index after airborne.
    """
    segments = []
    indices = data.index.tolist()
    predictions = data[pred_col].values

    i = 0
    while i < len(indices):
        if predictions[i] == 1:
            start_grounded_idx = indices[i]
            i += 1
            
            while i < len(indices) and predictions[i] == 0:
                i += 1
            
            if i < len(indices) and predictions[i] == 1:
                end_grounded_idx = indices[i]
                if end_grounded_idx - start_grounded_idx > 1:
                    segments.append((start_grounded_idx, end_grounded_idx))
        else:
            i += 1

    return segments


def _compute_parabolic_z(
    data: pd.DataFrame,
    z_col: str,
    z_offset: float,
    start_idx: int,
    end_idx: int,
) -> pd.DataFrame:
    """
    Compute parabolic z trajectory for an airborne segment.

    Given start and end positions at z_offset and the time elapsed,
    computes the unique parabolic curve where:
    z(t) = z_offset + h * 4 * (t/T) * (1 - t/T)
    
    where T is total time and h is the maximum height reached at t=T/2.
    The parabola is symmetric with z=z_offset at t=0 and t=T.

    Args:
        data: DataFrame with time and position data.
        z_col: Name of the z coordinate column.
        z_offset: Height at grounded positions.
        start_idx: Index of last grounded position before airborne.
        end_idx: Index of first grounded position after airborne.

    Returns:
        DataFrame with z values for the airborne segment.
    """
    result = data.copy()

    start_time = float(result.loc[start_idx, "time"])  # type: ignore[arg-type]
    end_time = float(result.loc[end_idx, "time"])  # type: ignore[arg-type]
    total_time = end_time - start_time

    if total_time <= 0:
        return result

    gravity = 9.81
    max_height = 0.5 * gravity * (total_time / 2) ** 2

    for idx in range(start_idx + 1, end_idx):
        current_time = float(result.loc[idx, "time"])  # type: ignore[arg-type]
        t = current_time - start_time
        normalized_t = t / total_time
        
        z_value = z_offset + max_height * 4 * normalized_t * (1 - normalized_t)
        result.loc[idx, z_col] = z_value

    return result
