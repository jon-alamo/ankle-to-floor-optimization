"""
Module for interpolating ankle positions.
When ankles are in contact with the floor, x/y positions are linearly interpolated
between consecutive grounded positions. 
"""

import pandas as pd


def interpolate_positions(
    data: pd.DataFrame,
    ankle_columns: dict[str, dict[str, str]]
) -> pd.DataFrame:
    """
    Interpolate x/y positions for grounded ankles and compute z-coordinates.

    For each ankle:
    - X and Y positions are linearly interpolated between consecutive floor contacts.

    Args:
        data: DataFrame with ankle positions and predictions.
        ankle_columns: Mapping of ankle names to column coordinates.

    Returns:
        DataFrame with interpolated positions for each ankle.
    """
    result = data.copy()

    for ankle_name in ankle_columns.keys():
        result = _interpolate_ankle_positions(result, ankle_name)

    return result


def _interpolate_ankle_positions(
    data: pd.DataFrame,
    ankle_name: str,
) -> pd.DataFrame:
    """
    Interpolate positions and compute z for a single ankle.

    Args:
        data: DataFrame with ankle data.
        ankle_name: Name of the ankle to process.

    Returns:
        DataFrame with interpolated x, y for the ankle.
    """
    result = data.copy()
    
    pred_col = f"{ankle_name}_pred"
    x_col = f"{ankle_name}_x"
    y_col = f"{ankle_name}_y"


    grounded_mask = result[pred_col] == 1
    grounded_indices = result.index[grounded_mask].tolist()

    if len(grounded_indices) == 0:
        return result

    result = _interpolate_xy_for_grounded(result, x_col, y_col, grounded_indices)

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

