"""
Core module with the main public API for ankle-to-floor classification.
"""

import pandas as pd

from ankles2floor.config import best_parameters
from ankles2floor.pipeline.smooth_positions import smooth_positions
from ankles2floor.pipeline.compute_velocity import compute_velocity
from ankles2floor.pipeline.compute_acceleration import compute_acceleration
from ankles2floor.pipeline.determine_ankle_state import determine_ankle_state
from ankles2floor.pipeline.interpolate_positions import interpolate_positions


def get_ankles_in_floor(
    data: pd.DataFrame,
    ankle_columns: dict[str, dict[str, str]],
    window: int | None = None,
    vel_x_threshold: float | None = None,
    vel_y_threshold: float | None = None,
    acc_x_threshold: float | None = None,
    acc_y_threshold: float | None = None,
    use_acceleration: bool | None = None,
    min_distance_factor: float | None = None
) -> pd.DataFrame:
    """
    Classify ankle movements to determine when feet are grounded.

    Takes a DataFrame with ankle x/y coordinates in pixels and returns predictions
    for each ankle indicating whether the foot is on the floor (1) or not (0).

    When z_offset is provided, the function also:
    - Linearly interpolates x/y positions between consecutive grounded positions
    - Computes z-coordinates: z_offset when grounded, parabolic trajectory when airborne

    Args:
        data: DataFrame containing ankle position columns and a "time" column
            (required when z_offset is provided).
        ankle_columns: Dictionary mapping ankle names to their x/y column names.
            Example:
            {
                "left_ankle": {"x": "left_ankle_x", "y": "left_ankle_y"},
                "right_ankle": {"x": "right_ankle_x", "y": "right_ankle_y"},
            }
        window: Smoothing window size for rolling mean. Defaults to optimized value.
        vel_x_threshold: Horizontal velocity threshold (pixels/frame).
            Defaults to optimized value.
        vel_y_threshold: Vertical velocity threshold (pixels/frame).
            Defaults to optimized value.
        acc_x_threshold: Horizontal acceleration threshold (pixels/frame²).
            Defaults to optimized value.
        acc_y_threshold: Vertical acceleration threshold (pixels/frame²).
            Defaults to optimized value.
        min_distance_factor: Minimum distance factor for perspective adjustment.
            Defaults to optimized value.
        z_offset: Height value for ankles in contact with the floor. When provided,
            enables position interpolation and z-coordinate computation.

    Returns:
        DataFrame with original data plus prediction columns for each ankle.
        Prediction columns are named "{ankle_name}_pred" with values 0 or 1.
        When z_offset is provided, also includes "{ankle_name}_z" columns.

    Example:
        >>> import pandas as pd
        >>> from ankles2floor import get_ankles_in_floor
        >>> 
        >>> data = pd.DataFrame({
        ...     "frame": range(100),
        ...     "time": [...],
        ...     "left_ankle_x": [...],
        ...     "left_ankle_y": [...],
        ...     "right_ankle_x": [...],
        ...     "right_ankle_y": [...],
        ... })
        >>> 
        >>> ankle_columns = {
        ...     "left_ankle": {"x": "left_ankle_x", "y": "left_ankle_y"},
        ...     "right_ankle": {"x": "right_ankle_x", "y": "right_ankle_y"},
        ... }
        >>> 
        >>> result = get_ankles_in_floor(data, ankle_columns, z_offset=0.0)
        >>> # result contains "left_ankle_pred", "right_ankle_pred",
        >>> # "left_ankle_z", and "right_ankle_z" columns
    """
    params = {
        "window": window if window is not None else best_parameters["window"],
        "vel_x_threshold": vel_x_threshold if vel_x_threshold is not None else best_parameters["vel_x_threshold"],
        "vel_y_threshold": vel_y_threshold if vel_y_threshold is not None else best_parameters["vel_y_threshold"],
        "acc_x_threshold": acc_x_threshold if acc_x_threshold is not None else best_parameters["acc_x_threshold"],
        "acc_y_threshold": acc_y_threshold if acc_y_threshold is not None else best_parameters["acc_y_threshold"],
        "use_acceleration": use_acceleration if use_acceleration is not None else best_parameters["use_acceleration"],
        "min_distance_factor": min_distance_factor if min_distance_factor is not None else best_parameters["min_distance_factor"],
    }

    result = _prepare_data(data, ankle_columns)
    result = smooth_positions(result, ankle_columns, params["window"])
    result = compute_velocity(result, ankle_columns)
    result = compute_acceleration(result, ankle_columns)
    result = determine_ankle_state(
        result,
        ankle_columns,
        params["vel_x_threshold"],
        params["vel_y_threshold"],
        params["acc_x_threshold"],
        params["acc_y_threshold"],
        params['use_acceleration'],
        params["min_distance_factor"],
    )


    result = interpolate_positions(result, ankle_columns)

    return result


def _prepare_data(
    data: pd.DataFrame,
    ankle_columns: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """
    Prepare input data by extracting ankle columns into standardized format.

    Args:
        data: Input DataFrame with ankle coordinates.
        ankle_columns: Mapping of ankle names to column names.

    Returns:
        DataFrame with standardized column names for processing.
    """
    result = data.copy()

    for ankle_name, coords in ankle_columns.items():
        result[f"{ankle_name}_x"] = result[coords["x"]]
        result[f"{ankle_name}_y"] = result[coords["y"]]

    return result
