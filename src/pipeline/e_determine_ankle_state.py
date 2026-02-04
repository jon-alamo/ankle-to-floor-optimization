"""
Módulo para determinar si un tobillo está anclado al suelo o no.

Utiliza umbrales de velocidad y aceleración para clasificar el estado del tobillo.
"""

import numpy as np
import pandas as pd

from src.pipeline.a_load_data import ANKLE_COLUMNS_MAP


def compute_distance_factor(
    data: pd.DataFrame,
    min_distance_factor: float,
) -> pd.DataFrame:
    """
    Calcula el factor de distancia basado en la coordenada y.

    Los objetos más lejanos (y menor) tienen un factor menor, mientras que
    los más cercanos (y mayor) tienen factor 1.

    Args:
        data: DataFrame con posiciones de tobillos.
        min_distance_factor: Factor mínimo para la posición más lejana.

    Returns:
        DataFrame con columna de factor de distancia para cada tobillo.
    """
    result = data.copy()

    all_y_values = []
    for ankle_name in ANKLE_COLUMNS_MAP.keys():
        y_col = f"{ankle_name}_y"
        all_y_values.extend(result[y_col].dropna().tolist())

    y_min = min(all_y_values)
    y_max = max(all_y_values)
    y_range = y_max - y_min

    for ankle_name in ANKLE_COLUMNS_MAP.keys():
        y_col = f"{ankle_name}_y"
        factor_col = f"{ankle_name}_distance_factor"

        normalized_y = (result[y_col] - y_min) / y_range
        result[factor_col] = min_distance_factor + normalized_y * (1 - min_distance_factor)

    return result


def determine_ankle_state(
    data: pd.DataFrame,
    vel_x_threshold: float,
    vel_y_threshold: float,
    acc_x_threshold: float,
    acc_y_threshold: float,
    min_distance_factor: float,
) -> pd.DataFrame:
    """
    Determina si cada tobillo está anclado (1) o en movimiento (0).

    Un tobillo se considera anclado si tanto su velocidad como su aceleración
    (ajustadas por el factor de distancia) están por debajo de los umbrales.

    Args:
        data: DataFrame con velocidades y aceleraciones calculadas.
        vel_x_threshold: Umbral de velocidad horizontal (píxeles/frame).
        vel_y_threshold: Umbral de velocidad vertical (píxeles/frame).
        acc_x_threshold: Umbral de aceleración horizontal (píxeles/frame²).
        acc_y_threshold: Umbral de aceleración vertical (píxeles/frame²).
        min_distance_factor: Factor mínimo de distancia para ajuste por perspectiva.

    Returns:
        DataFrame con columnas de estado predicho para cada tobillo.
    """
    result = compute_distance_factor(data, min_distance_factor)

    for ankle_name in ANKLE_COLUMNS_MAP.keys():
        vel_x_col = f"{ankle_name}_vel_x"
        vel_y_col = f"{ankle_name}_vel_y"
        acc_x_col = f"{ankle_name}_acc_x"
        acc_y_col = f"{ankle_name}_acc_y"
        factor_col = f"{ankle_name}_distance_factor"
        pred_col = f"{ankle_name}_pred"

        adjusted_vel_x_threshold = vel_x_threshold * result[factor_col]
        adjusted_vel_y_threshold = vel_y_threshold * result[factor_col]
        adjusted_acc_x_threshold = acc_x_threshold * result[factor_col]
        adjusted_acc_y_threshold = acc_y_threshold * result[factor_col]

        vel_x_ok = np.abs(result[vel_x_col]) <= adjusted_vel_x_threshold
        vel_y_ok = np.abs(result[vel_y_col]) <= adjusted_vel_y_threshold
        acc_x_ok = np.abs(result[acc_x_col]) <= adjusted_acc_x_threshold
        acc_y_ok = np.abs(result[acc_y_col]) <= adjusted_acc_y_threshold

        result[pred_col] = (vel_x_ok & vel_y_ok & acc_x_ok & acc_y_ok).astype(int)

    return result
