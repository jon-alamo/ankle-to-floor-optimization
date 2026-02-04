"""
Módulo para calcular la velocidad de los tobillos.

La velocidad se calcula como la diferencia de posición entre frames consecutivos.
"""

import pandas as pd

from src.pipeline.a_load_data import ANKLE_COLUMNS_MAP


def compute_velocity(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la velocidad en x e y para cada tobillo.

    La velocidad se define como la diferencia de posición entre frames consecutivos
    (en píxeles/frame).

    Args:
        data: DataFrame con posiciones de tobillos (suavizadas o no).

    Returns:
        DataFrame con columnas adicionales de velocidad para cada tobillo.
    """
    result = data.copy()

    for ankle_name in ANKLE_COLUMNS_MAP.keys():
        x_col = f"{ankle_name}_x"
        y_col = f"{ankle_name}_y"
        vel_x_col = f"{ankle_name}_vel_x"
        vel_y_col = f"{ankle_name}_vel_y"

        result[vel_x_col] = result[x_col].diff()
        result[vel_y_col] = result[y_col].diff()

    return result
