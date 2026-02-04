"""
Módulo para calcular la aceleración de los tobillos.

La aceleración se calcula como la diferencia de velocidad entre frames consecutivos.
"""

import pandas as pd

from src.pipeline.a_load_data import ANKLE_COLUMNS_MAP


def compute_acceleration(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la aceleración en x e y para cada tobillo.

    La aceleración se define como la diferencia de velocidad entre frames consecutivos
    (en píxeles/frame²).

    Args:
        data: DataFrame con velocidades de tobillos calculadas.

    Returns:
        DataFrame con columnas adicionales de aceleración para cada tobillo.
    """
    result = data.copy()

    for ankle_name in ANKLE_COLUMNS_MAP.keys():
        vel_x_col = f"{ankle_name}_vel_x"
        vel_y_col = f"{ankle_name}_vel_y"
        acc_x_col = f"{ankle_name}_acc_x"
        acc_y_col = f"{ankle_name}_acc_y"

        result[acc_x_col] = result[vel_x_col].diff()
        result[acc_y_col] = result[vel_y_col].diff()

    return result
