"""
Módulo para suavizar las posiciones de los tobillos.

Aplica un rolling mean centrado para reducir el ruido en las coordenadas.
"""

import pandas as pd

from src.pipeline.a_load_data import ANKLE_COLUMNS_MAP


def smooth_positions(
    data: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Suaviza las posiciones x e y de cada tobillo usando rolling mean centrado.

    Args:
        data: DataFrame con posiciones de tobillos.
        window: Tamaño de la ventana para el rolling mean.

    Returns:
        DataFrame con las posiciones suavizadas.
    """
    result = data.copy()

    for ankle_name in ANKLE_COLUMNS_MAP.keys():
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
