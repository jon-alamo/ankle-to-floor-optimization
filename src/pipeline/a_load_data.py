"""
Módulo para cargar y preparar los datos de tobillos y anotaciones.

Este módulo se encarga de leer los archivos CSV, hacer el merge entre ellos
y filtrar únicamente los tobillos y frames que están anotados.
"""

import pandas as pd


ANKLE_COLUMNS_MAP = {
    "low_left_left_ankle": {
        "x": "player_low_left_left_ankle_x",
        "y": "player_low_left_left_ankle_y",
    },
    "low_left_right_ankle": {
        "x": "player_low_left_right_ankle_x",
        "y": "player_low_left_right_ankle_y",
    },
    "up_drive_left_ankle": {
        "x": "player_up_drive_left_ankle_x",
        "y": "player_up_drive_left_ankle_y",
    },
    "up_drive_right_ankle": {
        "x": "player_up_drive_right_ankle_x",
        "y": "player_up_drive_right_ankle_y",
    },
}


def load_ankle_data(ankle_data_path: str) -> pd.DataFrame:
    """
    Carga el dataset de posiciones de tobillos.

    Args:
        ankle_data_path: Ruta al archivo CSV con datos de tobillos.

    Returns:
        DataFrame con los datos de tobillos.
    """
    return pd.read_csv(ankle_data_path)


def load_annotations(annotations_path: str) -> pd.DataFrame:
    """
    Carga el dataset de anotaciones de tobillos en el suelo.

    Args:
        annotations_path: Ruta al archivo CSV con anotaciones.

    Returns:
        DataFrame con las anotaciones.
    """
    df = pd.read_csv(annotations_path)
    df = df.dropna(subset=["source_index"])
    df["source_index"] = df["source_index"].astype(int)
    return df


def merge_data(
    ankle_data: pd.DataFrame,
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combina los datos de tobillos con las anotaciones usando source_frame/source_index.

    Args:
        ankle_data: DataFrame con posiciones de tobillos.
        annotations: DataFrame con anotaciones de estado.

    Returns:
        DataFrame combinado con solo los frames anotados.
    """
    merged = pd.merge(
        ankle_data,
        annotations,
        left_on="source_frame",
        right_on="source_index",
        how="inner",
    )
    return merged


def extract_ankle_positions(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae las posiciones x e y de los tobillos anotados en un formato estructurado.

    Args:
        merged_data: DataFrame combinado con datos y anotaciones.

    Returns:
        DataFrame con columnas para cada tobillo (x, y) y su anotación.
    """
    result = pd.DataFrame()
    result["frame_index"] = merged_data["frame_index"]
    result["source_frame"] = merged_data["source_frame"]

    for ankle_name, coords in ANKLE_COLUMNS_MAP.items():
        result[f"{ankle_name}_x"] = merged_data[coords["x"]]
        result[f"{ankle_name}_y"] = merged_data[coords["y"]]
        result[f"{ankle_name}_label"] = merged_data[ankle_name]

    return result


def load_and_prepare_data(
    ankle_data_path: str,
    annotations_path: str,
) -> pd.DataFrame:
    """
    Carga y prepara los datos completos para el pipeline.

    Args:
        ankle_data_path: Ruta al archivo CSV con datos de tobillos.
        annotations_path: Ruta al archivo CSV con anotaciones.

    Returns:
        DataFrame preparado con posiciones y anotaciones de los tobillos relevantes.
    """
    ankle_data = load_ankle_data(ankle_data_path)
    annotations = load_annotations(annotations_path)
    merged = merge_data(ankle_data, annotations)
    prepared = extract_ankle_positions(merged)
    return prepared
