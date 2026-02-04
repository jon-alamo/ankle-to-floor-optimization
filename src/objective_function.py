"""
Módulo con la función objetivo para la optimización.

Ejecuta el pipeline completo y calcula las métricas de evaluación.
"""

import numpy as np
import pandas as pd

from src.pipeline.a_load_data import ANKLE_COLUMNS_MAP, load_and_prepare_data
from src.pipeline.b_smooth_positions import smooth_positions
from src.pipeline.c_compute_vel import compute_velocity
from src.pipeline.d_compute_acc import compute_acceleration
from src.pipeline.e_determine_ankle_state import determine_ankle_state
from src.file_handler import get_ankle_data_path, get_annotations_path


def run_pipeline(
    window: int,
    vel_x_threshold: float,
    vel_y_threshold: float,
    acc_x_threshold: float,
    acc_y_threshold: float,
    min_distance_factor: float,
) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de clasificación de tobillos.

    Args:
        window: Tamaño de ventana para suavizado.
        vel_x_threshold: Umbral de velocidad horizontal.
        vel_y_threshold: Umbral de velocidad vertical.
        acc_x_threshold: Umbral de aceleración horizontal.
        acc_y_threshold: Umbral de aceleración vertical.
        min_distance_factor: Factor mínimo de distancia.

    Returns:
        DataFrame con predicciones y etiquetas para cada tobillo.
    """
    data = load_and_prepare_data(
        str(get_ankle_data_path()),
        str(get_annotations_path()),
    )

    data = smooth_positions(data, window)
    data = compute_velocity(data)
    data = compute_acceleration(data)
    data = determine_ankle_state(
        data,
        vel_x_threshold,
        vel_y_threshold,
        acc_x_threshold,
        acc_y_threshold,
        min_distance_factor,
    )

    return data


def flatten_predictions_and_labels(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplana todas las predicciones y etiquetas de todos los tobillos en arrays únicos.

    Concatena los datos de todos los tobillos tratándolos como un único conjunto
    para calcular métricas globales.

    Args:
        data: DataFrame con predicciones y etiquetas para cada tobillo.

    Returns:
        Tupla con (labels, predictions) como arrays concatenados.
    """
    valid_data = data.dropna()

    all_labels = []
    all_predictions = []

    for ankle_name in ANKLE_COLUMNS_MAP.keys():
        label_col = f"{ankle_name}_label"
        pred_col = f"{ankle_name}_pred"

        all_labels.extend(valid_data[label_col].values.tolist())
        all_predictions.extend(valid_data[pred_col].values.tolist())

    return np.array(all_labels), np.array(all_predictions)


def compute_metrics_with_tolerance(
    labels: np.ndarray,
    predictions: np.ndarray,
    tolerance: int = 1,
) -> dict:
    """
    Calcula métricas de precisión y recall con tolerancia temporal.

    Un TP se considera si la predicción coincide con alguna etiqueta
    dentro de +-tolerance frames.

    Args:
        labels: Array con las etiquetas reales.
        predictions: Array con las predicciones.
        tolerance: Número de frames de tolerancia.

    Returns:
        Diccionario con TP, FP, FN, precision y recall.
    """
    n = len(labels)
    tp = 0
    fp = 0

    matched_labels = set()

    for i in range(n):
        if predictions[i] == 1:
            found_match = False
            for j in range(max(0, i - tolerance), min(n, i + tolerance + 1)):
                if labels[j] == 1 and j not in matched_labels:
                    found_match = True
                    matched_labels.add(j)
                    break
            if found_match:
                tp += 1
            else:
                fp += 1

    fn = 0
    for i in range(n):
        if labels[i] == 1 and i not in matched_labels:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
    }


def compute_metrics(data: pd.DataFrame, tolerance: int = 1) -> dict:
    """
    Calcula métricas globales para todos los tobillos como un único conjunto.

    Args:
        data: DataFrame con predicciones y etiquetas.
        tolerance: Frames de tolerancia para considerar un TP.

    Returns:
        Diccionario con TP, FP, FN, precision y recall globales.
    """
    labels, predictions = flatten_predictions_and_labels(data)
    return compute_metrics_with_tolerance(labels, predictions, tolerance)


def objective_function(
    window: int,
    vel_x_threshold: float,
    vel_y_threshold: float,
    acc_x_threshold: float,
    acc_y_threshold: float,
    min_distance_factor: float,
    tolerance: int = 1,
) -> float:
    """
    Función objetivo para la optimización.

    Ejecuta el pipeline y devuelve una métrica a maximizar (F1-score).

    Args:
        window: Tamaño de ventana para suavizado.
        vel_x_threshold: Umbral de velocidad horizontal.
        vel_y_threshold: Umbral de velocidad vertical.
        acc_x_threshold: Umbral de aceleración horizontal.
        acc_y_threshold: Umbral de aceleración vertical.
        min_distance_factor: Factor mínimo de distancia.
        tolerance: Frames de tolerancia para métricas.

    Returns:
        F1-score como métrica a maximizar.
    """
    data = run_pipeline(
        window,
        vel_x_threshold,
        vel_y_threshold,
        acc_x_threshold,
        acc_y_threshold,
        min_distance_factor,
    )

    metrics = compute_metrics(data, tolerance)
    precision = metrics["precision"]
    recall = metrics["recall"]

    if precision + recall == 0:
        return 0.0

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score
