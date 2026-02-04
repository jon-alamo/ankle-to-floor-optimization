"""
Módulo para manejar rutas de archivos del proyecto.
"""

from pathlib import Path


def get_project_root() -> Path:
    """
    Obtiene la ruta raíz del proyecto.

    Returns:
        Path al directorio raíz del proyecto.
    """
    return Path(__file__).parent.parent


def get_datasets_path() -> Path:
    """
    Obtiene la ruta al directorio de datasets.

    Returns:
        Path al directorio de datasets.
    """
    return get_project_root() / "datasets"


def get_ankle_data_path() -> Path:
    """
    Obtiene la ruta al archivo de datos de tobillos.

    Returns:
        Path al archivo CSV de datos de tobillos.
    """
    return get_datasets_path() / "bcn-finals-2022-fem-ankle-and-shot-data.csv"


def get_annotations_path() -> Path:
    """
    Obtiene la ruta al archivo de anotaciones.

    Returns:
        Path al archivo CSV de anotaciones.
    """
    return get_datasets_path() / "bcn-finals-2022-fem-ankles-in-floor-annotations.csv"


