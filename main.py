"""
Módulo principal para la optimización de parámetros del clasificador de tobillos.

Utiliza Optuna para encontrar los mejores parámetros que maximizan el F1-score
en la clasificación de tobillos anclados al suelo.
"""

import optuna
from optuna.trial import Trial

from src.objective_function import objective_function, run_pipeline, compute_metrics


def create_objective(tolerance: int = 1):
    """
    Crea la función objetivo para Optuna.

    Args:
        tolerance: Frames de tolerancia para las métricas.

    Returns:
        Función objetivo que recibe un trial de Optuna.
    """
    def objective(trial: Trial) -> float:
        window = trial.suggest_int("window", 2, 4)
        vel_x_threshold = trial.suggest_float("vel_x_threshold", 1.5, 2.5)
        vel_y_threshold = trial.suggest_float("vel_y_threshold", 3., 4.)
        acc_x_threshold = trial.suggest_float("acc_x_threshold", 10, 20.0)
        acc_y_threshold = trial.suggest_float("acc_y_threshold", 10, 20.0)
        min_distance_factor = trial.suggest_float("min_distance_factor", 0.5, 8.0)

        return objective_function(
            window=window,
            vel_x_threshold=vel_x_threshold,
            vel_y_threshold=vel_y_threshold,
            acc_x_threshold=acc_x_threshold,
            acc_y_threshold=acc_y_threshold,
            min_distance_factor=min_distance_factor,
            tolerance=tolerance,
        )

    return objective


def run_optimization(
    n_trials: int = 100,
    tolerance: int = 1,
    show_progress_bar: bool = True,
) -> optuna.Study:
    """
    Ejecuta la optimización de hiperparámetros.

    Args:
        n_trials: Número de trials a ejecutar.
        tolerance: Frames de tolerancia para las métricas.
        show_progress_bar: Si mostrar barra de progreso.

    Returns:
        Estudio de Optuna con los resultados.
    """
    study = optuna.create_study(
        direction="maximize",
        study_name="ankle-floor-classification",
    )

    study.optimize(
        create_objective(tolerance),
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )

    return study


def print_results(study: optuna.Study, tolerance: int = 1) -> None:
    """
    Imprime los resultados de la optimización.

    Args:
        study: Estudio de Optuna completado.
        tolerance: Frames de tolerancia usados.
    """
    print("\n" + "=" * 60)
    print("RESULTADOS DE LA OPTIMIZACIÓN")
    print("=" * 60)

    print(f"\nMejor F1-score: {study.best_value:.4f}")
    print("\nMejores parámetros:")

    for param_name, param_value in study.best_params.items():
        if isinstance(param_value, float):
            print(f"  {param_name}: {param_value:.4f}")
        else:
            print(f"  {param_name}: {param_value}")

    print("\n" + "-" * 60)
    print("Métricas con los mejores parámetros:")
    print("-" * 60)

    data = run_pipeline(**study.best_params)
    metrics = compute_metrics(data, tolerance)

    print(f"\n  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  TP: {metrics['tp']}")
    print(f"  FP: {metrics['fp']}")
    print(f"  FN: {metrics['fn']}")


def main() -> None:
    """
    Punto de entrada principal de la aplicación.
    """
    print("Iniciando optimización de parámetros para clasificación de tobillos...")
    print("Esto puede tardar varios minutos dependiendo del número de trials.\n")

    n_trials = 1000
    tolerance = 1

    study = run_optimization(
        n_trials=n_trials,
        tolerance=tolerance,
        show_progress_bar=True,
    )

    print_results(study, tolerance)


if __name__ == "__main__":
    main()
