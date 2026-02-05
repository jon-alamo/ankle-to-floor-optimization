"""
Optimization module for finding best ankle classification parameters.

Uses Optuna for hyperparameter optimization with the annotated dataset.
This module is optional and requires the 'optimization' extra dependency.
"""

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


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


def _get_datasets_path():
    """Get path to datasets directory."""
    from pathlib import Path
    return Path(__file__).parent / "datasets"


def load_and_prepare_data(
    ankle_data_path: str,
    annotations_path: str,
) -> pd.DataFrame:
    """
    Load and prepare data for optimization pipeline.

    Args:
        ankle_data_path: Path to ankle data CSV file.
        annotations_path: Path to annotations CSV file.

    Returns:
        Prepared DataFrame with positions and annotations.
    """
    ankle_data = pd.read_csv(ankle_data_path)
    annotations = pd.read_csv(annotations_path)
    annotations = annotations.dropna(subset=["source_index"])
    annotations["source_index"] = annotations["source_index"].astype(int)

    merged = pd.merge(
        ankle_data,
        annotations,
        left_on="source_frame",
        right_on="source_index",
        how="inner",
    )

    result = pd.DataFrame()
    result["frame_index"] = merged["frame_index"]
    result["source_frame"] = merged["source_frame"]

    for ankle_name, coords in ANKLE_COLUMNS_MAP.items():
        result[f"{ankle_name}_x"] = merged[coords["x"]]
        result[f"{ankle_name}_y"] = merged[coords["y"]]
        result[f"{ankle_name}_label"] = merged[ankle_name]

    return result


def run_optimization_pipeline(
    data: pd.DataFrame,
    window: int,
    vel_x_threshold: float,
    vel_y_threshold: float,
    acc_x_threshold: float | None,
    acc_y_threshold: float | None,
    use_acceleration: bool,
    min_distance_factor: float,
) -> pd.DataFrame:
    """
    Run the classification pipeline for optimization.

    Args:
        data: Prepared DataFrame with ankle positions and labels.
        window: Smoothing window size.
        vel_x_threshold: Horizontal velocity threshold.
        vel_y_threshold: Vertical velocity threshold.
        acc_x_threshold: Horizontal acceleration threshold.
        acc_y_threshold: Vertical acceleration threshold.
        min_distance_factor: Minimum distance factor.

    Returns:
        DataFrame with predictions and labels.
    """
    from ankles2floor.pipeline.smooth_positions import smooth_positions
    from ankles2floor.pipeline.compute_velocity import compute_velocity
    from ankles2floor.pipeline.compute_acceleration import compute_acceleration
    from ankles2floor.pipeline.determine_ankle_state import determine_ankle_state

    result = smooth_positions(data, ANKLE_COLUMNS_MAP, window)
    result = compute_velocity(result, ANKLE_COLUMNS_MAP)
    if use_acceleration:
        result = compute_acceleration(result, ANKLE_COLUMNS_MAP)
    else:
        acc_x_threshold = None
        acc_y_threshold = None 

    result = determine_ankle_state(
        result,
        ANKLE_COLUMNS_MAP,
        vel_x_threshold,
        vel_y_threshold,
        acc_x_threshold,
        acc_y_threshold,
        min_distance_factor,
    )

    return result


def compute_metrics_with_tolerance(
    labels: np.ndarray,
    predictions: np.ndarray,
    tolerance: int = 1,
) -> dict:
    """
    Compute precision and recall metrics with temporal tolerance.

    A TP is counted if the prediction matches any label within +-tolerance frames.

    Args:
        labels: Array with ground truth labels.
        predictions: Array with predictions.
        tolerance: Number of tolerance frames.

    Returns:
        Dictionary with TP, FP, FN, precision, and recall.
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
    Compute global metrics for all ankles as a single set.

    Args:
        data: DataFrame with predictions and labels.
        tolerance: Tolerance frames to consider a TP.

    Returns:
        Dictionary with global TP, FP, FN, precision, and recall.
    """
    valid_data = data.dropna()
    all_labels = []
    all_predictions = []

    for ankle_name in ANKLE_COLUMNS_MAP.keys():
        label_col = f"{ankle_name}_label"
        pred_col = f"{ankle_name}_pred"
        all_labels.extend(valid_data[label_col].values.tolist())
        all_predictions.extend(valid_data[pred_col].values.tolist())

    return compute_metrics_with_tolerance(
        np.array(all_labels),
        np.array(all_predictions),
        tolerance,
    )


def objective_function(
    data: pd.DataFrame,
    window: int,
    vel_x_threshold: float,
    vel_y_threshold: float,
    acc_x_threshold: float,
    acc_y_threshold: float,
    min_distance_factor: float,
    use_acceleration: bool,
    tolerance: int = 1,
) -> float:
    """
    Objective function for optimization.

    Runs the pipeline and returns a metric to maximize (F1-score).

    Args:
        data: Prepared DataFrame with ankle positions and labels.
        window: Smoothing window size.
        vel_x_threshold: Horizontal velocity threshold.
        vel_y_threshold: Vertical velocity threshold.
        acc_x_threshold: Horizontal acceleration threshold.
        acc_y_threshold: Vertical acceleration threshold.
        min_distance_factor: Minimum distance factor.
        tolerance: Tolerance frames for metrics.

    Returns:
        F1-score as metric to maximize.
    """
    result = run_optimization_pipeline(
        data,
        window,
        vel_x_threshold,
        vel_y_threshold,
        acc_x_threshold,
        acc_y_threshold,
        use_acceleration,
        min_distance_factor
    )

    metrics = compute_metrics(result, tolerance)
    precision = metrics["precision"]
    recall = metrics["recall"]

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def create_objective(data: pd.DataFrame, tolerance: int = 1):
    """
    Create the objective function for Optuna.

    Args:
        data: Prepared DataFrame with ankle positions and labels.
        tolerance: Tolerance frames for metrics.

    Returns:
        Objective function that receives an Optuna trial.
    """
    def objective(trial: Trial) -> float:
        window = trial.suggest_int("window", 2, 4)
        vel_x_threshold = trial.suggest_float("vel_x_threshold", 1.5, 2.5)
        vel_y_threshold = trial.suggest_float("vel_y_threshold", 2.5, 3.5)
        acc_y_threshold = trial.suggest_float("acc_y_threshold", 0.01, 0.01)
        acc_x_threshold = trial.suggest_float("acc_x_threshold", 0.01, 0.01)
        use_acceleration = trial.suggest_categorical("use_acceleration", [True, False])
        min_distance_factor = trial.suggest_float("min_distance_factor", 0.7, 0.9)

        return objective_function(
            data=data,
            window=window,
            vel_x_threshold=vel_x_threshold,
            vel_y_threshold=vel_y_threshold,
            acc_x_threshold=acc_x_threshold,
            acc_y_threshold=acc_y_threshold,
            min_distance_factor=min_distance_factor,
            use_acceleration=use_acceleration,
            tolerance=tolerance,
        )

    return objective


def run_optimization(
    ankle_data_path: str | None = None,
    annotations_path: str | None = None,
    n_trials: int = 100,
    tolerance: int = 1,
    show_progress_bar: bool = True,
) -> "optuna.Study":
    """
    Run hyperparameter optimization.

    Args:
        ankle_data_path: Path to ankle data CSV. Uses default if None.
        annotations_path: Path to annotations CSV. Uses default if None.
        n_trials: Number of trials to run.
        tolerance: Tolerance frames for metrics.
        show_progress_bar: Whether to show progress bar.

    Returns:
        Optuna study with results.

    Raises:
        ImportError: If optuna is not installed.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for optimization. "
            "Install with: pip install ankles2floor[optimization]"
        )

    if ankle_data_path is None:
        ankle_data_path = str(_get_datasets_path() / "bcn-finals-2022-fem-ankle-and-shot-data.csv")
    if annotations_path is None:
        annotations_path = str(_get_datasets_path() / "bcn-finals-2022-fem-ankles-in-floor-annotations.csv")

    data = load_and_prepare_data(ankle_data_path, annotations_path)

    study = optuna.create_study(
        direction="maximize",
        study_name="ankle-floor-classification",
    )

    study.optimize(
        create_objective(data, tolerance),
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )

    return study


def print_results(
    study: "optuna.Study",
    tolerance: int = 1,
    ankle_data_path: str | None = None,
    annotations_path: str | None = None,
) -> None:
    """
    Print optimization results.

    Args:
        study: Completed Optuna study.
        tolerance: Tolerance frames used.
        ankle_data_path: Path to ankle data CSV. Uses default if None.
        annotations_path: Path to annotations CSV. Uses default if None.
    """
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nBest F1-score: {study.best_value:.4f}")
    print("\nBest parameters:")

    for param_name, param_value in study.best_params.items():
        if isinstance(param_value, float):
            print(f"  {param_name}: {param_value:.4f}")
        else:
            print(f"  {param_name}: {param_value}")

    # Recompute metrics with best parameters to show detailed results
    if ankle_data_path is None:
        ankle_data_path = str(_get_datasets_path() / "bcn-finals-2022-fem-ankle-and-shot-data.csv")
    if annotations_path is None:
        annotations_path = str(_get_datasets_path() / "bcn-finals-2022-fem-ankles-in-floor-annotations.csv")

    data = load_and_prepare_data(ankle_data_path, annotations_path)
    
    best_params = study.best_params
    result = run_optimization_pipeline(
        data,
        window=best_params["window"],
        vel_x_threshold=best_params["vel_x_threshold"],
        vel_y_threshold=best_params["vel_y_threshold"],
        acc_x_threshold=best_params.get("acc_x_threshold"),
        acc_y_threshold=best_params.get("acc_y_threshold"),
        use_acceleration=best_params.get("use_acceleration", False),
        min_distance_factor=best_params["min_distance_factor"],
    )

    metrics = compute_metrics(result, tolerance)

    print("\nDetailed metrics:")
    print(f"  TP: {metrics['tp']}")
    print(f"  FP: {metrics['fp']}")
    print(f"  FN: {metrics['fn']}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")


def main() -> None:
    """
    Main entry point for optimization script.
    """
    if not OPTUNA_AVAILABLE:
        print("Error: Optuna is required for optimization.")
        print("Install with: pip install ankles2floor[optimization]")
        return

    print("Starting parameter optimization for ankle classification...")
    print("This may take several minutes depending on the number of trials.\n")

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
