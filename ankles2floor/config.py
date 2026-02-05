"""
Configuration module with optimized parameters.

Contains the best parameters found through Optuna optimization for ankle-to-floor
classification.

Best parameters found through Optuna optimization for ankle-to-floor classification.
Best F1-score: 0.8821

Best parameters:
  window: 3
  vel_x_threshold: 2.2639
  vel_y_threshold: 3.0747
  acc_y_threshold: 0.0100
  acc_x_threshold: 0.0100
  use_acceleration: False
  min_distance_factor: 0.7555

Detailed metrics:
  TP: 430
  FP: 72
  FN: 43
  Precision: 0.8566
  Recall: 0.9091

"""

best_parameters = {
    "window": 3,
    "vel_x_threshold": 2.2639,
    "vel_y_threshold": 3.0747,
    "acc_x_threshold": 0,
    "acc_y_threshold": 0,
    "use_acceleration": False,
    "min_distance_factor": 0.7555
}
