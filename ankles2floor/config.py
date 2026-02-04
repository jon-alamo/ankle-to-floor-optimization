"""
Configuration module with optimized parameters.

Contains the best parameters found through Optuna optimization for ankle-to-floor
classification.
"""

best_parameters = {
    "window": 3,
    "vel_x_threshold": 2.0875,
    "vel_y_threshold": 3.5748,
    "acc_x_threshold": 15.3652,
    "acc_y_threshold": 14.6945,
    "min_distance_factor": 0.6257
}
