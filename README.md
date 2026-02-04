# ankles2floor

A Python library for classifying ankle movements to determine when a padel player's foot is grounded, using COCO17 pose detection coordinates.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/joni/ankles2floor.git
```

To include optimization capabilities (requires Optuna):

```bash
pip install "ankles2floor[optimization] @ git+https://github.com/joni/ankles2floor.git"
```

## Quick Start

```python
import pandas as pd
from ankles2floor import get_ankles_in_floor

# Load your data with ankle coordinates
data = pd.DataFrame({
    "frame": range(100),
    "time": [...], # total time in seconds
    "left_ankle_x": [...],  # x coordinates in pixels
    "left_ankle_y": [...],  # y coordinates in pixels
    "right_ankle_x": [...],
    "right_ankle_y": [...],
})

# Define ankle column mapping
ankle_columns = {
    "left_ankle": {"x": "left_ankle_x", "y": "left_ankle_y"},
    "right_ankle": {"x": "right_ankle_x", "y": "right_ankle_y"},
}

# Get predictions
result = get_ankles_in_floor(data, ankle_columns)

# Result contains original data plus prediction columns:
# - "left_ankle_pred": 1 if foot is on floor, 0 otherwise
# - "right_ankle_pred": 1 if foot is on floor, 0 otherwise

# Optionally, compute z-coordinates with interpolation
result_with_z = get_ankles_in_floor(data, ankle_columns, z_offset=0.0)

# Result now also contains:
# - "left_ankle_z": z-coordinate (0.0 when grounded, parabolic trajectory when airborne)
# - "right_ankle_z": z-coordinate for right ankle
# - Interpolated x/y positions between consecutive grounded positions
```

## How It Works

The library uses a motion-based approach to classify whether an ankle (and thus the foot) is grounded:

1. **Position Smoothing**: Applies a centered rolling mean to reduce noise in the x/y coordinates.
2. **Velocity Calculation**: Computes horizontal and vertical velocity as the difference between consecutive frames (pixels/frame).
3. **Acceleration Calculation**: Computes acceleration as the difference of velocities between consecutive frames (pixels/frame²).
4. **State Classification**: An ankle is considered grounded if both velocity and acceleration (adjusted by a perspective distance factor) are below configured thresholds.

### Distance Factor (Perspective Adjustment)

Objects farther from the camera appear smaller and move fewer pixels for the same real-world movement. The library accounts for this by applying a distance factor based on the y-coordinate:
- Objects at the bottom of the image (closer) have a factor of 1.0
- Objects at the top (farther) have a reduced factor (configurable via `min_distance_factor`)

## API Reference

### `get_ankles_in_floor`

```python
def get_ankles_in_floor(
    data: pd.DataFrame,
    ankle_columns: dict[str, dict[str, str]],
    window: int | None = None,
    vel_x_threshold: float | None = None,
    vel_y_threshold: float | None = None,
    acc_x_threshold: float | None = None,
    acc_y_threshold: float | None = None,
    min_distance_factor: float | None = None,
    z_offset: float | None = None,
) -> pd.DataFrame:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | Required | DataFrame containing ankle position columns and a "time" column (required when z_offset is provided) |
| `ankle_columns` | `dict` | Required | Mapping of ankle names to their x/y column names |
| `window` | `int` | 3 | Smoothing window size for rolling mean |
| `vel_x_threshold` | `float` | 2.0875 | Horizontal velocity threshold (pixels/frame) |
| `vel_y_threshold` | `float` | 3.5748 | Vertical velocity threshold (pixels/frame) |
| `acc_x_threshold` | `float` | 15.3652 | Horizontal acceleration threshold (pixels/frame²) |
| `acc_y_threshold` | `float` | 14.6945 | Vertical acceleration threshold (pixels/frame²) |
| `min_distance_factor` | `float` | 0.6257 | Minimum distance factor for perspective adjustment |
| `z_offset` | `float` | None | Height value for ankles on the floor. When provided, enables position interpolation and z-coordinate computation |

**Returns:** DataFrame with original data plus `{ankle_name}_pred` columns (0 or 1). When `z_offset` is provided, also includes `{ankle_name}_z` columns and interpolated x/y positions.

### Default Parameters

The default parameters were optimized using Optuna on an annotated dataset of padel players. You can access them directly:

```python
from ankles2floor import best_parameters

print(best_parameters)
# {
#     "window": 3,
#     "vel_x_threshold": 2.0875,
#     "vel_y_threshold": 3.5748,
#     "acc_x_threshold": 15.3652,
#     "acc_y_threshold": 14.6945,
#     "min_distance_factor": 0.6257
# }
```

## Optimization (Optional)

If you want to find optimal parameters for your own annotated dataset, install with the optimization extra and use the CLI:

```bash
pip install "ankles2floor[optimization] @ git+https://github.com/joni/ankles2floor.git"
ankles2floor-optimize
```

Or programmatically:

```python
from ankles2floor.optimization import run_optimization, print_results

study = run_optimization(
    ankle_data_path="path/to/ankle_data.csv",
    annotations_path="path/to/annotations.csv",
    n_trials=1000,
)

print_results(study)
```

## Requirements

- Python >= 3.10
- pandas >= 1.5.0
- numpy >= 1.21.0
- optuna >= 3.0.0 (optional, for optimization)

## License

MIT
