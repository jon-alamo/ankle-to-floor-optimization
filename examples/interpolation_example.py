"""
Example demonstrating ankle position interpolation and z-coordinate computation.

This script shows how to use the z_offset parameter to:
1. Linearly interpolate x/y positions between grounded ankle positions
2. Compute z-coordinates with parabolic trajectories for airborne segments
"""

import pandas as pd
import numpy as np

from ankles2floor import get_ankles_in_floor


def create_sample_data() -> pd.DataFrame:
    """
    Create sample ankle tracking data simulating a player movement.
    
    Returns:
        DataFrame with time, and left/right ankle x/y coordinates.
    """
    num_frames = 50
    fps = 30
    
    time = np.arange(num_frames) / fps
    
    left_ankle_x = np.linspace(100, 150, num_frames)
    left_ankle_y = np.full(num_frames, 400.0)
    left_ankle_y[10:20] = 400 - 30 * np.sin(np.linspace(0, np.pi, 10))
    
    right_ankle_x = np.linspace(130, 180, num_frames)
    right_ankle_y = np.full(num_frames, 400.0)
    right_ankle_y[25:35] = 400 - 25 * np.sin(np.linspace(0, np.pi, 10))
    
    data = pd.DataFrame({
        "time": time,
        "left_ankle_x": left_ankle_x,
        "left_ankle_y": left_ankle_y,
        "right_ankle_x": right_ankle_x,
        "right_ankle_y": right_ankle_y,
    })
    
    return data


def main() -> None:
    """Run the interpolation example."""
    data = create_sample_data()
    
    ankle_columns = {
        "left_ankle": {"x": "left_ankle_x", "y": "left_ankle_y"},
        "right_ankle": {"x": "right_ankle_x", "y": "right_ankle_y"},
    }
    
    z_offset = 0.0
    
    result = get_ankles_in_floor(data, ankle_columns, z_offset=z_offset)
    
    print("Sample of results (first 20 frames):")
    print("-" * 80)
    
    columns_to_show = [
        "time",
        "left_ankle_x",
        "left_ankle_y",
        "left_ankle_pred",
        "left_ankle_z",
        "right_ankle_pred",
        "right_ankle_z",
    ]
    
    print(result[columns_to_show].head(20).to_string())
    
    print("\n" + "-" * 80)
    print("\nSummary:")
    print(f"Total frames: {len(result)}")
    print(f"Left ankle grounded frames: {(result['left_ankle_pred'] == 1).sum()}")
    print(f"Right ankle grounded frames: {(result['right_ankle_pred'] == 1).sum()}")
    
    left_z_max = result["left_ankle_z"].max()
    right_z_max = result["right_ankle_z"].max()
    print(f"\nMax z-coordinate (left ankle): {left_z_max:.4f}")
    print(f"Max z-coordinate (right ankle): {right_z_max:.4f}")


if __name__ == "__main__":
    main()
