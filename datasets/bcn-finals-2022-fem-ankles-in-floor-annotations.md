# Dataset Documentation: bcn-finals-2022-fem-ankles-in-floor-annotations.csv

## General Information
- **Rows**: 240
- **Columns**: 5

## Frame Information
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `frame_index` | Integer | - | Frame index number after removing phantom frames. |
| `source_index` | Integer |  | Frame index extracted from the original file name before cleansing. |
| `low_left_left_ankle` | Integer | - | Whether the left ankle from the lower team's player in the left position is sitting in the floor (1) or not (0) based on the ankle's movement. |
| `low_left_right_ankle` | Integer | - | Whether the right ankle from the lower team's player in the left position is sitting in the floor (1) or not (0) based on the ankle's movement. |
| `up_drive_left_ankle` | Integer | - | Whether the left ankle from the upper team's player in the drive position is sitting in the floor (1) or not (0) based on the ankle's movement. |
| `up_drive_right_ankle` | Integer | - | Whether the right ankle from the lower team's player in the drive position is sitting in the floor (1) or not (0) based on the ankle's movement. |
