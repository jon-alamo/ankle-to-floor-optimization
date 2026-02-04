# Dataset Documentation: bcn-finals-2022-fem-ankle-and-shot-data

## General Information
- **Rows**: 16937
- **Columns**: 38

## Frame Information
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `source_frame` | Integer | - | Frame number extracted from the file name. |
| `time` | Float | Seconds | Time elapsed since the start of the video. |
| `file_name` | String | - | Name of the original image file. |

## Shots & Events
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `has_shot` | Integer | Binary | Indicator if a shot occurred (1=Yes, 0=No). |

## Other
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `source_frame_file` | str | - | No description available. |
| `player_low_drive_x` | float64 | - | No description available. |
| `player_low_left_x` | float64 | - | No description available. |
| `player_up_drive_x` | float64 | - | No description available. |
| `player_up_left_x` | float64 | - | No description available. |
| `player_low_drive_y` | float64 | - | No description available. |
| `player_low_left_y` | float64 | - | No description available. |
| `player_up_drive_y` | float64 | - | No description available. |
| `player_up_left_y` | float64 | - | No description available. |
| `player_low_drive_w` | float64 | - | No description available. |
| `player_low_left_w` | float64 | - | No description available. |
| `player_up_drive_w` | float64 | - | No description available. |
| `player_up_left_w` | float64 | - | No description available. |
| `player_low_drive_h` | float64 | - | No description available. |
| `player_low_left_h` | float64 | - | No description available. |
| `player_up_drive_h` | float64 | - | No description available. |
| `player_up_left_h` | float64 | - | No description available. |
| `player_low_drive_left_ankle_x` | float64 | - | Lower team drive player's ankle x position in pixels (higher y). |
| `player_low_left_left_ankle_x` | float64 | - | Lower team left player's ankle x position in pixels (higher y). |
| `player_up_drive_left_ankle_x` | float64 | - | Upper team player's ankle x position in pixels (lower y). |
| `player_up_left_left_ankle_x` | float64 | - | Upper team player's ankle x position in pixels (lower y). |
| `player_low_drive_left_ankle_y` | float64 | - | Lower team player's ankle y position in pixels (higher y). |
| `player_low_left_left_ankle_y` | float64 | - | Lower team player's ankle y position in pixels (higher y). |
| `player_up_drive_left_ankle_y` | float64 | - | Upper team player's ankle y position in pixels (lower y). |
| `player_up_left_left_ankle_y` | float64 | - | Upper team player's ankle y position in pixels (lower y). |
| `player_low_drive_right_ankle_x` | float64 | - | Lower team player's ankle x position in pixels (higher y). |
| `player_low_left_right_ankle_x` | float64 | - | Lower team player's ankle x position in pixels (higher y). |
| `player_up_drive_right_ankle_x` | float64 | - | Upper team player's ankle x position in pixels (lower y). |
| `player_up_left_right_ankle_x` | float64 | - | Upper team player's ankle x position in pixels (lower y). |
| `player_low_drive_right_ankle_y` | float64 | - | Lower team player's ankle y position in pixels (higher y). |
| `player_low_left_right_ankle_y` | float64 | - | Lower team player's ankle y position in pixels (higher y). |
| `player_up_drive_right_ankle_y` | float64 | - | Upper team player's ankle y position in pixels (lower y). |
| `player_up_left_right_ankle_y` | float64 | - | Upper team player's ankle y position in pixels (lower y). |
| `category` | String | - | Type of shot detected. |
