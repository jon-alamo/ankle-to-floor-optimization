# Instructions

When developing the application, you must follow what is defined in this file. Any change that modifies or adds functionality must also be reflected in this file. For simplicity, some technical details such as function signatures, constraints, or implementation details may be defined in the code itself as docstrings. Avoid using comments whenever possible, prioritizing documentation in docstrings or this file for general topics. Always consider what is already implemented and treat it with great care, trying to minimize modifications when necessary. Additionally, there is no need to maintain any type of backward compatibility in favor of simple and readable code. When implementing a change that may affect previous versions, the old code is cleaned up.


# Style

- Code should always be self-explanatory, avoiding unnecessary comments. It should be well encapsulated and prioritize the use of pure functions over classes for better readability, maintenance, and ease of testing.
- It is essential to follow the single responsibility principle applicable to any unit of code, whether it is a function, method, class, or module.
- Class, function, and variable names should always be as descriptive as possible, and in general, a Pythonic code style should be followed. Avoid very dense statements on single lines. Do things explicitly and clearly, avoiding for example list comprehensions.
- Add docstrings to each function, class, or module.
- Always use type hints.


# Ankle-to-Floor Classifier

This application is a classifier for ankle movements based on flat x and y coordinates in pixels provided by pose detectors in COCO17 format applied to padel players. The goal is to determine when a foot (the joint closest to the ground that COCO17 considers) is resting on the ground based on ankle movement.

## Dataset

There is a dataset with x and y coordinates in pixels for 4 players during approximately 16,000 frames (`datasets/bcn-finals-2022-fem-ankle-and-shot-data.csv`), whose structure is detailed in: `datasets/bcn-finals-2022-fem-ankle-and-shot-data.md`. Additionally, for 2 of the 4 players, there is a sample of 240 annotated frames that determine whether the left and right foot of each of the two players is resting on the ground or not based on their ankle movements.

## Parameters

The parameters considered for classification are:

- **window**: Smoothing window for x and y position of each ankle (centered rolling mean of size "window").
- **vel_x_threshold**: Horizontal velocity threshold that determines if an ankle is anchored (1) or not (0).
- **vel_y_threshold**: Vertical velocity threshold that determines if an ankle is anchored (1) or not (0).
- **acc_x_threshold**: Horizontal acceleration threshold that determines if an ankle is starting to move or not.
- **acc_y_threshold**: Vertical acceleration threshold that determines if an ankle is starting to move or not.
- **min_distance_factor**: Due to image perspective, the sensitivity to the number of pixels/frame in the movement of objects farther from the camera is greater, as they are farther away and occupy less space in the image. A distance factor is applied based on the total range of y-axis movement, which is the coordinate on which distance depends. To calculate the maximum range in y (pixels), the highest and lowest positions of all ankle "y" coordinates are examined. The factor will be 1 for the lowest position (highest "y" in pixels since it increases downward) and the minimum value "min_distance_factor" for the farthest position, i.e., the highest "y" coordinate (lowest value) of all ankle "y"s. The distance factor is calculated linearly between "min_distance_factor" and 1 in the range obtained for the "y" coordinate.

## Pipeline

The pipeline to classify ankle movements is simple:

1. Load data with ankle x/y coordinates.
2. Smooth ankle positions using a centered rolling mean with a window of size indicated by "window".
3. Calculate velocity in "x" and "y" as the difference between consecutive frames.
4. Calculate acceleration as the difference of velocities in consecutive frames.
5. Compare thresholds vel_x_threshold, vel_y_threshold, acc_x_threshold, acc_y_threshold to determine if the ankle is still or not.
6. (Optional) When `z_offset` is provided, interpolate positions and compute z-coordinates:
   - X and Y positions are linearly interpolated between consecutive grounded positions (where `<side>_ankle_pred` is 1).
   - Z-coordinate is set to `z_offset` for grounded positions.
   - Z-coordinate follows a parabolic trajectory for airborne segments (between grounded positions), computed using the "time" column in the dataframe. Given the start and end times at z_offset, there is a unique parabolic curve: z(t) = z_offset + h * 4 * (t/T) * (1 - t/T), where T is total airborne time and h = 0.5 * g * (T/2)² is the maximum height at t=T/2.

## Objective Function (Optimization)

The objective function executes the pipeline that performs calculations for a parameter set and returns the metrics to optimize. When calculating precision TP/(TP + FP) and recall TP/(TP + FN), a tolerance is applied, which by default is 1. This means a TP will be counted if the determined value matches the actual value with a tolerance of +-1 frame. That is, if it detected an anchored ankle (foot on the ground, value 1) in frame 6, it will be a TP if there is a 1 in the sample in frames 6+-1, i.e., in frames 5, 6, or 7.

