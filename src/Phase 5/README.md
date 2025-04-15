# Traditional Gyro Trajectory Segment Analysis with Corrected Turn Point Matching

This analysis identifies and visualizes different segments in the traditional gyroscope-based trajectory, comparing them with ground truth positions. It includes corrected error measurements for ground truth points that align with gyroscope turn points.

## Methodology

1. **Turn Detection**: Significant turns are detected using angular velocity thresholds (15°/sec).
2. **Segment Identification**: Straight segments (non-turn regions) are identified between detected turns.
3. **Coordinate Transformation**: Ground truth coordinates are transformed to match the local coordinate system of the gyro trajectory.
4. **Corrected Error Measurement**: For GT1, GT2, GT4, and GT6, errors are calculated by comparing with specific turn points rather than the nearest points, to better represent the comparison between trajectory turns and ground truth.
5. **Trajectory Visualization**: The trajectory is visualized with different segments colored differently and labeled, with error measurements shown.

## Key Findings

- When measuring errors to the closest points, the traditional gyro trajectory has an average error of 3.30m at ground truth points.
- When measuring errors to the corresponding turn points for GT1, GT2, GT4, and GT6, the average error increases to 4.88m, which better represents the actual turn accuracy.
- Most ground truth points (6 out of 8) are located in turn regions rather than straight segments.
- The largest errors using corrected turn point matching occur at GT1 (7.43m) and GT7 (5.33m).

## Output Files

- `traditional_gyro_segments.png`: Visualization of the traditional gyro trajectory with identified segments.
- `traditional_gyro_segments_zoomed.png`: Zoomed version of the trajectory visualization.
- `ground_truth_error_detail.png`: Detailed error visualization for each ground truth point.
- `corrected_error_measurements.png`: Visualization of the trajectory with corrected error measurements.
- `corrected_error_measurements_zoomed.png`: Zoomed visualization of corrected measurements.
- `corrected_ground_truth_errors.png`: Bar chart showing corrected errors at each ground truth point.
- `corrected_gt_error_stats.csv`: Detailed statistics about the corrected error measurements.
- `identified_segments.csv`: Information about each identified segment (start/end indices, time, duration).

## Turn Point Matching Logic

For specific ground truth points, we use specialized matching logic to identify the corresponding turn points in the gyro trajectory:

1. **GT1**: Matched with the topmost turn point in the upper section of the trajectory.
2. **GT2**: Matched with the turn point in the upper left section of the trajectory.
3. **GT4**: Matched with the same turn point as GT3, at the corner of the trajectory.
4. **GT6**: Matched with the leftmost point of the turn in the left section of the trajectory.

This approach provides a more accurate representation of how the gyro's turn handling compares to the ground truth, rather than simply using the closest points.

## Implications for Heading Correction

This segment analysis with corrected turn point matching provides valuable insights for developing more effective heading correction methods:

1. **Turn-specific Correction**: Different correction methods may be needed for turn regions versus straight segments.
2. **Error Analysis at Turns**: The higher errors at turn points highlight the need for specialized correction in these regions.
3. **Contextual Correction**: Correction algorithms should take into account the type of path segment and turn characteristics.

By understanding the different segments of the trajectory and how they relate to ground truth points, we can develop more targeted correction approaches rather than applying a one-size-fits-all correction method. 