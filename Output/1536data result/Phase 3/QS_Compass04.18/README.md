# Quasi-Static Interval Detection Results

This directory contains the results of quasi-static interval detection on compass heading data.

## Overview

Quasi-static intervals are periods during which the compass heading remains relatively stable (low variance). These intervals are useful for:
- Calibrating compass measurements
- Analyzing heading accuracy
- Identifying points where the user may have paused or moved slowly

## Parameters

The optimal parameters for QS detection were determined through a parameter search:

- **Stability Threshold**: 20 degrees² (maximum variance allowed for a QS interval)
- **Window Size**: 50 samples (number of consecutive readings to consider for variance calculation)
- **Step Difference Threshold**: 0.5 steps (minimum number of steps required for a valid QS interval)

## Results

- **Detected Intervals**: 3 quasi-static intervals
- **Coverage**: Approximately 6.15% of the data points are within quasi-static intervals
- **Mean Heading Difference**: The average absolute difference between compass heading and true heading within QS intervals is approximately 4 degrees

## Files

- `quasi_static_data.csv`: All data points from the detected quasi-static intervals
- `quasi_static_averages.csv`: Mean headings and other statistics for each interval
- `quasi_static_summary.csv`: Summary statistics for each interval
- `QS_analysis_report.txt`: Text report with detection results
- `parameter_search_results.csv`: Results from different parameter combinations
- `best_parameters.txt`: Best parameter combinations from the search

## Visualizations

- `positions_with_QS_intervals.png`: Map showing the trajectory with QS intervals highlighted
- `heading_steps_with_QS_intervals.png`: Plot of compass heading vs steps with QS intervals highlighted
- `variance_history.png`: Plot of heading variance over time with QS thresholds
- `qs_interval_summary_table.png`: Table summarizing the properties of each QS interval
- `num_intervals_by_parameters.png`: Plot showing how the number of detected intervals varies with different parameters
- `coverage_by_parameters.png`: Plot showing how the coverage percentage varies with different parameters

## Parameter Search

The `tsd_*_st_*_ws_*/` directories contain results from different parameter combinations:
- `tsd`: Threshold step difference
- `st`: Stability threshold
- `ws`: Window size

## Analysis

The detected quasi-static intervals show:
1. Consistent periods where heading remains stable
2. Reasonable agreement between compass heading and true heading
3. Distribution in different regions of the trajectory

These intervals could be used for:
- Calibrating the compass measurements
- Adjusting for systematic errors in the heading data
- Analyzing the stability of the heading during different movement phases

## Notes

- The optimal parameters were chosen to balance between detecting too many short intervals and missing longer stable periods
- The quasi-static detection algorithm examines the variance of the heading within a sliding window
- Intervals must exceed the minimum step threshold to be considered valid 