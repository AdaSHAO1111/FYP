# Quasi-Static (QS) Detection Approach Roadmap

## Overview

The QS detection method in `quasi_static_detection_1536.py` primarily relies on gyroscope data to identify periods when the device is not rotating significantly, but also uses compass heading data as a complementary measure to validate these periods. This hybrid approach helps filter out false positives and improves detection accuracy.

## Primary Detection Method: Gyroscope-Based

The code primarily uses **gyroscope data** for QS detection through these key steps:

1. **Gyroscope Magnitude Calculation**:
   - Computes the total angular velocity magnitude from the 3-axis gyroscope readings
   - Formula: `gyro_magnitude = sqrt(value_1² + value_2² + value_3²)`

2. **Rolling Statistics**:
   - Calculates moving average and standard deviation of gyro magnitude using a sliding window
   - Window size is configurable (default: 15 samples)
   - Low standard deviation indicates stable rotation rate (potential QS period)

3. **Adaptive Thresholding**:
   - Automatically determines appropriate stability thresholds based on data characteristics
   - Uses sample statistics to set `gyro_stability_threshold` and `gyro_change_threshold`
   - Threshold calculation: `gyro_change_threshold = std_gyro * 2.0`

## Secondary Validation: Compass-Based

Compass heading data is used as a secondary validation measure:

1. **Heading Stability Analysis**:
   - Calculates differences between consecutive compass heading readings
   - Accounts for 359° → 0° wraparound with: `angle_diff = abs((angle2 - angle1 + 180) % 360 - 180)`
   - Computes rolling standard deviation of heading differences

2. **Heading Change Rate Detection**:
   - Identifies turns by calculating heading change rate per step: `heading_change_rate = angle_diff / step_diff`
   - Marks regions with significant heading changes (>20° per step) as turns, which disqualifies them from being QS intervals

3. **Correlation Analysis**:
   - For each candidate QS interval, calculates correlation between heading changes and gyro changes
   - Low correlation (<0.7) confirms true QS intervals (heading remains stable when gyro is stable)
   - High correlation often indicates false QS (both gyro and heading changing together)

## Filtering Mechanisms

Several filtering methods are applied to remove false positives:

1. **Turn Exclusion**:
   - Regions identified as turns are automatically excluded from QS candidates
   - Implemented via the `is_turn` flag in the data

2. **Minimum Duration Requirement**:
   - QS intervals must contain at least `window_size` samples to be considered valid
   - This helps exclude brief periods of apparent stability

3. **Region-Specific Filtering**:
   - Special handling for problematic regions like GT5
   - Additional correlation thresholds (>0.6) combined with high gyro values identify false QS intervals

4. **Correlation-Based Validation**:
   - Final validation uses normalized correlation between heading and gyro changes
   - Filters out intervals where heading changes follow gyro changes too closely

## Output and Results

The detection process provides several outputs:

1. **Interval Information**:
   - Step range and mean for each QS interval
   - Mean compass heading and standard deviation
   - Mean gyro magnitude
   - Association with ground truth regions

2. **Visualizations**:
   - Compass headings over time with QS intervals highlighted
   - Gyro magnitude over time with QS intervals highlighted
   - Step progression with QS intervals and ground truth regions
   - Spatial locations with QS intervals marked

3. **Data Products**:
   - CSV files with detailed QS interval data
   - Statistical summaries of intervals
   - Reference data for heading correction algorithms

## Technical Implementation

The core detection is implemented in two main functions:

1. `detect_quasi_static_intervals()`: Primary detection based on gyro stability and heading consistency
2. `filter_intervals()`: Secondary filtering to remove false positives, especially in problematic regions

This hybrid approach leveraging both gyroscope and compass data produces more reliable QS detection than using either data source alone, particularly in challenging scenarios with sensor noise or environmental interference. 