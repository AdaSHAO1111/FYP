# Gyro Heading and Position Correction

This package provides tools for correcting gyroscope heading measurements and using them to generate improved position estimates for indoor navigation.

## Table of Contents
1. [Overview](#overview)
2. [Scripts](#scripts)
3. [Usage](#usage)
4. [Workflow](#workflow)
5. [Output Files](#output-files)
6. [Performance](#performance)
7. [Troubleshooting](#troubleshooting)

## Overview

The gyroscope in mobile devices tends to accumulate drift over time, which affects both heading and position accuracy. This package provides two main components:

1. **Heading Correction**: Corrects gyro heading drift using ground truth measurements as reference points
2. **Position Correction**: Uses the corrected heading to calculate improved position estimates

## Scripts

- `gyro_heading_correction.py`: Corrects gyro heading drift using ground truth calibration points
- `gyro_position_correction.py`: Uses corrected heading data to generate improved position estimates

## Usage

### Step 1: Correct Gyro Heading

```bash
python src/Phase\ 5/gyro_heading_correction.py
```

This script will:
- Load gyro heading data and ground truth data
- Apply heading correction based on ground truth calibration
- Generate corrected heading values
- Create visualizations comparing original and corrected headings
- Output statistical analysis of the heading corrections

### Step 2: Generate Corrected Position

```bash
python src/Phase\ 5/gyro_position_correction.py
```

This script will:
- Load the corrected heading data
- Generate position estimates using adaptive step sizes
- Scale the positions to match ground truth coordinates
- Create trajectory visualizations
- Output statistical analysis of the position accuracy

## Workflow

```
                      ┌───────────────┐
                      │  Gyro Heading │
                      │     Data      │
                      └───────┬───────┘
                              │
                              ▼
┌──────────────┐     ┌───────────────┐
│ Ground Truth │────▶│    Heading    │
│     Data     │     │   Correction  │
└──────────────┘     └───────┬───────┘
                              │
                              ▼
                      ┌───────────────┐
                      │   Corrected   │
                      │    Heading    │
                      └───────┬───────┘
                              │
                              ▼
                      ┌───────────────┐
                      │    Position   │
                      │   Estimation  │
                      └───────┬───────┘
                              │
                              ▼
                      ┌───────────────┐
                      │   Trajectory  │
                      │ Visualization │
                      └───────────────┘
```

## Output Files

### Heading Correction Outputs
- `gyro_heading_corrected.csv`: Original gyro data with corrected heading values
- `gyro_heading_correction_plot.png`: Visualization of original, corrected, and ground truth headings
- `heading_error_comparison.png`: Comparison of heading errors before and after correction

### Position Correction Outputs
- `gyro_position_corrected.csv`: Position data with original and corrected coordinates
- `trajectory_comparison.png`: Complete trajectory visualization
- `trajectory_comparison_zoomed.png`: Zoomed view of the trajectories
- `trajectory_detail_comparison.png`: Detailed view of trajectories near ground truth points

## Performance

The heading correction achieves:
- 65.81% reduction in mean heading error
- 88.04% of entries have improved accuracy

The position correction achieves:
- 24.33% reduction in average position error at ground truth points
- 71.43% of ground truth points show improved positioning

## Troubleshooting

### Common Issues

1. **Missing or incorrect path errors**:
   - Ensure all input files exist at the specified paths
   - Check that the file format and structure match the expected format

2. **Ground truth point misalignment**:
   - Verify that the ground truth data is properly formatted
   - Check for any timestamp inconsistencies between datasets

3. **Scaling issues in position visualization**:
   - Adjust the `base_step_size` parameter to better match the scale
   - Consider modifying the scaling approach for your specific dataset

### Configuration Tips

- Modify the step size calculation in `gyro_position_correction.py` if your movement patterns differ
- Adjust the visualization parameters for better readability with your specific dataset
- For large datasets, consider downsampling the data for faster processing

For any further questions or issues, please refer to the analysis documents or contact the development team. 