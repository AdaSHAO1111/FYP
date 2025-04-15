# Phase 1: Data Preprocessing and Classification

This directory contains the implementation of Phase 1 of the Indoor Navigation System project, focused on data preprocessing and classification of sensor data (Gyroscope, Compass, and Ground Truth).

## Files

- `main.py`: Main script that runs the complete Phase 1 pipeline
- `data_parser_integrated.py`: Script for parsing, cleaning, and visualizing sensor data with advanced ground truth generation

## Requirements

The scripts require the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn

You can install them using pip:
```
pip install pandas numpy matplotlib seaborn
```

## Usage

### Running the Complete Pipeline

To run the complete Phase 1 pipeline with the default data file (1536_CompassGyroSumHeadingData.txt):

```
python main.py
```

To specify a different input file or output directory:

```
python main.py --input ../../Data_collected/1578_CompassGyroSumHeadingData.txt --output ../../Output/Phase1_Custom
```

### Running the Parser Directly

To run only the data parser:

```
python data_parser_integrated.py --input ../../Data_collected/1536_CompassGyroSumHeadingData.txt --output ../../Output/Phase\ 1
```

## Output

The scripts generate the following outputs in the specified output directory:

1. Cleaned Data Files:
   - `{data_id}_cleaned_gyro_data.csv`
   - `{data_id}_cleaned_compass_data.csv`
   - `{data_id}_cleaned_ground_truth_data.csv`
   - `{data_id}_cleaned_all_data.csv`

2. Visualizations:
   - `{data_id}_data_visualization.png`: Basic visualization of the cleaned data
   - `{data_id}_ground_truth_path.png`: Visualization of the ground truth path
   - `{data_id}_data_distributions.png`: Distribution visualizations of the sensor data

3. Documentation:
   - `{data_id}_data_processing_flowchart.md`: A flowchart documenting the data processing steps

## Data Cleaning Approaches

The following techniques are used for data cleaning:

1. **Duplicate Removal**: Removes duplicate records from the sensor data
2. **Missing Value Handling**: 
   - Uses forward-fill and backward-fill for gyro and compass data
   - Uses linear interpolation for ground truth location data
3. **Outlier Detection and Removal**:
   - For gyroscope data: Uses the Interquartile Range (IQR) method to detect and replace outliers
   - Outliers are replaced with the median value of their respective axis
4. **Smoothing**:
   - For compass data: Uses a rolling window average to smooth the heading values
   - For ground truth locations: Uses a rolling window to smooth the position coordinates

## Ground Truth Generation

When no explicit ground truth data is available in the input file, the system will automatically generate a realistic ground truth path based on:

1. The compass heading data
2. The initial position (default is 0,0 if not specified)
3. A realistic movement model that takes into account heading changes

This generated path is then used for visualizations and will be available for subsequent phases of the project.

## Notes

- The data ID is extracted from the input filename (e.g., "1536" from "1536_CompassGyroSumHeadingData.txt")
- All output files are prefixed with this data ID to maintain traceability
- The visualizations highlight the differences between raw and cleaned data, and show the ground truth path
- The system uses advanced error handling to deal with various file formats and data issues 