# Indoor Navigation System - Phase 1: Data Preprocessing and Classification

This project implements an indoor navigation system that combines gyroscope and compass data to improve heading estimation. This repository contains the Phase 1 implementation, focusing on data preprocessing and classification.

## Project Overview

The indoor navigation system aims to improve heading estimation by combining gyroscope and compass data. The project follows a phased approach:

1. **Phase 1: Data Preprocessing and Classification** (Current Phase)
2. Phase 2: Sensor Fusion and Improved Heading Estimation
3. Phase 3: Advanced Position Tracking and Navigation
4. Phase 4: Adaptive Quasi-Static Detection
5. Phase 5: System Integration and Deployment

## Phase 1 Implementation

The Phase 1 implementation includes:

1. **Data Parsing**: Automatically identify and classify different sensor data types (Gyroscope, Compass, Ground Truth)
2. **Data Cleaning**: Handle duplicates, outliers, and inconsistencies in the sensor data
3. **Visualization Pipeline**: Create visualizations for raw data exploration
4. **Anomaly Detection**: Detect and classify unclassifiable data or anomalies

### Project Structure

```
.
├── data/                      # Raw sensor data
├── output/                    # Output directory for processed data and visualizations
│   ├── anomalies/             # Anomaly detection results
│   ├── data/                  # Processed data
│   └── plots/                 # Data visualizations
├── src/                       # Source code
│   ├── data_parser.py         # Module for parsing and classifying sensor data
│   ├── data_cleaner.py        # Module for cleaning sensor data
│   ├── data_visualizer.py     # Module for creating visualizations
│   └── anomaly_detector.py    # Module for detecting anomalies
├── main.py                    # Main program
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd indoor-navigation-system
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main program can be run with various command-line arguments:

```bash
python main.py --data_dir data --output_dir output --visualize --detect_anomalies --interpolate
```

### Command-line Arguments

- `--data_dir`: Directory containing the sensor data files (default: 'data')
- `--file`: Specific data file to process (if not specified, the first file will be used)
- `--output_dir`: Directory to save output files (default: 'output')
- `--visualize`: Generate visualizations
- `--detect_anomalies`: Detect anomalies in the data
- `--interpolate`: Interpolate ground truth positions

## Data Format

The system expects data files in CSV format with semicolon (;) as the delimiter. The data should have the following columns:
- `Timestamp_(ms)`: Timestamp in milliseconds
- `Type`: Sensor type ('Gyro', 'Compass', 'Ground_truth_Location', 'Initial_Location')
- `step`: Step number
- Additional columns specific to each sensor type

## Example Visualizations

The visualization pipeline creates several plots to explore the data:

1. Heading comparison between gyroscope, compass, and ground truth
2. Sensor data over time
3. Interpolated ground truth positions
4. Magnetic field variation over time
5. Anomaly detection visualizations

## Next Steps

The next phase (Phase 2) will focus on sensor fusion and improved heading estimation:
- Implement Kalman Filter variants (Extended, Unscented)
- Explore deep learning approaches (LSTM, GRU networks)
- Design end-to-end neural networks for heading estimation
- Create a benchmark system to evaluate heading estimation accuracy