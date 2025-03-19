# Indoor Navigation System with AI/ML Sensor Fusion

This project implements advanced AI/ML algorithms for indoor navigation using sensor fusion techniques to combine gyroscope and compass data for improved heading estimation and positioning.

## Project Overview

Indoor navigation is challenging due to the absence of GPS signals. This system addresses this challenge by combining data from various sensors (primarily gyroscope and compass) using sophisticated sensor fusion algorithms to achieve accurate heading estimation, which is critical for reliable indoor positioning.

The system is developed in a phased approach:

- **Phase 1**: Data Preprocessing and Classification
- **Phase 2**: Sensor Fusion and Improved Heading Estimation
- **Phase 3**: Advanced Position Tracking and Navigation (Future)
- **Phase 4**: Adaptive Quasi-Static Detection (Future)
- **Phase 5**: System Integration and Deployment (Future)

## System Architecture

The system is structured as follows:

```
Indoor Navigation System
├── Data Collection
├── Data Preprocessing
│   ├── Data Parsing
│   ├── Data Cleaning
│   └── Anomaly Detection
├── Sensor Fusion
│   ├── Extended Kalman Filter (EKF)
│   ├── Unscented Kalman Filter (UKF) 
│   ├── LSTM Neural Networks
│   ├── Adaptive Filtering
│   └── Context-Aware Models
├── Evaluation
│   ├── Benchmark System
│   └── Performance Metrics
└── Visualization
```

## Key Concepts

### Sensor Fusion

Sensor fusion combines data from multiple sensors to achieve more accurate and reliable information than would be possible when using these sensors individually. In this project, we focus on fusing gyroscope and compass data to estimate heading accurately.

**Why Sensor Fusion is Necessary:**
- **Gyroscope**: Provides precise short-term angular velocity measurements but suffers from drift over time
- **Compass**: Provides absolute heading measurements but is susceptible to magnetic disturbances
- **Fusion**: Combines the strengths of both sensors while minimizing their weaknesses

### Kalman Filtering

Kalman filtering is a recursive algorithm that uses a series of measurements over time to estimate unknown variables with higher precision than would be possible using a single measurement.

**Extended Kalman Filter (EKF):**
- Handles non-linear systems through linearization around the current state estimate
- Uses a two-step process: prediction and update
- In our system, it predicts heading using gyroscope data and corrects it using compass data

**Unscented Kalman Filter (UKF):**
- Improves upon EKF by using sigma points to approximate probability distributions
- Better handles non-linearities without explicit linearization
- Provides more accurate state estimation in highly non-linear systems

### Deep Learning Approaches

**LSTM Neural Networks:**
- Long Short-Term Memory networks are a type of recurrent neural network
- Can learn temporal patterns in sequential sensor data
- Automatically handle complex non-linearities in the sensor relationships
- Can be trained to ignore unreliable sensor readings based on patterns

### Adaptive and Context-Aware Models

**Adaptive Filtering:**
- Dynamically adjusts filter parameters based on detected movement patterns
- Identifies stationary vs. moving states
- Applies different weights to gyroscope and compass data based on the current motion state

**Context-Aware Models:**
- Detect and adapt to different environmental conditions (stable/unstable magnetic fields)
- Leverage environmental information to optimize sensor fusion parameters
- Handle different magnetic disturbance scenarios by adjusting trust in compass data

## Implemented Functionality

### Phase 1: Data Preprocessing and Classification

1. **Data Parsing**: Automatically identifies and classifies different sensor data
   - Gyroscope data: Contains angular velocity measurements
   - Compass data: Contains heading information based on the Earth's magnetic field
   - Ground Truth data: Contains reference positions for validation

2. **Data Cleaning**:
   - Removes duplicate timestamps
   - Identifies and handles outliers
   - Interpolates missing data when necessary

3. **Anomaly Detection**:
   - Uses Isolation Forest and Local Outlier Factor algorithms to detect anomalies
   - Identifies sudden jumps in gyroscope and compass readings
   - Flags anomalous magnetic field measurements

4. **Visualization**:
   - Creates visualizations for raw sensor data
   - Displays detected anomalies
   - Shows the relationship between different sensors

### Phase 2: Sensor Fusion and Improved Heading Estimation

1. **Extended Kalman Filter (EKF)**:
   - Implements a classic sensor fusion approach
   - Predicts heading using gyroscope data
   - Updates predictions using compass data
   - Handles sensor uncertainty through covariance matrices

2. **Unscented Kalman Filter (UKF)**:
   - Implements a more advanced Kalman filter variant
   - Uses sigma points to better approximate non-linear transformations
   - Provides improved accuracy for highly non-linear systems

3. **LSTM-Based Fusion**:
   - Implements deep learning approach using LSTM networks
   - Learns temporal patterns from sequences of sensor readings
   - Outputs heading estimation that accounts for sensor biases and drift
   - Requires ground truth data for training

4. **Adaptive Filtering**:
   - Implements motion state detection
   - Adjusts process and measurement noise based on detected movement
   - Optimizes the fusion process for different scenarios

5. **Context-Aware Models**:
   - Detects the environment type (stable, moderate, unstable)
   - Adjusts trust in sensors based on magnetic field stability
   - Provides reliability assessment for compass measurements

6. **Benchmark System**:
   - Evaluates and compares different fusion methods
   - Calculates performance metrics (MAE, RMSE, etc.)
   - Visualizes comparison results

## Workflow

The system follows this workflow:

1. **Data Acquisition**: Collect sensor data from gyroscope and compass
2. **Data Preprocessing**:
   - Parse and classify the data
   - Clean the data (remove outliers, duplicates)
   - Detect anomalies in the data
3. **Sensor Fusion**:
   - Apply one or more fusion algorithms
   - Combine gyroscope and compass data
   - Output estimated heading
4. **Evaluation**:
   - Compare estimated heading with ground truth
   - Calculate error metrics
   - Visualize results

## Usage

### Basic Usage

```bash
python main.py --fusion ekf --visualize
```

This command processes the sensor data using the Extended Kalman Filter and generates visualizations.

### Advanced Usage

```bash
python main.py --fusion ukf --visualize --detect_anomalies --interpolate
```

This command:
- Processes the sensor data using the Unscented Kalman Filter
- Detects anomalies in the data
- Interpolates ground truth positions
- Generates visualizations

### Benchmarking

```bash
python main.py --benchmark --visualize
```

This command runs all implemented fusion methods and compares their performance.

## Command Line Arguments

- `--data_dir`: Directory containing sensor data files
- `--file`: Specific data file to process
- `--output_dir`: Directory to save output files
- `--visualize`: Generate visualizations
- `--detect_anomalies`: Detect anomalies in the data
- `--interpolate`: Interpolate ground truth positions
- `--fusion`: Fusion method to use (ekf, ukf, lstm, adaptive, context)
- `--benchmark`: Run benchmark comparison of all fusion methods

## Results and Performance

The system generates various outputs:

- **Fusion Results**: CSV files containing fused heading data
- **Visualizations**: PNG files showing comparisons between different sensors and fusion methods
- **Benchmark Reports**: Comparison of different fusion methods using various metrics

## Future Development

Future phases will focus on:

- Advanced position tracking using deep learning
- Quasi-static detection optimization
- Unified pipeline for real-time heading correction
- Deployment framework for mobile devices

## Conclusion

This indoor navigation system demonstrates the power of combining traditional signal processing techniques with modern AI/ML approaches. By fusing data from multiple sensors and adapting to different environmental conditions, the system provides robust heading estimation for indoor navigation applications.