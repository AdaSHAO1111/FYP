# Indoor Navigation System with AI/ML Sensor Fusion

This project implements advanced AI/ML algorithms for indoor navigation using sensor fusion techniques to combine gyroscope and compass data for improved heading estimation and positioning.

## Project Overview

Indoor navigation is challenging due to the absence of GPS signals. This system addresses this challenge by combining data from various sensors (primarily gyroscope and compass) using sophisticated sensor fusion algorithms to achieve accurate heading estimation, which is critical for reliable indoor positioning.

The system is developed in a phased approach:

- **Phase 1**: Data Preprocessing and Classification
- **Phase 2**: Sensor Fusion and Improved Heading Estimation
- **Phase 3**: Advanced Position Tracking and Navigation
- **Phase 4**: Adaptive Quasi-Static Detection
- **Phase 5**: System Integration and Deployment (Current Phase)

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
├── Position Tracking
│   ├── LSTM Models
│   ├── CNN-LSTM Models
│   └── Bidirectional LSTM Models
├── Quasi-Static Detection
│   ├── Base Detector
│   ├── Genetic Algorithm Optimization
│   ├── Reinforcement Learning Adaptation
│   └── CNN-Based Classification
├── System Integration
│   ├── Context-Aware Algorithm Selection
│   ├── Unified Navigation Pipeline
│   ├── Real-time Processing
│   └── Performance Monitoring
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

**CNN-LSTM Models:**
- Combines Convolutional Neural Networks with LSTM architectures
- CNNs extract local features from sensor data
- LSTMs model the temporal dependencies in the extracted features
- Provides better feature extraction for position prediction

**Bidirectional LSTM Models:**
- Process data in both forward and backward directions
- Capture future context in addition to past context
- Improve prediction accuracy by utilizing information from both directions
- Particularly effective for complex trajectory modeling

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

### Phase 3: Advanced Position Tracking and Navigation

1. **LSTM-based Dead Reckoning Algorithm** (Phase 3.1):
   - Implementation in `src/phase3_position_tracking.py`
   - Uses LSTM networks to predict position changes (dx, dy) from sensor data
   - Processes sequences of sensor readings to capture motion patterns
   - Reconstructs complete trajectories by accumulating predicted changes
   - Includes position trajectory visualization features
   - Provides various metrics for evaluating prediction accuracy

2. **Neural Network for Step Length Estimation** (Phase 3.1):
   - Implementation in `src/phase3_position_tracking.py`
   - Uses LSTM architecture to predict step length from sensor data
   - Processes sensor sequence data to understand walking patterns
   - Outputs positive step length values (using ReLU activation)
   - Includes step length visualization and evaluation metrics
   - Creates foundation for more accurate position tracking

3. **Unified Position Tracking Pipeline** (Phase 3.1):
   - Implementation in `src/run_phase3.py`
   - Orchestrates the position tracking process from data loading to evaluation
   - Handles data preprocessing, model training, and result visualization
   - Automatically processes sensor data to create feature sequences
   - Saves model artifacts and performance metrics
   - Updates roadmap to track implementation progress

4. **Position Tracking Models** (Phase 3.1):
   - Implements various LSTM architectures for position prediction:
     - Basic LSTM: Standard LSTM layers for sequence processing
     - CNN-LSTM: Combines convolutional layers for feature extraction with LSTM
     - Bidirectional LSTM: Processes data in both forward and backward directions
   - Compares model performance using visualization and metrics
   - Each model provides different trade-offs between accuracy and complexity

5. **Position Data Processing** (Phase 3.1):
   - Handles special processing required for position data
   - Creates sequences suitable for LSTM processing
   - Normalizes sensor and position data for effective training
   - Implements feature engineering to improve position prediction
   - Leverages step data and heading information from multiple sensors

6. **Trajectory Reconstruction and Evaluation** (Phase 3.1):
   - Reconstructs complete movement trajectories from predicted position changes
   - Visualizes actual vs. predicted paths
   - Calculates error metrics including:
     - Average distance error
     - Median distance error
     - 90th percentile error
   - Identifies areas of high prediction error for further improvement

Note: The remaining Phase 3 components (weighted multi-model ensemble, real-time error correction, and transfer learning) are planned for future implementation.

### Phase 4: Adaptive Quasi-Static Detection

1. **Base Quasi-Static Detector**:
   - Implementation in `adaptive_quasi_static_detection.py`
   - Uses a sliding window approach to identify periods of low heading variance
   - Two key parameters control the detection sensitivity:
     - **Stability Threshold**: The maximum variance allowed for quasi-static classification
     - **Window Size**: The number of heading samples to consider when calculating variance

2. **Genetic Algorithm Optimization**:
   - Implementation in `adaptive_quasi_static_detection.py`
   - Uses a genetic algorithm approach to evolve parameter sets
   - Fitness function based on:
     - Number of detected intervals (rewards ~10 intervals)
     - Low variance within detected intervals
     - Accuracy of compass heading vs. ground truth during intervals

3. **Reinforcement Learning Approach**:
   - Implementation in `adaptive_quasi_static_detection.py`
   - Uses Q-learning to adapt parameters in real-time
   - State space: current variance and recent interval count
   - Action space: adjustments to stability threshold and window size
   - Rewards based on variance reduction and heading accuracy

4. **CNN-Based Classification**:
   - Implementation in `cnn_quasi_static_classifier.py`
   - Directly learns to classify quasi-static states from sensor data
   - Uses 1D convolutional layers to extract temporal patterns
   - Fuses multiple sensor inputs:
     - Compass data (heading, magnitude)
     - Gyroscope data (accumulated rotation)
     - Contextual features (step count, floor)

### Phase 5: System Integration and Deployment

1. **Integrated Navigation System** (Phase 5):
   - Implementation in `integrated_navigation_system.py`
   - Creates a unified pipeline that combines all previous components into a cohesive system
   - Uses context-aware algorithm selection to choose the best method for each situation
   - Provides real-time heading correction through adaptive switching between methods
   - Implements smart calibration processes using genetic algorithms from Phase 4
   - Features comprehensive performance monitoring and visualization tools

2. **Context-Aware Navigation** (Phase 5):
   - Implementation in `integrated_navigation_system.py` (NavigationContext class)
   - Tracks contextual factors such as:
     - Movement state (moving, quasi-static)
     - Magnetic disturbance levels
     - Position and heading confidence
     - Device orientation
     - Step frequency
   - Makes intelligent decisions about which algorithms to use based on context
   - Dynamically adjusts weights for sensor fusion based on environmental conditions
   - Prioritizes compass data during quasi-static periods and gyro during magnetic disturbances

3. **Hybrid Model Switching** (Phase 5):
   - Automatically selects between multiple heading estimation models:
     - Compass calibration during quasi-static periods
     - Gyro-only during high magnetic disturbance
     - Adaptive fusion during normal operation
     - CNN-LSTM for complex scenarios
   - Calculates confidence metrics for each algorithm in real-time
   - Provides seamless transitions between methods to maintain accuracy
   - Evaluates and logs the performance of each method for continued improvement

4. **Efficient Deployment Framework** (Phase 5):
   - Configurable system parameters through JSON-based configuration
   - Battery optimization options for mobile deployment
   - Real-time processing with low computational overhead
   - Performance logging and diagnostics
   - Error tracking and visualization

5. **Data Collection and Model Update Mechanism** (Phase 5):
   - System calibration using genetic algorithms to optimize parameters
   - Ability to save and reload optimized parameters
   - Framework for collecting performance metrics during operation
   - Visualization tools for analyzing system performance

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
4. **Position Tracking**:
   - Process fused sensor data with deep learning models
   - Predict x,y coordinates based on sensor patterns
   - Generate position trajectory
5. **Evaluation**:
   - Compare estimated positions with ground truth
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

### Position Tracking

```bash
python main.py --fusion lstm --position_tracking --model_type cnn_lstm --visualize
```

This command:
- Processes the sensor data using LSTM fusion
- Applies CNN-LSTM position tracking
- Generates visualizations for both processes

### Integrated Navigation System (Phase 5)

```bash
python integrated_navigation_system.py --data_file data/1740735201447_CompassGyroSumHeadingData.TXT --calibrate --visualize --save_results
```

This command:
- Initializes the integrated navigation system
- Loads and processes the specified data file
- Calibrates the system using genetic algorithm optimization
- Processes the data stream with adaptive algorithm selection
- Generates visualizations and saves results

### Benchmarking

```bash
python main.py --benchmark --benchmark_position --visualize
```

This command:
- Runs all implemented fusion methods and compares their performance
- Benchmarks different position tracking models
- Generates comparison visualizations

## Command Line Arguments

- `--data_dir`: Directory containing sensor data files
- `--file`: Specific data file to process
- `--output_dir`: Directory to save output files
- `--visualize`: Generate visualizations
- `--detect_anomalies`: Detect anomalies in the data
- `--interpolate`: Interpolate ground truth positions
- `--fusion`: Fusion method to use (ekf, ukf, lstm, adaptive, context)
- `--benchmark`: Run benchmark comparison of all fusion methods
- `--position_tracking`: Apply deep learning-based position tracking
- `--model_type`: Type of model for position tracking (lstm, cnn_lstm, bidirectional)
- `--seq_length`: Sequence length for position tracking models
- `--epochs`: Number of training epochs for position tracking
- `--batch_size`: Batch size for training position tracking models
- `--benchmark_position`: Run benchmark comparison of position tracking models

## Results and Performance

The system generates various outputs:

- **Fusion Results**: CSV files containing fused heading data
- **Position Tracking Results**: CSV files with predicted positions
- **Visualizations**: PNG files showing comparisons between different sensors, fusion methods, and position predictions
- **Benchmark Reports**: Comparison of different fusion and position tracking methods using various metrics

## Future Development

Future work will focus on:

- Federated learning for collaborative model training across devices
- Transfer learning to adapt models to new environments
- Edge optimization for resource-constrained deployments
- Integration with mapping systems for enhanced navigation

## Conclusion

This indoor navigation system demonstrates the power of combining traditional signal processing techniques with modern AI/ML approaches. By integrating multiple algorithms and using context-aware selection, the system provides robust navigation capabilities for indoor environments where GPS is unavailable. The adaptive quasi-static detection and hybrid model switching enable the system to optimize heading estimation under varying conditions, resulting in improved positioning accuracy and reliability.

## Dataset Comparison

The system includes a comprehensive dataset comparison framework for evaluating the performance of different quasi-static detection methods across multiple datasets.

### Comparison Framework

The comparison framework automatically:
- Processes multiple datasets with different detection methods
- Measures heading accuracy, processing efficiency, and other metrics
- Generates visualizations and detailed reports
- Identifies optimal parameters for each dataset

### Running Dataset Comparisons

```bash
python compare_datasets.py --calibrate --visualize --save_results
```

This command:
- Discovers and processes all datasets in the `data` directory
- Applies both genetic algorithm optimization and basic detection methods
- Performs calibration for the genetic algorithm method
- Generates visualizations and saves detailed results
- Creates a comprehensive HTML comparison report

### Command Line Arguments

- `--data_dir`: Directory containing data files (default: "data")
- `--output_dir`: Directory to save output files (default: "output/comparison")
- `--calibrate`: Run system calibration for each dataset
- `--visualize`: Generate visualizations for each dataset
- `--save_results`: Save detailed results for each dataset

### Comparison Results

The comparison generates:
- **Comparison CSV**: Raw data table with all metrics for each dataset and method
- **Heading Error Charts**: Bar charts comparing average and median heading errors
- **Position Error Charts**: Bar charts comparing average and median position errors
- **Processing Metrics Charts**: Bar charts showing processing times and quasi-static interval counts
- **HTML Report**: A comprehensive report with tables, charts, and analysis

### Key Metrics Compared

- **Heading Accuracy**: Average and median heading errors (degrees)
- **Position Accuracy**: Average and median position errors (meters)
- **Processing Efficiency**: Time required to process each dataset (seconds)
- **Quasi-Static Detection**: Number of intervals detected in each dataset
- **Optimal Parameters**: The best stability threshold and window size for each dataset

### Insights

The comparison reveals:
- Different datasets require different optimal parameters for best performance
- Genetic algorithm optimization provides varying benefits depending on the dataset
- There's a trade-off between processing time and accuracy
- Some datasets yield more reliable heading estimations than others

This comparison framework is particularly useful for:
- Selecting the best datasets for training robust models
- Understanding how the system performs in different environments
- Fine-tuning parameters for specific navigation scenarios
- Evaluating the cost-benefit of optimization methods