# Roadmap for Developing AI/ML Algorithms for Indoor Navigation System

This roadmap outlines the development plan for implementing AI/ML algorithms to improve an indoor navigation system that combines gyroscope and compass data.

## Phase 1: Data Preprocessing and Classification

### Status and Tasks
- [x] Develop a data parser that can automatically identify and classify different sensor data
- [x] Create a visualization comparing raw and cleaned data to illustrate the impact of the cleaning process on anomaly amplitude/frequency and data reliability
- [x] Generate a flowchart documenting the data processing steps in Phase 1

### Implementation Procedure

#### 1. Data Parser Development
- **Objective**: Extract and classify sensor data from raw files
- **Steps**:
  - [x] Read raw data from `collected_data` files (CSV, JSON, TXT formats)
  - [x] Analyze and identify data structure for gyroscope, compass, and ground truth values
  - [x] Classify each data record by sensor type and store in appropriate data structures
  - [x] Add step count or timestamp identifiers to each data record
  - [x] Set initial position as navigation starting point
  - [x] Output processed data as `cleaned_data` for subsequent analysis

#### 2. Data Visualization
- **Objective**: Visually compare raw and cleaned data to validate processing effectiveness
- **Steps**:
  - [x] Select visualization libraries (Matplotlib, Seaborn, Plotly)
  - [x] Create time-series plots of raw sensor data highlighting anomalies and inconsistencies
  - [x] Generate corresponding plots of cleaned data using the same scale and metrics
  - [x] Display comparative visualizations side-by-side or overlaid with color differentiation
  - [x] Calculate and report statistical metrics (mean, standard deviation, min/max values) to quantify cleaning effectiveness

#### 3. Process Documentation
- **Objective**: Create comprehensive flowchart of the data processing workflow
- **Steps**:
  - [x] Identify key processing steps (data reading, parsing, classification, cleaning)
  - [x] Use flowchart tools (draw.io, Lucidchart) to document the process flow
  - [x] Represent processing steps with appropriate symbols and connect with directional arrows
  - [x] Add brief explanations for each processing node

## Phase 2: Processing and Visualization of Different Data Categories

### Status and Tasks
- [x] Visualize the Ground Truth position path with interpolation using AI/ML or environment adaptation techniques
- [x] Create models for Heading and position estimation
- [x] Explore deep learning approaches (LSTM, GRU networks) for the Heading estimation
- [x] Generate a flowchart documenting the data processing steps in ML
- [x] Visualize the position paths of Ground Truth, Gyroscope, and Compass on a single coordinate graph for direct comparison

### Implementation Procedure

#### 1. Ground Truth Visualization
- **Objective**: Create visual representation of the actual movement path
- **Steps**:
  - [x] Extract ground truth position data from `cleaned_data`
  - [x] Apply interpolation techniques for sparse or discontinuous data points
  - [x] Plot trajectory in 2D/3D coordinate system with appropriate markers
  - [x] Add visual indicators for start/end points and significant path segments

#### 2. Heading Model Development
- **Objective**: Create models to estimate device heading from sensor data
- **Steps**:
  - **Gyroscope-based Heading**:
    - [x] Extract angular velocity data from cleaned gyroscope readings
    - [x] Establish initial heading reference from compass or ground truth
    - [x] Implement time-based integration of angular velocity to calculate heading changes
    - [x] Apply drift compensation techniques
  - **Compass-based Heading**:
    - [x] Extract magnetic field strength data (three-axis)
    - [x] Convert magnetic readings to heading values relative to magnetic north
    - [x] Apply tilt compensation to account for device orientation effects
    - [x] Filter readings to reduce noise and magnetic interference

#### 3. Position Model Development
- **Objective**: Create models to estimate device position from sensor data
- **Steps**:
  - **Gyroscope-based Position** (Dead Reckoning):
    - [x] Detect steps using accelerometer data or other sensor inputs
    - [x] Estimate step length based on user characteristics or adaptive models
    - [x] Use calculated heading as direction reference for each step
    - [x] Implement position updating algorithm from initial position
  - **Compass-assisted Position**:
    - [x] Use compass heading as direction constraint
    - [x] Integrate with step detection for improved position estimation
    - [x] Implement filtering to reduce position estimation errors

#### 4. Comparative Trajectory Visualization
- **Objective**: Directly compare estimated positions with ground truth
- **Steps**:
  - [x] Plot ground truth, gyroscope-based, and compass-assisted position trajectories on same graph
  - [x] Use distinct colors and line styles to differentiate trajectory sources
  - [x] Add legend and reference points for clarity
  - [x] Analyze divergence patterns and error accumulation characteristics

## Phase 3: Sensor Fusion and Improved Heading Estimation

### Status and Tasks
- [x] Develop QS detection methods for sensor behavior assessment
   - [x] Implement gyroscope-based QS detection
   - [x] Add compass-based validation for QS intervals
   - [x] Develop filtering mechanisms for false positive reduction
- [x] Create tools for QS detection evaluation and visualization
- [x] Design adaptive thresholding mechanisms for various environmental conditions

### Implementation Procedure

#### 1. QS Detection Methods Development
- **Objective**: Identify quasi-static intervals for sensor calibration
- **Steps**:
  - [x] Calculate gyroscope magnitude from 3-axis readings
  - [x] Implement rolling statistics (moving average, standard deviation)
  - [x] Develop adaptive thresholding based on data characteristics
  - [x] Create heading stability analysis with wraparound handling
  - [x] Implement heading change rate detection for turn identification
  - [x] Develop correlation analysis between heading and gyro data

#### 2. Filtering and Validation System
- **Objective**: Ensure QS detection reliability by removing false positives
- **Steps**:
  - [x] Implement turn exclusion mechanisms
  - [x] Add minimum duration requirements for valid QS intervals
  - [x] Create region-specific filtering for problematic areas
  - [x] Develop correlation-based validation for final QS confirmation
  - [x] Integrate all filtering mechanisms into a comprehensive system

#### 3. QS Detection Outputs and Visualization
- **Objective**: Document QS intervals and provide visual verification tools
- **Steps**:
  - [x] Generate interval information (step range, mean values)
  - [x] Create visualizations of compass headings with QS intervals
  - [x] Develop gyro magnitude plots with QS intervals highlighted
  - [x] Generate spatial location maps with QS intervals marked
  - [x] Export CSV files with detailed QS interval data

## Phase 4: Adaptive Quasi-Static Detection and Heading Correction

### Status and Tasks
- [x] Implement traditional Quasi-Static detection on Gyro data
- [x] Develop trajectory segment analysis for context-aware heading correction
- [x] Create heading correction methods using QS intervals
   - [x] Implement basic gyro heading correction
   - [x] Develop improved heading correction with turn detection
   - [x] Add compass-gyro hybrid heading correction
- [x] Compare and visualize the trajectories using different correction methods

### Implementation Procedure

#### 1. Traditional QS Detection Implementation
- **Objective**: Establish baseline QS detection using conventional methods
- **Steps**:
  - [x] Analyze gyroscope data to identify periods of minimal angular velocity
  - [x] Implement sliding window approach for QS detection
  - [x] Apply thresholding techniques to distinguish static from dynamic states
  - [x] Identify tunable parameters (window size, threshold values)

#### 2. Trajectory Analysis and Segmentation
- **Objective**: Understand trajectory characteristics for targeted correction
- **Steps**:
  - [x] Implement turn detection using angular velocity thresholds
  - [x] Identify straight segments between detected turns
  - [x] Transform coordinates for comparative analysis
  - [x] Develop corrected error measurement for turn points
  - [x] Create segment-specific visualization and analysis tools

#### 3. Heading Correction Development
- **Objective**: Create methods to correct heading data using QS information
- **Steps**:
  - [x] Develop basic QS-based heading correction
  - [x] Implement improved correction with turn awareness
  - [x] Add differential correction strategies for turns vs. straight segments
  - [x] Create proper angle normalization and difference calculations
  - [x] Implement compass-gyro hybrid correction methods

#### 4. Comparative Analysis and Visualization
- **Objective**: Evaluate different heading correction approaches
- **Steps**:
  - [x] Compare heading errors before and after correction
  - [x] Analyze position errors at ground truth points
  - [x] Identify strengths and weaknesses of each correction method
  - [x] Visualize corrected trajectories for direct comparison
  - [x] Document improvement statistics and recommendations

## Phase 5: LSTM-based Sensor Fusion

### Status and Tasks
- [x] Develop LSTM-based sensor fusion model for heading estimation
- [x] Create data preparation pipeline for sequence-based learning
- [x] Implement trajectory generation and evaluation from fused headings
- [x] Compare fusion results with individual sensor approaches

### Implementation Procedure

#### 1. Data Preparation for LSTM Fusion
- **Objective**: Create properly formatted sequential data for LSTM training
- **Steps**:
  - [x] Merge corrected compass and gyroscope data
  - [x] Calculate ground truth headings from positions
  - [x] Implement sine/cosine encoding for circular heading data
  - [x] Create sequence-based training examples
  - [x] Split data into training, validation, and test sets

#### 2. LSTM Fusion Model Development
- **Objective**: Create deep learning model to fuse sensor data
- **Steps**:
  - [x] Design bidirectional LSTM architecture
  - [x] Implement custom angle difference loss function
  - [x] Create training pipeline with early stopping
  - [x] Optimize hyperparameters for optimal performance
  - [x] Train model to predict ground truth from sensor inputs

#### 3. Trajectory Generation and Evaluation
- **Objective**: Generate improved trajectories using fused heading estimates
- **Steps**:
  - [x] Apply trained model to generate fused heading predictions
  - [x] Calculate positions from fused headings
  - [x] Compare fused trajectory with corrected trajectories
  - [x] Quantify improvements through error metrics
  - [x] Visualize comparative performance of different approaches

#### 4. Model Finalization and Deployment
- **Objective**: Create ready-to-use fusion solution
- **Steps**:
  - [x] Finalize optimal model configuration
  - [x] Create application script for model deployment
  - [x] Document model usage and performance characteristics
  - [x] Implement real-time prediction capabilities
  - [x] Create example usage scenarios and demonstrations