# Data Collection and Preprocessing Methodology

## Data Collection Protocol

### Participants
- 25 participants (14 male, 11 female)
- Age range: 19-65 years
- Height range: 155-192 cm
- Various walking styles and gaits

### Equipment
- **Smartphone Models**: iPhone 12/13/14 series, Samsung Galaxy S21/S22, Google Pixel 6/7
- **Data Collection App**: Custom-built iOS/Android application
- **Ground Truth**: Motion capture system (OptiTrack) in lab settings, predefined waypoints with tape markings for outdoor settings
- **Reference System**: High-precision IMU (Xsens MTi-G-710) for comparison

### Collection Environments
1. **Indoor Laboratory** (controlled environment)
   - Open space with minimal magnetic interference
   - Flat, uniform surface
   - Motion capture system for ground truth

2. **Indoor Office Building**
   - Corridors, open spaces, staircases
   - Various floor surfaces (carpet, tile, wood)
   - Presence of magnetic interference from electronics and structure

3. **Indoor Shopping Mall**
   - Multiple levels connected by escalators and elevators
   - Crowded environment with varying walking speeds
   - Complex magnetic landscape

4. **Outdoor Urban Environment**
   - Sidewalks and crosswalks
   - Mixture of open spaces and areas between tall buildings
   - Various weather conditions (sunny, cloudy, light rain)

5. **Outdoor Park/Trail**
   - Natural terrain with slight elevation changes
   - Combination of paved paths and grass/dirt trails
   - Minimal magnetic interference

### Walking Patterns
1. **Straight Line Walking**
   - 20m straight lines at consistent pace
   - 3 speeds: slow (0.8 m/s), normal (1.2 m/s), fast (1.8 m/s)
   - 5 repetitions per speed

2. **Figure-8 Pattern**
   - Continuous figure-8 walking for 2 minutes
   - Consistent turning radius (approximately 1.5m)
   - 3 repetitions

3. **Random Walk**
   - Unstructured walking for 5 minutes
   - Natural pace and direction changes
   - 2 repetitions

4. **Stop-and-Go**
   - Walking with intermittent stops (3-5 seconds)
   - 3 minutes duration
   - 2 repetitions

5. **Stair Climbing**
   - Ascending and descending stairs
   - 3 floors up and down
   - 2 repetitions

### Data Acquisition
- Sampling rate: 100 Hz for all sensors
- Data logged: Raw accelerometer, gyroscope, magnetometer readings
- Device position: Hand-held in front of body, consistently oriented
- Total collection time per participant: approximately 1.5 hours
- Total raw dataset size: ~120 hours (approximately 43.2 million data points)

## Data Preprocessing Pipeline

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Raw Sensor    │     │ Calibration & │     │ Noise         │
│ Data (100Hz)  │────▶│ Orientation   │────▶│ Filtering     │
└───────────────┘     │ Normalization │     └───────┬───────┘
                      └───────────────┘             │
                                                    ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Feature       │     │ Segmentation  │     │ Outlier       │
│ Extraction    │◀────│ into Fixed    │◀────│ Detection &   │
└───────┬───────┘     │ Windows       │     │ Removal       │
        │             └───────────────┘     └───────────────┘
        ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Feature       │     │ Train/Val/Test│     │ Data          │
│ Normalization │────▶│ Split         │────▶│ Augmentation  │
└───────────────┘     └───────────────┘     └───────┬───────┘
                                                    │
                                                    ▼
                                            ┌───────────────┐
                                            │ Final Dataset │
                                            │ for Training  │
                                            └───────────────┘
```

### 1. Calibration and Orientation Normalization
- **Accelerometer Calibration**: Zero-g offset correction and scale factor calibration
- **Gyroscope Calibration**: Zero-rate offset removal
- **Magnetometer Calibration**: Hard and soft iron calibration
- **Orientation Normalization**: Transformation to Earth frame (gravity-aligned coordinate system)

### 2. Noise Filtering
- **Low-pass Filter**: 4th order Butterworth filter with 10Hz cutoff for accelerometer and gyroscope
- **Median Filter**: 3-point median filter for magnetometer to reduce spike noise
- **Complementary Filter**: Initial sensor fusion for orientation estimation

### 3. Outlier Detection and Removal
- **Statistical Outlier Removal**: Z-score method (threshold: 3σ)
- **Magnitude Check**: Rejection of physically impossible values (e.g., accelerations > 8g)
- **Discontinuity Detection**: Identification and correction of timestamp discontinuities

### 4. Segmentation into Fixed Windows
- Window size: 1 second (100 samples at 100Hz)
- Overlap: 50% (50 samples)
- Approximately 8,000 windows per hour of data

### 5. Feature Extraction
- **Time-domain Features**:
  - Mean, variance, skewness, kurtosis
  - Min/max values and ranges
  - Zero-crossing rate
  - Peak-to-peak intervals
  
- **Frequency-domain Features**:
  - FFT coefficients (first 10 components)
  - Spectral energy
  - Dominant frequencies
  
- **Derived Features**:
  - Integrated gyroscope for rotation angle
  - Jerk (derivative of acceleration)
  - Signal magnitude area
  - Heading from magnetometer

### 6. Feature Normalization
- Min-max scaling to range [-1, 1]
- Standard scaling (z-score normalization)
- Applied per feature across the entire dataset
- Normalization parameters saved for inference time

### 7. Train/Validation/Test Split
- **Train**: 70% (84 hours)
- **Validation**: 15% (18 hours)
- **Test**: 15% (18 hours)
- Split by participant to ensure no data leakage (20 participants for training/validation, 5 for testing)

### 8. Data Augmentation (Training Set Only)
- **Rotation**: Random rotation of data in 3D space (±15°)
- **Scaling**: Random amplitude scaling (±10%)
- **Noise Addition**: Gaussian noise (σ = 0.05)
- **Time Warping**: Random stretching/compressing of time series (±5%)
- **Sensor Dropout**: Random masking of sensor values to simulate temporary sensor failure

## Ground Truth Preparation

### Lab Setting (with Motion Capture)
- 12 infrared cameras tracking reflective markers
- Spatial accuracy: ±2mm
- Temporal synchronization with smartphone sensors (NTP)
- Direct heading and position measurements

### Field Setting (with Waypoint System)
- Predefined waypoints marked on the ground
- Participants instructed to walk through waypoints
- Video recording for verification
- Heading derived from consecutive waypoint positions
- Position error calculated as distance from waypoint at timestamp

### Data Alignment and Synchronization
- Timestamp alignment between sensors and ground truth
- Cross-correlation for fine-tuning temporal alignment
- Spatial alignment of coordinate systems
- Resampling to uniform 100Hz grid

## Quality Control Measures

### Data Validation Checks
- Sensor range verification
- Completeness check (missing data points < 0.1%)
- Consistency between devices
- Ground truth correlation check

### Participant Instructions
- Standardized briefing for all participants
- Consistent device placement and orientation
- Practice trials before actual data collection
- Post-collection verification interview

## Ethics and Privacy

- IRB approval obtained for human subjects research
- Informed consent from all participants
- Anonymization of all collected data
- Option for participants to withdraw their data

## Dataset Statistics

| Metric                            | Value        |
|-----------------------------------|--------------|
| Total collection time             | ~120 hours   |
| Number of participants            | 25           |
| Number of environments            | 5            |
| Number of walking patterns        | 5            |
| Total raw data points             | ~43.2 million|
| Processed sliding windows         | ~864,000     |
| Final dataset size                | 7.5 GB       |
| Ground truth accuracy (position)  | ±10 cm (lab) / ±30 cm (field) |
| Ground truth accuracy (heading)   | ±1° (lab) / ±3° (field) | 