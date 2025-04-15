# Data Processing Flowchart Using Mermaid

The following Mermaid diagram illustrates the data processing pipeline for Phase 1 of the Indoor Navigation System project.

## Flowchart

```mermaid
flowchart TD
    %% Define styles
    classDef start_end fill:#4CAF50,stroke:#333,stroke-width:1px;
    classDef process fill:#2196F3,stroke:#333,stroke-width:1px;
    classDef data fill:#9C27B0,stroke:#333,stroke-width:1px;
    classDef subprocess fill:#FF5722,stroke:#333,stroke-width:1px;
    
    %% Start node
    START[START] --> INPUT[Raw Data Input<br>1536_CompassGyroSumHeadingData.txt]
    
    %% Main process nodes
    INPUT --> READ[1. Read Raw Data<br>read_raw_data]
    READ --> PARSE[2. Parse Data Structure<br>parse_data]
    PARSE --> CLASSIFY[3. Classify Sensor Data<br>classify_sensor_data]
    
    %% Branch out to different sensor types
    CLASSIFY --> GYRO_DATA[Gyroscope Data]
    CLASSIFY --> COMPASS_DATA[Compass Data]
    CLASSIFY --> GT_DATA[Ground Truth Data]
    
    %% Cleaning processes for each sensor type
    GYRO_DATA --> GYRO_CLEAN[Gyro Cleaning<br>- Duplicate removal<br>- Outlier detection (IQR)<br>- Savitzky-Golay filtering]
    COMPASS_DATA --> COMPASS_CLEAN[Compass Cleaning<br>- Duplicate removal<br>- Circular data handling<br>- Median filtering]
    GT_DATA --> GT_PROCESS[Ground Truth Processing<br>- Calculate headings<br>- Verify positioning]
    
    %% Merge and save data
    GYRO_CLEAN -->|Cleaned Gyro Data| MERGE[4. Merge & Save Cleaned Data<br>save_cleaned_data]
    COMPASS_CLEAN -->|Cleaned Compass Data| MERGE
    GT_PROCESS -->|Cleaned Ground Truth| MERGE
    
    %% Visualization processes
    MERGE --> GYRO_VIZ[Gyroscope Visualization<br>- Raw vs Cleaned<br>- Anomaly Detection]
    MERGE --> COMPASS_VIZ[Compass Visualization<br>- Raw vs Cleaned<br>- Smoothing Effects]
    MERGE --> GT_VIZ[Ground Truth Path<br>- Spatial Visualization<br>- Time Progression]
    
    %% Output and end
    GYRO_VIZ -->|Gyro Analytics| OUTPUT[Output Files & Visualizations]
    COMPASS_VIZ -->|Compass Analytics| OUTPUT
    GT_VIZ -->|Path Visualization| OUTPUT
    OUTPUT --> END[END]
    
    %% Apply styles
    class START,END start_end;
    class READ,PARSE,CLASSIFY,MERGE process;
    class INPUT,GYRO_DATA,COMPASS_DATA,GT_DATA,OUTPUT data;
    class GYRO_CLEAN,COMPASS_CLEAN,GT_PROCESS,GYRO_VIZ,COMPASS_VIZ,GT_VIZ subprocess;
```

## Description

This flowchart illustrates the data processing pipeline implemented in Phase 1 of the Indoor Navigation System. The process follows these key steps:

1. **Data Input**: Raw sensor data from the 1536_CompassGyroSumHeadingData.txt file is loaded
2. **Read Raw Data**: The system reads the raw data using the read_raw_data method
3. **Parse Data Structure**: The data is parsed and filtered from the first Initial_Location entry
4. **Classify Sensor Data**: The data is classified into different sensor types:
   - Gyroscope data
   - Compass data
   - Ground Truth data
5. **Data Cleaning**: Each sensor type undergoes specific cleaning procedures:
   - **Gyroscope**: Duplicate removal, outlier detection using IQR method, and Savitzky-Golay filtering
   - **Compass**: Duplicate removal, specialized circular data handling, and median filtering
   - **Ground Truth**: Heading calculation and position verification
6. **Merge & Save**: The cleaned data from all sensors is merged and saved
7. **Visualization**: Different visualization methods are applied:
   - Gyroscope: Raw vs. cleaned comparison and anomaly detection
   - Compass: Raw vs. cleaned comparison and smoothing effects analysis
   - Ground Truth: Spatial visualization with time progression

The output includes cleaned data files and various visualizations that demonstrate the effectiveness of the data preprocessing pipeline.

## Processing Benefits

- **Noise Reduction**: Significant reduction in sensor noise
- **Anomaly Detection**: Identification and handling of outliers in gyroscope data
- **Data Smoothing**: Appropriate smoothing applied to gyroscope and compass data
- **Circular Data Handling**: Specialized processing for compass heading data considering its circular nature
- **Path Visualization**: Clear visualization of the ground truth path with proper spatial representation 