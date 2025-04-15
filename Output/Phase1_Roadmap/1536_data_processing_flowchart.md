
        # Data Processing Flowchart for Indoor Navigation System

        ```
        +---------------------------+
        |   Raw Sensor Data         |
        | (Gyro, Compass, etc.)     |
        +---------------------------+
                    |
                    v
        +---------------------------+
        |   Data Parsing            |
        | - Identify sensor type    |
        | - Extract timestamps      |
        | - Classify data           |
        | - Extract ground truth    |
        +---------------------------+
                    |
                    v
        +---------------------------+
        |   Data Cleaning           |
        | - Remove duplicates       |
        | - Handle missing values   |
        | - Remove outliers         |
        | - Smooth data             |
        | - Clean location data     |
        +---------------------------+
                    |
                    v
        +---------------------------+
        |   Data Visualization      |
        | - Plot raw/clean data     |
        | - Compare data types      |
        | - Visualize position paths|
        | - Identify patterns       |
        +---------------------------+
                    |
                    v
        +---------------------------+
        |   Processed Data          |
        | Ready for analysis        |
        +---------------------------+
        ```

        ## Data Processing Details

        1. **Data Parsing Stage**
           - Load raw data from text files
           - Parse column structure based on file format
           - Classify into Gyroscope, Compass, and Ground Truth data
           - Add step or timestamp identifiers
           - Extract ground truth location data (when available)
           - Create realistic ground truth path (when real data is unavailable)
           - Calculate headings from consecutive positions

        2. **Data Cleaning Stage**
           - Remove duplicate records
           - Handle missing values using interpolation methods
           - Identify and handle outliers using statistical methods (IQR)
           - Apply smoothing filters to reduce noise (rolling window)
           - Clean location data using appropriate techniques

        3. **Data Visualization Stage**
           - Generate time-series plots of sensor readings
           - Visualize position paths and trajectories
           - Compare raw and cleaned data
           - Highlight anomalies and their reduction

        4. **Output Generation**
           - Save cleaned data to CSV files
           - Generate visualization graphics
           - Create documentation of the process
        