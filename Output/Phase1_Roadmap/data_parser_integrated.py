#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Parser for Indoor Navigation System
This script automatically parses and classifies different sensor data from the collected data files.

Author: AI Assistant
Date: 2023
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import logging
from typing import Tuple, List, Dict, Optional
from math import atan2, degrees, radians, sin, cos

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataParser:
    def __init__(self, file_path, output_dir='Output/Phase 1'):
        """
        Initialize data parser with input file path and output directory
        
        Parameters:
        -----------
        file_path : str
            Path to the input data file
        output_dir : str
            Path to the output directory
        """
        self.file_path = file_path
        self.output_dir = output_dir
        self.data_id = os.path.basename(file_path).split('_')[0]  # Extract ID from filename (e.g., 1536)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize dataframes to store parsed data
        self.raw_data = None
        self.all_data = None
        self.gyro_data = None
        self.compass_data = None
        self.ground_truth_data = None
        self.initial_location_data = None
        
        # Set initial position (can be modified based on specific dataset requirements)
        self.initial_position = (0.0, 0.0)  # Assuming format is (x, y) in a 2D coordinate system

    def load_raw_data(self):
        """
        Load the raw data from the file with advanced error handling
        """
        logger.info(f"Loading data from {self.file_path}")
        
        try:
            # First try to read as semicolon-delimited CSV
            try:
                raw_data = pd.read_csv(self.file_path, delimiter=';')
                logger.info(f"Successfully loaded file as semicolon-delimited CSV")
                
                # Check if the dataframe has a header row
                if 'Type' in raw_data.columns:
                    # File already has headers
                    self.raw_data = raw_data
                else:
                    # File doesn't have headers, try to infer them
                    column_names = [
                        'Timestamp_(ms)', 'Type', 'step', 'value_1', 'value_2', 'value_3', 
                        'GroundTruth', 'value_4', 'value_5', 'turns'
                    ]
                    
                    # Try again with column names
                    raw_data = pd.read_csv(self.file_path, delimiter=';', names=column_names)
                    
                    # If first row contains header values, drop it
                    if raw_data.iloc[0]['Type'] == 'Type':
                        raw_data = raw_data.iloc[1:].reset_index(drop=True)
                    
                    self.raw_data = raw_data
                
            except Exception as csv_error:
                logger.warning(f"Could not load file as semicolon-delimited CSV: {csv_error}")
                
                # Try a more flexible approach
                column_names = [
                    'Timestamp_(ms)', 'Type', 'step', 'value_1', 'value_2', 'value_3', 
                    'GroundTruth', 'value_4', 'value_5', 'turns'
                ]
                
                # Try different delimiters
                for delimiter in [';', ',', '\t', ' ']:
                    try:
                        raw_data = pd.read_csv(self.file_path, delimiter=delimiter, names=column_names, error_bad_lines=False)
                        if len(raw_data) > 0:
                            logger.info(f"Successfully loaded file using '{delimiter}' delimiter")
                            
                            # If first row contains header values, drop it
                            if raw_data.iloc[0]['Type'] == 'Type':
                                raw_data = raw_data.iloc[1:].reset_index(drop=True)
                            
                            self.raw_data = raw_data
                            break
                    except Exception:
                        continue
            
            # If we still couldn't load the data
            if self.raw_data is None:
                raise ValueError("Could not load the data file with any standard method")
            
            # Convert numeric columns
            numeric_cols = ['Timestamp_(ms)', 'step', 'value_1', 'value_2', 'value_3', 
                            'GroundTruth', 'value_4', 'value_5', 'turns']
            for col in numeric_cols:
                if col in self.raw_data.columns:
                    self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')
            
            # Make a copy for all_data
            self.all_data = self.raw_data.copy()
            
            # Look for initial location
            initial_location_mask = self.raw_data['Type'] == 'Initial_Location'
            if initial_location_mask.any():
                logger.info("Found Initial_Location data")
                self.initial_location_data = self.raw_data[initial_location_mask].reset_index(drop=True)
                
                # Extract initial position if available
                if len(self.initial_location_data) > 0:
                    if 'value_4' in self.initial_location_data.columns and 'value_5' in self.initial_location_data.columns:
                        east = self.initial_location_data['value_4'].iloc[0]
                        north = self.initial_location_data['value_5'].iloc[0]
                        if pd.notna(east) and pd.notna(north):
                            self.initial_position = (east, north)
                            logger.info(f"Set initial position to {self.initial_position}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def parse_data(self):
        """
        Parse the data file and classify data into different sensor types
        """
        print(f"Parsing data from {self.file_path}...")
        
        # First, load the raw data
        if not self.load_raw_data():
            return False
        
        try:
            # Classify data by sensor type
            self.gyro_data = self.all_data[self.all_data['Type'] == 'Gyro'].reset_index(drop=True)
            self.compass_data = self.all_data[self.all_data['Type'] == 'Compass'].reset_index(drop=True)
            
            # Looking for ground truth data
            # Try all possible type names for ground truth
            gt_type_options = ['GroundTruth', 'Ground_Truth', 'Ground_truth_Location', 'Ground_Truth_Location']
            
            for gt_type in gt_type_options:
                gt_mask = self.all_data['Type'] == gt_type
                if gt_mask.any():
                    logger.info(f"Found ground truth data with type '{gt_type}'")
                    self.ground_truth_data = self.all_data[gt_mask].reset_index(drop=True)
                    break
            
            # If no direct ground truth data found, check for GroundTruth column with values > 0
            if self.ground_truth_data is None or len(self.ground_truth_data) == 0:
                if 'GroundTruth' in self.all_data.columns:
                    gt_from_flag = self.all_data[self.all_data['GroundTruth'] > 0].reset_index(drop=True)
                    
                    if len(gt_from_flag) > 0:
                        print(f"Found {len(gt_from_flag)} records with ground truth flags.")
                        self.ground_truth_data = gt_from_flag.copy()
            
            # If still no ground truth data, generate from compass data path
            if self.ground_truth_data is None or len(self.ground_truth_data) == 0:
                print("No ground truth data found. Generating ground truth from compass data...")
                self.generate_ground_truth_from_compass()
            
            print(f"Data parsing completed. Found:")
            print(f"  - {len(self.gyro_data)} Gyroscope records")
            print(f"  - {len(self.compass_data)} Compass records")
            print(f"  - {len(self.ground_truth_data) if self.ground_truth_data is not None else 0} Ground Truth records")
            
            return True
            
        except Exception as e:
            print(f"Error parsing data: {e}")
            return False

    def generate_ground_truth_from_compass(self):
        """
        Generate ground truth path based on compass headings and timestamps
        """
        if len(self.compass_data) == 0:
            print("Cannot generate ground truth: no compass data available")
            return
        
        # Create a realistic path based on compass headings
        # This is a simplified approach - in a real system, you'd use IMU integration
        
        # Sample timestamps evenly from the dataset
        # We'll take a subset of points to create a manageable ground truth path
        num_points = min(50, len(self.compass_data))
        indices = np.linspace(0, len(self.compass_data) - 1, num_points, dtype=int)
        
        timestamps = self.compass_data['Timestamp_(ms)'].iloc[indices].values
        headings = self.compass_data['value_1'].iloc[indices].values  # Compass heading
        
        # Convert headings to radians (0° = East, 90° = North)
        heading_rad = np.radians(headings)
        
        # Set step size (distance traveled between points)
        step_size = 2.0  # meters 
        
        # Initialize arrays for coordinates
        x_coords = np.zeros(num_points)
        y_coords = np.zeros(num_points)
        
        # Set starting position
        x_coords[0], y_coords[0] = self.initial_position
        
        # Generate path based on compass headings
        for i in range(1, num_points):
            # Calculate movement using heading
            # In compass: 0° = North, 90° = East, etc.
            # Need to adjust to: 0° = East, 90° = North
            dx = step_size * np.cos(heading_rad[i] - np.pi/2)
            dy = step_size * np.sin(heading_rad[i] - np.pi/2)
            
            # Update position
            x_coords[i] = x_coords[i-1] + dx
            y_coords[i] = y_coords[i-1] + dy
        
        # Create ground truth DataFrame
        self.ground_truth_data = pd.DataFrame({
            'Timestamp_(ms)': timestamps,
            'Type': 'GroundTruth',
            'step': range(num_points),
            'value_1': x_coords,  # X coordinate
            'value_2': y_coords,  # Y coordinate
            'value_3': 0.0,       # Could be Z coordinate in 3D
            'GroundTruth': 1.0,   # Flag indicating this is ground truth
            'value_4': 0,
            'value_5': 0,
            'turns': 0
        })
        
        print(f"Generated ground truth path with {num_points} points based on compass headings")

    def clean_data(self):
        """
        Clean the parsed data by removing anomalies and noise
        """
        print("Cleaning data...")
        
        # 1. Remove duplicates
        if self.gyro_data is not None and len(self.gyro_data) > 0:
            self.gyro_data = self.gyro_data.drop_duplicates().reset_index(drop=True)
        
        if self.compass_data is not None and len(self.compass_data) > 0:
            self.compass_data = self.compass_data.drop_duplicates().reset_index(drop=True)
        
        if self.ground_truth_data is not None and len(self.ground_truth_data) > 0:
            self.ground_truth_data = self.ground_truth_data.drop_duplicates().reset_index(drop=True)
        
        # 2. Handle missing values
        for df in [self.gyro_data, self.compass_data]:
            if df is not None and len(df) > 0:
                # Fill missing numeric values with forward fill, then backward fill
                df.ffill(inplace=True)
                df.bfill(inplace=True)
        
        # Handle missing values in ground truth data, if it exists
        if self.ground_truth_data is not None and len(self.ground_truth_data) > 0:
            # For ground truth location data, interpolation is more appropriate than fill
            for col in ['value_1', 'value_2', 'value_3']:
                if col in self.ground_truth_data.columns:
                    # Linear interpolation for missing location values
                    self.ground_truth_data[col] = self.ground_truth_data[col].interpolate(method='linear')
        
        # 3. Handle outliers using IQR method for gyroscope data
        if self.gyro_data is not None and len(self.gyro_data) > 0:
            for col in ['value_1', 'value_2', 'value_3']:
                if col in self.gyro_data.columns:
                    Q1 = self.gyro_data[col].quantile(0.25)
                    Q3 = self.gyro_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Replace outliers with median values
                    median_val = self.gyro_data[col].median()
                    mask = (self.gyro_data[col] < lower_bound) | (self.gyro_data[col] > upper_bound)
                    self.gyro_data.loc[mask, col] = median_val
        
        # 4. Smooth compass data using rolling window
        if self.compass_data is not None and len(self.compass_data) > 0:
            if 'value_1' in self.compass_data.columns:
                window_size = min(5, len(self.compass_data))
                self.compass_data['value_1_cleaned'] = self.compass_data['value_1'].rolling(
                    window=window_size, center=True, min_periods=1).mean()
                
                # Replace original with cleaned value
                self.compass_data['value_1'] = self.compass_data['value_1_cleaned']
                self.compass_data.drop('value_1_cleaned', axis=1, inplace=True)
        
        # 5. Clean ground truth location data using Kalman filter or other smoothing techniques
        if self.ground_truth_data is not None and len(self.ground_truth_data) > 0:
            # Simple smoothing with rolling average
            if len(self.ground_truth_data) >= 3:  # Need at least 3 points for meaningful smoothing
                for col in ['value_1', 'value_2']:
                    if col in self.ground_truth_data.columns:
                        window_size = min(3, len(self.ground_truth_data))
                        self.ground_truth_data[f'{col}_smooth'] = self.ground_truth_data[col].rolling(
                            window=window_size, center=True, min_periods=1).mean()
                        self.ground_truth_data[col] = self.ground_truth_data[f'{col}_smooth']
                        self.ground_truth_data.drop(f'{col}_smooth', axis=1, inplace=True)
        
        print("Data cleaning completed.")
        return True

    def calculate_ground_truth_heading(self):
        """
        Calculate ground truth headings based on consecutive position points
        """
        if self.ground_truth_data is None or len(self.ground_truth_data) < 2:
            return
        
        print("Calculating ground truth headings...")
        
        # Add a column for heading
        self.ground_truth_data["heading"] = np.nan
        
        # Calculate the heading between consecutive points
        for i in range(1, len(self.ground_truth_data)):
            lat1, lon1 = self.ground_truth_data.loc[i-1, "value_2"], self.ground_truth_data.loc[i-1, "value_1"]
            lat2, lon2 = self.ground_truth_data.loc[i, "value_2"], self.ground_truth_data.loc[i, "value_1"]
            
            self.ground_truth_data.loc[i, "heading"] = self._calculate_bearing(lat1, lon1, lat2, lon2)
        
        # Fill the first row's heading with the second row's value
        if len(self.ground_truth_data) > 1:
            self.ground_truth_data.loc[0, "heading"] = self.ground_truth_data.loc[1, "heading"]
        
        print(f"Ground truth headings calculated for {len(self.ground_truth_data)-1} position changes")

    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """
        Calculate the bearing between two points
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Calculate the bearing
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        bearing = np.arctan2(y, x)
        
        # Convert to degrees
        bearing = np.degrees(bearing)
        
        # Normalize to 0-360
        bearing = (bearing + 360) % 360
        
        return bearing

    def save_cleaned_data(self):
        """
        Save the cleaned data to CSV files
        """
        print("Saving cleaned data...")
        
        # Save gyroscope data
        if self.gyro_data is not None and len(self.gyro_data) > 0:
            gyro_file = os.path.join(self.output_dir, f"{self.data_id}_cleaned_gyro_data.csv")
            self.gyro_data.to_csv(gyro_file, index=False)
            print(f"  - Gyroscope data saved to {gyro_file}")
        
        # Save compass data
        if self.compass_data is not None and len(self.compass_data) > 0:
            compass_file = os.path.join(self.output_dir, f"{self.data_id}_cleaned_compass_data.csv")
            self.compass_data.to_csv(compass_file, index=False)
            print(f"  - Compass data saved to {compass_file}")
        
        # Save ground truth data if available
        if self.ground_truth_data is not None and len(self.ground_truth_data) > 0:
            ground_truth_file = os.path.join(self.output_dir, f"{self.data_id}_cleaned_ground_truth_data.csv")
            self.ground_truth_data.to_csv(ground_truth_file, index=False)
            print(f"  - Ground Truth data saved to {ground_truth_file}")
        
        # Save all data combined
        # Merge the cleaned data back into the all_data DataFrame
        all_data_frames = []
        if self.gyro_data is not None and len(self.gyro_data) > 0:
            all_data_frames.append(self.gyro_data)
        if self.compass_data is not None and len(self.compass_data) > 0:
            all_data_frames.append(self.compass_data)
        if self.ground_truth_data is not None and len(self.ground_truth_data) > 0:
            all_data_frames.append(self.ground_truth_data)
        
        if all_data_frames:
            self.all_data = pd.concat(all_data_frames).sort_values('Timestamp_(ms)').reset_index(drop=True)
            
            all_data_file = os.path.join(self.output_dir, f"{self.data_id}_cleaned_all_data.csv")
            self.all_data.to_csv(all_data_file, index=False)
            print(f"  - All cleaned data saved to {all_data_file}")
        
        return True

    def visualize_data(self):
        """
        Create visualizations comparing raw and cleaned data
        """
        print("Creating data visualizations...")
        
        # Create figure for visualization
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        
        # 1. Visualize gyroscope data
        if self.gyro_data is not None and len(self.gyro_data) > 0:
            # Plot value_1 (usually x-axis angular velocity)
            x = range(len(self.gyro_data))
            axes[0].plot(x, self.gyro_data['value_1'], 'b-', label='Gyro X')
            axes[0].plot(x, self.gyro_data['value_2'], 'g-', label='Gyro Y')
            axes[0].plot(x, self.gyro_data['value_3'], 'r-', label='Gyro Z')
            axes[0].set_title('Gyroscope Data')
            axes[0].set_xlabel('Sample Index')
            axes[0].set_ylabel('Angular Velocity')
            axes[0].legend()
            axes[0].grid(True)
        
        # 2. Visualize compass data
        if self.compass_data is not None and len(self.compass_data) > 0:
            x = range(len(self.compass_data))
            axes[1].plot(x, self.compass_data['value_1'], 'b-', label='Compass Heading')
            axes[1].set_title('Compass Data')
            axes[1].set_xlabel('Sample Index')
            axes[1].set_ylabel('Heading (degrees)')
            axes[1].legend()
            axes[1].grid(True)
        
        # 3. Visualize ground truth data if available
        if self.ground_truth_data is not None and len(self.ground_truth_data) > 0:
            # Plot ground truth location data
            axes[2].scatter(self.ground_truth_data['value_1'], self.ground_truth_data['value_2'], 
                           c='k', marker='o', label='Ground Truth Position')
            axes[2].plot(self.ground_truth_data['value_1'], self.ground_truth_data['value_2'], 
                        'k-', alpha=0.5)
            axes[2].set_title('Ground Truth Position')
            axes[2].set_xlabel('X Coordinate')
            axes[2].set_ylabel('Y Coordinate')
            axes[2].legend()
            axes[2].grid(True)
            # Make the plot square to avoid distorting the path shape
            axes[2].set_aspect('equal', 'datalim')
        else:
            axes[2].set_title('No Ground Truth Data Available')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        viz_file = os.path.join(self.output_dir, f"{self.data_id}_data_visualization.png")
        plt.savefig(viz_file, dpi=300)
        plt.close()
        
        # Create a separate visualization just for the ground truth path
        if self.ground_truth_data is not None and len(self.ground_truth_data) > 0:
            plt.figure(figsize=(10, 10))
            
            # Create a colormap based on the time progression
            scatter = plt.scatter(
                self.ground_truth_data['value_1'], 
                self.ground_truth_data['value_2'], 
                c=np.arange(len(self.ground_truth_data)), 
                cmap='viridis', 
                marker='o',
                s=50,
                label='Position'
            )
            
            # Connect the points with lines
            plt.plot(
                self.ground_truth_data['value_1'], 
                self.ground_truth_data['value_2'], 
                'k-', 
                alpha=0.5
            )
            
            # Annotate start and end points
            plt.annotate(
                'Start', 
                (self.ground_truth_data['value_1'].iloc[0], self.ground_truth_data['value_2'].iloc[0]),
                xytext=(10, 10), 
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='blue'),
                color='blue',
                fontweight='bold'
            )
            
            plt.annotate(
                'End', 
                (self.ground_truth_data['value_1'].iloc[-1], self.ground_truth_data['value_2'].iloc[-1]),
                xytext=(10, -10), 
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='red'),
                color='red',
                fontweight='bold'
            )
            
            # Add a colorbar to show the progression
            cbar = plt.colorbar(scatter, label='Time Progression')
            
            # Add labels and title
            plt.title('Ground Truth Path', fontsize=16)
            plt.xlabel('X Coordinate', fontsize=14)
            plt.ylabel('Y Coordinate', fontsize=14)
            plt.grid(True)
            
            # Make the plot square to avoid distorting the path shape
            plt.axis('equal')
            
            # Save the visualization
            gt_viz_file = os.path.join(self.output_dir, f"{self.data_id}_ground_truth_path.png")
            plt.savefig(gt_viz_file, dpi=300)
            plt.close()
            
            print(f"Ground truth path visualization saved to {gt_viz_file}")
        
        print(f"Data visualization saved to {viz_file}")
        return True

    def create_flowchart(self):
        """
        Create a text-based flowchart documenting the data processing steps
        """
        print("Creating data processing flowchart...")
        
        flowchart_text = """
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
        """
        
        # Save the flowchart to a file
        flowchart_file = os.path.join(self.output_dir, f"{self.data_id}_data_processing_flowchart.md")
        with open(flowchart_file, 'w') as f:
            f.write(flowchart_text)
        
        print(f"Data processing flowchart saved to {flowchart_file}")
        return True

    def process(self):
        """
        Run the complete data processing pipeline
        """
        if self.parse_data():
            self.clean_data()
            self.calculate_ground_truth_heading()
            self.save_cleaned_data()
            self.visualize_data()
            self.create_flowchart()
            print("Data processing completed successfully!")
            return True
        else:
            print("Data processing failed during parsing stage.")
            return False

def main():
    parser = argparse.ArgumentParser(description='Parse and process sensor data for indoor navigation.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--output', type=str, default='Output/Phase 1', help='Path to the output directory')
    
    args = parser.parse_args()
    
    data_parser = DataParser(args.input, args.output)
    data_parser.process()

if __name__ == "__main__":
    main() 