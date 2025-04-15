#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Parser for Indoor Navigation System
This script automatically identifies and classifies different sensor data 
(Gyroscope, Compass, Ground Truth) according to the roadmap requirements.

Author: AI Assistant
Date: 2023
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SensorDataParser:
    """
    Parser for sensor data that automatically identifies and classifies different sensor types
    including Gyroscope, Compass, and Ground Truth location data.
    
    Based on the roadmap requirements and raw_code.py reference implementation.
    """
    
    def __init__(self, file_path, output_dir='Output/Phase 1'):
        """
        Initialize the parser with file path and output directory
        
        Args:
            file_path (str): Path to the input data file
            output_dir (str): Directory to save output files
        """
        self.file_path = file_path
        self.output_dir = output_dir
        self.data_id = os.path.basename(file_path).split('_')[0]  # Extract ID from filename
        
        # Initialize data containers
        self.collected_data = None  # Raw data as loaded from file
        self.data = None            # Data filtered from first Initial_Location
        self.gyro_data = None       # Gyroscope data
        self.compass_data = None    # Compass data
        self.ground_truth_data = None  # Ground truth location data
        self.initial_location_data = None  # Initial location data
        self.initial_position = None  # Initial position coordinates
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def read_raw_data(self):
        """
        Step 1: Read the raw data from the input file
        """
        logger.info(f"Reading raw data from: {self.file_path}")
        try:
            # Read the CSV file using semicolon delimiter (based on reference code)
            self.collected_data = pd.read_csv(self.file_path, delimiter=';')
            logger.info(f"Successfully loaded data with {len(self.collected_data)} records")
            return True
        except Exception as e:
            logger.error(f"Error reading data: {e}")
            return False

    def parse_data(self):
        """
        Step 2: Parse data structure and filter from first Initial_Location
        """
        if self.collected_data is None:
            logger.error("No data to parse. Please read raw data first.")
            return False
        
        try:
            # Find the index of the first occurrence of 'Initial_Location'
            initial_location_indices = self.collected_data[self.collected_data['Type'] == 'Initial_Location'].index
            
            if len(initial_location_indices) == 0:
                logger.warning("No 'Initial_Location' record found. Using all data.")
                self.data = self.collected_data.copy()
            else:
                initial_location_index = initial_location_indices[0]
                # Slice the DataFrame from the first occurrence onwards
                self.data = self.collected_data.iloc[initial_location_index:].reset_index(drop=True)
                logger.info(f"Data filtered from first Initial_Location (index {initial_location_index})")
            
            # Convert numeric columns if needed
            numeric_cols = ['step', 'value_1', 'value_2', 'value_3', 'value_4', 'value_5', 'GroundTruth', 'turns']
            for col in numeric_cols:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            return True
        except Exception as e:
            logger.error(f"Error parsing data: {e}")
            return False

    def classify_sensor_data(self):
        """
        Step 3: Classify data into different sensor types
        """
        if self.data is None:
            logger.error("No parsed data available. Please parse data first.")
            return False
        
        try:
            # Extract initial location data
            self.initial_location_data = self.data[self.data['Type'] == 'Initial_Location'].reset_index(drop=True)
            
            # Extract gyroscope data
            self.gyro_data = self.data[self.data['Type'] == 'Gyro'].reset_index(drop=True)
            
            # Extract compass data
            self.compass_data = self.data[self.data['Type'] == 'Compass'].reset_index(drop=True)
            
            # Extract ground truth location data (including Initial_Location as first point)
            self.ground_truth_data = self.data[(self.data['Type'] == 'Ground_truth_Location') | 
                                               (self.data['Type'] == 'Initial_Location')].reset_index(drop=True)
            
            # Sort the ground truth data by step
            if len(self.ground_truth_data) > 0:
                self.ground_truth_data.sort_values(by='step', inplace=True)
                
                # Drop duplicates based on step, keeping the last occurrence
                self.ground_truth_data.drop_duplicates(subset='step', keep='last', inplace=True)
                self.ground_truth_data.reset_index(drop=True, inplace=True)
            
            logger.info(f"Classified sensor data:")
            logger.info(f"  - Initial Location: {len(self.initial_location_data)} records")
            logger.info(f"  - Gyroscope: {len(self.gyro_data)} records")
            logger.info(f"  - Compass: {len(self.compass_data)} records")
            logger.info(f"  - Ground Truth: {len(self.ground_truth_data)} records")
            
            return True
        except Exception as e:
            logger.error(f"Error classifying sensor data: {e}")
            return False

    def add_timestamps(self):
        """
        Step 4: Add step numbers or timestamps to data records
        """
        # Step numbers are already in the data (step column)
        # Verify steps are properly numbered in each dataset
        for dataset_name, dataset in [('gyro_data', self.gyro_data), 
                                      ('compass_data', self.compass_data),
                                      ('ground_truth_data', self.ground_truth_data)]:
            if dataset is not None and len(dataset) > 0:
                if 'step' not in dataset.columns:
                    logger.warning(f"No step column in {dataset_name}. Adding sequential step numbers.")
                    dataset['step'] = range(len(dataset))
        
        logger.info("Verified timestamps/step numbers in all datasets")
        return True

    def set_initial_position(self):
        """
        Step 5: Set initial position for navigation
        """
        if self.initial_location_data is None or len(self.initial_location_data) == 0:
            logger.warning("No initial location data found. Setting default position (0, 0).")
            self.initial_position = (0.0, 0.0)
        else:
            # Extract initial position from initial_location_data
            self.initial_position = (
                self.initial_location_data['value_4'].iloc[0],
                self.initial_location_data['value_5'].iloc[0]
            )
            logger.info(f"Set initial position to {self.initial_position}")
        
        return True

    def calculate_ground_truth_heading(self):
        """
        Calculate ground truth heading based on consecutive ground truth locations
        """
        if self.ground_truth_data is None or len(self.ground_truth_data) < 2:
            logger.warning("Not enough ground truth points to calculate heading")
            return
        
        # Add a column for ground truth heading
        self.ground_truth_data["GroundTruthHeadingComputed"] = np.nan
        
        # Calculate the heading between consecutive points
        for i in range(1, len(self.ground_truth_data)):
            self.ground_truth_data.loc[i, "GroundTruthHeadingComputed"] = self._calculate_bearing(
                self.ground_truth_data.loc[i-1, "value_5"], self.ground_truth_data.loc[i-1, "value_4"],
                self.ground_truth_data.loc[i, "value_5"], self.ground_truth_data.loc[i, "value_4"]
            )
        
        logger.info(f"Ground truth headings calculated for {len(self.ground_truth_data)-1} position changes")

    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """
        Calculate the bearing (azimuth) between two points
        Based on the reference implementation
        """
        from math import atan2, degrees, radians, sin, cos
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        delta_lon = lon2 - lon1
        x = atan2(
            sin(delta_lon) * cos(lat2),
            cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(delta_lon)
        )
        
        bearing = (degrees(x) + 360) % 360  # Normalize to 0-360 degrees
        return bearing

    def save_cleaned_data(self):
        """
        Step 6: Save cleaned and classified data to output files
        """
        # Save gyroscope data
        if self.gyro_data is not None and len(self.gyro_data) > 0:
            gyro_file = os.path.join(self.output_dir, f"{self.data_id}_cleaned_gyro_data.csv")
            self.gyro_data.to_csv(gyro_file, index=False)
            logger.info(f"Saved gyroscope data to {gyro_file}")
        
        # Save compass data
        if self.compass_data is not None and len(self.compass_data) > 0:
            compass_file = os.path.join(self.output_dir, f"{self.data_id}_cleaned_compass_data.csv")
            self.compass_data.to_csv(compass_file, index=False)
            logger.info(f"Saved compass data to {compass_file}")
        
        # Save ground truth data
        if self.ground_truth_data is not None and len(self.ground_truth_data) > 0:
            gt_file = os.path.join(self.output_dir, f"{self.data_id}_cleaned_ground_truth_data.csv")
            self.ground_truth_data.to_csv(gt_file, index=False)
            logger.info(f"Saved ground truth data to {gt_file}")
        
        # Save all cleaned data combined
        all_cleaned_file = os.path.join(self.output_dir, f"{self.data_id}_cleaned_all_data.csv")
        self.data.to_csv(all_cleaned_file, index=False)
        logger.info(f"Saved all cleaned data to {all_cleaned_file}")
        
        return True

    def visualize_ground_truth_path(self):
        """
        Create a visualization of the ground truth path
        """
        if self.ground_truth_data is None or len(self.ground_truth_data) < 2:
            logger.warning("Not enough ground truth points to visualize path")
            return False
        
        try:
            # Set up matplotlib figure
            plt.figure(figsize=(12, 10))
            
            # Get x and y coordinates
            x_coords = self.ground_truth_data['value_4'].values
            y_coords = self.ground_truth_data['value_5'].values
            
            # Calculate base values for better readability
            x_base = np.floor(np.min(x_coords) / 1000) * 1000
            y_base = np.floor(np.min(y_coords) / 1000) * 1000
            
            # Adjust coordinates relative to base
            x_adjusted = x_coords - x_base
            y_adjusted = y_coords - y_base
            
            # Create a scatter plot with color mapping for time progression
            scatter = plt.scatter(
                x_adjusted, 
                y_adjusted,
                c=range(len(self.ground_truth_data)), 
                cmap='viridis', 
                s=80, 
                zorder=5
            )
            
            # Plot the path connecting the points
            plt.plot(
                x_adjusted, 
                y_adjusted, 
                'gray', 
                alpha=0.7, 
                linewidth=1.5,
                zorder=4
            )
            
            # Mark start and end points with simple colored circles
            plt.scatter(
                x_adjusted[0], 
                y_adjusted[0],
                c='blue', 
                s=180, 
                zorder=10, 
                label='Start'
            )
            
            plt.scatter(
                x_adjusted[-1], 
                y_adjusted[-1],
                c='red', 
                s=180, 
                zorder=10, 
                label='End'
            )
            
            # Add labels and title
            plt.title(f'Ground Truth Path - Dataset {self.data_id}')
            plt.xlabel(f'East (meters) +{x_base:.1f}')
            plt.ylabel(f'North (meters) +{y_base:.1f}')
            plt.grid(True)
            plt.legend(loc='upper right')
            
            # Make axes equal to preserve the geometry
            plt.axis('equal')
            
            # Add colorbar for time progression
            cbar = plt.colorbar(scatter)
            cbar.set_label('Time Progression')
            
            # Adjust layout to make room for the axes labels
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(self.output_dir, f"{self.data_id}_ground_truth_path.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved ground truth path visualization to {output_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error visualizing ground truth path: {e}")
            return False

    def clean_data(self):
        """
        Step 5: Clean data by removing duplicates, handling missing values,
        detecting and handling outliers, and applying smoothing
        """
        logger.info("Cleaning sensor data...")
        
        # Clean gyroscope data
        if self.gyro_data is not None and len(self.gyro_data) > 0:
            logger.info("Cleaning gyroscope data...")
            
            # Drop duplicates based on step and Type
            original_len = len(self.gyro_data)
            self.gyro_data.drop_duplicates(subset=['step', 'Type'], keep='first', inplace=True)
            logger.info(f"Removed {original_len - len(self.gyro_data)} duplicate records from gyroscope data")
            
            # Interpolate missing values
            for col in ['value_1', 'value_2', 'value_3']:
                missing_count = self.gyro_data[col].isna().sum()
                if missing_count > 0:
                    self.gyro_data[col] = self.gyro_data[col].interpolate(method='linear')
                    logger.info(f"Interpolated {missing_count} missing values in column {col}")
            
            # Handle outliers for each axis using IQR method
            for col in ['value_1', 'value_2', 'value_3']:
                self.gyro_data[col] = self._handle_outliers(self.gyro_data[col], method='clip')
            
            # Apply Savitzky-Golay filter for smoothing
            window_length = min(21, len(self.gyro_data) - 1 if len(self.gyro_data) % 2 == 0 else len(self.gyro_data))
            if window_length > 3:  # Minimum window length for Savitzky-Golay filter
                if window_length % 2 == 0:  # Must be odd
                    window_length -= 1
                
                for col in ['value_1', 'value_2', 'value_3']:
                    original_values = self.gyro_data[col].copy()
                    try:
                        from scipy.signal import savgol_filter
                        self.gyro_data[col] = savgol_filter(
                            self.gyro_data[col], 
                            window_length=window_length, 
                            polyorder=3
                        )
                        logger.info(f"Applied Savitzky-Golay filter to gyroscope {col}")
                    except Exception as e:
                        logger.warning(f"Could not apply Savitzky-Golay filter to gyroscope {col}: {e}")
                        # Fallback to moving average if Savitzky-Golay fails
                        self.gyro_data[col] = original_values.rolling(window=5, center=True).mean()
                        self.gyro_data[col] = self.gyro_data[col].fillna(original_values)  # Fill NaNs introduced by rolling
                        logger.info(f"Applied moving average filter to gyroscope {col} as fallback")
        
        # Clean compass data
        if self.compass_data is not None and len(self.compass_data) > 0:
            logger.info("Cleaning compass data...")
            
            # Drop duplicates based on step and Type
            original_len = len(self.compass_data)
            self.compass_data.drop_duplicates(subset=['step', 'Type'], keep='first', inplace=True)
            logger.info(f"Removed {original_len - len(self.compass_data)} duplicate records from compass data")
            
            # Handle compass heading interpolation (needs special handling for circular data)
            if 'value_1' in self.compass_data.columns:
                missing_count = self.compass_data['value_1'].isna().sum()
                if missing_count > 0:
                    # Linear interpolation for compass readings must handle the circular nature
                    self.compass_data['value_1'] = self._interpolate_circular(self.compass_data['value_1'], 360)
                    logger.info(f"Interpolated {missing_count} missing values in compass heading")
            
            # Handle outliers in compass data
            if 'value_1' in self.compass_data.columns:
                # For compass, use a specialized method that respects circular values (0-360)
                self.compass_data['value_1'] = self._handle_circular_outliers(self.compass_data['value_1'])
            
            # Apply median filter to avoid oversmoothing heading changes
            if len(self.compass_data) >= 5:  # Need at least 5 points for a reasonable median filter
                try:
                    from scipy.signal import medfilt
                    original_values = self.compass_data['value_1'].copy()
                    # Convert to sin/cos, filter, then convert back to avoid discontinuity at 0/360
                    sin_vals = np.sin(np.radians(original_values))
                    cos_vals = np.cos(np.radians(original_values))
                    
                    filtered_sin = medfilt(sin_vals, kernel_size=5)
                    filtered_cos = medfilt(cos_vals, kernel_size=5)
                    
                    filtered_heading = np.degrees(np.arctan2(filtered_sin, filtered_cos)) % 360
                    self.compass_data['value_1'] = filtered_heading
                    logger.info("Applied median filter to compass heading")
                except Exception as e:
                    logger.warning(f"Could not apply median filter to compass data: {e}")
                    # Fallback to a simple moving average
                    window_size = 5
                    original_values = self.compass_data['value_1'].copy()
                    
                    # Convert to sin/cos components for averaging
                    sin_vals = np.sin(np.radians(original_values))
                    cos_vals = np.cos(np.radians(original_values))
                    
                    # Apply moving average to sin/cos components
                    sin_smooth = sin_vals.rolling(window=window_size, center=True).mean()
                    cos_smooth = cos_vals.rolling(window=window_size, center=True).mean()
                    
                    # Convert back to angles
                    heading_smooth = np.degrees(np.arctan2(sin_smooth, cos_smooth)) % 360
                    
                    # Fill in NaN values from rolling window
                    mask = heading_smooth.isna()
                    heading_smooth[mask] = original_values[mask]
                    
                    self.compass_data['value_1'] = heading_smooth
                    logger.info("Applied moving average to compass heading as fallback")
        
        # Clean ground truth data
        if self.ground_truth_data is not None and len(self.ground_truth_data) > 0:
            logger.info("Cleaning ground truth data...")
            # For ground truth, focus on duplicate removal
            original_len = len(self.ground_truth_data)
            self.ground_truth_data.drop_duplicates(subset=['step'], keep='last', inplace=True)
            logger.info(f"Removed {original_len - len(self.ground_truth_data)} duplicate records from ground truth data")
            
            # Sort by step to ensure chronological order
            self.ground_truth_data.sort_values(by='step', inplace=True)
            self.ground_truth_data.reset_index(drop=True, inplace=True)
            
            # No smoothing for ground truth as it should be accurate by definition
        
        return True
    
    def _handle_outliers(self, series, method='clip', factor=1.5):
        """
        Handle outliers in a pandas Series using IQR method
        
        Args:
            series: Pandas Series to clean
            method: Method to handle outliers ('clip', 'remove', or 'interpolate')
            factor: Factor to multiply IQR for defining outliers (default: 1.5)
            
        Returns:
            Cleaned pandas Series
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Count outliers
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            logger.info(f"Detected {outlier_count} outliers in data")
            
            if method == 'clip':
                # Clip outliers to bounds
                return series.clip(lower=lower_bound, upper=upper_bound)
            
            elif method == 'remove':
                # Replace outliers with NaN and then interpolate
                series_clean = series.copy()
                series_clean[outlier_mask] = np.nan
                return series_clean.interpolate(method='linear')
            
            elif method == 'interpolate':
                # Replace outliers with interpolated values
                series_clean = series.copy()
                series_clean[outlier_mask] = np.nan
                interpolated = series_clean.interpolate(method='linear')
                # Handle edge cases (beginning and end)
                interpolated.fillna(method='ffill', inplace=True)
                interpolated.fillna(method='bfill', inplace=True)
                return interpolated
            
            else:
                logger.warning(f"Unknown outlier handling method: {method}. Using clip.")
                return series.clip(lower=lower_bound, upper=upper_bound)
        else:
            return series
    
    def _interpolate_circular(self, series, period=360):
        """
        Interpolate missing values in circular data (like angles)
        
        Args:
            series: Pandas Series with circular data
            period: Period of the circular data (e.g. 360 for degrees)
            
        Returns:
            Interpolated pandas Series
        """
        # Convert to sin/cos components
        sin_vals = np.sin(np.radians(series))
        cos_vals = np.cos(np.radians(series))
        
        # Interpolate sin/cos components
        sin_interp = sin_vals.interpolate(method='linear')
        cos_interp = cos_vals.interpolate(method='linear')
        
        # Convert back to angles
        angles = np.degrees(np.arctan2(sin_interp, cos_interp)) % period
        
        # Handle any remaining missing values
        return angles.fillna(method='ffill').fillna(method='bfill')
    
    def _handle_circular_outliers(self, series, period=360, window_size=5, threshold=30):
        """
        Handle outliers in circular data (like compass headings)
        
        Args:
            series: Pandas Series with circular data
            period: Period of the circular data (e.g. 360 for degrees)
            window_size: Size of the window for median calculation
            threshold: Threshold angle difference for outlier detection
            
        Returns:
            Cleaned pandas Series
        """
        # Create a copy to avoid modifying the original
        clean_series = series.copy()
        
        # Calculate rolling median using circular statistics
        sin_vals = np.sin(np.radians(series))
        cos_vals = np.cos(np.radians(series))
        
        sin_median = sin_vals.rolling(window=window_size, center=True).median()
        cos_median = cos_vals.rolling(window=window_size, center=True).median()
        
        median_angles = np.degrees(np.arctan2(sin_median, cos_median)) % period
        
        # Fill NaN values at edges
        median_angles = median_angles.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate the minimum angular distance
        dist = np.minimum(
            (clean_series - median_angles) % period,
            (median_angles - clean_series) % period
        )
        
        # Identify outliers where the angular distance exceeds the threshold
        outliers = dist > threshold
        
        # Replace outliers with the median value
        clean_series[outliers] = median_angles[outliers]
        
        outlier_count = outliers.sum()
        if outlier_count > 0:
            logger.info(f"Corrected {outlier_count} outliers in circular data")
        
        return clean_series

    def process(self):
        """
        Process the data through all steps
        """
        if not self.read_raw_data():
            logger.error("Failed to read raw data. Aborting.")
            return False
        
        if not self.parse_data():
            logger.error("Failed to parse data. Aborting.")
            return False
        
        if not self.classify_sensor_data():
            logger.error("Failed to classify sensor data. Aborting.")
            return False
        
        if not self.add_timestamps():
            logger.error("Failed to verify timestamps. Aborting.")
            return False
        
        if not self.set_initial_position():
            logger.error("Failed to set initial position. Aborting.")
            return False
        
        # Add clean_data step
        if not self.clean_data():
            logger.error("Failed to clean data. Aborting.")
            return False
        
        # Calculate ground truth heading
        self.calculate_ground_truth_heading()
        
        if not self.save_cleaned_data():
            logger.error("Failed to save cleaned data. Aborting.")
            return False
        
        if not self.visualize_ground_truth_path():
            logger.warning("Failed to visualize ground truth path.")
        
        logger.info("Data processing completed successfully.")
        return True


def main():
    """
    Main entry point for the script
    """
    parser = argparse.ArgumentParser(description="Process sensor data according to roadmap requirements")
    parser.add_argument("--input", required=True, help="Path to input data file")
    parser.add_argument("--output", default="Output/Phase 1", help="Directory for output files")
    
    args = parser.parse_args()
    
    # Create and run parser
    data_parser = SensorDataParser(args.input, args.output)
    success = data_parser.process()
    
    if success:
        print(f"Data processing completed successfully. Results saved to {args.output}")
    else:
        print("Data processing failed.")


if __name__ == "__main__":
    main() 