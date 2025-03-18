import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SensorDataParser:
    """
    A class for parsing and classifying sensor data from raw text files.
    Handles Gyroscope, Compass, and Ground Truth location data.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the parser with a file path.
        
        Args:
            file_path: Path to the raw data file
        """
        self.file_path = file_path
        self.data = None
        self.gyro_data = None
        self.compass_data = None
        self.ground_truth_data = None
        self.initial_location_data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from the file and preprocess it.
        
        Returns:
            DataFrame containing the processed data
        """
        try:
            logger.info(f"Loading data from {self.file_path}")
            raw_data = pd.read_csv(self.file_path, delimiter=';')
            
            # Find the index of the first occurrence of 'Initial_Location'
            initial_location_indices = raw_data[raw_data['Type'] == 'Initial_Location'].index
            
            if len(initial_location_indices) == 0:
                logger.warning("No 'Initial_Location' found in the data")
                self.data = raw_data
            else:
                initial_location_index = initial_location_indices[0]
                # Slice the DataFrame from the first occurrence onwards
                self.data = raw_data.iloc[initial_location_index:].reset_index(drop=True)
                logger.info(f"Data loaded and filtered from first Initial_Location (index {initial_location_index})")
            
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def classify_sensor_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Classify data into different sensor types.
        
        Returns:
            Tuple of DataFrames (gyro_data, compass_data, ground_truth_data, initial_location_data)
        """
        if self.data is None:
            self.load_data()
        
        try:
            # Extract specific data types
            self.gyro_data = self.data[self.data['Type'] == 'Gyro'].reset_index(drop=True)
            self.compass_data = self.data[self.data['Type'] == 'Compass'].reset_index(drop=True)
            self.ground_truth_data = self.data[self.data['Type'] == 'Ground_truth_Location'].reset_index(drop=True)
            self.initial_location_data = self.data[self.data['Type'] == 'Initial_Location'].reset_index(drop=True)
            
            # Rename columns for better clarity
            if not self.gyro_data.empty:
                self.gyro_data = self.gyro_data.rename(columns={
                    'value_1': 'axisZAngle',
                    'value_2': 'gyroSumFromstart0',
                    'value_3': 'compass'
                })
            
            if not self.compass_data.empty:
                self.compass_data = self.compass_data.rename(columns={
                    'value_1': 'Magnetic_Field_Magnitude',
                    'value_2': 'gyroSumFromstart0',
                    'value_3': 'compass'
                })
            
            logger.info(f"Data classified: {len(self.gyro_data)} gyro records, {len(self.compass_data)} compass records, " +
                       f"{len(self.ground_truth_data)} ground truth records, {len(self.initial_location_data)} initial location records")
            
            return self.gyro_data, self.compass_data, self.ground_truth_data, self.initial_location_data
        
        except Exception as e:
            logger.error(f"Error classifying sensor data: {str(e)}")
            raise
    
    def get_initial_position(self) -> Tuple[float, float]:
        """
        Get the initial position from the data.
        
        Returns:
            Tuple of (east, north) coordinates
        """
        if self.initial_location_data is None:
            _, _, _, self.initial_location_data = self.classify_sensor_data()
        
        if len(self.initial_location_data) == 0:
            logger.warning("No initial location data found, returning (0,0)")
            return (0.0, 0.0)
        
        # Initial position is stored in value_4 (east) and value_5 (north)
        initial_position = (
            self.initial_location_data['value_4'].iloc[0],
            self.initial_location_data['value_5'].iloc[0]
        )
        logger.info(f"Initial position: {initial_position}")
        
        return initial_position
    
    def calculate_ground_truth_heading(self) -> pd.DataFrame:
        """
        Calculate the ground truth heading based on consecutive ground truth locations.
        
        Returns:
            DataFrame with ground truth heading information
        """
        if self.ground_truth_data is None or self.initial_location_data is None:
            _, _, self.ground_truth_data, self.initial_location_data = self.classify_sensor_data()
        
        # Combine initial location and ground truth data
        combined_data = pd.concat([self.initial_location_data, self.ground_truth_data]).reset_index(drop=True)
        combined_data.sort_values(by='step', inplace=True)
        
        # Drop duplicates based on the 'step' column, keeping the last occurrence
        combined_data.drop_duplicates(subset='step', keep='last', inplace=True)
        combined_data.reset_index(drop=True, inplace=True)
        
        # Add a column for ground truth heading (azimuth)
        combined_data["GroundTruthHeadingComputed"] = np.nan
        
        # Calculate the heading between consecutive points
        for i in range(1, len(combined_data)):
            combined_data.loc[i, "GroundTruthHeadingComputed"] = self._calculate_bearing(
                combined_data.loc[i-1, "value_5"], combined_data.loc[i-1, "value_4"],
                combined_data.loc[i, "value_5"], combined_data.loc[i, "value_4"]
            )
        
        logger.info(f"Ground truth headings calculated for {len(combined_data)-1} position changes")
        return combined_data
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the bearing (azimuth) between two points.
        
        Args:
            lat1: Latitude of first point
            lon1: Longitude of first point
            lat2: Latitude of second point
            lon2: Longitude of second point
            
        Returns:
            Bearing in degrees (0-360)
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
    
    def interpolate_ground_truth_positions(self, step_size: float = 0.5) -> pd.DataFrame:
        """
        Interpolate ground truth positions between manually labeled points.
        
        Args:
            step_size: The step size for interpolation (in half steps)
            
        Returns:
            DataFrame with interpolated positions
        """
        combined_data = self.calculate_ground_truth_heading()
        
        # Initialize an empty list to store the interpolated positions
        interpolated_positions = []
        
        # Iterate over each pair of successive ground truth locations
        for i in range(len(combined_data) - 1):
            current_row = combined_data.iloc[i]
            next_row = combined_data.iloc[i + 1]
            
            # Calculate the number of half steps between the current and next ground truth locations
            num_half_steps = int((next_row['step'] - current_row['step']) * (1/step_size))
            
            # Calculate the step size for each half step
            interp_step_size = (next_row['step'] - current_row['step']) / num_half_steps
            
            # Iterate over each half step and compute the interpolated position
            for j in range(num_half_steps):
                # Calculate the step for the current half step
                half_step = current_row['step'] + j * interp_step_size
                
                # Calculate the interpolation factor
                t = (half_step - current_row['step']) / (next_row['step'] - current_row['step'])
                
                # Calculate the interpolated position using linear interpolation
                interpolated_position = {
                    'Timestamp_(ms)': np.nan,
                    'Type': 'Interpolated_Location',
                    'step': half_step,
                    'value_1': np.nan,
                    'value_2': np.nan,
                    'value_3': np.nan,
                    'GroundTruth': np.nan,
                    'value_4': current_row['value_4'] + t * (next_row['value_4'] - current_row['value_4']),
                    'value_5': current_row['value_5'] + t * (next_row['value_5'] - current_row['value_5']),
                    'GroundTruthHeadingComputed': np.nan
                }
                
                # Append the interpolated position to the list
                interpolated_positions.append(interpolated_position)
        
        # Append the last ground truth location to the list
        interpolated_positions.append(combined_data.iloc[-1].to_dict())
        
        # Create a DataFrame from the list of interpolated positions
        interpolated_positions_df = pd.DataFrame(interpolated_positions)
        
        # Sort the DataFrame by the 'step' column
        interpolated_positions_df.sort_values(by='step', inplace=True)
        
        # Reset the index of the DataFrame
        interpolated_positions_df.reset_index(drop=True, inplace=True)
        
        logger.info(f"Generated {len(interpolated_positions_df)} interpolated ground truth positions")
        return interpolated_positions_df

# Helper function to list available sensor data files
def list_available_data_files(data_dir: str = 'data') -> List[str]:
    """
    List all available sensor data files in the data directory.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        List of filenames that match the expected pattern
    """
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory {data_dir} does not exist")
        return []
    
    data_files = [f for f in os.listdir(data_dir) 
                 if f.endswith('.TXT') or f.endswith('.txt') and 'CompassGyro' in f]
    
    logger.info(f"Found {len(data_files)} sensor data files in {data_dir}")
    return data_files 