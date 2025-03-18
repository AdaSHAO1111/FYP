import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
import logging
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SensorDataCleaner:
    """
    A class for cleaning sensor data to handle duplicates, outliers, and inconsistencies.
    """
    
    def __init__(self, data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize the cleaner with optional data.
        
        Args:
            data: Dictionary containing dataframes for each sensor type
                 (keys: 'gyro', 'compass', 'ground_truth', 'initial_location')
        """
        self.data = data if data is not None else {}
        
    def clean_all_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Clean all types of sensor data.
        
        Args:
            data: Dictionary containing dataframes for each sensor type
            
        Returns:
            Dictionary containing cleaned dataframes
        """
        self.data = data
        cleaned_data = {}
        
        # Clean each type of data
        if 'gyro' in self.data and not self.data['gyro'].empty:
            cleaned_data['gyro'] = self.clean_gyro_data(self.data['gyro'])
        
        if 'compass' in self.data and not self.data['compass'].empty:
            cleaned_data['compass'] = self.clean_compass_data(self.data['compass'])
            
        if 'ground_truth' in self.data and not self.data['ground_truth'].empty:
            cleaned_data['ground_truth'] = self.clean_ground_truth_data(self.data['ground_truth'])
            
        if 'initial_location' in self.data and not self.data['initial_location'].empty:
            cleaned_data['initial_location'] = self.data['initial_location']  # No special cleaning needed
        
        logger.info("All data cleaned")
        return cleaned_data
    
    def clean_gyro_data(self, gyro_data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean gyroscope data.
        
        Args:
            gyro_data: DataFrame containing gyroscope data
            
        Returns:
            Cleaned gyroscope data
        """
        logger.info(f"Cleaning gyro data: {len(gyro_data)} records")
        
        # Create a copy to avoid modifying the original
        cleaned_data = gyro_data.copy()
        
        # Convert string columns to numeric
        numeric_columns = ['axisZAngle', 'gyroSumFromstart0', 'compass']
        for col in numeric_columns:
            if col in cleaned_data.columns:
                # Try to convert to numeric, setting errors to coerce will replace invalid parsing with NaN
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                # Report how many values couldn't be converted
                null_count = cleaned_data[col].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Could not convert {null_count} values in '{col}' to numeric")
        
        # Sort by timestamp to ensure chronological order
        cleaned_data.sort_values(by='Timestamp_(ms)', inplace=True)
        
        # Remove duplicates based on timestamp
        orig_len = len(cleaned_data)
        cleaned_data.drop_duplicates(subset='Timestamp_(ms)', keep='first', inplace=True)
        logger.info(f"Removed {orig_len - len(cleaned_data)} duplicate timestamps from gyro data")
        
        # Check for missing values in important columns
        missing_vals = cleaned_data[['axisZAngle', 'gyroSumFromstart0']].isnull().sum()
        if missing_vals.sum() > 0:
            logger.warning(f"Missing values detected in gyro data: {missing_vals}")
            
            # Interpolate missing values
            cleaned_data['axisZAngle'] = cleaned_data['axisZAngle'].interpolate(method='linear')
            cleaned_data['gyroSumFromstart0'] = cleaned_data['gyroSumFromstart0'].interpolate(method='linear')
        
        # Remove outliers in axisZAngle using Z-score method
        # Only calculate z-scores for non-null values
        non_null_indices = cleaned_data['axisZAngle'].dropna().index
        non_null_values = cleaned_data.loc[non_null_indices, 'axisZAngle'].values
        
        if len(non_null_values) > 0:
            # Calculate z-scores only for non-null values
            z_scores = stats.zscore(non_null_values)
            abs_z_scores = np.abs(z_scores)
            
            # Create an outlier flag column
            cleaned_data['axisZAngle_is_outlier'] = False
            
            # Find indices where z-score is greater than threshold
            outlier_mask = abs_z_scores > 3  # Keep only entries with z-score < 3
            outlier_indices = non_null_indices[outlier_mask]
            
            if len(outlier_indices) > 0:
                cleaned_data.loc[outlier_indices, 'axisZAngle_is_outlier'] = True
                logger.info(f"Identified {len(outlier_indices)} outliers in gyro data (axisZAngle)")
        
        return cleaned_data
    
    def clean_compass_data(self, compass_data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean compass data.
        
        Args:
            compass_data: DataFrame containing compass data
            
        Returns:
            Cleaned compass data
        """
        logger.info(f"Cleaning compass data: {len(compass_data)} records")
        
        # Create a copy to avoid modifying the original
        cleaned_data = compass_data.copy()
        
        # Convert string columns to numeric
        numeric_columns = ['Magnetic_Field_Magnitude', 'gyroSumFromstart0', 'compass']
        for col in numeric_columns:
            if col in cleaned_data.columns:
                # Try to convert to numeric, setting errors to coerce will replace invalid parsing with NaN
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                # Report how many values couldn't be converted
                null_count = cleaned_data[col].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Could not convert {null_count} values in '{col}' to numeric")
        
        # Sort by timestamp to ensure chronological order
        cleaned_data.sort_values(by='Timestamp_(ms)', inplace=True)
        
        # Remove duplicates based on timestamp
        orig_len = len(cleaned_data)
        cleaned_data.drop_duplicates(subset='Timestamp_(ms)', keep='first', inplace=True)
        logger.info(f"Removed {orig_len - len(cleaned_data)} duplicate timestamps from compass data")
        
        # Check for missing values in important columns
        missing_vals = cleaned_data[['Magnetic_Field_Magnitude', 'compass']].isnull().sum()
        if missing_vals.sum() > 0:
            logger.warning(f"Missing values detected in compass data: {missing_vals}")
            
            # Interpolate missing values
            cleaned_data['Magnetic_Field_Magnitude'] = cleaned_data['Magnetic_Field_Magnitude'].interpolate(method='linear')
            cleaned_data['compass'] = cleaned_data['compass'].interpolate(method='linear')
        
        # Handle compass heading outliers 
        # Compass readings can be cyclical (0-360), so we need a different approach than simple z-scores
        
        # Method 1: Moving median filter to detect outliers
        compass_headings = cleaned_data['compass'].values
        window_size = 5  # Use a window of 5 samples
        
        # Calculate moving median
        median_headings = np.zeros_like(compass_headings)
        for i in range(len(compass_headings)):
            window_start = max(0, i - window_size // 2)
            window_end = min(len(compass_headings), i + window_size // 2 + 1)
            window_headings = compass_headings[window_start:window_end]
            median_headings[i] = np.median(window_headings)
        
        # Calculate absolute differences between compass readings and median
        # Need to account for cyclicity (e.g., 359° is close to 1°)
        heading_diffs = np.minimum(
            np.abs(compass_headings - median_headings),
            360 - np.abs(compass_headings - median_headings)
        )
        
        # Mark outliers where difference is greater than threshold
        threshold = 30  # 30 degrees
        cleaned_data['compass_is_outlier'] = heading_diffs > threshold
        
        outliers_count = cleaned_data['compass_is_outlier'].sum()
        if outliers_count > 0:
            logger.info(f"Identified {outliers_count} outliers in compass data")
            
            # For extreme outliers, replace with moving median
            extreme_outliers = heading_diffs > 60  # 60 degrees
            cleaned_data.loc[extreme_outliers, 'compass'] = median_headings[extreme_outliers]
            logger.info(f"Corrected {extreme_outliers.sum()} extreme outliers in compass data")
        
        return cleaned_data
    
    def clean_ground_truth_data(self, ground_truth_data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean ground truth location data.
        
        Args:
            ground_truth_data: DataFrame containing ground truth location data
            
        Returns:
            Cleaned ground truth location data
        """
        logger.info(f"Cleaning ground truth data: {len(ground_truth_data)} records")
        
        # Create a copy to avoid modifying the original
        cleaned_data = ground_truth_data.copy()
        
        # Convert string columns to numeric
        numeric_columns = ['value_4', 'value_5', 'step']
        for col in numeric_columns:
            if col in cleaned_data.columns:
                # Try to convert to numeric, setting errors to coerce will replace invalid parsing with NaN
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                # Report how many values couldn't be converted
                null_count = cleaned_data[col].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Could not convert {null_count} values in '{col}' to numeric")
        
        # Sort by step to ensure correct order
        cleaned_data.sort_values(by='step', inplace=True)
        
        # Remove duplicates based on step
        orig_len = len(cleaned_data)
        cleaned_data.drop_duplicates(subset='step', keep='last', inplace=True)
        logger.info(f"Removed {orig_len - len(cleaned_data)} duplicate steps from ground truth data")
        
        # Check for missing values in important columns
        missing_vals = cleaned_data[['value_4', 'value_5']].isnull().sum()
        if missing_vals.sum() > 0:
            logger.warning(f"Missing values detected in ground truth data: {missing_vals}")
            # For ground truth data, we don't want to interpolate missing position values
            # Instead, we'll drop rows with missing east/north coordinates
            cleaned_data = cleaned_data.dropna(subset=['value_4', 'value_5'])
            logger.info(f"Dropped {missing_vals.sum()} rows with missing position values")
        
        # Check for plausible coordinates (based on the context of your navigation system)
        # This is a simple check; you might need a more sophisticated one based on your data
        if len(cleaned_data) >= 2:
            # Calculate distances between consecutive points
            east_diffs = cleaned_data['value_4'].diff()
            north_diffs = cleaned_data['value_5'].diff()
            distances = np.sqrt(east_diffs**2 + north_diffs**2)
            
            # Identify potential jumps/teleports in position
            max_plausible_distance = 10.0  # Example: max 10 units between consecutive points
            implausible_jumps = distances > max_plausible_distance
            
            if implausible_jumps.sum() > 0:
                logger.warning(f"Found {implausible_jumps.sum()} implausible position jumps in ground truth data")
                cleaned_data['implausible_jump'] = False
                cleaned_data.loc[implausible_jumps.index, 'implausible_jump'] = True
        
        return cleaned_data
    
    @staticmethod
    def remove_outliers_iqr(data: pd.Series, column: str, k: float = 1.5) -> pd.Series:
        """
        Remove outliers using the Interquartile Range (IQR) method.
        
        Args:
            data: DataFrame containing the data
            column: Column name to check for outliers
            k: Multiplier for IQR (default 1.5)
            
        Returns:
            Boolean mask where False indicates outliers
        """
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        
        return ~((data[column] < lower_bound) | (data[column] > upper_bound))
    
    @staticmethod
    def remove_outliers_zscore(data: pd.Series, column: str, threshold: float = 3.0) -> pd.Series:
        """
        Remove outliers using the Z-score method.
        
        Args:
            data: DataFrame containing the data
            column: Column name to check for outliers
            threshold: Z-score threshold (default 3.0)
            
        Returns:
            Boolean mask where False indicates outliers
        """
        z_scores = stats.zscore(data[column])
        abs_z_scores = np.abs(z_scores)
        return abs_z_scores < threshold 