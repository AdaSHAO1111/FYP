import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SensorAnomalyDetector:
    """
    A class to detect anomalies in sensor data that can't be classified normally.
    Implements different anomaly detection methods for different sensor types.
    """
    
    def __init__(self, output_dir: str = 'output/anomalies'):
        """
        Initialize the anomaly detector.
        
        Args:
            output_dir: Directory to save visualizations and results
        """
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
    def detect_all_anomalies(self, 
                           gyro_data: pd.DataFrame, 
                           compass_data: pd.DataFrame,
                           ground_truth_data: pd.DataFrame,
                           visualize: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Detect anomalies in all sensor data types.
        
        Args:
            gyro_data: DataFrame with gyroscope data
            compass_data: DataFrame with compass data
            ground_truth_data: DataFrame with ground truth data
            visualize: Whether to create visualizations
            
        Returns:
            Dictionary of DataFrames with anomaly detection results
        """
        results = {}
        
        # Detect anomalies in gyro data
        if not gyro_data.empty:
            results['gyro'] = self.detect_gyro_anomalies(gyro_data, visualize=visualize)
            
        # Detect anomalies in compass data
        if not compass_data.empty:
            results['compass'] = self.detect_compass_anomalies(compass_data, visualize=visualize)
            
        # Detect anomalies in ground truth data
        if not ground_truth_data.empty:
            results['ground_truth'] = self.detect_ground_truth_anomalies(ground_truth_data, visualize=visualize)
        
        return results
    
    def detect_gyro_anomalies(self, gyro_data: pd.DataFrame, visualize: bool = True) -> pd.DataFrame:
        """
        Detect anomalies in gyroscope data.
        
        Args:
            gyro_data: DataFrame with gyroscope data
            visualize: Whether to create visualizations
            
        Returns:
            DataFrame with anomaly detection results
        """
        logger.info(f"Detecting anomalies in gyro data ({len(gyro_data)} records)")
        
        # Create a copy to avoid modifying the original
        result_data = gyro_data.copy()
        
        # Method 1: Isolation Forest for multivariate anomaly detection
        # Prepare features for anomaly detection
        features = ['axisZAngle', 'gyroSumFromstart0']
        if all(f in result_data.columns for f in features):
            X = result_data[features].copy()
            
            # Fill missing values with median to not affect the anomaly detection
            for col in features:
                X[col] = X[col].fillna(X[col].median())
            
            # Fit isolation forest model
            model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
            result_data['isolation_forest_anomaly'] = model.fit_predict(X)
            
            # Convert to boolean (True for anomalies)
            result_data['isolation_forest_anomaly'] = result_data['isolation_forest_anomaly'] == -1
            
            anomaly_count = result_data['isolation_forest_anomaly'].sum()
            logger.info(f"Identified {anomaly_count} anomalies in gyro data using Isolation Forest")
            
            if visualize and anomaly_count > 0:
                self._visualize_gyro_anomalies(result_data)
        
        # Method 2: Time series specific anomalies - detect sudden jumps
        if 'Timestamp_(ms)' in result_data.columns and 'gyroSumFromstart0' in result_data.columns:
            # Sort by timestamp to ensure chronological order
            result_data.sort_values(by='Timestamp_(ms)', inplace=True)
            
            # Calculate the difference between consecutive readings
            result_data['gyro_diff'] = result_data['gyroSumFromstart0'].diff()
            
            # Detect large jumps in the gyro readings (more than 3 std devs)
            std_gyro_diff = result_data['gyro_diff'].std()
            result_data['jump_anomaly'] = abs(result_data['gyro_diff']) > 3 * std_gyro_diff
            
            jump_anomaly_count = result_data['jump_anomaly'].sum()
            logger.info(f"Identified {jump_anomaly_count} sudden jump anomalies in gyro data")
        
        # Combine all anomaly detection methods
        if 'isolation_forest_anomaly' in result_data.columns and 'jump_anomaly' in result_data.columns:
            result_data['is_anomaly'] = result_data['isolation_forest_anomaly'] | result_data['jump_anomaly']
            logger.info(f"Total of {result_data['is_anomaly'].sum()} anomalies detected in gyro data")
        elif 'isolation_forest_anomaly' in result_data.columns:
            result_data['is_anomaly'] = result_data['isolation_forest_anomaly']
        elif 'jump_anomaly' in result_data.columns:
            result_data['is_anomaly'] = result_data['jump_anomaly']
        
        return result_data
    
    def detect_compass_anomalies(self, compass_data: pd.DataFrame, visualize: bool = True) -> pd.DataFrame:
        """
        Detect anomalies in compass data.
        
        Args:
            compass_data: DataFrame with compass data
            visualize: Whether to create visualizations
            
        Returns:
            DataFrame with anomaly detection results
        """
        logger.info(f"Detecting anomalies in compass data ({len(compass_data)} records)")
        
        # Create a copy to avoid modifying the original
        result_data = compass_data.copy()
        
        # Method 1: Local Outlier Factor for detecting local anomalies
        # Prepare features for anomaly detection
        features = ['compass', 'Magnetic_Field_Magnitude']
        if all(f in result_data.columns for f in features):
            X = result_data[features].copy()
            
            # Fill missing values with median to not affect the anomaly detection
            for col in features:
                X[col] = X[col].fillna(X[col].median())
            
            # Normalize compass to handle circular data
            # Convert degrees to radians first
            compass_rad = np.radians(X['compass'])
            # Represent as sine and cosine components
            X['compass_sin'] = np.sin(compass_rad)
            X['compass_cos'] = np.cos(compass_rad)
            # Drop original compass column
            X = X.drop('compass', axis=1)
            
            # Fit LOF model
            model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
            result_data['lof_anomaly'] = model.fit_predict(X)
            
            # Convert to boolean (True for anomalies)
            result_data['lof_anomaly'] = result_data['lof_anomaly'] == -1
            
            anomaly_count = result_data['lof_anomaly'].sum()
            logger.info(f"Identified {anomaly_count} anomalies in compass data using Local Outlier Factor")
            
            if visualize and anomaly_count > 0:
                self._visualize_compass_anomalies(result_data)
        
        # Method 2: Detect sudden heading jumps
        if 'Timestamp_(ms)' in result_data.columns and 'compass' in result_data.columns:
            # Sort by timestamp to ensure chronological order
            result_data.sort_values(by='Timestamp_(ms)', inplace=True)
            
            # Handle circular data when calculating differences
            # Convert to radians
            compass_rad = np.radians(result_data['compass'])
            # Calculate sin and cos
            sin_compass = np.sin(compass_rad)
            cos_compass = np.cos(compass_rad)
            
            # Calculate the difference in complex form and then get the angle
            complex_diff = (cos_compass + 1j*sin_compass) / (cos_compass.shift(1) + 1j*sin_compass.shift(1))
            angle_diff = np.abs(np.angle(complex_diff, deg=True))
            
            # Mark as anomaly if the heading change is too large
            threshold = 45  # 45 degrees threshold for sudden changes
            result_data['heading_jump_anomaly'] = angle_diff > threshold
            
            jump_anomaly_count = result_data['heading_jump_anomaly'].sum()
            logger.info(f"Identified {jump_anomaly_count} sudden heading jump anomalies in compass data")
        
        # Combine all anomaly detection methods
        if 'lof_anomaly' in result_data.columns and 'heading_jump_anomaly' in result_data.columns:
            result_data['is_anomaly'] = result_data['lof_anomaly'] | result_data['heading_jump_anomaly']
            logger.info(f"Total of {result_data['is_anomaly'].sum()} anomalies detected in compass data")
        elif 'lof_anomaly' in result_data.columns:
            result_data['is_anomaly'] = result_data['lof_anomaly']
        elif 'heading_jump_anomaly' in result_data.columns:
            result_data['is_anomaly'] = result_data['heading_jump_anomaly']
        
        return result_data
    
    def detect_ground_truth_anomalies(self, ground_truth_data: pd.DataFrame, visualize: bool = True) -> pd.DataFrame:
        """
        Detect anomalies in ground truth location data.
        
        Args:
            ground_truth_data: DataFrame with ground truth data
            visualize: Whether to create visualizations
            
        Returns:
            DataFrame with anomaly detection results
        """
        logger.info(f"Detecting anomalies in ground truth data ({len(ground_truth_data)} records)")
        
        # Create a copy to avoid modifying the original
        result_data = ground_truth_data.copy()
        
        # Detect spatial anomalies in position data
        if 'value_4' in result_data.columns and 'value_5' in result_data.columns:
            # Extract coordinates
            coords = result_data[['value_4', 'value_5']].copy()
            
            # Drop rows with missing values
            coords = coords.dropna()
            
            if len(coords) > 10:  # Need enough data points for meaningful analysis
                # Detect outliers using Isolation Forest
                model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
                
                # Get indexes after dropping NA rows
                valid_indices = coords.index
                
                # Fit model and predict
                anomaly_labels = model.fit_predict(coords)
                
                # Create anomaly column
                result_data['position_anomaly'] = False
                
                # Set anomaly flags for valid indices
                result_data.loc[valid_indices, 'position_anomaly'] = (anomaly_labels == -1)
                
                anomaly_count = result_data['position_anomaly'].sum()
                logger.info(f"Identified {anomaly_count} position anomalies in ground truth data")
                
                if visualize and anomaly_count > 0:
                    self._visualize_ground_truth_anomalies(result_data)
            else:
                logger.warning("Not enough valid position data for anomaly detection")
        
        # Method 2: Detect implausible travel speeds between consecutive points
        if 'step' in result_data.columns and 'value_4' in result_data.columns and 'value_5' in result_data.columns:
            # Sort by step to ensure correct order
            result_data.sort_values(by='step', inplace=True)
            
            # Calculate distances between consecutive points
            east_diffs = result_data['value_4'].diff()
            north_diffs = result_data['value_5'].diff()
            step_diffs = result_data['step'].diff()
            
            # Calculate distance and speed (distance per step)
            distances = np.sqrt(east_diffs**2 + north_diffs**2)
            speeds = distances / step_diffs
            
            # Replace inf with NaN
            speeds = speeds.replace([np.inf, -np.inf], np.nan)
            
            # Detect implausible speeds (more than 3 std devs from mean)
            mean_speed = speeds.mean()
            std_speed = speeds.std()
            threshold = mean_speed + 3 * std_speed
            
            result_data['speed_anomaly'] = (speeds > threshold)
            
            speed_anomaly_count = result_data['speed_anomaly'].sum()
            logger.info(f"Identified {speed_anomaly_count} travel speed anomalies in ground truth data")
        
        # Combine all anomaly detection methods
        if 'position_anomaly' in result_data.columns and 'speed_anomaly' in result_data.columns:
            result_data['is_anomaly'] = result_data['position_anomaly'] | result_data['speed_anomaly']
            logger.info(f"Total of {result_data['is_anomaly'].sum()} anomalies detected in ground truth data")
        elif 'position_anomaly' in result_data.columns:
            result_data['is_anomaly'] = result_data['position_anomaly']
        elif 'speed_anomaly' in result_data.columns:
            result_data['is_anomaly'] = result_data['speed_anomaly']
        
        return result_data
    
    def detect_unclassifiable_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect unclassifiable data that doesn't match known sensor types.
        
        Args:
            data: Raw DataFrame from data parser
            
        Returns:
            DataFrame with unclassifiable data detection results
        """
        logger.info(f"Detecting unclassifiable data in {len(data)} records")
        
        # Create a copy to avoid modifying the original
        result_data = data.copy()
        
        # Define known sensor types
        known_types = ['Gyro', 'Compass', 'Ground_truth_Location', 'Initial_Location']
        
        # Mark rows that don't belong to known types
        result_data['unclassifiable'] = ~result_data['Type'].isin(known_types)
        
        unclassifiable_count = result_data['unclassifiable'].sum()
        
        if unclassifiable_count > 0:
            logger.warning(f"Found {unclassifiable_count} unclassifiable data records")
            
            # Group by unknown types to see what we have
            unknown_types = result_data[result_data['unclassifiable']]['Type'].value_counts()
            logger.info(f"Unknown data types: {unknown_types.to_dict()}")
        else:
            logger.info("No unclassifiable data detected")
        
        return result_data
    
    def _visualize_gyro_anomalies(self, gyro_data: pd.DataFrame) -> None:
        """
        Create visualizations for gyro anomalies.
        
        Args:
            gyro_data: DataFrame with gyro anomaly detection results
        """
        # Plot gyro data with anomalies highlighted
        plt.figure(figsize=(12, 6))
        
        # Plot all points
        plt.plot(gyro_data['Timestamp_(ms)'], gyro_data['gyroSumFromstart0'], 
                 color='blue', alpha=0.7, label='Gyro Data')
        
        # Highlight anomalies
        if 'is_anomaly' in gyro_data.columns:
            anomalies = gyro_data[gyro_data['is_anomaly']]
            plt.scatter(anomalies['Timestamp_(ms)'], anomalies['gyroSumFromstart0'], 
                       color='red', s=50, label='Anomalies')
        
        plt.xlabel('Timestamp (ms)')
        plt.ylabel('Gyro Reading')
        plt.title('Gyroscope Anomalies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        import os
        plt.savefig(os.path.join(self.output_dir, 'gyro_anomalies.png'), dpi=300)
        logger.info(f"Saved gyro anomaly plot to {self.output_dir}/gyro_anomalies.png")
        
        plt.close()
    
    def _visualize_compass_anomalies(self, compass_data: pd.DataFrame) -> None:
        """
        Create visualizations for compass anomalies.
        
        Args:
            compass_data: DataFrame with compass anomaly detection results
        """
        # Plot compass data with anomalies highlighted
        plt.figure(figsize=(12, 6))
        
        # Plot all points
        plt.plot(compass_data['Timestamp_(ms)'], compass_data['compass'], 
                 color='green', alpha=0.7, label='Compass Data')
        
        # Highlight anomalies
        if 'is_anomaly' in compass_data.columns:
            anomalies = compass_data[compass_data['is_anomaly']]
            plt.scatter(anomalies['Timestamp_(ms)'], anomalies['compass'], 
                       color='red', s=50, label='Anomalies')
        
        plt.xlabel('Timestamp (ms)')
        plt.ylabel('Compass Heading (degrees)')
        plt.title('Compass Anomalies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        import os
        plt.savefig(os.path.join(self.output_dir, 'compass_anomalies.png'), dpi=300)
        logger.info(f"Saved compass anomaly plot to {self.output_dir}/compass_anomalies.png")
        
        plt.close()
        
        # Also plot magnetic field magnitude if available
        if 'Magnetic_Field_Magnitude' in compass_data.columns:
            plt.figure(figsize=(12, 6))
            
            # Plot all points
            plt.plot(compass_data['Timestamp_(ms)'], compass_data['Magnetic_Field_Magnitude'], 
                     color='purple', alpha=0.7, label='Magnetic Field')
            
            # Highlight anomalies
            if 'is_anomaly' in compass_data.columns:
                anomalies = compass_data[compass_data['is_anomaly']]
                plt.scatter(anomalies['Timestamp_(ms)'], anomalies['Magnetic_Field_Magnitude'], 
                           color='red', s=50, label='Anomalies')
            
            plt.xlabel('Timestamp (ms)')
            plt.ylabel('Magnetic Field Magnitude')
            plt.title('Magnetic Field Anomalies')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, 'magnetic_field_anomalies.png'), dpi=300)
            logger.info(f"Saved magnetic field anomaly plot to {self.output_dir}/magnetic_field_anomalies.png")
            
            plt.close()
    
    def _visualize_ground_truth_anomalies(self, ground_truth_data: pd.DataFrame) -> None:
        """
        Create visualizations for ground truth anomalies.
        
        Args:
            ground_truth_data: DataFrame with ground truth anomaly detection results
        """
        # Plot ground truth positions with anomalies highlighted
        plt.figure(figsize=(10, 8))
        
        # Plot all positions
        plt.scatter(ground_truth_data['value_4'], ground_truth_data['value_5'], 
                   color='blue', alpha=0.7, label='Ground Truth Positions')
        
        # Highlight anomalies
        if 'is_anomaly' in ground_truth_data.columns:
            anomalies = ground_truth_data[ground_truth_data['is_anomaly']]
            plt.scatter(anomalies['value_4'], anomalies['value_5'], 
                       color='red', s=100, label='Anomalies')
        
        plt.xlabel('East')
        plt.ylabel('North')
        plt.title('Ground Truth Position Anomalies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Make axis equal
        plt.axis('equal')
        
        # Save plot
        import os
        plt.savefig(os.path.join(self.output_dir, 'ground_truth_anomalies.png'), dpi=300)
        logger.info(f"Saved ground truth anomaly plot to {self.output_dir}/ground_truth_anomalies.png")
        
        plt.close() 