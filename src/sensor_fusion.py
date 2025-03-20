import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import logging
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SensorFusion:
    """
    Class for implementing various sensor fusion algorithms to combine 
    gyroscope and compass data for improved heading estimation.
    """
    
    def __init__(self, output_dir: str = 'output/fusion'):
        """
        Initialize the sensor fusion class.
        
        Args:
            output_dir: Directory to save outputs and visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        
    def fuse_sensors(self, 
                    gyro_data: pd.DataFrame, 
                    compass_data: pd.DataFrame,
                    method: str = 'ekf',
                    ground_truth_data: Optional[pd.DataFrame] = None,
                    visualize: bool = True) -> pd.DataFrame:
        """
        Fuse gyroscope and compass data using the specified method.
        
        Args:
            gyro_data: DataFrame with gyroscope data
            compass_data: DataFrame with compass data
            method: Fusion method to use: 'ekf' (Extended Kalman Filter), 
                   'ukf' (Unscented Kalman Filter), 'lstm' (LSTM neural network),
                   'adaptive' (Adaptive filtering), 'context' (Context-aware models)
            ground_truth_data: Optional DataFrame with ground truth data for evaluation
            visualize: Whether to generate visualizations
            
        Returns:
            DataFrame with fused heading data
        """
        if method.lower() == 'ekf':
            return self.extended_kalman_filter(gyro_data, compass_data, ground_truth_data, visualize)
        elif method.lower() == 'ukf':
            return self.unscented_kalman_filter(gyro_data, compass_data, ground_truth_data, visualize)
        elif method.lower() == 'lstm':
            return self.lstm_fusion(gyro_data, compass_data, ground_truth_data, visualize)
        elif method.lower() == 'adaptive':
            return self.adaptive_filtering(gyro_data, compass_data, ground_truth_data, visualize)
        elif method.lower() == 'context':
            return self.context_aware_fusion(gyro_data, compass_data, ground_truth_data, visualize)
        else:
            logger.warning(f"Unknown fusion method '{method}', using EKF instead")
            return self.extended_kalman_filter(gyro_data, compass_data, ground_truth_data, visualize)
    
    def extended_kalman_filter(self, 
                              gyro_data: pd.DataFrame, 
                              compass_data: pd.DataFrame,
                              ground_truth_data: Optional[pd.DataFrame] = None,
                              visualize: bool = True) -> pd.DataFrame:
        """
        Implement Extended Kalman Filter for sensor fusion between gyroscope and compass data.
        
        The EKF fuses:
        - Gyroscope data for short-term accuracy (but it can drift over time)
        - Compass data for absolute heading reference (but it can be influenced by magnetic disturbances)
        
        Args:
            gyro_data: DataFrame with gyroscope data
            compass_data: DataFrame with compass data
            ground_truth_data: Optional DataFrame with ground truth data for evaluation
            visualize: Whether to generate visualizations
            
        Returns:
            DataFrame with fused heading data
        """
        logger.info("Applying Extended Kalman Filter for sensor fusion")
        
        # Ensure data is sorted by timestamp
        gyro_data = gyro_data.sort_values('Timestamp_(ms)').reset_index(drop=True)
        compass_data = compass_data.sort_values('Timestamp_(ms)').reset_index(drop=True)
        
        # Create a common timeline for all measurements
        all_timestamps = np.union1d(gyro_data['Timestamp_(ms)'].values, 
                                   compass_data['Timestamp_(ms)'].values)
        
        # Initialize state and covariance
        # State = [heading, gyro_bias]
        state = np.zeros(2)
        
        # Initial heading from compass if available, otherwise use 0
        if len(compass_data) > 0:
            state[0] = compass_data['compass'].iloc[0]
        
        # Covariance matrix
        covariance = np.diag([np.radians(10)**2, np.radians(1)**2])  # Initial uncertainty
        
        # Process noise covariance
        # Q = diag([heading_process_noise, gyro_bias_process_noise])
        q_heading = np.radians(0.1)**2  # Process noise for heading
        q_bias = np.radians(0.01)**2    # Process noise for gyro bias
        Q = np.diag([q_heading, q_bias])
        
        # Measurement noise for compass
        compass_noise = np.radians(5)**2
        
        # Storage for results
        results = []
        
        # Previous timestamp for delta time calculation
        prev_timestamp = all_timestamps[0]
        
        # Process each timestamp
        for timestamp in all_timestamps:
            # Calculate time difference in seconds
            dt = (timestamp - prev_timestamp) / 1000.0  # Convert ms to seconds
            prev_timestamp = timestamp
            
            # Skip if dt is too large (gap in data)
            if dt > 1.0:  # More than 1 second gap
                continue
            
            # --- Prediction step ---
            # Find closest gyro measurement
            gyro_idx = np.argmin(np.abs(gyro_data['Timestamp_(ms)'].values - timestamp))
            gyro_reading = gyro_data['gyroSumFromstart0'].iloc[gyro_idx]
            gyro_timestamp = gyro_data['Timestamp_(ms)'].iloc[gyro_idx]
            
            # Only use gyro if it's close enough to the current timestamp
            if abs(gyro_timestamp - timestamp) < 100:  # Within 100ms
                # Jacobian of state transition
                F = np.array([[1, -dt], [0, 1]])
                
                # Predict state: x = F * x + B * u
                # u is the gyro reading, B is the control matrix
                gyro_rate = np.radians(gyro_reading)  # Convert to radians
                state = F @ state + np.array([dt * gyro_rate, 0])
                
                # Normalize heading to [0, 2π]
                state[0] = state[0] % (2 * np.pi)
                
                # Update covariance: P = F * P * F^T + Q
                covariance = F @ covariance @ F.T + Q
            
            # --- Update step (if compass data is available) ---
            compass_idx = np.argmin(np.abs(compass_data['Timestamp_(ms)'].values - timestamp))
            compass_timestamp = compass_data['Timestamp_(ms)'].iloc[compass_idx]
            
            # Only use compass if it's close enough to the current timestamp
            if abs(compass_timestamp - timestamp) < 100:  # Within 100ms
                compass_reading = compass_data['compass'].iloc[compass_idx]
                compass_reading_rad = np.radians(compass_reading)
                
                # Check magnetic field magnitude for anomalies
                if 'Magnetic_Field_Magnitude' in compass_data.columns:
                    mag_magnitude = compass_data['Magnetic_Field_Magnitude'].iloc[compass_idx]
                    
                    # Skip update if magnetic field magnitude is too extreme
                    if mag_magnitude < 20 or mag_magnitude > 100:  # Example thresholds
                        logger.debug(f"Skipping compass update at {timestamp} due to anomalous magnetic field: {mag_magnitude}")
                        continue
                
                # Check if compass measurement is an anomaly (if flag exists)
                if 'is_anomaly' in compass_data.columns and compass_data['is_anomaly'].iloc[compass_idx]:
                    logger.debug(f"Skipping compass update at {timestamp} due to flagged anomaly")
                    continue
                
                # Measurement matrix
                H = np.array([1, 0]).reshape(1, 2)
                
                # Calculate innovation (difference between measurement and prediction)
                # Need to handle heading wraparound for correct error calculation
                predicted_heading = state[0]
                error = compass_reading_rad - predicted_heading
                
                # Normalize error to [-π, π]
                while error > np.pi:
                    error -= 2 * np.pi
                while error < -np.pi:
                    error += 2 * np.pi
                
                # Calculate Kalman gain
                S = H @ covariance @ H.T + compass_noise
                K = covariance @ H.T / S
                
                # Update state
                state = state + K.flatten() * error
                
                # Normalize heading to [0, 2π]
                state[0] = state[0] % (2 * np.pi)
                
                # Update covariance
                covariance = (np.eye(2) - K @ H) @ covariance
            
            # Store result
            heading_deg = np.degrees(state[0])
            gyro_bias_deg = np.degrees(state[1])
            
            results.append({
                'Timestamp_(ms)': timestamp,
                'fused_heading': heading_deg,
                'gyro_bias': gyro_bias_deg,
                'heading_variance': covariance[0, 0]
            })
        
        # Create DataFrame from results
        fused_data = pd.DataFrame(results)
        
        # Evaluate against ground truth if available
        if ground_truth_data is not None and len(ground_truth_data) > 0 and visualize:
            self._evaluate_against_ground_truth(fused_data, ground_truth_data, gyro_data, compass_data, method='ekf')
        
        # Visualize results if requested
        if visualize:
            self._visualize_fusion_results(fused_data, gyro_data, compass_data, ground_truth_data)
        
        # Save results
        fused_data.to_csv(os.path.join(self.output_dir, 'data', 'ekf_fused_heading.csv'), index=False)
        
        return fused_data
    
    def unscented_kalman_filter(self, 
                               gyro_data: pd.DataFrame, 
                               compass_data: pd.DataFrame,
                               ground_truth_data: Optional[pd.DataFrame] = None,
                               visualize: bool = True) -> pd.DataFrame:
        """
        Implement Unscented Kalman Filter for sensor fusion between gyroscope and compass data.
        
        The UKF differs from the EKF in how it handles non-linearities:
        - UKF uses sigma points to approximate the probability distribution
        - This allows for better handling of non-linearities and non-Gaussian noise
        
        Args:
            gyro_data: DataFrame with gyroscope data
            compass_data: DataFrame with compass data
            ground_truth_data: Optional DataFrame with ground truth data for evaluation
            visualize: Whether to generate visualizations
            
        Returns:
            DataFrame with fused heading data
        """
        logger.info("Applying Unscented Kalman Filter for sensor fusion")
        
        # Ensure data is sorted by timestamp
        gyro_data = gyro_data.sort_values('Timestamp_(ms)').reset_index(drop=True)
        compass_data = compass_data.sort_values('Timestamp_(ms)').reset_index(drop=True)
        
        # Create a common timeline for all measurements
        all_timestamps = np.union1d(gyro_data['Timestamp_(ms)'].values, 
                                   compass_data['Timestamp_(ms)'].values)
        
        # Initialize state and covariance
        # State = [heading, gyro_bias]
        state_dim = 2
        state = np.zeros(state_dim)
        
        # Initial heading from compass if available, otherwise use 0
        if len(compass_data) > 0:
            state[0] = np.radians(compass_data['compass'].iloc[0])  # Convert to radians
        
        # Covariance matrix
        covariance = np.diag([np.radians(10)**2, np.radians(1)**2])  # Initial uncertainty
        
        # UKF parameters
        alpha = 1e-3  # Determines spread of sigma points (usually small, e.g., 1e-3)
        beta = 2.0    # Optimal for Gaussian distributions
        kappa = 0.0   # Secondary scaling parameter (usually 0 or 3-n)
        
        # Calculate lambda parameter for UKF
        lambd = alpha**2 * (state_dim + kappa) - state_dim
        
        # Calculate weights for mean and covariance
        weights_m = np.zeros(2 * state_dim + 1)
        weights_c = np.zeros(2 * state_dim + 1)
        
        weights_m[0] = lambd / (state_dim + lambd)
        weights_c[0] = lambd / (state_dim + lambd) + (1 - alpha**2 + beta)
        
        for i in range(1, 2 * state_dim + 1):
            weights_m[i] = 1.0 / (2 * (state_dim + lambd))
            weights_c[i] = 1.0 / (2 * (state_dim + lambd))
        
        # Process noise covariance
        q_heading = np.radians(0.1)**2  # Process noise for heading
        q_bias = np.radians(0.01)**2    # Process noise for gyro bias
        Q = np.diag([q_heading, q_bias])
        
        # Measurement noise for compass
        compass_noise = np.radians(5)**2
        R = np.array([[compass_noise]])
        
        # Storage for results
        results = []
        
        # Previous timestamp for delta time calculation
        prev_timestamp = all_timestamps[0]
        
        # Process each timestamp
        for timestamp in all_timestamps:
            # Calculate time difference in seconds
            dt = (timestamp - prev_timestamp) / 1000.0  # Convert ms to seconds
            prev_timestamp = timestamp
            
            # Skip if dt is too large (gap in data)
            if dt > 1.0:  # More than 1 second gap
                continue
            
            # Find closest gyro measurement
            gyro_idx = np.argmin(np.abs(gyro_data['Timestamp_(ms)'].values - timestamp))
            gyro_reading = gyro_data['gyroSumFromstart0'].iloc[gyro_idx]
            gyro_timestamp = gyro_data['Timestamp_(ms)'].iloc[gyro_idx]
            
            # Only use gyro if it's close enough to the current timestamp
            if abs(gyro_timestamp - timestamp) < 100:  # Within 100ms
                # --- Prediction step ---
                # Generate sigma points
                sigma_points = self._generate_sigma_points(state, covariance, lambd)
                
                # Convert gyro_reading to radians per second
                gyro_rate = np.radians(gyro_reading)
                
                # Propagate sigma points through process model
                propagated_sigmas = np.zeros_like(sigma_points)
                for i in range(sigma_points.shape[0]):
                    # Apply process model to each sigma point
                    # State transition: x = [heading + dt * (gyro_rate - bias), bias]
                    propagated_sigmas[i, 0] = sigma_points[i, 0] + dt * (gyro_rate - sigma_points[i, 1])
                    propagated_sigmas[i, 1] = sigma_points[i, 1]  # Bias remains the same
                    
                    # Normalize heading to [0, 2π]
                    propagated_sigmas[i, 0] = propagated_sigmas[i, 0] % (2 * np.pi)
                
                # Compute predicted state and covariance
                predicted_state = np.zeros(state_dim)
                for i in range(sigma_points.shape[0]):
                    predicted_state += weights_m[i] * propagated_sigmas[i]
                
                # Normalize heading to [0, 2π]
                predicted_state[0] = predicted_state[0] % (2 * np.pi)
                
                predicted_covariance = np.zeros((state_dim, state_dim))
                for i in range(sigma_points.shape[0]):
                    # Compute difference with normalized heading
                    diff = propagated_sigmas[i] - predicted_state
                    
                    # Handle wrapping for heading
                    if diff[0] > np.pi:
                        diff[0] -= 2 * np.pi
                    elif diff[0] < -np.pi:
                        diff[0] += 2 * np.pi
                    
                    predicted_covariance += weights_c[i] * np.outer(diff, diff)
                
                # Add process noise
                predicted_covariance += Q
                
                # Update state and covariance with predictions
                state = predicted_state
                covariance = predicted_covariance
            
            # --- Update step (if compass data is available) ---
            compass_idx = np.argmin(np.abs(compass_data['Timestamp_(ms)'].values - timestamp))
            compass_timestamp = compass_data['Timestamp_(ms)'].iloc[compass_idx]
            
            # Only use compass if it's close enough to the current timestamp
            if abs(compass_timestamp - timestamp) < 100:  # Within 100ms
                compass_reading = compass_data['compass'].iloc[compass_idx]
                compass_reading_rad = np.radians(compass_reading)
                
                # Check magnetic field magnitude for anomalies
                if 'Magnetic_Field_Magnitude' in compass_data.columns:
                    mag_magnitude = compass_data['Magnetic_Field_Magnitude'].iloc[compass_idx]
                    
                    # Skip update if magnetic field magnitude is too extreme
                    if mag_magnitude < 20 or mag_magnitude > 100:  # Example thresholds
                        logger.debug(f"Skipping compass update at {timestamp} due to anomalous magnetic field: {mag_magnitude}")
                        continue
                
                # Check if compass measurement is an anomaly (if flag exists)
                if 'is_anomaly' in compass_data.columns and compass_data['is_anomaly'].iloc[compass_idx]:
                    logger.debug(f"Skipping compass update at {timestamp} due to flagged anomaly")
                    continue
                
                # Generate sigma points
                sigma_points = self._generate_sigma_points(state, covariance, lambd)
                
                # Measurement model: only the heading is observed
                measurement_dim = 1
                predicted_measurements = np.zeros((sigma_points.shape[0], measurement_dim))
                
                for i in range(sigma_points.shape[0]):
                    # Measurement function: z = h(x) = [heading]
                    predicted_measurements[i, 0] = sigma_points[i, 0]
                
                # Predict measurement and measurement covariance
                predicted_measurement = np.zeros(measurement_dim)
                for i in range(sigma_points.shape[0]):
                    predicted_measurement += weights_m[i] * predicted_measurements[i]
                
                measurement_covariance = np.zeros((measurement_dim, measurement_dim))
                cross_covariance = np.zeros((state_dim, measurement_dim))
                
                for i in range(sigma_points.shape[0]):
                    # Handle heading wrapping for predicted measurement
                    diff_z = predicted_measurements[i] - predicted_measurement
                    if diff_z[0] > np.pi:
                        diff_z[0] -= 2 * np.pi
                    elif diff_z[0] < -np.pi:
                        diff_z[0] += 2 * np.pi
                    
                    # Handle heading wrapping for state
                    diff_x = sigma_points[i] - state
                    if diff_x[0] > np.pi:
                        diff_x[0] -= 2 * np.pi
                    elif diff_x[0] < -np.pi:
                        diff_x[0] += 2 * np.pi
                    
                    measurement_covariance += weights_c[i] * np.outer(diff_z, diff_z)
                    cross_covariance += weights_c[i] * np.outer(diff_x, diff_z)
                
                # Add measurement noise
                measurement_covariance += R
                
                # Calculate Kalman gain
                K = cross_covariance @ np.linalg.inv(measurement_covariance)
                
                # Calculate innovation (difference between measurement and prediction)
                # Need to handle heading wraparound for correct error calculation
                error = compass_reading_rad - predicted_measurement[0]
                
                # Normalize error to [-π, π]
                while error > np.pi:
                    error -= 2 * np.pi
                while error < -np.pi:
                    error += 2 * np.pi
                
                # Update state
                state = state + K @ np.array([error])
                
                # Normalize heading to [0, 2π]
                state[0] = state[0] % (2 * np.pi)
                
                # Update covariance
                covariance = covariance - K @ measurement_covariance @ K.T
            
            # Store result
            heading_deg = np.degrees(state[0])
            gyro_bias_deg = np.degrees(state[1])
            
            results.append({
                'Timestamp_(ms)': timestamp,
                'fused_heading': heading_deg,
                'gyro_bias': gyro_bias_deg,
                'heading_variance': covariance[0, 0]
            })
        
        # Create DataFrame from results
        fused_data = pd.DataFrame(results)
        
        # Evaluate against ground truth if available
        if ground_truth_data is not None and len(ground_truth_data) > 0 and visualize:
            self._evaluate_against_ground_truth(fused_data, ground_truth_data, gyro_data, compass_data, method='ukf')
        
        # Visualize results if requested
        if visualize:
            self._visualize_fusion_results(fused_data, gyro_data, compass_data, ground_truth_data, method='ukf')
        
        # Save results
        fused_data.to_csv(os.path.join(self.output_dir, 'data', 'ukf_fused_heading.csv'), index=False)
        
        return fused_data
    
    def _generate_sigma_points(self, mean, covariance, lambd):
        """
        Generate sigma points for the Unscented Kalman Filter.
        
        Args:
            mean: State mean vector
            covariance: State covariance matrix
            lambd: UKF lambda parameter
            
        Returns:
            Array of sigma points
        """
        n = mean.shape[0]
        sigma_points = np.zeros((2 * n + 1, n))
        
        # Calculate matrix square root of covariance
        L = np.linalg.cholesky((n + lambd) * covariance)
        
        # Set first sigma point to the mean
        sigma_points[0] = mean
        
        # Set remaining sigma points
        for i in range(n):
            sigma_points[i + 1] = mean + L[i]
            sigma_points[n + i + 1] = mean - L[i]
        
        return sigma_points
    
    def _evaluate_against_ground_truth(self, 
                                      fused_data: pd.DataFrame, 
                                      ground_truth_data: pd.DataFrame,
                                      gyro_data: pd.DataFrame,
                                      compass_data: pd.DataFrame,
                                      method: str = 'ekf') -> Dict[str, float]:
        """
        Evaluate fusion results against ground truth data.
        
        Args:
            fused_data: DataFrame with fused heading data
            ground_truth_data: DataFrame with ground truth data
            gyro_data: DataFrame with gyroscope data
            compass_data: DataFrame with compass data
            method: Fusion method used (for logging purposes)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {method.upper()} fusion results against ground truth")
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Find the heading column in ground truth data
        heading_column = None
        if 'GroundTruthHeadingComputed' in ground_truth_data.columns:
            heading_column = 'GroundTruthHeadingComputed'
        else:
            # Check for interpolated heading column
            for col in ground_truth_data.columns:
                if 'heading' in col.lower():
                    heading_column = col
                    break
        
        # Ensure ground truth has the required column
        if heading_column is None:
            logger.warning("Ground truth data does not have heading information, skipping evaluation")
            return metrics
            
        logger.info(f"Using ground truth heading from column: {heading_column}")
        
        # Create a common timeline for evaluation
        # Merge fused data with ground truth based on closest timestamps
        merged_data = pd.DataFrame()
        merged_data['Timestamp_(ms)'] = fused_data['Timestamp_(ms)']
        merged_data['fused_heading'] = fused_data['fused_heading']
        
        # For each timestamp in fused data, find the closest ground truth timestamp
        gt_headings = []
        for timestamp in merged_data['Timestamp_(ms)']:
            gt_idx = np.argmin(np.abs(ground_truth_data['Timestamp_(ms)'].values - timestamp))
            gt_heading = ground_truth_data[heading_column].iloc[gt_idx]
            gt_headings.append(gt_heading)
        
        merged_data['ground_truth_heading'] = gt_headings
        
        # Calculate error metrics
        # We need to handle heading wrapping when calculating errors
        errors = []
        for i, row in merged_data.iterrows():
            error = row['fused_heading'] - row['ground_truth_heading']
            # Normalize error to [-180, 180]
            if error > 180:
                error -= 360
            elif error < -180:
                error += 360
            errors.append(error)
        
        merged_data['heading_error'] = errors
        merged_data['abs_heading_error'] = np.abs(errors)
        
        # Calculate metrics
        mae = merged_data['abs_heading_error'].mean()
        rmse = np.sqrt((merged_data['heading_error'] ** 2).mean())
        max_error = merged_data['abs_heading_error'].max()
        error_90th_percentile = np.percentile(merged_data['abs_heading_error'], 90)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'max_error': max_error,
            'error_90th_percentile': error_90th_percentile
        }
        
        # Log metrics
        logger.info(f"{method.upper()} Heading Estimation Metrics:")
        logger.info(f"  Mean Absolute Error: {mae:.2f} degrees")
        logger.info(f"  Root Mean Square Error: {rmse:.2f} degrees")
        logger.info(f"  Maximum Error: {max_error:.2f} degrees")
        logger.info(f"  90th Percentile Error: {error_90th_percentile:.2f} degrees")
        
        # Save evaluation results
        merged_data.to_csv(os.path.join(self.output_dir, 'data', f'{method}_evaluation.csv'), index=False)
        
        # Visualize error distribution
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        plt.hist(merged_data['heading_error'], bins=50, alpha=0.75, color='skyblue')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
        plt.axvline(x=mae, color='green', linestyle='-', linewidth=1, label=f'MAE: {mae:.2f}°')
        plt.axvline(x=-mae, color='green', linestyle='-', linewidth=1)
        plt.xlabel('Heading Error (degrees)')
        plt.ylabel('Frequency')
        plt.title(f'{method.upper()} Heading Estimation Error Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'plots', f'{method}_error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualize error over time
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        plt.plot(merged_data['Timestamp_(ms)'], merged_data['heading_error'], color='blue', linewidth=1)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
        plt.xlabel('Timestamp (ms)')
        plt.ylabel('Heading Error (degrees)')
        plt.title(f'{method.upper()} Heading Estimation Error Over Time')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'plots', f'{method}_error_over_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return metrics
    
    def lstm_fusion(self, 
                   gyro_data: pd.DataFrame, 
                   compass_data: pd.DataFrame,
                   ground_truth_data: Optional[pd.DataFrame] = None,
                   visualize: bool = True) -> pd.DataFrame:
        """
        Implement LSTM-based sensor fusion between gyroscope and compass data.
        
        The LSTM approach:
        - Uses sequences of sensor readings to learn temporal patterns
        - Automatically handles complex non-linearities
        - Can learn to ignore unreliable sensor readings
        
        Args:
            gyro_data: DataFrame with gyroscope data
            compass_data: DataFrame with compass data
            ground_truth_data: Optional DataFrame with ground truth data for evaluation
            visualize: Whether to generate visualizations
            
        Returns:
            DataFrame with fused heading data
        """
        logger.info("Applying LSTM-based sensor fusion")
        
        # Check if we have ground truth for training
        if ground_truth_data is None or len(ground_truth_data) == 0:
            logger.warning("LSTM fusion requires ground truth data for training, falling back to EKF")
            return self.extended_kalman_filter(gyro_data, compass_data, ground_truth_data, visualize)
        
        # Check for heading information in ground truth data
        heading_column = None
        if 'GroundTruthHeadingComputed' in ground_truth_data.columns:
            heading_column = 'GroundTruthHeadingComputed'
        else:
            # Check for interpolated heading column
            for col in ground_truth_data.columns:
                if 'heading' in col.lower():
                    heading_column = col
                    break
        
        if heading_column is None:
            logger.warning("Ground truth data does not have heading information, falling back to EKF")
            return self.extended_kalman_filter(gyro_data, compass_data, ground_truth_data, visualize)
            
        logger.info(f"Using ground truth heading from column: {heading_column}")
        
        # Ensure data is sorted by timestamp
        gyro_data = gyro_data.sort_values('Timestamp_(ms)').reset_index(drop=True)
        compass_data = compass_data.sort_values('Timestamp_(ms)').reset_index(drop=True)
        ground_truth_data = ground_truth_data.sort_values('Timestamp_(ms)').reset_index(drop=True)
        
        # Create a common timeline for all measurements
        all_timestamps = np.union1d(
            np.union1d(gyro_data['Timestamp_(ms)'].values, compass_data['Timestamp_(ms)'].values),
            ground_truth_data['Timestamp_(ms)'].values
        )
        all_timestamps = np.sort(all_timestamps)
        
        # Prepare merged dataset with interpolated values
        merged_data = pd.DataFrame()
        merged_data['Timestamp_(ms)'] = all_timestamps
        
        # Interpolate gyro data
        for col in gyro_data.columns:
            if col != 'Timestamp_(ms)':
                f = lambda x: np.interp(x, gyro_data['Timestamp_(ms)'].values, gyro_data[col].values)
                merged_data[f'gyro_{col}'] = f(all_timestamps)
        
        # Interpolate compass data
        for col in compass_data.columns:
            if col != 'Timestamp_(ms)':
                # Special handling for compass heading (circular interpolation)
                if col == 'compass':
                    # Convert to Cartesian coordinates for interpolation
                    compass_rad = np.radians(compass_data[col].values)
                    cos_vals = np.cos(compass_rad)
                    sin_vals = np.sin(compass_rad)
                    
                    # Interpolate cos and sin components
                    interp_cos = np.interp(all_timestamps, compass_data['Timestamp_(ms)'].values, cos_vals)
                    interp_sin = np.interp(all_timestamps, compass_data['Timestamp_(ms)'].values, sin_vals)
                    
                    # Convert back to degrees
                    interp_heading = np.degrees(np.arctan2(interp_sin, interp_cos)) % 360
                    merged_data[f'compass_{col}'] = interp_heading
                else:
                    f = lambda x: np.interp(x, compass_data['Timestamp_(ms)'].values, compass_data[col].values)
                    merged_data[f'compass_{col}'] = f(all_timestamps)
        
        # Interpolate ground truth data
        for col in ground_truth_data.columns:
            if col != 'Timestamp_(ms)':
                # Special handling for heading (circular interpolation)
                if col == heading_column:
                    # Convert to Cartesian coordinates for interpolation
                    heading_rad = np.radians(ground_truth_data[col].values)
                    cos_vals = np.cos(heading_rad)
                    sin_vals = np.sin(heading_rad)
                    
                    # Interpolate cos and sin components
                    interp_cos = np.interp(all_timestamps, ground_truth_data['Timestamp_(ms)'].values, cos_vals)
                    interp_sin = np.interp(all_timestamps, ground_truth_data['Timestamp_(ms)'].values, sin_vals)
                    
                    # Convert back to degrees
                    interp_heading = np.degrees(np.arctan2(interp_sin, interp_cos)) % 360
                    merged_data[f'gt_{col}'] = interp_heading
                else:
                    f = lambda x: np.interp(x, ground_truth_data['Timestamp_(ms)'].values, ground_truth_data[col].values)
                    merged_data[f'gt_{col}'] = f(all_timestamps)
        
        # Add target variable (ground truth heading in sin/cos form)
        merged_data['gt_sin'] = np.sin(np.radians(merged_data[f'gt_{heading_column}']))
        merged_data['gt_cos'] = np.cos(np.radians(merged_data[f'gt_{heading_column}']))
        
        # Calculate features
        # Derive gyro features like rate of change
        if 'gyro_gyroSumFromstart0' in merged_data.columns:
            # Calculate gyro rate (degrees per second)
            gyro_rates = []
            for i in range(1, len(merged_data)):
                dt = (merged_data['Timestamp_(ms)'].iloc[i] - merged_data['Timestamp_(ms)'].iloc[i-1]) / 1000.0
                if dt > 0:
                    rate = (merged_data['gyro_gyroSumFromstart0'].iloc[i] - merged_data['gyro_gyroSumFromstart0'].iloc[i-1]) / dt
                else:
                    rate = 0
                gyro_rates.append(rate)
            
            # Prepend a 0 for the first row
            gyro_rates.insert(0, 0)
            merged_data['gyro_rate'] = gyro_rates
        
        # Add some sinusoidal features to help the model with heading's circular nature
        merged_data['compass_sin'] = np.sin(np.radians(merged_data['compass_compass']))
        merged_data['compass_cos'] = np.cos(np.radians(merged_data['compass_compass']))
        
        # Prepare data for LSTM
        # Select features
        feature_cols = [
            'gyro_gyroSumFromstart0', 'gyro_rate',
            'compass_compass', 'compass_sin', 'compass_cos'
        ]
        
        # Add magnetic field magnitude if available
        if 'compass_Magnetic_Field_Magnitude' in merged_data.columns:
            feature_cols.append('compass_Magnetic_Field_Magnitude')
        
        # Add anomaly flags if available
        if 'compass_is_anomaly' in merged_data.columns:
            feature_cols.append('compass_is_anomaly')
        
        # Target columns: sine and cosine of ground truth heading
        target_cols = ['gt_sin', 'gt_cos']
        
        # Scale features
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(merged_data[feature_cols])
        
        # Reshape data for LSTM (samples, time steps, features)
        seq_length = 10  # Number of time steps to look back
        X, y = [], []
        
        for i in range(seq_length, len(merged_data)):
            X.append(scaled_features[i-seq_length:i])
            y.append(merged_data[target_cols].iloc[i].values)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data into train and test sets (80/20 split)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(seq_length, len(feature_cols))))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2))  # Output: sin and cos of heading
        
        # Compile model
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
        logger.info("Training LSTM model for sensor fusion")
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Save the model
        model.save(os.path.join(self.output_dir, 'models', 'lstm_fusion_model.keras'))
        
        # Predict on all data
        # First, create sequences for all data
        all_X = []
        for i in range(seq_length, len(merged_data)):
            all_X.append(scaled_features[i-seq_length:i])
        
        all_X = np.array(all_X)
        
        # Make predictions
        predictions = model.predict(all_X)
        
        # Convert predictions back to heading angles
        predicted_sin = predictions[:, 0]
        predicted_cos = predictions[:, 1]
        
        predicted_heading = np.degrees(np.arctan2(predicted_sin, predicted_cos)) % 360
        
        # Create a DataFrame with the results
        results_timestamps = merged_data['Timestamp_(ms)'].iloc[seq_length:].values
        
        fused_data = pd.DataFrame({
            'Timestamp_(ms)': results_timestamps,
            'fused_heading': predicted_heading,
            'heading_sin': predicted_sin,
            'heading_cos': predicted_cos
        })
        
        # Add estimated uncertainty based on model validation
        # Use the validation loss as a proxy for uncertainty
        val_loss = history.history['val_loss'][-1]
        fused_data['heading_variance'] = np.ones(len(fused_data)) * val_loss
        
        # Evaluate against ground truth
        if ground_truth_data is not None and len(ground_truth_data) > 0 and visualize:
            self._evaluate_against_ground_truth(fused_data, ground_truth_data, gyro_data, compass_data, method='lstm')
        
        # Visualize results if requested
        if visualize:
            self._visualize_fusion_results(fused_data, gyro_data, compass_data, ground_truth_data, method='lstm')
            
            # Visualize training history
            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Squared Error')
            plt.title('LSTM Training History')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, 'plots', 'lstm_training_history.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save results
        fused_data.to_csv(os.path.join(self.output_dir, 'data', 'lstm_fused_heading.csv'), index=False)
        
        return fused_data
    
    def adaptive_filtering(self, 
                         gyro_data: pd.DataFrame, 
                         compass_data: pd.DataFrame,
                         ground_truth_data: Optional[pd.DataFrame] = None,
                         visualize: bool = True) -> pd.DataFrame:
        """
        Implement adaptive filtering to handle different movement scenarios.
        
        Adaptive filtering:
        - Dynamically adjusts filter parameters based on movement patterns
        - Detects stationary vs. moving states
        - Changes weighting of sensors based on detected scenario
        
        Args:
            gyro_data: DataFrame with gyroscope data
            compass_data: DataFrame with compass data
            ground_truth_data: Optional DataFrame with ground truth data for evaluation
            visualize: Whether to generate visualizations
            
        Returns:
            DataFrame with fused heading data
        """
        logger.info("Applying adaptive filtering for sensor fusion")
        
        # Ensure data is sorted by timestamp
        gyro_data = gyro_data.sort_values('Timestamp_(ms)').reset_index(drop=True)
        compass_data = compass_data.sort_values('Timestamp_(ms)').reset_index(drop=True)
        
        # Create a common timeline for all measurements
        all_timestamps = np.union1d(gyro_data['Timestamp_(ms)'].values, 
                                   compass_data['Timestamp_(ms)'].values)
        
        # Initialize state and covariance
        # State = [heading, gyro_bias]
        state = np.zeros(2)
        
        # Initial heading from compass if available, otherwise use 0
        if len(compass_data) > 0:
            state[0] = compass_data['compass'].iloc[0]
        
        # Covariance matrix
        covariance = np.diag([np.radians(10)**2, np.radians(1)**2])  # Initial uncertainty
        
        # Storage for results
        results = []
        
        # Previous timestamp for delta time calculation
        prev_timestamp = all_timestamps[0]
        
        # Window for motion detection
        window_size = 10
        gyro_window = []
        
        # Process each timestamp
        for timestamp in all_timestamps:
            # Calculate time difference in seconds
            dt = (timestamp - prev_timestamp) / 1000.0  # Convert ms to seconds
            prev_timestamp = timestamp
            
            # Skip if dt is too large (gap in data)
            if dt > 1.0:  # More than 1 second gap
                continue
            
            # Find closest gyro measurement
            gyro_idx = np.argmin(np.abs(gyro_data['Timestamp_(ms)'].values - timestamp))
            gyro_reading = gyro_data['gyroSumFromstart0'].iloc[gyro_idx]
            gyro_timestamp = gyro_data['Timestamp_(ms)'].iloc[gyro_idx]
            
            # Maintain a window of gyro readings for motion state detection
            if abs(gyro_timestamp - timestamp) < 100:  # Within 100ms
                # Add to window
                if len(gyro_window) >= window_size:
                    gyro_window.pop(0)
                gyro_window.append(gyro_reading)
            
            # Detect motion state
            is_moving = False
            if len(gyro_window) > 0:
                # Calculate gyro variance in the window
                gyro_variance = np.var(gyro_window)
                # If variance is above threshold, we're moving
                is_moving = gyro_variance > 0.5  # Threshold can be tuned
            
            # Adjust process noise based on motion state
            if is_moving:
                # Higher process noise during movement
                q_heading = np.radians(0.5)**2  # Process noise for heading during movement
                q_bias = np.radians(0.05)**2    # Process noise for gyro bias during movement
            else:
                # Lower process noise when stationary
                q_heading = np.radians(0.05)**2  # Process noise for heading when stationary
                q_bias = np.radians(0.01)**2     # Process noise for gyro bias when stationary
            
            Q = np.diag([q_heading, q_bias])
            
            # Only use gyro if it's close enough to the current timestamp
            if abs(gyro_timestamp - timestamp) < 100:  # Within 100ms
                # Jacobian of state transition
                F = np.array([[1, -dt], [0, 1]])
                
                # Predict state: x = F * x + B * u
                # u is the gyro reading, B is the control matrix
                gyro_rate = np.radians(gyro_reading)  # Convert to radians
                state = F @ state + np.array([dt * gyro_rate, 0])
                
                # Normalize heading to [0, 2π]
                state[0] = state[0] % (2 * np.pi)
                
                # Update covariance: P = F * P * F^T + Q
                covariance = F @ covariance @ F.T + Q
            
            # Find closest compass measurement
            compass_idx = np.argmin(np.abs(compass_data['Timestamp_(ms)'].values - timestamp))
            compass_reading = compass_data['compass'].iloc[compass_idx]
            compass_timestamp = compass_data['Timestamp_(ms)'].iloc[compass_idx]
            
            # Adjust measurement noise based on motion state and magnetic field
            compass_noise_base = np.radians(5)**2
            
            # Check magnetic field magnitude for anomalies
            mag_field_ok = True
            if 'Magnetic_Field_Magnitude' in compass_data.columns:
                mag_magnitude = compass_data['Magnetic_Field_Magnitude'].iloc[compass_idx]
                
                # Skip update if magnetic field magnitude is too extreme
                if mag_magnitude < 20 or mag_magnitude > 100:  # Example thresholds
                    logger.debug(f"Skipping compass update at {timestamp} due to anomalous magnetic field: {mag_magnitude}")
                    mag_field_ok = False
            
            # Check if compass measurement is an anomaly (if flag exists)
            is_anomaly = False
            if 'is_anomaly' in compass_data.columns and compass_data['is_anomaly'].iloc[compass_idx]:
                logger.debug(f"Skipping compass update at {timestamp} due to flagged anomaly")
                is_anomaly = True
            
            # Only use compass if conditions are met
            if (abs(compass_timestamp - timestamp) < 100) and mag_field_ok and not is_anomaly:
                # Adjust compass noise based on motion
                if is_moving:
                    # Higher noise during movement (trust gyro more)
                    compass_noise = compass_noise_base * 2.0
                else:
                    # Lower noise when stationary (trust compass more)
                    compass_noise = compass_noise_base * 0.5
                
                compass_reading_rad = np.radians(compass_reading)
                
                # Measurement matrix
                H = np.array([1, 0]).reshape(1, 2)
                
                # Calculate innovation (difference between measurement and prediction)
                # Need to handle heading wraparound for correct error calculation
                predicted_heading = state[0]
                error = compass_reading_rad - predicted_heading
                
                # Normalize error to [-π, π]
                while error > np.pi:
                    error -= 2 * np.pi
                while error < -np.pi:
                    error += 2 * np.pi
                
                # Calculate Kalman gain
                S = H @ covariance @ H.T + compass_noise
                K = covariance @ H.T / S
                
                # Update state
                state = state + K.flatten() * error
                
                # Normalize heading to [0, 2π]
                state[0] = state[0] % (2 * np.pi)
                
                # Update covariance
                covariance = (np.eye(2) - K @ H) @ covariance
            
            # Store result
            heading_deg = np.degrees(state[0])
            gyro_bias_deg = np.degrees(state[1])
            
            results.append({
                'Timestamp_(ms)': timestamp,
                'fused_heading': heading_deg,
                'gyro_bias': gyro_bias_deg,
                'heading_variance': covariance[0, 0],
                'is_moving': is_moving
            })
        
        # Create DataFrame from results
        fused_data = pd.DataFrame(results)
        
        # Evaluate against ground truth if available
        if ground_truth_data is not None and len(ground_truth_data) > 0 and visualize:
            self._evaluate_against_ground_truth(fused_data, ground_truth_data, gyro_data, compass_data, method='adaptive')
        
        # Visualize results if requested
        if visualize:
            self._visualize_fusion_results(fused_data, gyro_data, compass_data, ground_truth_data, method='adaptive')
            
            # Visualize motion state
            fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
            plt.scatter(fused_data['Timestamp_(ms)'], fused_data['is_moving'], c=fused_data['is_moving'], cmap='coolwarm', s=10)
            plt.xlabel('Timestamp (ms)')
            plt.ylabel('Motion State')
            plt.title('Detected Motion States (0: Stationary, 1: Moving)')
            plt.grid(True, alpha=0.3)
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, 'plots', 'adaptive_motion_states.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save results
        fused_data.to_csv(os.path.join(self.output_dir, 'data', 'adaptive_fused_heading.csv'), index=False)
        
        return fused_data
    
    def _visualize_fusion_results(self, 
                                 fused_data: pd.DataFrame, 
                                 gyro_data: pd.DataFrame, 
                                 compass_data: pd.DataFrame,
                                 ground_truth_data: Optional[pd.DataFrame] = None,
                                 method: str = 'ekf') -> None:
        """
        Visualize fusion results.
        
        Args:
            fused_data: DataFrame with fused heading data
            gyro_data: DataFrame with gyroscope data
            compass_data: DataFrame with compass data
            ground_truth_data: Optional DataFrame with ground truth data
            method: Fusion method used (for visualization titles and filenames)
        """
        logger.info(f"Visualizing {method.upper()} fusion results")
        
        # Create figure for IEEE column width
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        
        # Plot fused heading
        plt.plot(fused_data['Timestamp_(ms)'], fused_data['fused_heading'], 
                 color='red', linewidth=1.5, label=f'{method.upper()} Fused Heading')
        
        # Plot gyro-based heading
        if 'gyroSumFromstart0' in gyro_data.columns:
            # Get the first ground truth value if available to start gyro from
            first_gt = 0
            if ground_truth_data is not None and len(ground_truth_data) > 0 and 'GroundTruthHeadingComputed' in ground_truth_data.columns:
                first_gt = ground_truth_data['GroundTruthHeadingComputed'].iloc[0]
            
            plt.plot(gyro_data['Timestamp_(ms)'], 
                     gyro_data['gyroSumFromstart0'].values - gyro_data['gyroSumFromstart0'].iloc[0] + first_gt, 
                     color='blue', linewidth=1, label='Gyro Heading')
        
        # Plot compass heading
        if 'compass' in compass_data.columns:
            plt.plot(compass_data['Timestamp_(ms)'], compass_data['compass'], 
                     color='green', linewidth=1, alpha=0.7, label='Compass Heading')
        
        # Plot ground truth if available
        if ground_truth_data is not None and len(ground_truth_data) > 0 and 'GroundTruthHeadingComputed' in ground_truth_data.columns:
            plt.plot(ground_truth_data['Timestamp_(ms)'], ground_truth_data['GroundTruthHeadingComputed'], 
                     color='black', linewidth=1.5, linestyle='--', label='Ground Truth')
        
        plt.xlabel('Timestamp (ms)')
        plt.ylabel('Heading (degrees)')
        plt.title(f'Sensor Fusion Results: {method.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'plots', f'{method}_fusion_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot gyro bias if available
        if 'gyro_bias' in fused_data.columns:
            fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
            plt.plot(fused_data['Timestamp_(ms)'], fused_data['gyro_bias'], color='purple', linewidth=1.5)
            plt.xlabel('Timestamp (ms)')
            plt.ylabel('Gyro Bias (degrees/s)')
            plt.title(f'Estimated Gyroscope Bias: {method.upper()}')
            plt.grid(True, alpha=0.3)
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, 'plots', f'{method}_gyro_bias.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot heading variance
        if 'heading_variance' in fused_data.columns:
            fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
            plt.plot(fused_data['Timestamp_(ms)'], fused_data['heading_variance'], color='orange', linewidth=1.5)
            plt.xlabel('Timestamp (ms)')
            plt.ylabel('Heading Variance')
            plt.title(f'Heading Estimation Uncertainty: {method.upper()}')
            plt.grid(True, alpha=0.3)
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, 'plots', f'{method}_heading_variance.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def context_aware_fusion(self, 
                            gyro_data: pd.DataFrame, 
                            compass_data: pd.DataFrame,
                            ground_truth_data: Optional[pd.DataFrame] = None,
                            visualize: bool = True) -> pd.DataFrame:
        """
        Implement context-aware models that leverage environmental information.
        
        The context-aware approach:
        - Uses available environmental data to adjust sensor fusion parameters
        - Detects and adapts to different environments (indoor/outdoor, etc.)
        - Can handle different magnetic disturbance scenarios
        
        Args:
            gyro_data: DataFrame with gyroscope data
            compass_data: DataFrame with compass data
            ground_truth_data: Optional DataFrame with ground truth data for evaluation
            visualize: Whether to generate visualizations
            
        Returns:
            DataFrame with fused heading data
        """
        logger.info("Applying context-aware fusion")
        
        # Ensure data is sorted by timestamp
        gyro_data = gyro_data.sort_values('Timestamp_(ms)').reset_index(drop=True)
        compass_data = compass_data.sort_values('Timestamp_(ms)').reset_index(drop=True)
        
        # Create a common timeline for all measurements
        all_timestamps = np.union1d(gyro_data['Timestamp_(ms)'].values, 
                                   compass_data['Timestamp_(ms)'].values)
        
        # Initialize state and covariance
        # State = [heading, gyro_bias]
        state = np.zeros(2)
        
        # Initial heading from compass if available, otherwise use 0
        if len(compass_data) > 0:
            state[0] = compass_data['compass'].iloc[0]
        
        # Covariance matrix
        covariance = np.diag([np.radians(10)**2, np.radians(1)**2])  # Initial uncertainty
        
        # Storage for results
        results = []
        
        # Previous timestamp for delta time calculation
        prev_timestamp = all_timestamps[0]
        
        # Window for environment detection
        window_size = 20
        compass_window = []
        gyro_window = []
        
        # Context parameters
        environment_type = "unknown"  # Can be "stable", "unstable", or "unknown"
        
        # Process each timestamp
        for timestamp in all_timestamps:
            # Calculate time difference in seconds
            dt = (timestamp - prev_timestamp) / 1000.0  # Convert ms to seconds
            prev_timestamp = timestamp
            
            # Skip if dt is too large (gap in data)
            if dt > 1.0:  # More than 1 second gap
                continue
            
            # --- Get sensor readings for this timestamp ---
            
            # Find closest gyro measurement
            gyro_idx = np.argmin(np.abs(gyro_data['Timestamp_(ms)'].values - timestamp))
            gyro_reading = gyro_data['gyroSumFromstart0'].iloc[gyro_idx]
            gyro_timestamp = gyro_data['Timestamp_(ms)'].iloc[gyro_idx]
            
            # Find closest compass measurement
            compass_idx = np.argmin(np.abs(compass_data['Timestamp_(ms)'].values - timestamp))
            compass_reading = compass_data['compass'].iloc[compass_idx]
            compass_timestamp = compass_data['Timestamp_(ms)'].iloc[compass_idx]
            
            # Update sensor windows for context detection
            if abs(gyro_timestamp - timestamp) < 100:
                if len(gyro_window) >= window_size:
                    gyro_window.pop(0)
                gyro_window.append(gyro_reading)
                
            if abs(compass_timestamp - timestamp) < 100:
                if len(compass_window) >= window_size:
                    compass_window.pop(0)
                compass_window.append(compass_reading)
            
            # --- Detect environment context based on sensor characteristics ---
            
            # Check if we have enough data for context detection
            if len(compass_window) > window_size/2 and len(gyro_window) > window_size/2:
                # Calculate compass variance
                compass_variance = np.var(compass_window)
                
                # Calculate gyro variance
                gyro_variance = np.var(gyro_window)
                
                # Check for magnetic field magnitude if available
                mag_field_stable = True
                if 'Magnetic_Field_Magnitude' in compass_data.columns:
                    mag_window = []
                    
                    # Get magnetic field values for the window
                    for i in range(max(0, compass_idx - window_size//2), min(len(compass_data), compass_idx + window_size//2)):
                        if 'Magnetic_Field_Magnitude' in compass_data.columns:
                            mag_window.append(compass_data['Magnetic_Field_Magnitude'].iloc[i])
                    
                    if len(mag_window) > 0:
                        # Calculate variance of magnetic field
                        mag_variance = np.var(mag_window)
                        mag_mean = np.mean(mag_window)
                        
                        # If magnetic field is unstable, mark it
                        mag_field_stable = mag_variance < 10 and 20 < mag_mean < 80
                
                # Determine environment type
                if mag_field_stable and compass_variance < 10:
                    # Low magnetic variance indicates stable environment
                    environment_type = "stable"
                elif not mag_field_stable or compass_variance > 50:
                    # High magnetic variance indicates unstable environment
                    environment_type = "unstable"
                else:  # moderate or unknown
                    environment_type = "moderate"
            
            # --- Adjust filter parameters based on context ---
            
            # Set process noise based on environment
            if environment_type == "stable":
                q_heading = np.radians(0.05)**2  # Lower process noise in stable environment
                q_bias = np.radians(0.005)**2
                compass_noise = np.radians(2)**2  # Trust compass more in stable environment
            elif environment_type == "unstable":
                q_heading = np.radians(0.2)**2  # Higher process noise in unstable environment
                q_bias = np.radians(0.02)**2
                compass_noise = np.radians(20)**2  # Trust compass less in unstable environment
            else:  # moderate or unknown
                q_heading = np.radians(0.1)**2  # Default values
                q_bias = np.radians(0.01)**2
                compass_noise = np.radians(5)**2
            
            Q = np.diag([q_heading, q_bias])
            
            # --- Prediction step ---
            
            # Only use gyro if it's close enough to the current timestamp
            if abs(gyro_timestamp - timestamp) < 100:  # Within 100ms
                # Jacobian of state transition
                F = np.array([[1, -dt], [0, 1]])
                
                # Predict state: x = F * x + B * u
                # u is the gyro reading, B is the control matrix
                gyro_rate = np.radians(gyro_reading)  # Convert to radians
                state = F @ state + np.array([dt * gyro_rate, 0])
                
                # Normalize heading to [0, 2π]
                state[0] = state[0] % (2 * np.pi)
                
                # Update covariance: P = F * P * F^T + Q
                covariance = F @ covariance @ F.T + Q
            
            # --- Update step (if compass data is available) ---
            
            # Check conditions for using compass update
            use_compass = True
            compass_reliability = 1.0  # 1.0 = fully reliable, 0.0 = completely unreliable
            
            # Check if compass timestamp is close enough
            if abs(compass_timestamp - timestamp) > 100:  # Not within 100ms
                use_compass = False
            
            # Check for magnetic anomalies if available
            if 'Magnetic_Field_Magnitude' in compass_data.columns:
                mag_magnitude = compass_data['Magnetic_Field_Magnitude'].iloc[compass_idx]
                
                # Skip update if magnetic field magnitude is too extreme
                if mag_magnitude < 20 or mag_magnitude > 100:
                    logger.debug(f"Reducing compass reliability at {timestamp} due to anomalous magnetic field: {mag_magnitude}")
                    compass_reliability *= 0.5
                
                # Adjust compass reliability based on magnetic field
                if 30 <= mag_magnitude <= 70:  # Ideal range
                    pass  # Keep reliability as is
                elif 20 <= mag_magnitude < 30 or 70 < mag_magnitude <= 80:
                    compass_reliability *= 0.8  # Slightly reduce reliability
                elif mag_magnitude < 20 or mag_magnitude > 80:
                    compass_reliability *= 0.5  # Significantly reduce reliability
            
            # Check if compass measurement is an anomaly (if flag exists)
            if 'is_anomaly' in compass_data.columns and compass_data['is_anomaly'].iloc[compass_idx]:
                logger.debug(f"Skipping compass update at {timestamp} due to flagged anomaly")
                use_compass = False
            
            # Adjust measurement noise based on compass reliability
            adjusted_compass_noise = compass_noise / compass_reliability
            
            # Only perform update if compass should be used
            if use_compass:
                compass_reading_rad = np.radians(compass_reading)
                
                # Measurement matrix
                H = np.array([1, 0]).reshape(1, 2)
                
                # Calculate innovation (difference between measurement and prediction)
                # Need to handle heading wraparound for correct error calculation
                predicted_heading = state[0]
                error = compass_reading_rad - predicted_heading
                
                # Normalize error to [-π, π]
                while error > np.pi:
                    error -= 2 * np.pi
                while error < -np.pi:
                    error += 2 * np.pi
                
                # Calculate Kalman gain
                S = H @ covariance @ H.T + adjusted_compass_noise
                K = covariance @ H.T / S
                
                # Update state
                state = state + K.flatten() * error
                
                # Normalize heading to [0, 2π]
                state[0] = state[0] % (2 * np.pi)
                
                # Update covariance
                covariance = (np.eye(2) - K @ H) @ covariance
            
            # Store result
            heading_deg = np.degrees(state[0])
            gyro_bias_deg = np.degrees(state[1])
            
            results.append({
                'Timestamp_(ms)': timestamp,
                'fused_heading': heading_deg,
                'gyro_bias': gyro_bias_deg,
                'heading_variance': covariance[0, 0],
                'environment_type': environment_type,
                'compass_reliability': compass_reliability if use_compass else 0.0
            })
        
        # Create DataFrame from results
        fused_data = pd.DataFrame(results)
        
        # Evaluate against ground truth if available
        if ground_truth_data is not None and len(ground_truth_data) > 0 and visualize:
            self._evaluate_against_ground_truth(fused_data, ground_truth_data, gyro_data, compass_data, method='context')
        
        # Visualize results if requested
        if visualize:
            self._visualize_fusion_results(fused_data, gyro_data, compass_data, ground_truth_data, method='context')
            
            # Visualize environment context
            fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
            env_type_numeric = fused_data['environment_type'].map({'stable': 0, 'moderate': 1, 'unstable': 2, 'unknown': 3})
            plt.scatter(fused_data['Timestamp_(ms)'], env_type_numeric, c=env_type_numeric, cmap='viridis', s=10)
            plt.yticks([0, 1, 2, 3], ['Stable', 'Moderate', 'Unstable', 'Unknown'])
            plt.xlabel('Timestamp (ms)')
            plt.ylabel('Environment Type')
            plt.title('Detected Environment Types')
            plt.grid(True, alpha=0.3)
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, 'plots', 'context_environment_types.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Visualize compass reliability
            fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
            plt.plot(fused_data['Timestamp_(ms)'], fused_data['compass_reliability'], color='blue', linewidth=1)
            plt.xlabel('Timestamp (ms)')
            plt.ylabel('Compass Reliability')
            plt.title('Estimated Compass Reliability')
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3)
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, 'plots', 'context_compass_reliability.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save results
        fused_data.to_csv(os.path.join(self.output_dir, 'data', 'context_fused_heading.csv'), index=False)
        
        return fused_data
        
    def benchmark_fusion_methods(self, 
                                gyro_data: pd.DataFrame, 
                                compass_data: pd.DataFrame,
                                ground_truth_data: Optional[pd.DataFrame] = None,
                                methods: List[str] = ['ekf', 'ukf', 'lstm', 'adaptive', 'context'],
                                visualize: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Benchmark multiple fusion methods and compare their performance.
        
        Args:
            gyro_data: DataFrame with gyroscope data
            compass_data: DataFrame with compass data
            ground_truth_data: Optional DataFrame with ground truth data for evaluation
            methods: List of fusion methods to benchmark
            visualize: Whether to generate visualizations
            
        Returns:
            Dictionary of evaluation metrics for each method
        """
        logger.info(f"Benchmarking fusion methods: {methods}")
        
        # Check if ground truth is available
        if ground_truth_data is None or len(ground_truth_data) == 0:
            logger.warning("Benchmarking requires ground truth data, skipping")
            return {}
        
        # Ensure ground truth has the required column
        if 'GroundTruthHeadingComputed' not in ground_truth_data.columns:
            logger.warning("Ground truth data does not have heading information, skipping benchmarking")
            return {}
        
        # Dictionary to store results
        benchmark_results = {}
        
        # Run each method
        for method in methods:
            logger.info(f"Running {method.upper()} method for benchmark")
            
            # Apply fusion method
            fused_data = self.fuse_sensors(
                gyro_data,
                compass_data,
                method=method,
                ground_truth_data=ground_truth_data,
                visualize=visualize
            )
            
            # Evaluate against ground truth
            metrics = self._evaluate_against_ground_truth(
                fused_data, 
                ground_truth_data, 
                gyro_data, 
                compass_data, 
                method=method
            )
            
            benchmark_results[method] = metrics
        
        # Compare methods visually if requested
        if visualize and len(benchmark_results) > 0:
            self._visualize_benchmark_results(benchmark_results)
        
        return benchmark_results
    
    def _visualize_benchmark_results(self, benchmark_results: Dict[str, Dict[str, float]]) -> None:
        """
        Visualize benchmark results for different fusion methods.
        
        Args:
            benchmark_results: Dictionary of evaluation metrics for each method
        """
        logger.info("Visualizing benchmark results")
        
        # Extract methods and metrics
        methods = list(benchmark_results.keys())
        metrics = ['mae', 'rmse', 'max_error', 'error_90th_percentile']
        
        # Ensure all methods have all metrics
        valid_methods = []
        for method in methods:
            if all(metric in benchmark_results[method] for metric in metrics):
                valid_methods.append(method)
            else:
                logger.warning(f"Method {method} is missing metrics, excluding from visualization")
        
        # Skip if no valid methods
        if len(valid_methods) == 0:
            logger.warning("No valid methods to visualize")
            return
        
        # Create bar plots for each metric
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            
            # Extract metric values for each method
            values = [benchmark_results[method][metric] for method in valid_methods]
            
            # Define colors for methods
            colors = ['blue', 'green', 'red', 'purple', 'orange'][:len(valid_methods)]
            
            # Create bar plot
            bars = plt.bar(valid_methods, values, color=colors)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom')
            
            # Add labels and title
            plt.xlabel('Fusion Method')
            plt.ylabel(f'{metric.upper()} (degrees)')
            plt.title(f'Comparison of {metric.upper()} Across Fusion Methods')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, 'plots', f'benchmark_{metric}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create a combined metrics plot
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        
        # Width of a bar
        bar_width = 0.2
        
        # Position of bars on x-axis
        r = np.arange(len(valid_methods))
        
        # Create bars for each metric
        for i, metric in enumerate(metrics):
            values = [benchmark_results[method][metric] for method in valid_methods]
            position = [x + bar_width * i for x in r]
            plt.bar(position, values, width=bar_width, label=metric.upper())
        
        # Add labels and title
        plt.xlabel('Fusion Method')
        plt.ylabel('Error (degrees)')
        plt.title('Comparison of All Metrics Across Fusion Methods')
        plt.xticks([r + bar_width * 1.5 for r in range(len(valid_methods))], valid_methods)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'plots', 'benchmark_combined.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a summary table
        summary_data = []
        for method in valid_methods:
            row = [method]
            for metric in metrics:
                row.append(f"{benchmark_results[method][metric]:.2f}")
            summary_data.append(row)
        
        # Save summary table as text file
        with open(os.path.join(self.output_dir, 'data', 'benchmark_summary.txt'), 'w') as f:
            f.write("Method\t" + "\t".join(metrics) + "\n")
            for row in summary_data:
                f.write("\t".join(str(x) for x in row) + "\n") 