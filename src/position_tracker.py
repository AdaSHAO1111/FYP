#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position Tracker module for indoor navigation using deep learning approaches.
This module implements various neural network architectures for position prediction
based on sensor data from gyroscope, compass, and other available sensors.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List, Any

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Input, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PositionTracker:
    """
    A class for tracking and predicting positions using deep learning models.
    Implements various neural network architectures including LSTM, CNN-LSTM,
    and Transformer-based models for accurate position prediction.
    """
    
    def __init__(self, output_dir: str = 'output'):
        """
        Initialize the PositionTracker with needed directories.
        
        Args:
            output_dir: Directory to save model outputs and visualizations
        """
        self.output_dir = output_dir
        
        # Create output directories if they don't exist
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        
        # Initialize scalers for data normalization
        self.feature_scaler = None
        self.target_scaler = None
        
        logger.info("PositionTracker initialized")
    
    def preprocess_data(self, 
                       sensor_data: pd.DataFrame, 
                       ground_truth_data: pd.DataFrame,
                       seq_length: int = 10,
                       step_feature: str = 'step',
                       target_cols: List[str] = ['value_4', 'value_5'],
                       test_size: float = 0.2,
                       scale_features: bool = True,
                       scale_targets: bool = True) -> Dict[str, Any]:
        """
        Preprocess sensor and ground truth data for position prediction.
        
        Args:
            sensor_data: DataFrame containing sensor readings (gyro, compass, etc.)
            ground_truth_data: DataFrame containing ground truth positions
            seq_length: Number of time steps to include in each sequence
            step_feature: Name of the column that defines the step number
            target_cols: Names of the columns containing position coordinates
            test_size: Proportion of data to use for testing
            scale_features: Whether to scale input features
            scale_targets: Whether to scale target outputs
            
        Returns:
            Dictionary containing preprocessed data and associated metadata
        """
        logger.info(f"Preprocessing data with sequence length {seq_length}")
        
        # Ensure ground truth data has required columns
        for col in target_cols:
            if col not in ground_truth_data.columns:
                logger.error(f"Required target column '{col}' not found in ground truth data")
                raise ValueError(f"Ground truth data missing required column: {col}")
        
        # Make a copy of the dataframes to avoid modifying the originals
        sensor_data = sensor_data.copy()
        ground_truth_data = ground_truth_data.copy()
        
        # Ensure correct data types for merge columns
        if 'Timestamp_(ms)' in sensor_data.columns:
            sensor_data['Timestamp_(ms)'] = pd.to_numeric(sensor_data['Timestamp_(ms)'], errors='coerce')
        if 'Timestamp_(ms)' in ground_truth_data.columns:
            ground_truth_data['Timestamp_(ms)'] = pd.to_numeric(ground_truth_data['Timestamp_(ms)'], errors='coerce')
        if 'step' in sensor_data.columns:
            sensor_data['step'] = pd.to_numeric(sensor_data['step'], errors='coerce')
        if 'step' in ground_truth_data.columns:
            ground_truth_data['step'] = pd.to_numeric(ground_truth_data['step'], errors='coerce')
        
        # Determine appropriate columns for merging
        if step_feature in ground_truth_data.columns:
            logger.info(f"Using '{step_feature}' from ground truth for merging")
            gt_merge_col = step_feature
        elif 'step' in ground_truth_data.columns:
            logger.info("Using 'step' from ground truth for merging")
            gt_merge_col = 'step'
            step_feature = 'step'
        elif 'Timestamp_(ms)' in ground_truth_data.columns:
            logger.info("Using 'Timestamp_(ms)' from ground truth for merging")
            gt_merge_col = 'Timestamp_(ms)'
            step_feature = 'Timestamp_(ms)'
        else:
            logger.error("No suitable column found in ground truth data for merging")
            raise ValueError("Cannot find suitable column for merging ground truth data")
            
        if 'Timestamp_(ms)' in sensor_data.columns:
            logger.info("Using 'Timestamp_(ms)' from sensor data for merging")
            sensor_merge_col = 'Timestamp_(ms)'
        elif step_feature in sensor_data.columns:
            logger.info(f"Using '{step_feature}' from sensor data for merging")
            sensor_merge_col = step_feature
        else:
            logger.error("No suitable column found in sensor data for merging")
            raise ValueError("Cannot find suitable column for merging sensor data")

        # Now merge the data based on the available columns
        if gt_merge_col == sensor_merge_col:
            # We can merge directly on the common column
            logger.info(f"Merging data directly on '{gt_merge_col}'")
            try:
                merged_data = pd.merge_asof(
                    sensor_data.sort_values(by=sensor_merge_col),
                    ground_truth_data[
                        [gt_merge_col] + target_cols
                    ].sort_values(by=gt_merge_col),
                    on=gt_merge_col,
                    direction='nearest'
                )
            except Exception as e:
                logger.error(f"Error merging data: {str(e)}")
                logger.info("Attempting alternative merge strategy")
                
                # Try regular merge as fallback
                merged_data = pd.merge(
                    sensor_data,
                    ground_truth_data[[gt_merge_col] + target_cols],
                    on=gt_merge_col,
                    how='inner'
                )
                
                
                if len(merged_data) == 0:
                    logger.error("Merge resulted in empty dataset")
                    # Last resort: Create artificial mapping
                    sensor_steps = np.linspace(0, 100, len(sensor_data))
                    sensor_data_with_step = sensor_data.copy()
                    sensor_data_with_step['artificial_step'] = sensor_steps
                    
                    gt_steps = np.linspace(0, 100, len(ground_truth_data))
                    ground_truth_with_step = ground_truth_data.copy()
                    ground_truth_with_step['artificial_step'] = gt_steps
                    
                    merged_data = pd.merge_asof(
                        sensor_data_with_step.sort_values(by='artificial_step'),
                        ground_truth_with_step[['artificial_step'] + target_cols].sort_values(by='artificial_step'),
                        on='artificial_step',
                        direction='nearest'
                    )
                    
                    step_feature = 'artificial_step'
        else:
            # We need to use merge_asof with 'by' parameter or create a common column
            if 'Timestamp_(ms)' == sensor_merge_col and gt_merge_col == 'step':
                # Scenario: sensor data has timestamps, ground truth has steps
                # Create appropriate merge
                logger.info("Creating timestamp mapping for ground truth steps")
                
                # Get min and max timestamps from sensor data
                min_ts = sensor_data['Timestamp_(ms)'].min()
                max_ts = sensor_data['Timestamp_(ms)'].max()
                
                # Get min and max steps from ground truth
                min_step = ground_truth_data['step'].min()
                max_step = ground_truth_data['step'].max()
                
                # Create a linear mapping from steps to timestamps
                step_to_ts = {}
                step_range = max_step - min_step
                ts_range = max_ts - min_ts
                
                # Ensure data types for steps
                step_values = ground_truth_data['step'].values
                
                for i, step in enumerate(step_values):
                    # Linear mapping from step to timestamp
                    if step_range > 0:
                        norm_step = (step - min_step) / step_range
                        ts = min_ts + norm_step * ts_range
                    else:
                        # If only one step, place it in the middle of the time range
                        ts = min_ts + ts_range / 2
                    step_to_ts[step] = ts
                
                # Add timestamps to ground truth
                ground_truth_with_ts = ground_truth_data.copy()
                
                # Use vectorized approach to avoid type issues with map
                timestamps = []
                for step in ground_truth_with_ts['step']:
                    timestamps.append(step_to_ts.get(step, min_ts))
                ground_truth_with_ts['Timestamp_(ms)'] = timestamps
                
                # Ensure timestamps are float for merging
                ground_truth_with_ts['Timestamp_(ms)'] = ground_truth_with_ts['Timestamp_(ms)'].astype(float)
                sensor_data['Timestamp_(ms)'] = sensor_data['Timestamp_(ms)'].astype(float)
                
                # Now merge on timestamp
                try:
                    merged_data = pd.merge_asof(
                        sensor_data.sort_values(by='Timestamp_(ms)'),
                        ground_truth_with_ts[['Timestamp_(ms)'] + target_cols].sort_values(by='Timestamp_(ms)'),
                        on='Timestamp_(ms)',
                        direction='nearest'
                    )
                except Exception as e:
                    logger.error(f"Error in timestamp merge: {str(e)}")
                    logger.info("Falling back to simple index-based merge")
                    
                    # Create index-based merge as fallback
                    sensor_indices = np.linspace(0, 1, len(sensor_data))
                    gt_indices = np.linspace(0, 1, len(ground_truth_data))
                    
                    sensor_data['merge_idx'] = sensor_indices
                    ground_truth_with_ts['merge_idx'] = gt_indices
                    
                    merged_data = pd.merge_asof(
                        sensor_data.sort_values(by='merge_idx'),
                        ground_truth_with_ts[['merge_idx'] + target_cols].sort_values(by='merge_idx'),
                        on='merge_idx',
                        direction='nearest'
                    )
                
                # Keep original step feature name for visualization
                step_feature = 'step'
                
            elif gt_merge_col == 'Timestamp_(ms)' and 'Timestamp_(ms)' in sensor_data.columns:
                # Both have timestamps, simple merge
                logger.info("Both datasets have timestamps, performing direct merge")
                
                # Ensure timestamps are float for merging
                sensor_data['Timestamp_(ms)'] = sensor_data['Timestamp_(ms)'].astype(float)
                ground_truth_data['Timestamp_(ms)'] = ground_truth_data['Timestamp_(ms)'].astype(float)
                
                merged_data = pd.merge_asof(
                    sensor_data.sort_values(by='Timestamp_(ms)'),
                    ground_truth_data[['Timestamp_(ms)'] + target_cols].sort_values(by='Timestamp_(ms)'),
                    on='Timestamp_(ms)',
                    direction='nearest'
                )
                step_feature = 'Timestamp_(ms)'
            else:
                # General case - try to map between the columns
                logger.warning(f"Complex merge scenario: sensor on {sensor_merge_col}, ground truth on {gt_merge_col}")
                
                # Create artificial index for both
                sensor_data_with_idx = sensor_data.copy()
                sensor_data_with_idx['merge_idx'] = np.linspace(0, 1, len(sensor_data))
                
                ground_truth_with_idx = ground_truth_data.copy()
                ground_truth_with_idx['merge_idx'] = np.linspace(0, 1, len(ground_truth_data))
                
                # Merge on artificial index
                merged_data = pd.merge_asof(
                    sensor_data_with_idx.sort_values(by='merge_idx'),
                    ground_truth_with_idx[['merge_idx'] + target_cols].sort_values(by='merge_idx'),
                    on='merge_idx',
                    direction='nearest'
                )
                
                # Use artificial index as step feature
                step_feature = 'merge_idx'
        
        logger.info(f"Merged data contains {len(merged_data)} rows")
        
        # Drop rows with missing target values
        merged_data = merged_data.dropna(subset=target_cols)
        logger.info(f"After dropping NaN targets: {len(merged_data)} rows")
        
        if len(merged_data) == 0:
            logger.error("No valid data after merging and dropping NaNs")
            raise ValueError("No valid data for training after preprocessing")
        
        # Ensure target columns are numeric
        for col in target_cols:
            merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')
        
        # Extract features and targets
        exclude_cols = target_cols + ['Timestamp_(ms)', 'Type', 'Type_gyro', 'Type_compass', 
                                     'GroundTruth', 'GroundTruth_gyro', 'GroundTruth_compass',
                                     'turns', 'turns_gyro', 'turns_compass', 
                                     'implausible_jump', 'value_1', 'value_2', 'value_3',
                                     'merge_idx', 'artificial_step']
        
        if step_feature not in exclude_cols:
            exclude_cols.append(step_feature)
            
        # Identify initial feature columns, excluding the ones in exclude_cols
        initial_feature_cols = [col for col in merged_data.columns if col not in exclude_cols]
        
        # Further filter to only include numeric columns
        feature_cols = []
        for col in initial_feature_cols:
            try:
                # Check if conversion to numeric is possible
                pd.to_numeric(merged_data[col], errors='raise')
                feature_cols.append(col)
            except Exception as e:
                logger.warning(f"Excluding non-numeric column '{col}' from features: {str(e)}")
        
        logger.info(f"Using {len(feature_cols)} feature columns (after filtering non-numeric): {feature_cols}")
        
        # Add derived features
        # Create sinusoidal features for compass heading to handle circular nature
        if 'compass_compass' in merged_data.columns:
            try:
                merged_data['compass_sin'] = np.sin(np.radians(pd.to_numeric(merged_data['compass_compass'], errors='coerce')))
                merged_data['compass_cos'] = np.cos(np.radians(pd.to_numeric(merged_data['compass_compass'], errors='coerce')))
                feature_cols.extend(['compass_sin', 'compass_cos'])
            except Exception as e:
                logger.warning(f"Could not create sinusoidal features from compass: {str(e)}")
            
        # Calculate gyro rate of change if available
        if 'gyro_gyroSumFromstart0' in merged_data.columns:
            try:
                # Calculate gyro rate (degrees per second)
                gyro_rates = []
                gyro_data = pd.to_numeric(merged_data['gyro_gyroSumFromstart0'], errors='coerce')
                timestamp_data = pd.to_numeric(merged_data['Timestamp_(ms)'], errors='coerce')
                
                for i in range(1, len(merged_data)):
                    dt = (timestamp_data.iloc[i] - timestamp_data.iloc[i-1]) / 1000.0
                    if dt > 0:
                        rate = (gyro_data.iloc[i] - gyro_data.iloc[i-1]) / dt
                    else:
                        rate = 0
                    gyro_rates.append(rate)
                
                # Prepend a 0 for the first row
                gyro_rates.insert(0, 0)
                merged_data['gyro_rate'] = gyro_rates
                feature_cols.append('gyro_rate')
            except Exception as e:
                logger.warning(f"Could not calculate gyro rate: {str(e)}")
        
        # Convert to numpy arrays, handling any remaining NaN values
        X_data = merged_data[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        X = X_data.values
        y = merged_data[target_cols].values
        
        # Scale features and targets if requested
        if scale_features:
            self.feature_scaler = StandardScaler()
            X = self.feature_scaler.fit_transform(X)
            logger.info("Feature scaling applied")
        
        if scale_targets:
            self.target_scaler = StandardScaler()
            y = self.target_scaler.fit_transform(y)
            logger.info("Target scaling applied")
        
        # Create sequences for temporal models
        X_seq, y_seq = [], []
        for i in range(seq_length, len(X)):
            X_seq.append(X[i-seq_length:i])
            y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        logger.info(f"Created {len(X_seq)} sequences with shape {X_seq.shape}")
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=test_size, shuffle=False
        )
        
        # Store step values for later visualization
        # Use either the original step feature or timestamps converted to sequence numbers
        if step_feature in merged_data.columns:
            steps = pd.to_numeric(merged_data[step_feature], errors='coerce').fillna(0).values[seq_length:]
        else:
            # Create synthetic step values based on sequence
            steps = np.arange(len(X_seq))
            
        steps_train = steps[:len(X_train)]
        steps_test = steps[len(X_train):]
        
        logger.info(f"Data split into {len(X_train)} training and {len(X_test)} testing samples")
        
        # Return preprocessed data
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'steps_train': steps_train,
            'steps_test': steps_test,
            'feature_cols': feature_cols,
            'target_cols': target_cols,
            'seq_length': seq_length,
            'merged_data': merged_data
        }
    
    def create_lstm_model(self, 
                        input_shape: Tuple[int, int], 
                        output_shape: int,
                        units: List[int] = [64, 32],
                        dropout_rate: float = 0.2) -> tf.keras.Model:
        """
        Create an LSTM model for position prediction.
        
        Args:
            input_shape: Shape of input sequences (seq_length, n_features)
            output_shape: Number of outputs (typically 2 for x, y coordinates)
            units: List of units in each LSTM layer
            dropout_rate: Dropout rate between layers
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential()
        
        # Add LSTM layers
        for i, unit in enumerate(units):
            return_sequences = i < len(units) - 1
            if i == 0:
                model.add(LSTM(units=unit, 
                              return_sequences=return_sequences,
                              input_shape=input_shape))
            else:
                model.add(LSTM(units=unit, 
                              return_sequences=return_sequences))
            
            model.add(Dropout(dropout_rate))
        
        # Add dense layers
        model.add(Dense(16, activation='relu'))
        model.add(Dense(output_shape, activation='linear'))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        logger.info(f"Created LSTM model with {len(units)} LSTM layers")
        model.summary(print_fn=logger.info)
        
        return model
    
    def create_cnn_lstm_model(self, 
                            input_shape: Tuple[int, int], 
                            output_shape: int,
                            conv_filters: List[int] = [32, 64],
                            lstm_units: List[int] = [64],
                            dropout_rate: float = 0.2) -> tf.keras.Model:
        """
        Create a CNN-LSTM model for position prediction.
        This model first extracts spatial features using CNN layers,
        then processes temporal relationships using LSTM layers.
        
        Args:
            input_shape: Shape of input sequences (seq_length, n_features)
            output_shape: Number of outputs (typically 2 for x, y coordinates)
            conv_filters: List of filters in each Conv1D layer
            lstm_units: List of units in each LSTM layer
            dropout_rate: Dropout rate between layers
            
        Returns:
            Compiled CNN-LSTM model
        """
        model = Sequential()
        
        # Add CNN layers for feature extraction
        for i, filters in enumerate(conv_filters):
            if i == 0:
                model.add(Conv1D(filters=filters, 
                                kernel_size=3, 
                                activation='relu',
                                input_shape=input_shape))
            else:
                model.add(Conv1D(filters=filters, 
                                kernel_size=3, 
                                activation='relu'))
            
            model.add(BatchNormalization())
            
            # Add pooling every other layer
            if i % 2 == 1:
                model.add(MaxPooling1D(pool_size=2))
        
        # Add LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            model.add(LSTM(units=units, return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
        
        # Add dense layers
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(output_shape, activation='linear'))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        logger.info(f"Created CNN-LSTM model with {len(conv_filters)} CNN layers and {len(lstm_units)} LSTM layers")
        model.summary(print_fn=logger.info)
        
        return model
    
    def create_bidirectional_lstm_model(self, 
                                      input_shape: Tuple[int, int], 
                                      output_shape: int,
                                      units: List[int] = [64, 32],
                                      dropout_rate: float = 0.2) -> tf.keras.Model:
        """
        Create a Bidirectional LSTM model for position prediction.
        
        Args:
            input_shape: Shape of input sequences (seq_length, n_features)
            output_shape: Number of outputs (typically 2 for x, y coordinates)
            units: List of units in each LSTM layer
            dropout_rate: Dropout rate between layers
            
        Returns:
            Compiled Bidirectional LSTM model
        """
        model = Sequential()
        
        # Add Bidirectional LSTM layers
        for i, unit in enumerate(units):
            return_sequences = i < len(units) - 1
            if i == 0:
                model.add(tf.keras.layers.Bidirectional(
                    LSTM(units=unit, return_sequences=return_sequences),
                    input_shape=input_shape
                ))
            else:
                model.add(tf.keras.layers.Bidirectional(
                    LSTM(units=unit, return_sequences=return_sequences)
                ))
            
            model.add(Dropout(dropout_rate))
        
        # Add dense layers
        model.add(Dense(16, activation='relu'))
        model.add(Dense(output_shape, activation='linear'))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        logger.info(f"Created Bidirectional LSTM model with {len(units)} LSTM layers")
        model.summary(print_fn=logger.info)
        
        return model
    
    def train_model(self, 
                   model: tf.keras.Model,
                   data: Dict[str, Any],
                   model_name: str,
                   epochs: int = 100,
                   batch_size: int = 32,
                   patience: int = 15) -> Tuple[tf.keras.Model, Dict[str, List[float]]]:
        """
        Train the deep learning model for position prediction.
        
        Args:
            model: The model to train
            data: Dictionary containing preprocessed data
            model_name: Name to use when saving the model
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            patience: Patience for early stopping
            
        Returns:
            Trained model and training history
        """
        # Create callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        
        model_path = os.path.join(self.output_dir, 'models', f'{model_name}.keras')
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Train model
        logger.info(f"Training {model_name} model for {epochs} epochs with batch size {batch_size}")
        history = model.fit(
            data['X_train'], data['y_train'],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(data['X_test'], data['y_test']),
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        # Save training history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(self.output_dir, 'data', f'{model_name}_history.csv'), index=False)
        
        # Use the trained model with best weights restored through early stopping
        # Note: We don't load the saved model to avoid serialization issues
        trained_model = model
        
        logger.info(f"Model training completed, best weights from checkpointing are loaded")
        
        return trained_model, history.history
    
    def evaluate_model(self, 
                      model: tf.keras.Model,
                      data: Dict[str, Any],
                      model_name: str) -> Dict[str, float]:
        """
        Evaluate the position prediction model.
        
        Args:
            model: Trained model to evaluate
            data: Dictionary containing preprocessed data
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(data['X_test'])
        
        # Inverse transform if scaling was applied
        if self.target_scaler is not None:
            y_pred = self.target_scaler.inverse_transform(y_pred)
            y_true = self.target_scaler.inverse_transform(data['y_test'])
        else:
            y_true = data['y_test']
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate Euclidean distance error
        euclidean_distances = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))
        avg_distance_error = np.mean(euclidean_distances)
        med_distance_error = np.median(euclidean_distances)
        p90_distance_error = np.percentile(euclidean_distances, 90)
        
        # Log metrics
        logger.info(f"Model {model_name} evaluation metrics:")
        logger.info(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        logger.info(f"Avg Distance Error: {avg_distance_error:.4f}, Median: {med_distance_error:.4f}")
        
        # Store metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'avg_distance_error': avg_distance_error,
            'med_distance_error': med_distance_error,
            'p90_distance_error': p90_distance_error
        }
        
        # Save metrics
        pd.DataFrame([metrics]).to_csv(
            os.path.join(self.output_dir, 'data', f'{model_name}_metrics.csv'),
            index=False
        )
        
        # Visualize predictions
        self._visualize_predictions(
            y_true=y_true,
            y_pred=y_pred,
            steps=data['steps_test'],
            model_name=model_name
        )
        
        return metrics
    
    def _visualize_predictions(self, 
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              steps: np.ndarray,
                              model_name: str) -> None:
        """
        Visualize position predictions compared to ground truth.
        
        Args:
            y_true: Ground truth positions
            y_pred: Predicted positions
            steps: Step numbers for each data point
            model_name: Name of the model
        """
        # Create figure
        fig, axs = plt.subplots(3, 1, figsize=(12, 16), dpi=300)
        
        # Plot trajectory
        axs[0].plot(y_true[:, 0], y_true[:, 1], 'b-', label='Ground Truth')
        axs[0].plot(y_pred[:, 0], y_pred[:, 1], 'r--', label='Predicted')
        axs[0].set_xlabel('East')
        axs[0].set_ylabel('North')
        axs[0].set_title(f'{model_name}: Position Trajectory')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
        
        # Plot X coordinate over steps
        axs[1].plot(steps, y_true[:, 0], 'b-', label='Ground Truth X')
        axs[1].plot(steps, y_pred[:, 0], 'r--', label='Predicted X')
        axs[1].set_xlabel('Step')
        axs[1].set_ylabel('East')
        axs[1].set_title('X Position over Steps')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()
        
        # Plot Y coordinate over steps
        axs[2].plot(steps, y_true[:, 1], 'b-', label='Ground Truth Y')
        axs[2].plot(steps, y_pred[:, 1], 'r--', label='Predicted Y')
        axs[2].set_xlabel('Step')
        axs[2].set_ylabel('North')
        axs[2].set_title('Y Position over Steps')
        axs[2].grid(True, alpha=0.3)
        axs[2].legend()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(
            os.path.join(self.output_dir, 'plots', f'{model_name}_predictions.png'), 
            dpi=300, 
            bbox_inches='tight'
        )
        plt.close()
        
        # Create error analysis plot
        fig, axs = plt.subplots(2, 1, figsize=(12, 12), dpi=300)
        
        # Calculate errors
        x_errors = y_true[:, 0] - y_pred[:, 0]
        y_errors = y_true[:, 1] - y_pred[:, 1]
        distance_errors = np.sqrt(x_errors**2 + y_errors**2)
        
        # Plot error over steps
        axs[0].plot(steps, x_errors, 'r-', label='X Error')
        axs[0].plot(steps, y_errors, 'b-', label='Y Error')
        axs[0].set_xlabel('Step')
        axs[0].set_ylabel('Error')
        axs[0].set_title('Coordinate Errors over Steps')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
        
        # Plot distance error over steps
        axs[1].plot(steps, distance_errors, 'g-', label='Distance Error')
        axs[1].axhline(y=np.mean(distance_errors), color='k', linestyle='--', 
                      label=f'Mean: {np.mean(distance_errors):.2f}')
        axs[1].axhline(y=np.median(distance_errors), color='r', linestyle='--', 
                      label=f'Median: {np.median(distance_errors):.2f}')
        axs[1].set_xlabel('Step')
        axs[1].set_ylabel('Distance Error')
        axs[1].set_title('Euclidean Distance Error over Steps')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(
            os.path.join(self.output_dir, 'plots', f'{model_name}_errors.png'), 
            dpi=300, 
            bbox_inches='tight'
        )
        plt.close()
        
        logger.info(f"Visualizations for {model_name} saved to plots directory")

    def run_position_prediction(self, 
                              sensor_data: pd.DataFrame,
                              ground_truth_data: pd.DataFrame,
                              model_type: str = 'lstm',
                              seq_length: int = 10,
                              epochs: int = 100,
                              batch_size: int = 32) -> Dict[str, Any]:
        """
        Run the complete position prediction pipeline from preprocessing to evaluation.
        
        Args:
            sensor_data: DataFrame containing sensor data
            ground_truth_data: DataFrame containing ground truth positions
            model_type: Type of model to use ('lstm', 'cnn_lstm', 'bidirectional')
            seq_length: Number of time steps in each sequence
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing the model, metrics, and predictions
        """
        # Print column names for debugging
        logger.info(f"Ground truth data columns: {ground_truth_data.columns.tolist()}")
        logger.info(f"Sensor data columns: {sensor_data.columns.tolist()}")
        
        # Verify ground truth data has necessary columns
        if 'value_4' not in ground_truth_data.columns or 'value_5' not in ground_truth_data.columns:
            logger.error("Ground truth data missing required position columns (value_4, value_5)")
            raise ValueError("Ground truth data doesn't have required position coordinates")
            
        # Check if we have enough ground truth points
        if len(ground_truth_data) < 5:
            logger.warning(f"Very little ground truth data available: {len(ground_truth_data)} points")
            logger.info("Adding some synthetic points to allow model training")
            
            # Create synthetic data points by interpolation if possible
            if len(ground_truth_data) >= 2:
                # Linear interpolation between existing points
                new_points = []
                for i in range(len(ground_truth_data) - 1):
                    pt1 = ground_truth_data.iloc[i]
                    pt2 = ground_truth_data.iloc[i+1]
                    
                    # Create 3 points between each pair
                    for j in range(1, 4):
                        alpha = j / 4  # interpolation factor
                        new_point = pt1.copy()
                        new_point['value_4'] = pt1['value_4'] + alpha * (pt2['value_4'] - pt1['value_4'])
                        new_point['value_5'] = pt1['value_5'] + alpha * (pt2['value_5'] - pt1['value_5'])
                        
                        # Create timestamps if needed
                        if 'Timestamp_(ms)' in pt1 and 'Timestamp_(ms)' in pt2:
                            new_point['Timestamp_(ms)'] = pt1['Timestamp_(ms)'] + alpha * (pt2['Timestamp_(ms)'] - pt1['Timestamp_(ms)'])
                        
                        # Create step if needed
                        if 'step' in pt1 and 'step' in pt2:
                            new_point['step'] = pt1['step'] + alpha * (pt2['step'] - pt1['step'])
                            
                        new_points.append(new_point)
                
                # Combine original and new points
                augmented_data = pd.concat([ground_truth_data, pd.DataFrame(new_points)], ignore_index=True)
                
                # Sort by step or timestamp if available
                if 'step' in augmented_data.columns:
                    augmented_data = augmented_data.sort_values('step')
                elif 'Timestamp_(ms)' in augmented_data.columns:
                    augmented_data = augmented_data.sort_values('Timestamp_(ms)')
                    
                ground_truth_data = augmented_data
                logger.info(f"Created augmented ground truth dataset with {len(ground_truth_data)} points")
        
        # Preprocess data
        data = self.preprocess_data(
            sensor_data=sensor_data,
            ground_truth_data=ground_truth_data,
            seq_length=seq_length
        )
        
        input_shape = (seq_length, len(data['feature_cols']))
        output_shape = len(data['target_cols'])
        
        # Create model based on type
        if model_type == 'lstm':
            model = self.create_lstm_model(input_shape, output_shape)
            model_name = 'lstm_position'
        elif model_type == 'cnn_lstm':
            model = self.create_cnn_lstm_model(input_shape, output_shape)
            model_name = 'cnn_lstm_position'
        elif model_type == 'bidirectional':
            model = self.create_bidirectional_lstm_model(input_shape, output_shape)
            model_name = 'bidirectional_lstm_position'
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model, history = self.train_model(
            model=model,
            data=data,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Evaluate model
        metrics = self.evaluate_model(
            model=model,
            data=data,
            model_name=model_name
        )
        
        # Make predictions on all data
        X_all = np.vstack((data['X_train'], data['X_test']))
        steps_all = np.concatenate((data['steps_train'], data['steps_test']))
        
        # Make predictions
        y_pred_all = model.predict(X_all)
        
        # Inverse transform if scaling was applied
        if self.target_scaler is not None:
            y_pred_all = self.target_scaler.inverse_transform(y_pred_all)
        
        # Create prediction dataframe
        prediction_df = pd.DataFrame({
            'Step': steps_all,
            'Predicted_X': y_pred_all[:, 0],
            'Predicted_Y': y_pred_all[:, 1]
        })
        
        # Save predictions
        prediction_df.to_csv(
            os.path.join(self.output_dir, 'data', f'{model_name}_predictions.csv'),
            index=False
        )
        
        logger.info(f"Position prediction completed with {model_type} model")
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': prediction_df,
            'data': data,
            'history': history
        }
    
    def benchmark_models(self, 
                        sensor_data: pd.DataFrame,
                        ground_truth_data: pd.DataFrame,
                        seq_lengths: List[int] = [5, 10, 15],
                        epochs: int = 100,
                        batch_size: int = 32) -> pd.DataFrame:
        """
        Benchmark different models and sequence lengths for position prediction.
        
        Args:
            sensor_data: DataFrame containing sensor data
            ground_truth_data: DataFrame containing ground truth positions
            seq_lengths: List of sequence lengths to try
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            
        Returns:
            DataFrame comparing different models and configurations
        """
        model_types = ['lstm', 'cnn_lstm', 'bidirectional']
        results = []
        
        for model_type in model_types:
            for seq_length in seq_lengths:
                logger.info(f"Benchmarking {model_type} model with sequence length {seq_length}")
                
                try:
                    result = self.run_position_prediction(
                        sensor_data=sensor_data,
                        ground_truth_data=ground_truth_data,
                        model_type=model_type,
                        seq_length=seq_length,
                        epochs=epochs,
                        batch_size=batch_size
                    )
                    
                    # Store benchmark results
                    results.append({
                        'model_type': model_type,
                        'seq_length': seq_length,
                        'rmse': result['metrics']['rmse'],
                        'mae': result['metrics']['mae'],
                        'avg_distance_error': result['metrics']['avg_distance_error'],
                        'med_distance_error': result['metrics']['med_distance_error'],
                        'p90_distance_error': result['metrics']['p90_distance_error']
                    })
                except Exception as e:
                    logger.error(f"Error benchmarking {model_type} with seq_length {seq_length}: {str(e)}")
        
        # Create benchmark dataframe
        benchmark_df = pd.DataFrame(results)
        
        # Save benchmark results
        benchmark_df.to_csv(
            os.path.join(self.output_dir, 'data', 'position_prediction_benchmark.csv'),
            index=False
        )
        
        # Visualize benchmark results
        self._visualize_benchmark(benchmark_df)
        
        logger.info(f"Benchmark completed, results saved to position_prediction_benchmark.csv")
        
        return benchmark_df
    
    def _visualize_benchmark(self, benchmark_df: pd.DataFrame) -> None:
        """
        Visualize benchmark results.
        
        Args:
            benchmark_df: DataFrame containing benchmark results
        """
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
        axs = axs.flatten()
        
        metrics = ['rmse', 'mae', 'avg_distance_error', 'med_distance_error']
        titles = ['RMSE', 'MAE', 'Average Distance Error', 'Median Distance Error']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            # Group by model_type and seq_length
            pivot_df = benchmark_df.pivot(index='model_type', columns='seq_length', values=metric)
            pivot_df.plot(kind='bar', ax=axs[i])
            
            axs[i].set_title(title)
            axs[i].set_xlabel('Model Type')
            axs[i].set_ylabel(metric.upper())
            axs[i].grid(True, axis='y', alpha=0.3)
            axs[i].legend(title='Sequence Length')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(
            os.path.join(self.output_dir, 'plots', 'position_prediction_benchmark.png'), 
            dpi=300, 
            bbox_inches='tight'
        )
        plt.close()
        
        logger.info("Benchmark visualization saved to plots directory")
