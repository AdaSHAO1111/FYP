import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GRU, Bidirectional, Dropout, Input, Concatenate, Lambda, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import math

# Set output directory
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2'
os.makedirs(output_dir, exist_ok=True)

# Function to calculate bearing between two points
def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate the bearing (azimuth) between two points"""
    lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    
    delta_lon = lon2 - lon1
    x = math.atan2(
        math.sin(delta_lon) * math.cos(lat2),
        math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    )
    
    bearing = (math.degrees(x) + 360) % 360
    return bearing

# Function to create sequences for model input
def create_sequences(X, y, window_size):
    """Create input sequences and corresponding targets"""
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size])
    return np.array(X_seq), np.array(y_seq)

class ImprovedPositionModel:
    """
    Advanced model that directly optimizes for position tracking accuracy
    """
    
    def __init__(self, window_size=30, position_weight=0.7):
        self.window_size = window_size
        self.position_weight = position_weight  # Weight for position loss vs heading loss
        self.heading_model = None
        self.position_model = None
        self.integrated_model = None
        self.scalers = {
            'gyro_X': MinMaxScaler(),
            'compass_X': MinMaxScaler(),
            'steps_X': MinMaxScaler(),
            'heading_y': MinMaxScaler(),
            'position_y': MinMaxScaler()
        }
    
    def build_heading_model(self, gyro_shape, compass_shape):
        """Build a model that predicts heading by fusing gyro and compass data"""
        # Gyro input branch
        gyro_input = Input(shape=gyro_shape, name='gyro_input')
        gyro_gru1 = Bidirectional(GRU(64, return_sequences=True))(gyro_input)
        gyro_drop1 = Dropout(0.3)(gyro_gru1)
        gyro_gru2 = Bidirectional(GRU(32, return_sequences=False))(gyro_drop1)
        gyro_drop2 = Dropout(0.3)(gyro_gru2)
        
        # Compass input branch
        compass_input = Input(shape=compass_shape, name='compass_input')
        compass_gru1 = Bidirectional(GRU(64, return_sequences=True))(compass_input)
        compass_drop1 = Dropout(0.3)(compass_gru1)
        compass_gru2 = Bidirectional(GRU(32, return_sequences=False))(compass_drop1)
        compass_drop2 = Dropout(0.3)(compass_gru2)
        
        # Combine branches
        merged = Concatenate()([gyro_drop2, compass_drop2])
        
        # Output layers
        dense1 = Dense(64, activation='relu')(merged)
        drop3 = Dropout(0.2)(dense1)
        dense2 = Dense(32, activation='relu')(drop3)
        heading_output = Dense(1, name='heading_output')(dense2)
        
        # Build model
        model = Model(
            inputs=[gyro_input, compass_input],
            outputs=heading_output,
            name='heading_model'
        )
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_position_model(self, gyro_shape, compass_shape, steps_shape):
        """
        Build a model that directly predicts position changes based on 
        sensor data and step information
        """
        # Gyro input branch
        gyro_input = Input(shape=gyro_shape, name='gyro_input')
        gyro_gru1 = Bidirectional(GRU(64, return_sequences=True))(gyro_input)
        gyro_drop1 = Dropout(0.3)(gyro_gru1)
        gyro_gru2 = Bidirectional(GRU(32, return_sequences=False))(gyro_drop1)
        gyro_drop2 = Dropout(0.3)(gyro_gru2)
        
        # Compass input branch
        compass_input = Input(shape=compass_shape, name='compass_input')
        compass_gru1 = Bidirectional(GRU(64, return_sequences=True))(compass_input)
        compass_drop1 = Dropout(0.3)(compass_gru1)
        compass_gru2 = Bidirectional(GRU(32, return_sequences=False))(compass_drop1)
        compass_drop2 = Dropout(0.3)(compass_gru2)
        
        # Steps input branch - step detection affects position changes
        steps_input = Input(shape=steps_shape, name='steps_input')
        steps_gru1 = Bidirectional(GRU(32, return_sequences=True))(steps_input)
        steps_drop1 = Dropout(0.2)(steps_gru1)
        steps_gru2 = Bidirectional(GRU(16, return_sequences=False))(steps_drop1)
        steps_drop2 = Dropout(0.2)(steps_gru2)
        
        # Combine all branches
        merged = Concatenate()([gyro_drop2, compass_drop2, steps_drop2])
        
        # Output layers
        dense1 = Dense(64, activation='relu')(merged)
        drop3 = Dropout(0.2)(dense1)
        dense2 = Dense(32, activation='relu')(drop3)
        
        # Output two values: dx and dy (position changes)
        position_output = Dense(2, name='position_output')(dense2)
        
        # Build model
        model = Model(
            inputs=[gyro_input, compass_input, steps_input],
            outputs=position_output,
            name='position_model'
        )
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_integrated_model(self, gyro_shape, compass_shape, steps_shape):
        """
        Build a model that predicts both heading and position changes,
        with shared features between the two outputs
        """
        # Gyro input branch
        gyro_input = Input(shape=gyro_shape, name='gyro_input')
        gyro_gru1 = Bidirectional(GRU(64, return_sequences=True))(gyro_input)
        gyro_drop1 = Dropout(0.3)(gyro_gru1)
        gyro_gru2 = Bidirectional(GRU(32, return_sequences=False))(gyro_drop1)
        gyro_drop2 = Dropout(0.3)(gyro_gru2)
        
        # Compass input branch
        compass_input = Input(shape=compass_shape, name='compass_input')
        compass_gru1 = Bidirectional(GRU(64, return_sequences=True))(compass_input)
        compass_drop1 = Dropout(0.3)(compass_gru1)
        compass_gru2 = Bidirectional(GRU(32, return_sequences=False))(compass_drop1)
        compass_drop2 = Dropout(0.3)(compass_gru2)
        
        # Steps input branch - step detection affects position changes
        steps_input = Input(shape=steps_shape, name='steps_input')
        steps_gru1 = Bidirectional(GRU(32, return_sequences=True))(steps_input)
        steps_drop1 = Dropout(0.2)(steps_gru1)
        steps_gru2 = Bidirectional(GRU(16, return_sequences=False))(steps_drop1)
        steps_drop2 = Dropout(0.2)(steps_gru2)
        
        # Combine sensor branches for shared features
        sensor_merged = Concatenate()([gyro_drop2, compass_drop2])
        
        # Heading prediction branch
        heading_dense1 = Dense(64, activation='relu')(sensor_merged)
        heading_drop1 = Dropout(0.2)(heading_dense1)
        heading_dense2 = Dense(32, activation='relu')(heading_drop1)
        heading_output = Dense(1, name='heading_output')(heading_dense2)
        
        # Position prediction branch - also uses step data
        position_merged = Concatenate()([sensor_merged, steps_drop2])
        position_dense1 = Dense(64, activation='relu')(position_merged)
        position_drop1 = Dropout(0.2)(position_dense1)
        position_dense2 = Dense(32, activation='relu')(position_drop1)
        position_output = Dense(2, name='position_output')(position_dense2)
        
        # Build model with two outputs
        model = Model(
            inputs=[gyro_input, compass_input, steps_input],
            outputs=[heading_output, position_output],
            name='integrated_model'
        )
        
        # Use different loss weights to prioritize position accuracy
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss={
                'heading_output': 'mse',
                'position_output': 'mse'
            },
            loss_weights={
                'heading_output': 1.0 - self.position_weight,
                'position_output': self.position_weight
            },
            metrics={
                'heading_output': ['mae'],
                'position_output': ['mae']
            }
        )
        
        return model
    
    def train(self, gyro_data, compass_data, step_data, ground_truth_heading, ground_truth_positions,
              batch_size=32, epochs=100, validation_split=0.2):
        """
        Train the position-aware models
        
        Parameters:
        -----------
        gyro_data : DataFrame
            Gyro sensor data
        compass_data : DataFrame
            Compass sensor data
        step_data : DataFrame
            Step detection data
        ground_truth_heading : Series
            Ground truth heading values
        ground_truth_positions : array-like
            Ground truth position points
        """
        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-5,
                verbose=1
            )
        ]
        
        # Prepare gyro features
        gyro_features = gyro_data[['axisZAngle', 'gyroSumFromstart0', 'compass']].values
        gyro_features_scaled = self.scalers['gyro_X'].fit_transform(gyro_features)
        
        # Prepare compass features
        compass_features = compass_data[['Magnetic_Field_Magnitude', 'gyroSumFromstart0', 'compass']].values
        compass_features_scaled = self.scalers['compass_X'].fit_transform(compass_features)
        
        # Prepare step features
        step_features = step_data[['step', 'step_diff']].values
        step_features_scaled = self.scalers['steps_X'].fit_transform(step_features)
        
        # Prepare heading targets
        heading_target = ground_truth_heading.values.reshape(-1, 1)
        heading_target_scaled = self.scalers['heading_y'].fit_transform(heading_target)
        
        # Prepare position targets
        position_targets = np.array(ground_truth_positions)
        position_target_scaled = self.scalers['position_y'].fit_transform(position_targets)
        
        # Create sequences
        gyro_X_seq, _ = create_sequences(gyro_features_scaled, heading_target_scaled, self.window_size)
        compass_X_seq, _ = create_sequences(compass_features_scaled, heading_target_scaled, self.window_size)
        steps_X_seq, _ = create_sequences(step_features_scaled, heading_target_scaled, self.window_size)
        _, heading_y_seq = create_sequences(gyro_features_scaled, heading_target_scaled, self.window_size)
        _, position_y_seq = create_sequences(gyro_features_scaled, position_target_scaled, self.window_size)
        
        # Split data
        indices = np.arange(len(gyro_X_seq))
        train_idx, val_idx = train_test_split(indices, test_size=validation_split, shuffle=False)
        
        # Build and train integrated model
        print("Training integrated model...")
        self.integrated_model = self.build_integrated_model(
            gyro_X_seq.shape[1:], compass_X_seq.shape[1:], steps_X_seq.shape[1:]
        )
        
        integrated_history = self.integrated_model.fit(
            [gyro_X_seq[train_idx], compass_X_seq[train_idx], steps_X_seq[train_idx]],
            [heading_y_seq[train_idx], position_y_seq[train_idx]],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(
                [gyro_X_seq[val_idx], compass_X_seq[val_idx], steps_X_seq[val_idx]],
                [heading_y_seq[val_idx], position_y_seq[val_idx]]
            ),
            callbacks=callbacks,
            verbose=1
        )
        
        # Also train individual models for comparison
        print("Training heading model...")
        self.heading_model = self.build_heading_model(
            gyro_X_seq.shape[1:], compass_X_seq.shape[1:]
        )
        
        heading_history = self.heading_model.fit(
            [gyro_X_seq[train_idx], compass_X_seq[train_idx]],
            heading_y_seq[train_idx],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(
                [gyro_X_seq[val_idx], compass_X_seq[val_idx]],
                heading_y_seq[val_idx]
            ),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training position model...")
        self.position_model = self.build_position_model(
            gyro_X_seq.shape[1:], compass_X_seq.shape[1:], steps_X_seq.shape[1:]
        )
        
        position_history = self.position_model.fit(
            [gyro_X_seq[train_idx], compass_X_seq[train_idx], steps_X_seq[train_idx]],
            position_y_seq[train_idx],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(
                [gyro_X_seq[val_idx], compass_X_seq[val_idx], steps_X_seq[val_idx]],
                position_y_seq[val_idx]
            ),
            callbacks=callbacks,
            verbose=1
        )
        
        return {
            'integrated': integrated_history,
            'heading': heading_history,
            'position': position_history
        }
    
    def predict(self, gyro_data, compass_data, step_data):
        """
        Make predictions using the trained models
        
        Returns:
        --------
        dict containing predictions from each model
        """
        if self.integrated_model is None:
            raise ValueError("Models must be trained before prediction")
        
        # Prepare features
        gyro_features = gyro_data[['axisZAngle', 'gyroSumFromstart0', 'compass']].values
        gyro_features_scaled = self.scalers['gyro_X'].transform(gyro_features)
        
        compass_features = compass_data[['Magnetic_Field_Magnitude', 'gyroSumFromstart0', 'compass']].values
        compass_features_scaled = self.scalers['compass_X'].transform(compass_features)
        
        step_features = step_data[['step', 'step_diff']].values
        step_features_scaled = self.scalers['steps_X'].transform(step_features)
        
        # Create sequences
        gyro_X_seq = []
        compass_X_seq = []
        steps_X_seq = []
        
        for i in range(len(gyro_features_scaled) - self.window_size):
            gyro_X_seq.append(gyro_features_scaled[i:i + self.window_size])
            compass_X_seq.append(compass_features_scaled[i:i + self.window_size])
            steps_X_seq.append(step_features_scaled[i:i + self.window_size])
        
        gyro_X_seq = np.array(gyro_X_seq)
        compass_X_seq = np.array(compass_X_seq)
        steps_X_seq = np.array(steps_X_seq)
        
        # Make predictions with integrated model
        integrated_predictions = self.integrated_model.predict(
            [gyro_X_seq, compass_X_seq, steps_X_seq]
        )
        integrated_heading = self.scalers['heading_y'].inverse_transform(integrated_predictions[0])
        integrated_position = self.scalers['position_y'].inverse_transform(integrated_predictions[1])
        
        # Make predictions with individual models
        heading_predictions = self.heading_model.predict([gyro_X_seq, compass_X_seq])
        heading_predictions = self.scalers['heading_y'].inverse_transform(heading_predictions)
        
        position_predictions = self.position_model.predict([gyro_X_seq, compass_X_seq, steps_X_seq])
        position_predictions = self.scalers['position_y'].inverse_transform(position_predictions)
        
        # Initialize full-length arrays
        integrated_heading_full = np.zeros(len(gyro_data))
        integrated_position_full = np.zeros((len(gyro_data), 2))
        heading_predictions_full = np.zeros(len(gyro_data))
        position_predictions_full = np.zeros((len(gyro_data), 2))
        
        # Fill in predictions
        integrated_heading_full[self.window_size:self.window_size + len(integrated_heading)] = integrated_heading.flatten()
        integrated_position_full[self.window_size:self.window_size + len(integrated_position)] = integrated_position
        
        heading_predictions_full[self.window_size:self.window_size + len(heading_predictions)] = heading_predictions.flatten()
        position_predictions_full[self.window_size:self.window_size + len(position_predictions)] = position_predictions
        
        # Handle initialization (first window_size elements)
        if len(integrated_heading) > 0:
            integrated_heading_full[:self.window_size] = integrated_heading[0]
            integrated_position_full[:self.window_size] = integrated_position[0]
            
            heading_predictions_full[:self.window_size] = heading_predictions[0]
            position_predictions_full[:self.window_size] = position_predictions[0]
        
        # Normalize headings to 0-360 degrees
        integrated_heading_full = (integrated_heading_full + 360) % 360
        heading_predictions_full = (heading_predictions_full + 360) % 360
        
        return {
            'integrated_heading': integrated_heading_full,
            'integrated_position': integrated_position_full,
            'heading': heading_predictions_full,
            'position': position_predictions_full
        }
    
    def save_models(self, save_dir):
        """Save trained models"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if self.integrated_model:
            self.integrated_model.save(os.path.join(save_dir, 'integrated_model.keras'))
        
        if self.heading_model:
            self.heading_model.save(os.path.join(save_dir, 'heading_model.keras'))
        
        if self.position_model:
            self.position_model.save(os.path.join(save_dir, 'position_model.keras'))
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir):
        """Load trained models"""
        integrated_path = os.path.join(save_dir, 'integrated_model.keras')
        heading_path = os.path.join(save_dir, 'heading_model.keras')
        position_path = os.path.join(save_dir, 'position_model.keras')
        
        if os.path.exists(integrated_path):
            self.integrated_model = tf.keras.models.load_model(integrated_path)
            print(f"Loaded integrated model from {integrated_path}")
        
        if os.path.exists(heading_path):
            self.heading_model = tf.keras.models.load_model(heading_path)
            print(f"Loaded heading model from {heading_path}")
        
        if os.path.exists(position_path):
            self.position_model = tf.keras.models.load_model(position_path)
            print(f"Loaded position model from {position_path}")

# Function to calculate traditional positions from headings
def calculate_positions_from_heading(data, heading_column, step_length=0.66, initial_position=(0, 0)):
    """Calculate positions using step detection and heading"""
    positions = [initial_position]
    current_position = initial_position
    prev_step = data['step'].iloc[0]
    
    for i in range(1, len(data)):
        # Calculate step change
        change_in_step = data['step'].iloc[i] - prev_step
        
        # If step changes, calculate new position
        if change_in_step != 0:
            # Calculate distance change
            change_in_distance = change_in_step * step_length
            
            # Get heading value (0° is North, 90° is East)
            heading = data[heading_column].iloc[i]
            
            # Calculate new position (East is x-axis, North is y-axis)
            new_x = current_position[0] + change_in_distance * np.sin(np.radians(heading))
            new_y = current_position[1] + change_in_distance * np.cos(np.radians(heading))
            
            # Update current position
            current_position = (new_x, new_y)
            positions.append(current_position)
            
            # Update previous step
            prev_step = data['step'].iloc[i]
    
    return positions

# Function to interpolate positions from direct position predictions
def interpolate_positions(data, position_predictions, initial_position=(0, 0)):
    """Interpolate positions using direct position predictions"""
    positions = [initial_position]
    current_position = initial_position
    prev_step = data['step'].iloc[0]
    
    for i in range(1, len(data)):
        # If step changes, update position
        if data['step'].iloc[i] != prev_step:
            # Get predicted position changes
            dx = position_predictions[i][0]
            dy = position_predictions[i][1]
            
            # Update position
            new_x = current_position[0] + dx
            new_y = current_position[1] + dy
            
            # Store new position
            current_position = (new_x, new_y)
            positions.append(current_position)
            
            # Update previous step
            prev_step = data['step'].iloc[i]
    
    return positions 