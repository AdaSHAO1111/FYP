import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GRU, Bidirectional, Dropout, Input, Concatenate, Attention, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

# Custom attention layer for time series
class TimeAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TimeAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight", shape=(input_shape[-1], 1),
            initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias", shape=(input_shape[1], 1),
            initializer="zeros", trainable=True
        )
        super(TimeAttention, self).build(input_shape)
    
    def call(self, x):
        # Alignment scores
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        
        # Attention weights
        a = tf.nn.softmax(e, axis=1)
        
        # Weighted sum
        context = x * a
        context = tf.reduce_sum(context, axis=1)
        
        return context

# Custom loss function that incorporates both heading accuracy and position consistency
def combined_loss(ground_truth_positions):
    """Custom loss function that considers both heading error and position error"""
    def loss(y_true, y_pred):
        # Heading loss component (MSE)
        heading_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Convert to radians for position calculation
        y_true_rad = y_true * math.pi / 180
        y_pred_rad = y_pred * math.pi / 180
        
        # Calculate position difference (simplified for loss function)
        # This simulates how heading errors affect position
        dx_true = tf.cos(y_true_rad)
        dy_true = tf.sin(y_true_rad)
        dx_pred = tf.cos(y_pred_rad)
        dy_pred = tf.sin(y_pred_rad)
        
        # Position error component
        position_loss = tf.reduce_mean(tf.square(dx_true - dx_pred) + tf.square(dy_true - dy_pred))
        
        # Combined loss with weighting
        return heading_loss + 0.5 * position_loss
    
    return loss

class AdvancedHeadingModel:
    """Advanced model for heading prediction with attention and position awareness"""
    
    def __init__(self, window_size=20, use_attention=True, use_position_loss=True):
        self.window_size = window_size
        self.use_attention = use_attention
        self.use_position_loss = use_position_loss
        self.gyro_model = None
        self.compass_model = None
        self.fusion_model = None
        self.scalers = {
            'gyro_X': MinMaxScaler(),
            'gyro_y': MinMaxScaler(),
            'compass_X': MinMaxScaler(),
            'compass_y': MinMaxScaler(),
            'fusion_X': MinMaxScaler(),
            'fusion_y': MinMaxScaler()
        }
        self.ground_truth_positions = None
    
    def build_attention_model(self, input_shape, name=None):
        """Build a GRU model with attention mechanism"""
        inputs = Input(shape=input_shape)
        
        # Bidirectional GRU layers
        gru1 = Bidirectional(GRU(64, return_sequences=True))(inputs)
        drop1 = Dropout(0.25)(gru1)
        
        gru2 = Bidirectional(GRU(32, return_sequences=True))(drop1)
        drop2 = Dropout(0.25)(gru2)
        
        # Apply attention
        attention = TimeAttention()(drop2)
        
        # Output layers
        dense1 = Dense(32, activation='relu')(attention)
        drop3 = Dropout(0.2)(dense1)
        output = Dense(1)(drop3)
        
        model = Model(inputs=inputs, outputs=output, name=name)
        
        # Use combined loss if position-aware
        if self.use_position_loss and self.ground_truth_positions is not None:
            loss = combined_loss(self.ground_truth_positions)
        else:
            loss = 'mse'
            
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss,
            metrics=['mae']
        )
        
        return model
    
    def build_fusion_model(self, gyro_shape, compass_shape):
        """Build a fusion model that combines gyro and compass data"""
        # Gyro input branch
        gyro_input = Input(shape=gyro_shape, name='gyro_input')
        gyro_gru1 = Bidirectional(GRU(64, return_sequences=True))(gyro_input)
        gyro_drop1 = Dropout(0.25)(gyro_gru1)
        gyro_gru2 = Bidirectional(GRU(32, return_sequences=True))(gyro_drop1)
        
        # Compass input branch
        compass_input = Input(shape=compass_shape, name='compass_input')
        compass_gru1 = Bidirectional(GRU(64, return_sequences=True))(compass_input)
        compass_drop1 = Dropout(0.25)(compass_gru1)
        compass_gru2 = Bidirectional(GRU(32, return_sequences=True))(compass_drop1)
        
        # Apply attention to each branch
        gyro_attention = TimeAttention()(gyro_gru2)
        compass_attention = TimeAttention()(compass_gru2)
        
        # Concatenate the two branches
        merged = Concatenate()([gyro_attention, compass_attention])
        
        # Output layers
        dense1 = Dense(32, activation='relu')(merged)
        drop_final = Dropout(0.2)(dense1)
        output = Dense(1)(drop_final)
        
        model = Model(inputs=[gyro_input, compass_input], outputs=output, name='fusion_model')
        
        # Use combined loss if position-aware
        if self.use_position_loss and self.ground_truth_positions is not None:
            loss = combined_loss(self.ground_truth_positions)
        else:
            loss = 'mse'
            
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss=loss,
            metrics=['mae']
        )
        
        return model
    
    def train(self, gyro_data, compass_data, ground_truth_heading, ground_truth_positions=None, epochs=100):
        """
        Train the models using gyro and compass data
        
        Parameters:
        -----------
        gyro_data : DataFrame
            Gyro sensor data
        compass_data : DataFrame
            Compass sensor data
        ground_truth_heading : Series
            Ground truth heading values
        ground_truth_positions : array-like, optional
            Ground truth position points (for position-aware loss)
        epochs : int
            Number of training epochs
        """
        self.ground_truth_positions = ground_truth_positions
        
        # Prepare callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001)
        ]
        
        # Prepare gyro features
        gyro_features = gyro_data[['axisZAngle', 'gyroSumFromstart0', 'compass']].values
        gyro_target = ground_truth_heading[gyro_data.index].values.reshape(-1, 1)
        
        # Scale gyro data
        gyro_features_scaled = self.scalers['gyro_X'].fit_transform(gyro_features)
        gyro_target_scaled = self.scalers['gyro_y'].fit_transform(gyro_target)
        
        # Create sequences for gyro
        gyro_X_seq, gyro_y_seq = create_sequences(gyro_features_scaled, gyro_target_scaled, self.window_size)
        
        # Split gyro data
        gyro_X_train, gyro_X_val, gyro_y_train, gyro_y_val = train_test_split(
            gyro_X_seq, gyro_y_seq, test_size=0.2, shuffle=False
        )
        
        # Prepare compass features
        compass_features = compass_data[['Magnetic_Field_Magnitude', 'gyroSumFromstart0', 'compass']].values
        compass_target = ground_truth_heading[compass_data.index].values.reshape(-1, 1)
        
        # Scale compass data
        compass_features_scaled = self.scalers['compass_X'].fit_transform(compass_features)
        compass_target_scaled = self.scalers['compass_y'].fit_transform(compass_target)
        
        # Create sequences for compass
        compass_X_seq, compass_y_seq = create_sequences(compass_features_scaled, compass_target_scaled, self.window_size)
        
        # Split compass data
        compass_X_train, compass_X_val, compass_y_train, compass_y_val = train_test_split(
            compass_X_seq, compass_y_seq, test_size=0.2, shuffle=False
        )
        
        # Build and train gyro model
        print("Training gyro model...")
        self.gyro_model = self.build_attention_model((self.window_size, gyro_features.shape[1]), name='gyro_model')
        gyro_history = self.gyro_model.fit(
            gyro_X_train, gyro_y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(gyro_X_val, gyro_y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Build and train compass model
        print("Training compass model...")
        self.compass_model = self.build_attention_model((self.window_size, compass_features.shape[1]), name='compass_model')
        compass_history = self.compass_model.fit(
            compass_X_train, compass_y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(compass_X_val, compass_y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Build and train fusion model if we have matching data points
        if len(gyro_X_train) == len(compass_X_train):
            print("Training fusion model...")
            self.fusion_model = self.build_fusion_model(
                (self.window_size, gyro_features.shape[1]),
                (self.window_size, compass_features.shape[1])
            )
            
            fusion_history = self.fusion_model.fit(
                [gyro_X_train, compass_X_train], gyro_y_train,  # Using gyro target as reference
                epochs=epochs,
                batch_size=32,
                validation_data=([gyro_X_val, compass_X_val], gyro_y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            print("Skipping fusion model: data sequences length mismatch")
        
        return {
            'gyro': gyro_history,
            'compass': compass_history,
            'fusion': fusion_history if self.fusion_model else None
        }
    
    def predict_headings(self, gyro_data, compass_data):
        """
        Predict headings using trained models
        
        Parameters:
        -----------
        gyro_data : DataFrame
            Gyro sensor data
        compass_data : DataFrame
            Compass sensor data
            
        Returns:
        --------
        predictions : dict
            Dictionary containing predictions from each model
        """
        # Check if models are trained
        if self.gyro_model is None or self.compass_model is None:
            raise ValueError("Models must be trained before prediction")
        
        # Prepare gyro features
        gyro_features = gyro_data[['axisZAngle', 'gyroSumFromstart0', 'compass']].values
        gyro_features_scaled = self.scalers['gyro_X'].transform(gyro_features)
        
        # Create sequences for gyro
        gyro_X_seq = []
        for i in range(len(gyro_features_scaled) - self.window_size):
            gyro_X_seq.append(gyro_features_scaled[i:i + self.window_size])
        
        # Prepare compass features
        compass_features = compass_data[['Magnetic_Field_Magnitude', 'gyroSumFromstart0', 'compass']].values
        compass_features_scaled = self.scalers['compass_X'].transform(compass_features)
        
        # Create sequences for compass
        compass_X_seq = []
        for i in range(len(compass_features_scaled) - self.window_size):
            compass_X_seq.append(compass_features_scaled[i:i + self.window_size])
        
        # Initialize prediction arrays
        gyro_predictions = np.zeros(len(gyro_data))
        compass_predictions = np.zeros(len(compass_data))
        fusion_predictions = np.zeros(len(gyro_data))
        
        # Predict using gyro model
        if len(gyro_X_seq) > 0:
            gyro_X_seq = np.array(gyro_X_seq)
            gyro_pred_scaled = self.gyro_model.predict(gyro_X_seq)
            gyro_pred = self.scalers['gyro_y'].inverse_transform(gyro_pred_scaled)
            gyro_predictions[self.window_size:self.window_size+len(gyro_pred)] = gyro_pred.flatten()
        
        # Predict using compass model
        if len(compass_X_seq) > 0:
            compass_X_seq = np.array(compass_X_seq)
            compass_pred_scaled = self.compass_model.predict(compass_X_seq)
            compass_pred = self.scalers['compass_y'].inverse_transform(compass_pred_scaled)
            compass_predictions[self.window_size:self.window_size+len(compass_pred)] = compass_pred.flatten()
        
        # Predict using fusion model if available
        if self.fusion_model is not None and len(gyro_X_seq) > 0 and len(compass_X_seq) > 0:
            # Make sure we use same length sequences
            min_len = min(len(gyro_X_seq), len(compass_X_seq))
            fusion_pred_scaled = self.fusion_model.predict([gyro_X_seq[:min_len], compass_X_seq[:min_len]])
            fusion_pred = self.scalers['gyro_y'].inverse_transform(fusion_pred_scaled)
            fusion_predictions[self.window_size:self.window_size+len(fusion_pred)] = fusion_pred.flatten()
        
        # Fill the initial window_size values with first predictions
        if len(gyro_predictions) > self.window_size:
            gyro_predictions[:self.window_size] = gyro_predictions[self.window_size]
            compass_predictions[:self.window_size] = compass_predictions[self.window_size]
            fusion_predictions[:self.window_size] = fusion_predictions[self.window_size]
        
        # Normalize to 0-360 degrees
        gyro_predictions = (gyro_predictions + 360) % 360
        compass_predictions = (compass_predictions + 360) % 360
        fusion_predictions = (fusion_predictions + 360) % 360
        
        return {
            'gyro': gyro_predictions,
            'compass': compass_predictions,
            'fusion': fusion_predictions if self.fusion_model else None
        }
    
    def save_models(self, save_dir):
        """Save trained models to disk"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if self.gyro_model:
            self.gyro_model.save(os.path.join(save_dir, 'advanced_gyro_model.keras'))
        
        if self.compass_model:
            self.compass_model.save(os.path.join(save_dir, 'advanced_compass_model.keras'))
        
        if self.fusion_model:
            self.fusion_model.save(os.path.join(save_dir, 'advanced_fusion_model.keras'))
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir):
        """Load trained models from disk"""
        gyro_model_path = os.path.join(save_dir, 'advanced_gyro_model.keras')
        compass_model_path = os.path.join(save_dir, 'advanced_compass_model.keras')
        fusion_model_path = os.path.join(save_dir, 'advanced_fusion_model.keras')
        
        if os.path.exists(gyro_model_path):
            self.gyro_model = tf.keras.models.load_model(
                gyro_model_path, 
                custom_objects={'TimeAttention': TimeAttention}
            )
            print(f"Loaded gyro model from {gyro_model_path}")
        
        if os.path.exists(compass_model_path):
            self.compass_model = tf.keras.models.load_model(
                compass_model_path,
                custom_objects={'TimeAttention': TimeAttention}
            )
            print(f"Loaded compass model from {compass_model_path}")
        
        if os.path.exists(fusion_model_path):
            self.fusion_model = tf.keras.models.load_model(
                fusion_model_path,
                custom_objects={'TimeAttention': TimeAttention}
            )
            print(f"Loaded fusion model from {fusion_model_path}")

# Function to calculate positions from headings
def calculate_positions(data, heading_column, step_length=0.66, initial_position=(0, 0)):
    """Calculate positions using step detection and heading data"""
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