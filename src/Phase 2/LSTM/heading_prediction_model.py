import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

class HeadingPredictor:
    def __init__(self, window_size=30):
        """
        Initialize the Heading Predictor with LSTM model
        
        Parameters:
        -----------
        window_size : int
            The number of time steps to use as input for prediction
        """
        self.window_size = window_size
        self.gyro_model = None
        self.compass_model = None
        self.gyro_scaler_X = MinMaxScaler()
        self.gyro_scaler_y = MinMaxScaler()
        self.compass_scaler_X = MinMaxScaler()
        self.compass_scaler_y = MinMaxScaler()
        
    def create_sequences(self, X, y, window_size):
        """
        Create sequences for LSTM input
        
        Parameters:
        -----------
        X : array-like
            Input features
        y : array-like
            Target values
        window_size : int
            Sequence length
            
        Returns:
        --------
        X_seq : array
            Sequences of input features
        y_seq : array
            Target values corresponding to sequences
        """
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size):
            X_seq.append(X[i:i + window_size])
            y_seq.append(y[i + window_size])
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape):
        """
        Build LSTM model for heading prediction
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (window_size, n_features)
            
        Returns:
        --------
        model : Keras model
            Compiled LSTM model
        """
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train_gyro_model(self, gyro_data, ground_truth_heading, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train LSTM model for gyro heading prediction
        
        Parameters:
        -----------
        gyro_data : DataFrame
            Gyro sensor data
        ground_truth_heading : Series
            Ground truth heading values
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        validation_split : float
            Fraction of data to use for validation
            
        Returns:
        --------
        history : History object
            Training history
        """
        # Prepare features - we'll use axisZAngle and gyroSumFromstart0
        X = gyro_data[['axisZAngle', 'gyroSumFromstart0']].values
        y = ground_truth_heading.values.reshape(-1, 1)
        
        # Scale the data
        X_scaled = self.gyro_scaler_X.fit_transform(X)
        y_scaled = self.gyro_scaler_y.fit_transform(y)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, self.window_size)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=validation_split, shuffle=False)
        
        # Build and train the model
        self.gyro_model = self.build_model((self.window_size, X.shape[1]))
        history = self.gyro_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        return history
    
    def train_compass_model(self, compass_data, ground_truth_heading, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train LSTM model for compass heading prediction
        
        Parameters:
        -----------
        compass_data : DataFrame
            Compass sensor data
        ground_truth_heading : Series
            Ground truth heading values
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        validation_split : float
            Fraction of data to use for validation
            
        Returns:
        --------
        history : History object
            Training history
        """
        # Prepare features - we'll use Magnetic_Field_Magnitude and gyroSumFromstart0
        X = compass_data[['Magnetic_Field_Magnitude', 'gyroSumFromstart0']].values
        y = ground_truth_heading.values.reshape(-1, 1)
        
        # Scale the data
        X_scaled = self.compass_scaler_X.fit_transform(X)
        y_scaled = self.compass_scaler_y.fit_transform(y)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, self.window_size)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=validation_split, shuffle=False)
        
        # Build and train the model
        self.compass_model = self.build_model((self.window_size, X.shape[1]))
        history = self.compass_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        return history
    
    def predict_gyro_heading(self, gyro_data):
        """
        Predict heading using gyro data
        
        Parameters:
        -----------
        gyro_data : DataFrame
            Gyro sensor data
            
        Returns:
        --------
        predicted_heading : array
            Predicted heading values
        """
        if self.gyro_model is None:
            raise ValueError("Gyro model has not been trained yet.")
        
        # Prepare features
        X = gyro_data[['axisZAngle', 'gyroSumFromstart0']].values
        X_scaled = self.gyro_scaler_X.transform(X)
        
        # Create sequences
        X_seq = []
        for i in range(len(X_scaled) - self.window_size):
            X_seq.append(X_scaled[i:i + self.window_size])
        
        # Handle the initial window_size elements (where we can't make a full sequence)
        pad_predictions = np.zeros(self.window_size)
        
        # Predict using the model
        if len(X_seq) > 0:
            X_seq = np.array(X_seq)
            y_pred_scaled = self.gyro_model.predict(X_seq)
            y_pred = self.gyro_scaler_y.inverse_transform(y_pred_scaled)
            
            # Combine padding and predictions
            predictions = np.concatenate([pad_predictions, y_pred.flatten()])
        else:
            predictions = pad_predictions
            
        # Ensure we have the same number of predictions as input data
        if len(predictions) < len(gyro_data):
            # Pad beginning with the first prediction
            predictions = np.concatenate([np.ones(len(gyro_data) - len(predictions)) * predictions[0], predictions])
        
        # Normalize to 0-360 degrees
        predictions = (predictions + 360) % 360
        
        return predictions
    
    def predict_compass_heading(self, compass_data):
        """
        Predict heading using compass data
        
        Parameters:
        -----------
        compass_data : DataFrame
            Compass sensor data
            
        Returns:
        --------
        predicted_heading : array
            Predicted heading values
        """
        if self.compass_model is None:
            raise ValueError("Compass model has not been trained yet.")
        
        # Prepare features
        X = compass_data[['Magnetic_Field_Magnitude', 'gyroSumFromstart0']].values
        X_scaled = self.compass_scaler_X.transform(X)
        
        # Create sequences
        X_seq = []
        for i in range(len(X_scaled) - self.window_size):
            X_seq.append(X_scaled[i:i + self.window_size])
        
        # Handle the initial window_size elements (where we can't make a full sequence)
        pad_predictions = np.zeros(self.window_size)
        
        # Predict using the model
        if len(X_seq) > 0:
            X_seq = np.array(X_seq)
            y_pred_scaled = self.compass_model.predict(X_seq)
            y_pred = self.compass_scaler_y.inverse_transform(y_pred_scaled)
            
            # Combine padding and predictions
            predictions = np.concatenate([pad_predictions, y_pred.flatten()])
        else:
            predictions = pad_predictions
            
        # Ensure we have the same number of predictions as input data
        if len(predictions) < len(compass_data):
            # Pad beginning with the first prediction
            predictions = np.concatenate([np.ones(len(compass_data) - len(predictions)) * predictions[0], predictions])
        
        # Normalize to 0-360 degrees
        predictions = (predictions + 360) % 360
        
        return predictions
    
    def evaluate_models(self, gyro_data, compass_data, ground_truth_heading_gyro, ground_truth_heading_compass):
        """
        Evaluate models by comparing with ground truth heading
        
        Parameters:
        -----------
        gyro_data : DataFrame
            Gyro sensor data
        compass_data : DataFrame
            Compass sensor data
        ground_truth_heading_gyro : Series
            Ground truth heading values corresponding to gyro timestamps
        ground_truth_heading_compass : Series
            Ground truth heading values corresponding to compass timestamps
            
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        # Predict headings
        gyro_pred = self.predict_gyro_heading(gyro_data)
        compass_pred = self.predict_compass_heading(compass_data)
        
        # Traditional method - use the values already in the dataframes
        gyro_traditional = gyro_data['GyroStartByGroundTruth'].values
        compass_traditional = compass_data['compass'].values
        
        # Remove NaN values for evaluation
        gt_gyro = ground_truth_heading_gyro.dropna().values
        gt_compass = ground_truth_heading_compass.dropna().values
        
        # Ensure all arrays have matching lengths by using min length
        min_len_gyro = min(len(gt_gyro), len(gyro_pred), len(gyro_traditional))
        min_len_compass = min(len(gt_compass), len(compass_pred), len(compass_traditional))
        
        # Use the last part of each array for comparison
        gt_gyro = gt_gyro[-min_len_gyro:]
        gyro_pred = gyro_pred[-min_len_gyro:]
        gyro_traditional = gyro_traditional[-min_len_gyro:]
        
        gt_compass = gt_compass[-min_len_compass:]
        compass_pred = compass_pred[-min_len_compass:]
        compass_traditional = compass_traditional[-min_len_compass:]
        
        # Calculate metrics for gyro
        gyro_ml_mae = mean_absolute_error(gt_gyro, gyro_pred)
        gyro_traditional_mae = mean_absolute_error(gt_gyro, gyro_traditional)
        
        gyro_ml_rmse = np.sqrt(mean_squared_error(gt_gyro, gyro_pred))
        gyro_traditional_rmse = np.sqrt(mean_squared_error(gt_gyro, gyro_traditional))
        
        # Calculate metrics for compass
        compass_ml_mae = mean_absolute_error(gt_compass, compass_pred)
        compass_traditional_mae = mean_absolute_error(gt_compass, compass_traditional)
        
        compass_ml_rmse = np.sqrt(mean_squared_error(gt_compass, compass_pred))
        compass_traditional_rmse = np.sqrt(mean_squared_error(gt_compass, compass_traditional))
        
        metrics = {
            'gyro_ml_mae': gyro_ml_mae,
            'gyro_traditional_mae': gyro_traditional_mae,
            'gyro_ml_rmse': gyro_ml_rmse,
            'gyro_traditional_rmse': gyro_traditional_rmse,
            'compass_ml_mae': compass_ml_mae,
            'compass_traditional_mae': compass_traditional_mae,
            'compass_ml_rmse': compass_ml_rmse,
            'compass_traditional_rmse': compass_traditional_rmse
        }
        
        return metrics
    
    def save_models(self, output_dir):
        """
        Save trained models and scalers
        
        Parameters:
        -----------
        output_dir : str
            Directory to save models and scalers
        """
        # Save models
        if self.gyro_model is not None:
            self.gyro_model.save(os.path.join(output_dir, 'gyro_heading_lstm_model.keras'))
            
        if self.compass_model is not None:
            self.compass_model.save(os.path.join(output_dir, 'compass_heading_lstm_model.keras'))
        
        # Save scalers
        if hasattr(self, 'gyro_scaler_X') and hasattr(self, 'gyro_scaler_y'):
            import joblib
            joblib.dump(self.gyro_scaler_X, os.path.join(output_dir, 'gyro_scaler_X.pkl'))
            joblib.dump(self.gyro_scaler_y, os.path.join(output_dir, 'gyro_scaler_y.pkl'))
            
        if hasattr(self, 'compass_scaler_X') and hasattr(self, 'compass_scaler_y'):
            import joblib
            joblib.dump(self.compass_scaler_X, os.path.join(output_dir, 'compass_scaler_X.pkl'))
            joblib.dump(self.compass_scaler_y, os.path.join(output_dir, 'compass_scaler_y.pkl'))
        
        # Save window size
        with open(os.path.join(output_dir, 'lstm_config.txt'), 'w') as f:
            f.write(f"window_size={self.window_size}")
        
        print(f"Models and scalers saved to {output_dir}")
    
    def load_models(self, output_dir):
        """
        Load trained models and scalers
        
        Parameters:
        -----------
        output_dir : str
            Directory containing saved models and scalers
        """
        # Load models
        gyro_model_path = os.path.join(output_dir, 'gyro_heading_lstm_model.keras')
        compass_model_path = os.path.join(output_dir, 'compass_heading_lstm_model.keras')
        
        if os.path.exists(gyro_model_path):
            self.gyro_model = tf.keras.models.load_model(gyro_model_path)
            print(f"Loaded gyro model from {gyro_model_path}")
        
        if os.path.exists(compass_model_path):
            self.compass_model = tf.keras.models.load_model(compass_model_path)
            print(f"Loaded compass model from {compass_model_path}")
        
        # Load scalers
        try:
            import joblib
            
            gyro_scaler_X_path = os.path.join(output_dir, 'gyro_scaler_X.pkl')
            gyro_scaler_y_path = os.path.join(output_dir, 'gyro_scaler_y.pkl')
            compass_scaler_X_path = os.path.join(output_dir, 'compass_scaler_X.pkl')
            compass_scaler_y_path = os.path.join(output_dir, 'compass_scaler_y.pkl')
            
            if os.path.exists(gyro_scaler_X_path):
                self.gyro_scaler_X = joblib.load(gyro_scaler_X_path)
                print(f"Loaded gyro_scaler_X from {gyro_scaler_X_path}")
                
            if os.path.exists(gyro_scaler_y_path):
                self.gyro_scaler_y = joblib.load(gyro_scaler_y_path)
                print(f"Loaded gyro_scaler_y from {gyro_scaler_y_path}")
                
            if os.path.exists(compass_scaler_X_path):
                self.compass_scaler_X = joblib.load(compass_scaler_X_path)
                print(f"Loaded compass_scaler_X from {compass_scaler_X_path}")
                
            if os.path.exists(compass_scaler_y_path):
                self.compass_scaler_y = joblib.load(compass_scaler_y_path)
                print(f"Loaded compass_scaler_y from {compass_scaler_y_path}")
            
            # Load window size
            config_path = os.path.join(output_dir, 'lstm_config.txt')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    for line in f:
                        if line.startswith('window_size='):
                            self.window_size = int(line.split('=')[1])
                            print(f"Loaded window_size: {self.window_size}")
                            
        except Exception as e:
            print(f"Error loading scalers: {e}") 