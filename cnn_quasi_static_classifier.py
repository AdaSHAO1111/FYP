#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
import seaborn as sns
from tqdm import tqdm

class CNNQuasiStaticClassifier:
    """CNN-based model for direct quasi-static state classification"""
    
    def __init__(self, window_size=100, use_magnetometer=True, use_gyro=True, use_context=False, 
                 use_attention=True, use_accelerometer=False, ensemble_mode=False):
        """
        Initialize the CNN-based quasi-static classifier.
        
        Parameters:
        -----------
        window_size : int
            The size of the sliding window for creating input sequences
        use_magnetometer : bool
            Whether to use magnetometer data as input
        use_gyro : bool
            Whether to use gyroscope data as input
        use_context : bool
            Whether to use contextual features (step count, floor, etc.)
        use_attention : bool
            Whether to use attention mechanism in the model
        use_accelerometer : bool
            Whether to use accelerometer data as input
        ensemble_mode : bool
            Whether to use an ensemble of multiple models
        """
        self.window_size = window_size
        self.use_magnetometer = use_magnetometer
        self.use_gyro = use_gyro
        self.use_context = use_context
        self.use_attention = use_attention
        self.use_accelerometer = use_accelerometer
        self.ensemble_mode = ensemble_mode
        
        self.model = None
        self.ensemble_models = []
        self.input_dim = None
        self.history = None
        self.feature_names = []
    
    def _attention_layer(self, inputs):
        """
        Implements a simple attention mechanism
        
        Parameters:
        -----------
        inputs : tensor
            Input tensor
            
        Returns:
        --------
        tensor
            Attention-weighted tensor
        """
        from tensorflow.keras.layers import Dense, Multiply, Softmax, Reshape, Permute
        
        # Compute attention scores
        attention = Permute((2, 1))(inputs)  # Swap time and feature dimensions
        attention = Dense(inputs.shape[1], activation='tanh')(attention)
        attention = Permute((2, 1))(attention)  # Swap back
        
        # Apply softmax to get attention weights
        attention_weights = Dense(1)(attention)
        attention_weights = Softmax(axis=1)(attention_weights)
        
        # Apply attention weights to input
        weighted_input = Multiply()([inputs, attention_weights])
        
        return weighted_input
    
    def _build_model(self):
        """Build the CNN model architecture with optional attention mechanism"""
        from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
        from tensorflow.keras.layers import Concatenate, GlobalAveragePooling1D, SpatialDropout1D
        from tensorflow.keras.models import Model
        
        # Input layer
        input_tensor = Input(shape=(self.window_size, self.input_dim))
        
        # First Conv1D block
        x = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.1)(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # Second Conv1D block
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.1)(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # Third Conv1D block
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        
        # Apply attention if requested
        if self.use_attention:
            x = self._attention_layer(x)
        
        # Global pooling and flatten
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = MaxPooling1D(pool_size=x.shape[1])(x)
        max_pool = Flatten()(max_pool)
        
        # Combine pooling strategies
        x = Concatenate()([avg_pool, max_pool])
        
        # Dense layers
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Output layer - binary classification
        output = Dense(1, activation='sigmoid')(x)
        
        # Compile model
        model = Model(inputs=input_tensor, outputs=output)
        model.compile(loss='binary_crossentropy', 
                      optimizer='adam',
                      metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        
        return model
    
    def _create_sequences(self, data, labels=None):
        """
        Create sequences from the data for training or prediction.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The sensor data
        labels : pd.Series, optional
            The ground truth labels (1 for quasi-static, 0 for not)
            
        Returns:
        --------
        X : np.array
            The input sequences
        y : np.array
            The labels corresponding to each sequence (only if labels provided)
        """
        sequences = []
        sequence_labels = []
        
        # Determine which features to use
        features = []
        
        # Add magnetometer features if requested
        if self.use_magnetometer:
            mag_features = [col for col in data.columns if 'compass' in col.lower() or 'magnetic' in col.lower()]
            if mag_features:
                features.extend(mag_features)
            else:
                # Fallback to known column names
                mag_candidates = ['compass', 'Magnetic_Field_Magnitude']
                features.extend([col for col in mag_candidates if col in data.columns])
        
        # Add gyroscope features if requested
        if self.use_gyro:
            gyro_features = [col for col in data.columns if 'gyro' in col.lower()]
            if gyro_features:
                features.extend(gyro_features)
            else:
                # Fallback to known column names
                gyro_candidates = ['gyroSumFromstart0', 'GyroStartByGroundTruth']
                features.extend([col for col in gyro_candidates if col in data.columns])
        
        # Add accelerometer features if requested
        if self.use_accelerometer:
            accel_features = [col for col in data.columns if 'accel' in col.lower()]
            if accel_features:
                features.extend(accel_features)
        
        # Add context features if requested
        if self.use_context:
            if 'step' in data.columns:
                features.append('step')
            if 'value_4' in data.columns:  # Floor
                features.append('value_4')
            
            # Add derived features if available
            context_candidates = ['velocity', 'speed', 'distance_traveled']
            features.extend([col for col in context_candidates if col in data.columns])
        
        # Check if we have any features
        if not features:
            raise ValueError("No valid features found for the selected sensors")
        
        # Remove duplicates
        features = list(set(features))
        
        self.feature_names = features
        self.input_dim = len(features)
        
        # Create sequences
        for i in range(len(data) - self.window_size):
            try:
                seq = data.iloc[i:i+self.window_size][features].values
                sequences.append(seq)
                
                if labels is not None:
                    # Use the label of the last timestep in the window
                    sequence_labels.append(labels.iloc[i+self.window_size-1])
            except Exception as e:
                print(f"Error creating sequence at index {i}: {e}")
                continue
        
        if not sequences:
            raise ValueError("Failed to create any valid sequences")
        
        X = np.array(sequences)
        
        if labels is not None:
            y = np.array(sequence_labels)
            return X, y
        else:
            return X
    
    def fit(self, data, labels, epochs=50, batch_size=32, validation_split=0.2, verbose=1):
        """
        Train the CNN model.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The sensor data
        labels : pd.Series
            The ground truth labels (1 for quasi-static, 0 for not)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        validation_split : float
            Fraction of data to use for validation
        verbose : int
            Verbosity level for training
            
        Returns:
        --------
        self : object
            The trained model
        """
        # Create sequences
        X, y = self._create_sequences(data, labels)
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        # Build model
        self.model = self._build_model()
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint('best_cnn_model.h5', save_best_only=True)
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self
    
    def predict(self, data, threshold=0.5):
        """
        Predict quasi-static states.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The sensor data
        threshold : float
            Threshold for binary classification
            
        Returns:
        --------
        predictions : np.array
            The binary predictions
        probabilities : np.array
            The prediction probabilities
        """
        # Create sequences
        X = self._create_sequences(data)
        
        # Get predictions
        probabilities = self.model.predict(X).flatten()
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities
    
    def evaluate(self, data, labels):
        """
        Evaluate the model.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The sensor data
        labels : pd.Series
            The ground truth labels
            
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        # Create sequences
        X, y = self._create_sequences(data, labels)
        
        # Get predictions
        y_pred_proba = self.model.predict(X).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_true': y
        }
    
    def plot_training_history(self):
        """Plot the training history"""
        if self.history is None:
            print("No training history available")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


def generate_quasi_static_labels(data, stability_threshold=5.0, window_size=100):
    """
    Generate quasi-static labels for sensor data using traditional method.
    This function is used to create labeled data for training the CNN model.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The sensor data
    stability_threshold : float
        The threshold for determining quasi-static periods
    window_size : int
        The sliding window size for calculating variance
        
    Returns:
    --------
    labels : pd.Series
        Binary labels (1 for quasi-static, 0 for not)
    """
    # Extract compass headings
    compass_headings = data['compass'].values
    
    # Initialize labels
    labels = np.zeros(len(compass_headings))
    
    # Calculate variance using sliding window
    for i in range(len(compass_headings) - window_size + 1):
        window = compass_headings[i:i+window_size]
        variance = np.var(window)
        
        # Last element of the window gets the label
        labels[i+window_size-1] = 1 if variance < stability_threshold else 0
    
    # Initialize first window_size-1 elements (cannot be labeled accurately)
    labels[:window_size-1] = 0
    
    return pd.Series(labels)


def generate_synthetic_training_data(n_points=5000, n_quasi_static_regions=10):
    """
    Generate synthetic data for training the CNN model.
    
    Parameters:
    -----------
    n_points : int
        Number of data points to generate
    n_quasi_static_regions : int
        Number of quasi-static regions to create
        
    Returns:
    --------
    data : pd.DataFrame
        The synthetic sensor data
    labels : pd.Series
        The ground truth labels
    """
    # Create timestamps
    timestamps = np.arange(0, n_points * 100, 100)
    
    # Create compass headings with some noise
    ground_truth_headings = np.zeros(n_points)
    
    # Add some turns (change heading gradually)
    for i in range(1, 5):
        start_idx = i * n_points // 5
        end_idx = (i + 1) * n_points // 5
        ground_truth_headings[start_idx:end_idx] = (i * 90) % 360
    
    # Add noise to compass readings
    compass_headings = ground_truth_headings + np.random.normal(0, 10, n_points)
    compass_headings = compass_headings % 360
    
    # Create quasi-static periods (lower noise)
    is_static = np.zeros(n_points, dtype=bool)
    
    # Randomly place quasi-static regions
    static_region_length = 100  # Points per static region
    for i in range(n_quasi_static_regions):
        start_idx = np.random.randint(0, n_points - static_region_length)
        is_static[start_idx:start_idx+static_region_length] = True
    
    # Reduce noise in static periods
    compass_headings[is_static] = ground_truth_headings[is_static] + np.random.normal(0, 1, np.sum(is_static))
    
    # Create steps
    steps = np.floor(np.arange(0, n_points) / 5)  # 5 data points per step
    
    # Create gyro data (cumulative sum with drift)
    gyro_readings = np.zeros(n_points)
    for i in range(1, n_points):
        # More stable in quasi-static periods
        if is_static[i]:
            gyro_readings[i] = np.random.normal(0, 0.01)
        else:
            gyro_readings[i] = np.random.normal(0, 0.1)
    
    # Cumulative gyro readings
    gyro_cum = np.cumsum(gyro_readings)
    
    # Simulate gyro drift
    drift = np.cumsum(np.random.normal(0, 0.01, n_points))
    
    # Create East and North coordinates
    east = np.zeros(n_points)
    north = np.zeros(n_points)
    
    step_length = 0.66  # meters per step
    for i in range(1, n_points):
        if steps[i] > steps[i-1]:  # New step
            heading_rad = np.radians(ground_truth_headings[i])
            east[i] = east[i-1] + np.sin(heading_rad) * step_length
            north[i] = north[i-1] + np.cos(heading_rad) * step_length
        else:
            east[i] = east[i-1]
            north[i] = north[i-1]
    
    # Create floor information (mostly constant with occasional changes)
    floor = np.ones(n_points)
    for i in range(1, 5):
        floor[i * n_points // 5:] += 1
    
    # Create compass data DataFrame
    data = pd.DataFrame({
        'Timestamp_(ms)': timestamps,
        'Type': 'Compass',
        'step': steps,
        'compass': compass_headings,
        'GroundTruthHeadingComputed': ground_truth_headings,
        'Magnetic_Field_Magnitude': np.random.uniform(20, 50, n_points),
        'gyroSumFromstart0': gyro_cum,
        'GyroStartByGroundTruth': (ground_truth_headings[0] + gyro_cum) % 360,
        'value_4': floor,  # Floor
        'GroundTruth_X': east,  # East
        'GroundTruth_Y': north   # North
    })
    
    # Generate labels directly from is_static
    labels = pd.Series(is_static.astype(int))
    
    return data, labels


def main():
    """Main function to run CNN-based quasi-static detection"""
    print("CNN-based Quasi-Static State Detection")
    print("=" * 40)
    
    print("\n1. Generating synthetic training data...")
    data, labels = generate_synthetic_training_data(n_points=5000, n_quasi_static_regions=10)
    
    # Split data into train and test sets
    train_idx = int(len(data) * 0.8)
    train_data = data.iloc[:train_idx]
    test_data = data.iloc[train_idx:]
    train_labels = labels.iloc[:train_idx]
    test_labels = labels.iloc[train_idx:]
    
    print(f"Generated {len(data)} data points with {labels.sum()} quasi-static points")
    print(f"Training set: {len(train_data)} points, Test set: {len(test_data)} points")
    
    # Visualize a sample of the data
    plt.figure(figsize=(15, 10))
    
    # Plot compass headings and quasi-static periods
    plt.subplot(3, 1, 1)
    plt.plot(data['Timestamp_(ms)'][:1000], data['compass'][:1000], label='Compass Headings')
    plt.scatter(data['Timestamp_(ms)'][:1000][labels[:1000] == 1], 
                data['compass'][:1000][labels[:1000] == 1],
                color='red', label='Quasi-Static', alpha=0.5)
    plt.title('Compass Headings with Quasi-Static Periods')
    plt.legend()
    
    # Plot gyro readings
    plt.subplot(3, 1, 2)
    plt.plot(data['Timestamp_(ms)'][:1000], data['gyroSumFromstart0'][:1000], label='Gyro Cumulative')
    plt.title('Gyroscope Readings')
    plt.legend()
    
    # Plot step count
    plt.subplot(3, 1, 3)
    plt.plot(data['Timestamp_(ms)'][:1000], data['step'][:1000], label='Step Count')
    plt.scatter(data['Timestamp_(ms)'][:1000][labels[:1000] == 1], 
                data['step'][:1000][labels[:1000] == 1],
                color='red', label='Quasi-Static', alpha=0.5)
    plt.title('Step Count with Quasi-Static Periods')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n2. Training CNN model...")
    # Create and train the model
    model = CNNQuasiStaticClassifier(
        window_size=50,
        use_magnetometer=True,
        use_gyro=True,
        use_context=True
    )
    
    # Train the model
    model.fit(train_data, train_labels, epochs=20, batch_size=32, verbose=1)
    
    # Plot training history
    model.plot_training_history()
    
    print("\n3. Evaluating model on test data...")
    evaluation = model.evaluate(test_data, test_labels)
    
    print(f"Test Accuracy: {evaluation['accuracy']:.4f}")
    print("\nClassification Report:")
    for label, metrics in evaluation['classification_report'].items():
        if label in ['0', '1']:
            state = "Non-Static" if label == '0' else "Quasi-Static"
            print(f"{state}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(evaluation['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Static', 'Quasi-Static'],
                yticklabels=['Non-Static', 'Quasi-Static'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    print("\n4. Applying model to detect quasi-static periods...")
    # Apply the model to classify the entire dataset
    predictions, probabilities = model.predict(data)
    
    # Plot the results
    plt.figure(figsize=(15, 8))
    
    # Plot ground truth labels vs predictions
    plt.subplot(2, 1, 1)
    plt.plot(data['Timestamp_(ms)'][:1000], labels[:1000], 'b-', label='Ground Truth')
    plt.plot(data['Timestamp_(ms)'][:1000], predictions[:950], 'r--', label='CNN Predictions')
    plt.title('Ground Truth vs CNN Predictions')
    plt.legend()
    
    # Plot prediction probabilities
    plt.subplot(2, 1, 2)
    plt.plot(data['Timestamp_(ms)'][:1000], probabilities[:950], 'g-')
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title('Prediction Probabilities')
    plt.ylabel('Probability of Quasi-Static')
    
    plt.tight_layout()
    plt.show()
    
    print("\n5. Evaluating traditional vs CNN-based detection...")
    # Generate labels using traditional method
    traditional_labels = generate_quasi_static_labels(
        data, 
        stability_threshold=5.0, 
        window_size=50
    )
    
    # Calculate agreement between methods
    agreement = np.mean(predictions[:len(traditional_labels)] == traditional_labels)
    print(f"Agreement between traditional and CNN methods: {agreement:.4f}")
    
    # Plot the results
    plt.figure(figsize=(15, 8))
    
    # Comparison of detection methods
    plt.subplot(3, 1, 1)
    plt.plot(data['compass'][:1000], label='Compass Headings')
    plt.title('Compass Headings')
    
    plt.subplot(3, 1, 2)
    plt.plot(traditional_labels[:1000], label='Traditional Method')
    plt.title('Traditional Quasi-Static Detection')
    plt.ylabel('Is Quasi-Static')
    
    plt.subplot(3, 1, 3)
    plt.plot(predictions[:950], label='CNN Method')
    plt.title('CNN-based Quasi-Static Detection')
    plt.ylabel('Is Quasi-Static')
    
    plt.tight_layout()
    plt.show()
    
    # Plot the position trace with quasi-static points highlighted
    plt.figure(figsize=(10, 10))
    plt.plot(data['GroundTruth_X'], data['GroundTruth_Y'], 'b-', alpha=0.5, label='Full Trajectory')
    
    # Get indices where predictions are 1 (quasi-static)
    static_indices = np.where(predictions[:len(data)])[0]
    
    if len(static_indices) > 0:
        plt.scatter(
            data['GroundTruth_X'].iloc[static_indices], 
            data['GroundTruth_Y'].iloc[static_indices],
            color='red', s=50, label='CNN Quasi-Static Points'
        )
    
    plt.axis('equal')
    plt.title('Trajectory with CNN-Detected Quasi-Static Points')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\n6. Summary and Conclusion")
    print("-" * 60)
    print("CNN-based quasi-static detection can learn complex patterns in sensor data")
    print(f"Test accuracy: {evaluation['accuracy']:.4f}")
    print(f"Number of quasi-static periods detected: {np.sum(np.diff(predictions) > 0)}")
    print("-" * 60)
    print("The CNN approach can adapt to different scenarios and sensor behaviors")
    print("without requiring manual parameter tuning, making it suitable for")
    print("diverse indoor environments and movement patterns.")


if __name__ == "__main__":
    main() 