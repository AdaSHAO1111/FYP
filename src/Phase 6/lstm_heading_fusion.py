import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from prepare_fusion_data import prepare_dataset
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time

# Set paths
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 6/LSTM_fusion'
model_dir = os.path.join(output_dir, 'models')
os.makedirs(model_dir, exist_ok=True)

# Parameters
SEQUENCE_LENGTH = 10  # Number of time steps to include in each sequence
TEST_SIZE = 0.2       # Proportion of data to use for testing
VALIDATION_SPLIT = 0.2  # Proportion of training data to use for validation
RANDOM_STATE = 42     # Random seed for reproducibility
BATCH_SIZE = 32       # Batch size for training
EPOCHS = 200          # Maximum number of epochs to train for
LEARNING_RATE = 0.001  # Learning rate for the optimizer

def angle_difference(y_true, y_pred):
    """
    Custom loss function to handle the circular nature of angles
    Calculates the smallest angle between two angles in radians
    """
    # Extract sin and cos components
    sin_true, cos_true = y_true[:, 0], y_true[:, 1]
    sin_pred, cos_pred = y_pred[:, 0], y_pred[:, 1]
    
    # Calculate angular error using dot product and cross product
    dot_product = sin_true * sin_pred + cos_true * cos_pred
    cross_product = cos_true * sin_pred - sin_true * cos_pred
    
    # Return the angle between vectors
    return tf.reduce_mean(tf.abs(tf.atan2(cross_product, dot_product)))

def create_sequences(data, sequence_length):
    """Create sequences for LSTM input"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # Input features: heading_compass_sin, heading_compass_cos, heading_gyro_sin, heading_gyro_cos
        features = data.iloc[i:i+sequence_length, [data.columns.get_loc(col) for col in 
                                                 ['heading_compass_sin', 'heading_compass_cos', 
                                                  'heading_gyro_sin', 'heading_gyro_cos']]].values
        
        # Output: gt_heading_sin, gt_heading_cos (for the last timestep in the sequence)
        target = data.iloc[i+sequence_length, [data.columns.get_loc(col) for col in 
                                             ['gt_heading_sin', 'gt_heading_cos']]].values
        
        X.append(features)
        y.append(target)
    
    return np.array(X), np.array(y)

def build_lstm_model(sequence_length, n_features):
    """Build LSTM model for heading fusion"""
    model = Sequential([
        # Bidirectional LSTM layers
        Bidirectional(LSTM(64, return_sequences=True), 
                      input_shape=(sequence_length, n_features)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        # Dense layers
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        # Output layer: sin and cos components of the heading
        Dense(2)
    ])
    
    # Compile model with custom loss
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss=angle_difference, optimizer=optimizer)
    
    return model

def train_lstm_model(dataset_path=None):
    """Train the LSTM model for heading fusion"""
    print("\n=== Training LSTM Heading Fusion Model ===\n")
    
    # Load or prepare dataset
    if dataset_path and os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}...")
        data = pd.read_csv(dataset_path)
    else:
        print("Preparing dataset...")
        data = prepare_dataset()
    
    print(f"Dataset shape: {data.shape}")
    
    # Create sequences for LSTM
    print("Creating sequences for LSTM input...")
    X, y = create_sequences(data, SEQUENCE_LENGTH)
    print(f"Input shape: {X.shape}, Output shape: {y.shape}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Build model
    print("Building LSTM model...")
    model = build_lstm_model(SEQUENCE_LENGTH, X.shape[2])
    model.summary()
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, 'lstm_heading_fusion.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\nTraining model...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"Model training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    
    # Generate predictions
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    
    # Convert predictions back to angles
    pred_angles = np.arctan2(y_pred[:, 0], y_pred[:, 1])
    true_angles = np.arctan2(y_test[:, 0], y_test[:, 1])
    
    # Calculate error metrics
    angle_errors = np.abs(np.arctan2(np.sin(pred_angles - true_angles), np.cos(pred_angles - true_angles)))
    mean_error = np.mean(angle_errors)
    median_error = np.median(angle_errors)
    max_error = np.max(angle_errors)
    
    print(f"Mean angle error: {mean_error:.4f} radians ({np.degrees(mean_error):.2f} degrees)")
    print(f"Median angle error: {median_error:.4f} radians ({np.degrees(median_error):.2f} degrees)")
    print(f"Max angle error: {max_error:.4f} radians ({np.degrees(max_error):.2f} degrees)")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Plot prediction samples
    sample_indices = np.random.choice(len(pred_angles), min(100, len(pred_angles)), replace=False)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(sample_indices, np.degrees(true_angles[sample_indices]), 
                label='Ground Truth', alpha=0.7, color='green')
    plt.scatter(sample_indices, np.degrees(pred_angles[sample_indices]), 
                label='LSTM Prediction', alpha=0.7, color='blue')
    plt.title('Heading Prediction Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Heading (degrees)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'prediction_samples.png'))
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(np.degrees(angle_errors), bins=50, alpha=0.7)
    plt.axvline(np.degrees(mean_error), color='r', linestyle='--', 
                label=f'Mean Error: {np.degrees(mean_error):.2f}°')
    plt.axvline(np.degrees(median_error), color='g', linestyle='--', 
                label=f'Median Error: {np.degrees(median_error):.2f}°')
    plt.title('Distribution of Heading Errors')
    plt.xlabel('Error (degrees)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    
    # Save results to CSV
    results = {
        'Mean Error (rad)': [mean_error],
        'Median Error (rad)': [median_error],
        'Max Error (rad)': [max_error],
        'Mean Error (deg)': [np.degrees(mean_error)],
        'Median Error (deg)': [np.degrees(median_error)],
        'Max Error (deg)': [np.degrees(max_error)],
        'Training Time (s)': [training_time],
        'Sequence Length': [SEQUENCE_LENGTH],
        'Test Loss': [test_loss]
    }
    pd.DataFrame(results).to_csv(os.path.join(output_dir, 'model_evaluation.csv'), index=False)
    
    print(f"\nResults saved to: {output_dir}")
    
    return model, history, (true_angles, pred_angles)

def apply_fusion_to_trajectory(model=None):
    """Apply the trained LSTM model to generate a fused trajectory"""
    print("\n=== Applying LSTM Fusion to Generate Trajectory ===\n")
    
    # Load or prepare dataset
    dataset_path = os.path.join(output_dir, 'fusion_dataset.csv')
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}...")
        data = pd.read_csv(dataset_path)
    else:
        print("Preparing dataset...")
        data = prepare_dataset()
    
    # Load the model if not provided
    if model is None:
        model_path = os.path.join(model_dir, 'lstm_heading_fusion.h5')
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            model = load_model(model_path, custom_objects={'angle_difference': angle_difference})
        else:
            print("No trained model found. Please train the model first.")
            return None
    
    # Create sequences for the entire dataset
    print("Creating sequences for the entire dataset...")
    X_all, _ = create_sequences(data, SEQUENCE_LENGTH)
    
    # Generate predictions for all sequences
    print("Generating fused heading predictions...")
    y_pred_all = model.predict(X_all)
    
    # Convert predictions to angles
    pred_angles = np.arctan2(y_pred_all[:, 0], y_pred_all[:, 1])
    
    # Create a new dataframe for the fused trajectory
    trajectory_df = data.iloc[SEQUENCE_LENGTH:].copy().reset_index(drop=True)
    trajectory_df['fused_heading'] = pred_angles
    
    # Calculate positions based on fused heading
    trajectory_df['fused_x'] = 0.0
    trajectory_df['fused_y'] = 0.0
    
    # Set the initial position
    if len(trajectory_df) > 0:
        step_size = 0.7  # Assuming 0.7m per step
        
        # Calculate the position increments
        for i in range(1, len(trajectory_df)):
            prev_step = trajectory_df.iloc[i-1]['step']
            curr_step = trajectory_df.iloc[i]['step']
            
            # Calculate distance for this step
            distance = (curr_step - prev_step) * step_size
            
            # Calculate position increment using the fused heading
            dx = distance * np.cos(trajectory_df.iloc[i-1]['fused_heading'])
            dy = distance * np.sin(trajectory_df.iloc[i-1]['fused_heading'])
            
            # Update position
            trajectory_df.loc[i, 'fused_x'] = trajectory_df.iloc[i-1]['fused_x'] + dx
            trajectory_df.loc[i, 'fused_y'] = trajectory_df.iloc[i-1]['fused_y'] + dy
    
    # Save the fused trajectory
    print("Saving fused trajectory...")
    trajectory_df.to_csv(os.path.join(output_dir, 'fused_trajectory.csv'), index=False)
    
    # Plot trajectories
    print("Plotting trajectories...")
    plt.figure(figsize=(12, 10))
    
    # Original trajectories
    plt.plot(trajectory_df['x_compass'], trajectory_df['y_compass'], 
             'r-', linewidth=1.5, label='Corrected Compass')
    plt.plot(trajectory_df['x_gyro'], trajectory_df['y_gyro'], 
             'b-', linewidth=1.5, label='Corrected Gyro')
    
    # Ground truth trajectory
    plt.plot(trajectory_df['gt_x'], trajectory_df['gt_y'], 
             'g-', linewidth=2, label='Ground Truth')
    
    # Fused trajectory
    plt.plot(trajectory_df['fused_x'], trajectory_df['fused_y'], 
             'm-', linewidth=2, label='LSTM Fused')
    
    # Add markers for ground truth points
    gt_only = data[data['step'].isin(pd.read_csv(os.path.join(output_dir.replace('Phase 6/LSTM_fusion', 'Phase 4/QS_Compass4.19'), 'ground_truth_trajectory.csv'))['step'])]
    plt.scatter(gt_only['gt_x'], gt_only['gt_y'], 
                c='green', s=100, marker='o', edgecolors='black', label='Ground Truth Points')
    
    plt.title('Trajectory Comparison')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'fused_trajectory_comparison.png'), dpi=300)
    
    # Calculate error metrics for the fused trajectory
    print("Calculating error metrics...")
    trajectory_df['fused_error'] = np.sqrt(
        (trajectory_df['fused_x'] - trajectory_df['gt_x'])**2 + 
        (trajectory_df['fused_y'] - trajectory_df['gt_y'])**2
    )
    
    trajectory_df['compass_error'] = np.sqrt(
        (trajectory_df['x_compass'] - trajectory_df['gt_x'])**2 + 
        (trajectory_df['y_compass'] - trajectory_df['gt_y'])**2
    )
    
    trajectory_df['gyro_error'] = np.sqrt(
        (trajectory_df['x_gyro'] - trajectory_df['gt_x'])**2 + 
        (trajectory_df['y_gyro'] - trajectory_df['gt_y'])**2
    )
    
    # Calculate and save error statistics
    error_stats = pd.DataFrame({
        'Method': ['Corrected Compass', 'Corrected Gyro', 'LSTM Fused'],
        'Mean_Error': [
            trajectory_df['compass_error'].mean(),
            trajectory_df['gyro_error'].mean(),
            trajectory_df['fused_error'].mean()
        ],
        'Median_Error': [
            trajectory_df['compass_error'].median(),
            trajectory_df['gyro_error'].median(),
            trajectory_df['fused_error'].median()
        ],
        'Max_Error': [
            trajectory_df['compass_error'].max(),
            trajectory_df['gyro_error'].max(),
            trajectory_df['fused_error'].max()
        ],
        'Std_Dev': [
            trajectory_df['compass_error'].std(),
            trajectory_df['gyro_error'].std(),
            trajectory_df['fused_error'].std()
        ]
    })
    
    # Calculate error reduction percentages
    gyro_improvement = ((trajectory_df['gyro_error'].mean() - trajectory_df['fused_error'].mean()) / 
                       trajectory_df['gyro_error'].mean() * 100)
    compass_improvement = ((trajectory_df['compass_error'].mean() - trajectory_df['fused_error'].mean()) / 
                         trajectory_df['compass_error'].mean() * 100)
    
    error_stats.loc[error_stats['Method'] == 'LSTM Fused', 'Improvement_over_Compass_%'] = compass_improvement
    error_stats.loc[error_stats['Method'] == 'LSTM Fused', 'Improvement_over_Gyro_%'] = gyro_improvement
    
    error_stats.to_csv(os.path.join(output_dir, 'fused_trajectory_errors.csv'), index=False)
    
    print("\nFused Trajectory Error Statistics:")
    print(error_stats)
    print(f"\nImprovement over Compass: {compass_improvement:.2f}%")
    print(f"Improvement over Gyro: {gyro_improvement:.2f}%")
    
    print(f"\nResults saved to: {output_dir}")
    
    return trajectory_df

if __name__ == "__main__":
    # Train LSTM model
    model, history, angle_data = train_lstm_model()
    
    # Apply fusion to generate trajectory
    fused_trajectory = apply_fusion_to_trajectory(model) 