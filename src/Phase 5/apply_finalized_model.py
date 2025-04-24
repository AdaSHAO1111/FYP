import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
from scipy.interpolate import interp1d

# Define paths
MODEL_PATH = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 6/Finalized_Model/lstm_heading_fusion_final.h5'
OUTPUT_DIR = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 6/LSTM_Applied'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
SEQUENCE_LENGTH = 10  # Same as used during training

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    return ((angle + math.pi) % (2 * math.pi)) - math.pi

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
    X = []
    for i in range(len(data) - sequence_length):
        # Input features: heading_compass_sin, heading_compass_cos, heading_gyro_sin, heading_gyro_cos
        features = data.iloc[i:i+sequence_length, [data.columns.get_loc(col) for col in 
                                                 ['heading_compass_sin', 'heading_compass_cos', 
                                                  'heading_gyro_sin', 'heading_gyro_cos']]].values
        X.append(features)
    
    return np.array(X)

def preprocess_data(compass_data, gyro_data):
    """Preprocess compass and gyro data for LSTM model input"""
    print("Preprocessing data...")
    
    # Merge data by step
    if 'step' in compass_data.columns and 'step' in gyro_data.columns:
        data = pd.merge(compass_data, gyro_data, on='step', suffixes=('_compass', '_gyro'))
    else:
        raise ValueError("Both compass and gyro data must have a 'step' column")
    
    # Normalize headings
    if 'heading' in compass_data.columns and 'heading' in gyro_data.columns:
        data['heading_compass'] = compass_data['heading'].apply(normalize_angle)
        data['heading_gyro'] = gyro_data['heading'].apply(normalize_angle)
    else:
        raise ValueError("Both compass and gyro data must have a 'heading' column")
    
    # Convert to sin/cos components for circular data
    data['heading_compass_sin'] = np.sin(data['heading_compass'])
    data['heading_compass_cos'] = np.cos(data['heading_compass'])
    data['heading_gyro_sin'] = np.sin(data['heading_gyro'])
    data['heading_gyro_cos'] = np.cos(data['heading_gyro'])
    
    return data

def apply_model(compass_file, gyro_file, gt_file=None):
    """Apply the finalized LSTM model to new trajectory data"""
    print(f"\n=== Applying Finalized LSTM Model ===\n")
    
    # Load data
    print(f"Loading compass data from {compass_file}")
    compass_data = pd.read_csv(compass_file)
    
    print(f"Loading gyro data from {gyro_file}")
    gyro_data = pd.read_csv(gyro_file)
    
    # Preprocess data
    data = preprocess_data(compass_data, gyro_data)
    
    # Load ground truth data if available
    have_ground_truth = False
    if gt_file and os.path.exists(gt_file):
        print(f"Loading ground truth data from {gt_file}")
        gt_data = pd.read_csv(gt_file)
        
        # Check if we have ground truth data with appropriate columns
        if 'step' in gt_data.columns and ('x' in gt_data.columns or 'gt_x' in gt_data.columns) and ('y' in gt_data.columns or 'gt_y' in gt_data.columns):
            # Map ground truth data to steps
            x_col = 'x' if 'x' in gt_data.columns else 'gt_x'
            y_col = 'y' if 'y' in gt_data.columns else 'gt_y'
            
            # Interpolate ground truth
            f_x = interp1d(gt_data['step'].values, gt_data[x_col].values, 
                          kind='linear', bounds_error=False, fill_value='extrapolate')
            f_y = interp1d(gt_data['step'].values, gt_data[y_col].values, 
                          kind='linear', bounds_error=False, fill_value='extrapolate')
            
            data['gt_x'] = f_x(data['step'])
            data['gt_y'] = f_y(data['step'])
            have_ground_truth = True
    
    # Load model
    print("Loading LSTM model...")
    model = load_model(MODEL_PATH, custom_objects={'angle_difference': angle_difference})
    
    # Create input sequences
    print("Creating sequences for model input...")
    X = create_sequences(data, SEQUENCE_LENGTH)
    
    if len(X) == 0:
        print("Error: Not enough data points to create sequences.")
        return
    
    # Generate predictions
    print("Generating fused heading predictions...")
    y_pred = model.predict(X)
    
    # Convert predictions to angles
    pred_angles = np.arctan2(y_pred[:, 0], y_pred[:, 1])
    
    # Create result dataframe
    result_df = data.iloc[SEQUENCE_LENGTH:].copy().reset_index(drop=True)
    result_df['fused_heading'] = pred_angles
    
    # Calculate positions from fused heading
    result_df['fused_x'] = 0.0
    result_df['fused_y'] = 0.0
    
    # Calculate trajectory using the fused heading
    print("Calculating trajectory from fused headings...")
    step_size = 0.7  # Assuming 0.7m per step
    
    for i in range(1, len(result_df)):
        prev_step = result_df.iloc[i-1]['step']
        curr_step = result_df.iloc[i]['step']
        
        # Calculate distance for this step
        distance = (curr_step - prev_step) * step_size
        
        # Calculate position increment using the fused heading
        dx = distance * np.cos(result_df.iloc[i-1]['fused_heading'])
        dy = distance * np.sin(result_df.iloc[i-1]['fused_heading'])
        
        # Update position
        result_df.loc[i, 'fused_x'] = result_df.iloc[i-1]['fused_x'] + dx
        result_df.loc[i, 'fused_y'] = result_df.iloc[i-1]['fused_y'] + dy
    
    # Save results
    print("Saving results...")
    result_df.to_csv(os.path.join(OUTPUT_DIR, 'fused_trajectory_result.csv'), index=False)
    
    # Plot trajectories
    print("Generating visualizations...")
    plt.figure(figsize=(12, 10))
    
    # Plot original trajectories
    plt.plot(result_df['x_compass'], result_df['y_compass'], 
             'r-', linewidth=1.5, label='Compass')
    plt.plot(result_df['x_gyro'], result_df['y_gyro'], 
             'b-', linewidth=1.5, label='Gyro')
    
    # Plot fused trajectory
    plt.plot(result_df['fused_x'], result_df['fused_y'], 
             'm-', linewidth=2, label='LSTM Fused')
    
    # Plot ground truth if available
    if have_ground_truth:
        plt.plot(result_df['gt_x'], result_df['gt_y'], 
                'g-', linewidth=2, label='Ground Truth')
        
        # Calculate errors
        result_df['fused_error'] = np.sqrt(
            (result_df['fused_x'] - result_df['gt_x'])**2 + 
            (result_df['fused_y'] - result_df['gt_y'])**2
        )
        
        result_df['compass_error'] = np.sqrt(
            (result_df['x_compass'] - result_df['gt_x'])**2 + 
            (result_df['y_compass'] - result_df['gt_y'])**2
        )
        
        result_df['gyro_error'] = np.sqrt(
            (result_df['x_gyro'] - result_df['gt_x'])**2 + 
            (result_df['y_gyro'] - result_df['gt_y'])**2
        )
        
        # Calculate error statistics
        error_stats = pd.DataFrame({
            'Method': ['Compass', 'Gyro', 'LSTM Fused'],
            'Mean_Error': [
                result_df['compass_error'].mean(),
                result_df['gyro_error'].mean(),
                result_df['fused_error'].mean()
            ],
            'Median_Error': [
                result_df['compass_error'].median(),
                result_df['gyro_error'].median(),
                result_df['fused_error'].median()
            ],
            'Max_Error': [
                result_df['compass_error'].max(),
                result_df['gyro_error'].max(),
                result_df['fused_error'].max()
            ]
        })
        
        # Calculate improvement percentages
        compass_improvement = ((result_df['compass_error'].mean() - result_df['fused_error'].mean()) / 
                            result_df['compass_error'].mean() * 100)
        gyro_improvement = ((result_df['gyro_error'].mean() - result_df['fused_error'].mean()) / 
                         result_df['gyro_error'].mean() * 100)
        
        error_stats.loc[error_stats['Method'] == 'LSTM Fused', 'Improvement_over_Compass_%'] = compass_improvement
        error_stats.loc[error_stats['Method'] == 'LSTM Fused', 'Improvement_over_Gyro_%'] = gyro_improvement
        
        # Save error statistics
        error_stats.to_csv(os.path.join(OUTPUT_DIR, 'error_statistics.csv'), index=False)
        
        print("\nError Statistics:")
        print(error_stats)
        print(f"\nImprovement over Compass: {compass_improvement:.2f}%")
        print(f"Improvement over Gyro: {gyro_improvement:.2f}%")
    
    plt.title('Trajectory Comparison')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'trajectory_comparison.png'), dpi=300)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    return result_df

def main():
    """Main function to demonstrate model application"""
    # Define file paths - these should be changed to the actual paths for new data
    compass_trajectory_file = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 4/QS_Compass4.19/compass_corrected_trajectory.csv'
    gyro_trajectory_file = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 4/QS_Compass4.19/gyro_corrected_trajectory.csv'
    ground_truth_file = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 4/QS_Compass4.19/ground_truth_trajectory.csv'
    
    # Apply model
    result_df = apply_model(compass_trajectory_file, gyro_trajectory_file, ground_truth_file)
    
    print("\nModel application completed successfully!")

if __name__ == "__main__":
    main() 