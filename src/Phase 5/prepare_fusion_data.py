import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d

# Set paths
input_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 4/QS_Compass4.19'
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 6/LSTM_fusion'
os.makedirs(output_dir, exist_ok=True)

# File paths
compass_trajectory_file = os.path.join(input_dir, 'compass_corrected_trajectory.csv')
gyro_trajectory_file = os.path.join(input_dir, 'gyro_corrected_trajectory.csv')
ground_truth_file = os.path.join(input_dir, 'ground_truth_trajectory.csv')

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    return ((angle + math.pi) % (2 * math.pi)) - math.pi

def calculate_heading(x1, y1, x2, y2):
    """Calculate heading from (x1, y1) to (x2, y2) in radians"""
    dx = x2 - x1
    dy = y2 - y1
    heading = math.atan2(dy, dx)
    return heading

def interpolate_ground_truth(ground_truth_df, step_values):
    """Interpolate ground truth positions for each step"""
    gt_steps = ground_truth_df['step'].values
    gt_x = ground_truth_df['x'].values
    gt_y = ground_truth_df['y'].values
    
    # Create interpolation functions
    if len(gt_steps) < 2:
        raise ValueError("Not enough ground truth points for interpolation")
        
    f_x = interp1d(gt_steps, gt_x, kind='linear', bounds_error=False, fill_value='extrapolate')
    f_y = interp1d(gt_steps, gt_y, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Interpolate positions for each step
    interp_x = f_x(step_values)
    interp_y = f_y(step_values)
    
    return pd.DataFrame({
        'step': step_values,
        'gt_x': interp_x,
        'gt_y': interp_y
    })

def calculate_ground_truth_headings(gt_df):
    """Calculate ground truth headings from consecutive GT positions"""
    gt_df = gt_df.copy()
    gt_df['gt_heading'] = float('nan')
    
    # Calculate headings between consecutive points
    for i in range(1, len(gt_df)):
        prev_x, prev_y = gt_df.loc[i-1, 'gt_x'], gt_df.loc[i-1, 'gt_y']
        curr_x, curr_y = gt_df.loc[i, 'gt_x'], gt_df.loc[i, 'gt_y']
        
        # Skip calculating heading if positions are identical
        if abs(curr_x - prev_x) < 1e-10 and abs(curr_y - prev_y) < 1e-10:
            continue
            
        heading = calculate_heading(prev_x, prev_y, curr_x, curr_y)
        gt_df.loc[i, 'gt_heading'] = heading
    
    # Forward fill the first heading
    if not pd.isna(gt_df.loc[1, 'gt_heading']):
        gt_df.loc[0, 'gt_heading'] = gt_df.loc[1, 'gt_heading']
    
    # Forward fill nan values
    gt_df['gt_heading'] = gt_df['gt_heading'].fillna(method='ffill')
    
    return gt_df

def prepare_dataset():
    print("Loading trajectory data...")
    # Load corrected trajectory data
    compass_df = pd.read_csv(compass_trajectory_file)
    gyro_df = pd.read_csv(gyro_trajectory_file)
    ground_truth_df = pd.read_csv(ground_truth_file)
    
    print(f"Loaded {len(compass_df)} compass records, {len(gyro_df)} gyro records, and {len(ground_truth_df)} ground truth records")
    
    # Merge compass and gyro data based on step
    merged_df = pd.merge(compass_df, gyro_df, on='step', suffixes=('_compass', '_gyro'))
    
    # Get all step values from merged data
    step_values = merged_df['step'].values
    
    # Interpolate ground truth positions
    print("Interpolating ground truth positions...")
    gt_interp_df = interpolate_ground_truth(ground_truth_df, step_values)
    
    # Calculate ground truth headings
    print("Calculating ground truth headings...")
    gt_interp_df = calculate_ground_truth_headings(gt_interp_df)
    
    # Merge all data
    print("Merging all data...")
    final_df = pd.merge(merged_df, gt_interp_df, on='step')
    
    # Normalize all headings to be in [-pi, pi]
    print("Normalizing headings...")
    final_df['heading_compass'] = final_df['heading_compass'].apply(lambda x: normalize_angle(x))
    final_df['heading_gyro'] = final_df['heading_gyro'].apply(lambda x: normalize_angle(x))
    final_df['gt_heading'] = final_df['gt_heading'].apply(lambda x: normalize_angle(x))
    
    # Drop rows with NaN values
    final_df = final_df.dropna()
    
    # Convert to radians for consistency
    print("Converting angles to radians...")
    final_df['heading_compass_sin'] = np.sin(final_df['heading_compass'])
    final_df['heading_compass_cos'] = np.cos(final_df['heading_compass'])
    final_df['heading_gyro_sin'] = np.sin(final_df['heading_gyro'])
    final_df['heading_gyro_cos'] = np.cos(final_df['heading_gyro'])
    final_df['gt_heading_sin'] = np.sin(final_df['gt_heading'])
    final_df['gt_heading_cos'] = np.cos(final_df['gt_heading'])
    
    # Save the prepared dataset
    print("Saving prepared dataset...")
    final_df.to_csv(os.path.join(output_dir, 'fusion_dataset.csv'), index=False)
    
    # Create visualization
    print("Creating visualizations...")
    plt.figure(figsize=(12, 6))
    plt.plot(final_df['step'], final_df['heading_compass'], label='Corrected Compass Heading', color='red')
    plt.plot(final_df['step'], final_df['heading_gyro'], label='Corrected Gyro Heading', color='blue')
    plt.plot(final_df['step'], final_df['gt_heading'], label='Ground Truth Heading', color='green', linewidth=2)
    plt.xlabel('Steps')
    plt.ylabel('Heading (radians)')
    plt.title('Comparison of Headings')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'heading_comparison.png'))
    
    print(f"Dataset prepared with {len(final_df)} samples and saved to {os.path.join(output_dir, 'fusion_dataset.csv')}")
    return final_df

if __name__ == "__main__":
    prepare_dataset() 