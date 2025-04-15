"""
Apply Kalman Filter to preprocess gyroscope and compass data
for improved heading estimation and position tracking
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kalman_filter import HeadingKalmanFilter

# Set paths
data_file = '/Users/shaoxinyi/Downloads/FYP2/Data_collected/1536_CompassGyroSumHeadingData.txt'
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2/Filtered'
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load and preprocess sensor data
def load_and_preprocess_data(file_path):
    """Load and preprocess sensor data from file"""
    print("Loading data...")
    data = pd.read_csv(file_path, delimiter=';')
    
    # Extract Ground Truth location data
    ground_truth_location_data = data[data['Type'] == 'Ground_truth_Location'].copy()
    initial_location_data = data[data['Type'] == 'Initial_Location'].copy()
    
    # Sort by timestamp
    data.sort_values(by="Timestamp_(ms)", inplace=True)
    
    # Extract gyro and compass data
    gyro_data = data[data['Type'] == 'Gyro'].reset_index(drop=True)
    compass_data = data[data['Type'] == 'Compass'].reset_index(drop=True)
    
    # Rename columns for clarity
    compass_data.rename(columns={'value_1': 'Magnetic_Field_Magnitude'}, inplace=True)
    compass_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
    compass_data.rename(columns={'value_3': 'compass'}, inplace=True)
    
    gyro_data.rename(columns={'value_1': 'axisZAngle'}, inplace=True)
    gyro_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
    gyro_data.rename(columns={'value_3': 'compass'}, inplace=True)
    
    # Fill NaN values
    gyro_data = gyro_data.fillna(0)
    compass_data = compass_data.fillna(0)
    
    # Calculate traditional heading
    first_ground_truth = initial_location_data['GroundTruth'].iloc[0] if len(initial_location_data) > 0 else 0
    
    # Traditional heading for gyro data
    gyro_data['GyroStartByGroundTruth'] = first_ground_truth + gyro_data['gyroSumFromstart0'] - gyro_data['gyroSumFromstart0'].iloc[0]
    gyro_data['GyroStartByGroundTruth'] = (gyro_data['GyroStartByGroundTruth'] + 360) % 360
    
    # Traditional heading for compass data
    compass_data['GyroStartByGroundTruth'] = first_ground_truth + compass_data['gyroSumFromstart0'] - compass_data['gyroSumFromstart0'].iloc[0]
    compass_data['GyroStartByGroundTruth'] = (compass_data['GyroStartByGroundTruth'] + 360) % 360
    
    print("Data preprocessing completed")
    
    return gyro_data, compass_data, ground_truth_location_data, initial_location_data

# Step 2: Apply Kalman Filter to sensor data
def apply_kalman_filter(gyro_data, compass_data):
    """Apply Kalman filter to gyroscope and compass data for improved heading estimation"""
    print("Applying Kalman filter...")
    
    # Get timestamps
    gyro_timestamps = gyro_data['Timestamp_(ms)'].values
    compass_timestamps = compass_data['Timestamp_(ms)'].values
    
    # Initialize Kalman filter with first compass reading
    initial_heading = compass_data['compass'].iloc[0]
    initial_bias = 0.0  # Assume zero initial bias
    
    # Create a new DataFrame to store filtered results
    filtered_data = gyro_data.copy()
    filtered_data['KalmanFiltered_Heading'] = np.nan
    filtered_data['EstimatedBias'] = np.nan
    
    # Calculate time differences (dt) between gyro readings
    dt = np.diff(gyro_timestamps) / 1000.0  # Convert to seconds
    dt = np.append(dt, dt[-1])  # Repeat last dt for the last point
    
    # Initialize KF with more conservative parameters
    kf = HeadingKalmanFilter(initial_heading=initial_heading, initial_bias=initial_bias, dt=0.1)
    
    # Adjust Kalman filter parameters
    # Increase process noise to be more adaptive to changes
    kf.Q = np.array([[2.0, 0.0], [0.0, 0.02]])
    
    # Decrease compass measurement noise for more trust in compass
    kf.R = np.array([[5.0]])
    
    # Process each gyro reading
    for i in range(len(gyro_data)):
        # Calculate actual dt for this time step
        if i > 0:
            dt_actual = (gyro_timestamps[i] - gyro_timestamps[i-1]) / 1000.0
            kf.dt = dt_actual
        
        # Get gyro rate
        gyro_rate = gyro_data['axisZAngle'].iloc[i]
        
        # Print debug info for the first few iterations
        if i < 5:
            print(f"gyro_rate type: {type(gyro_rate)}, value: {gyro_rate}")
        
        # Convert gyro_rate to float if it's not
        if not isinstance(gyro_rate, (int, float)):
            try:
                gyro_rate = float(gyro_rate)
            except (ValueError, TypeError):
                print(f"Error: Cannot convert gyro_rate to float: {gyro_rate}")
                gyro_rate = 0.0
        
        # Update using gyro data
        kf.update_with_gyro(gyro_rate)
        
        # Find closest compass reading in time
        closest_compass_idx = np.argmin(np.abs(compass_timestamps - gyro_timestamps[i]))
        compass_heading = compass_data['compass'].iloc[closest_compass_idx]
        
        # Only update with compass if it's close in time to current gyro
        time_diff = abs(compass_timestamps[closest_compass_idx] - gyro_timestamps[i])
        if time_diff < 100:  # 100ms threshold
            kf.update_with_compass(compass_heading)
        
        # Store filtered values
        filtered_data.loc[i, 'KalmanFiltered_Heading'] = kf.get_heading()
        filtered_data.loc[i, 'EstimatedBias'] = kf.get_bias()
    
    print(f"Kalman filter applied: {len(gyro_data)} readings processed")
    
    return filtered_data

# Step 3: Calculate position based on filtered heading
def calculate_positions(data, heading_column, step_length=0.66, initial_position=(0, 0)):
    """Calculate positions using step detection and heading data"""
    print(f"Calculating positions using {heading_column}...")
    
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

# Step 4: Calculate ground truth positions
def calculate_ground_truth_positions(ground_truth_location_data, initial_location_data):
    """Calculate ground truth positions"""
    print("Calculating ground truth positions...")
    
    # Combine ground truth and initial locations
    df_gt = pd.concat([initial_location_data, ground_truth_location_data], ignore_index=True)
    df_gt.sort_values(by='Timestamp_(ms)', inplace=True)
    df_gt.reset_index(drop=True, inplace=True)
    
    # Extract coordinates
    gt_positions = []
    start_x = df_gt['value_4'].iloc[0]
    start_y = df_gt['value_5'].iloc[0]
    
    for i in range(len(df_gt)):
        # Use value_4 (longitude) and value_5 (latitude) for position, shifted to origin
        x = df_gt['value_4'].iloc[i] - start_x
        y = df_gt['value_5'].iloc[i] - start_y
        gt_positions.append((x, y))
    
    return gt_positions

# Step 5: Calculate position errors
def calculate_position_error(positions, gt_positions):
    """Calculate average position error"""
    min_len = min(len(positions), len(gt_positions))
    errors = []
    
    for i in range(min_len):
        error = np.sqrt((positions[i][0] - gt_positions[i][0])**2 + 
                      (positions[i][1] - gt_positions[i][1])**2)
        errors.append(error)
    
    return np.mean(errors) if errors else float('inf')

# Step 6: Visualize results
def visualize_results(filtered_data, gt_positions, positions_trad, positions_filtered, heading_error_trad, heading_error_filtered, position_error_trad, position_error_filtered):
    """Visualize heading and position results"""
    print("Generating visualizations...")
    
    # Plot heading comparison
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data['Timestamp_(ms)'], filtered_data['GyroStartByGroundTruth'], 'b-', alpha=0.7, 
             label=f'Traditional Heading (Error: {heading_error_trad:.2f}°)')
    plt.plot(filtered_data['Timestamp_(ms)'], filtered_data['KalmanFiltered_Heading'], 'r-', 
             label=f'Kalman Filtered Heading (Error: {heading_error_filtered:.2f}°)')
    plt.plot(filtered_data['Timestamp_(ms)'], filtered_data['compass'], 'g.', alpha=0.3, 
             label='Compass Readings')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Heading (degrees)')
    plt.title('Heading Comparison: Traditional vs. Kalman Filtered')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'heading_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot bias estimate
    plt.figure(figsize=(10, 4))
    plt.plot(filtered_data['Timestamp_(ms)'], filtered_data['EstimatedBias'], 'r-')
    plt.xlabel('Time (ms)')
    plt.ylabel('Estimated Gyro Bias (deg/s)')
    plt.title('Gyro Bias Estimation')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'gyro_bias_estimation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot position comparison
    plt.figure(figsize=(10, 10))
    
    # Extract coordinates
    x_gt = [pos[0] for pos in gt_positions]
    y_gt = [pos[1] for pos in gt_positions]
    
    x_trad = [pos[0] for pos in positions_trad]
    y_trad = [pos[1] for pos in positions_trad]
    
    x_filtered = [pos[0] for pos in positions_filtered]
    y_filtered = [pos[1] for pos in positions_filtered]
    
    # Plot trajectories
    plt.plot(x_gt, y_gt, 'k-', linewidth=2, label='Ground Truth')
    plt.plot(x_trad, y_trad, 'b--', linewidth=1, 
             label=f'Traditional Position (Error: {position_error_trad:.2f}m)')
    plt.plot(x_filtered, y_filtered, 'r-', linewidth=1.5, 
             label=f'Kalman Filtered Position (Error: {position_error_filtered:.2f}m)')
    
    # Mark start and end points
    plt.scatter(x_gt[0], y_gt[0], color='black', marker='o', s=100, label='Start')
    plt.scatter(x_gt[-1], y_gt[-1], color='black', marker='x', s=100, label='End')
    
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('Position Trajectory Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, 'position_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot position errors
    plt.figure(figsize=(8, 6))
    methods = ['Traditional', 'Kalman Filtered']
    position_errors = [position_error_trad, position_error_filtered]
    
    plt.bar(methods, position_errors, color=['blue', 'red'])
    for i, v in enumerate(position_errors):
        plt.text(i, v + 0.2, f"{v:.2f}m", ha='center')
    
    plt.ylabel('Average Position Error (m)')
    plt.title('Position Error Comparison')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'position_error_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to", output_dir)

# Step 7: Save processed data
def save_results(filtered_data, positions_trad, positions_filtered, gt_positions, heading_error_trad, heading_error_filtered, position_error_trad, position_error_filtered):
    """Save processed data and results to files"""
    print("Saving results...")
    
    # Save filtered data
    filtered_data.to_csv(os.path.join(output_dir, 'kalman_filtered_data.csv'), index=False)
    
    # Save position data in different files to avoid length mismatch
    # Ground truth positions
    gt_df = pd.DataFrame({
        'X': [pos[0] for pos in gt_positions],
        'Y': [pos[1] for pos in gt_positions]
    })
    gt_df.to_csv(os.path.join(output_dir, 'gt_positions.csv'), index=False)
    
    # Traditional positions
    trad_df = pd.DataFrame({
        'X': [pos[0] for pos in positions_trad],
        'Y': [pos[1] for pos in positions_trad]
    })
    trad_df.to_csv(os.path.join(output_dir, 'traditional_positions.csv'), index=False)
    
    # Filtered positions
    filtered_df = pd.DataFrame({
        'X': [pos[0] for pos in positions_filtered],
        'Y': [pos[1] for pos in positions_filtered]
    })
    filtered_df.to_csv(os.path.join(output_dir, 'filtered_positions.csv'), index=False)
    
    # Save error metrics
    error_df = pd.DataFrame({
        'Method': ['Traditional', 'Kalman Filtered'],
        'Heading_Error': [heading_error_trad, heading_error_filtered],
        'Position_Error': [position_error_trad, position_error_filtered],
        'Improvement_Percent': [0, ((position_error_trad - position_error_filtered) / position_error_trad) * 100]
    })
    error_df.to_csv(os.path.join(output_dir, 'error_metrics.csv'), index=False)
    
    print("Results saved to", output_dir)

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    gyro_data, compass_data, ground_truth_location_data, initial_location_data = load_and_preprocess_data(data_file)
    
    # Apply Kalman filter
    filtered_data = apply_kalman_filter(gyro_data, compass_data)
    
    # Calculate positions
    positions_trad = calculate_positions(filtered_data, 'GyroStartByGroundTruth')
    positions_filtered = calculate_positions(filtered_data, 'KalmanFiltered_Heading')
    
    # Calculate ground truth positions
    gt_positions = calculate_ground_truth_positions(ground_truth_location_data, initial_location_data)
    
    # Calculate heading errors
    if 'GroundTruthHeadingComputed' in filtered_data.columns:
        heading_data = filtered_data.dropna(subset=['GroundTruthHeadingComputed'])
        
        def calculate_heading_error(true_heading, pred_heading):
            diff = np.abs(true_heading - pred_heading)
            diff = np.minimum(diff, 360 - diff)  # Take smaller angle difference
            return np.mean(diff)
        
        heading_error_trad = calculate_heading_error(
            heading_data['GroundTruthHeadingComputed'], 
            heading_data['GyroStartByGroundTruth']
        )
        
        heading_error_filtered = calculate_heading_error(
            heading_data['GroundTruthHeadingComputed'],
            heading_data['KalmanFiltered_Heading']
        )
    else:
        # If no ground truth heading available, use placeholder values
        heading_error_trad = 0.0
        heading_error_filtered = 0.0
        print("Warning: No ground truth heading available for error calculation")
    
    # Calculate position errors
    position_error_trad = calculate_position_error(positions_trad, gt_positions)
    position_error_filtered = calculate_position_error(positions_filtered, gt_positions)
    
    # Print results
    print("\nResults:")
    print(f"Traditional Heading Error: {heading_error_trad:.2f} degrees")
    print(f"Filtered Heading Error: {heading_error_filtered:.2f} degrees")
    print(f"Traditional Position Error: {position_error_trad:.2f} meters")
    print(f"Filtered Position Error: {position_error_filtered:.2f} meters")
    
    if position_error_trad > 0:
        improvement = ((position_error_trad - position_error_filtered) / position_error_trad) * 100
        print(f"Position Improvement: {improvement:.2f}%")
    
    # Visualize results
    visualize_results(
        filtered_data, gt_positions, positions_trad, positions_filtered,
        heading_error_trad, heading_error_filtered, position_error_trad, position_error_filtered
    )
    
    # Save results
    save_results(
        filtered_data, positions_trad, positions_filtered, gt_positions,
        heading_error_trad, heading_error_filtered, position_error_trad, position_error_filtered
    )
    
    print("\nKalman filter processing completed!") 