import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

# Define paths
compass_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_compass_data.csv'
gyro_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_gyro_data.csv'
ground_truth_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_ground_truth_data.csv'
qs_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 4/QS_Compass4.19/compass_data_with_qs_intervals.csv'
output_folder = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 5/Compass_QS'

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def normalize_angle(angle):
    """
    Normalize angle to be in the range [0, 360)
    """
    return angle % 360

def angle_difference(angle1, angle2):
    """
    Calculate the smallest difference between two angles in degrees
    Correctly handles angle wrapping (e.g., 359° vs 1°)
    """
    diff = abs(normalize_angle(angle1) - normalize_angle(angle2)) % 360
    return min(diff, 360 - diff)

def angular_mean(angles, weights=None):
    """
    Calculate the circular mean of angles in degrees
    This correctly handles the circular nature of angles
    
    Parameters:
    -----------
    angles : array-like
        Array of angles in degrees
    weights : array-like, optional
        Weights for each angle
        
    Returns:
    --------
    mean_angle_deg : float
        Circular mean of angles in degrees
    """
    # Convert to radians for numpy's circular functions
    angles_rad = np.radians(angles)
    
    if weights is None:
        # Calculate mean of sin and cos components
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))
    else:
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        mean_sin = np.sum(np.sin(angles_rad) * weights)
        mean_cos = np.sum(np.cos(angles_rad) * weights)
    
    # Convert back to degrees
    mean_angle_rad = np.arctan2(mean_sin, mean_cos)
    mean_angle_deg = normalize_angle(np.degrees(mean_angle_rad))
    return mean_angle_deg

def detect_turns(headings, times, threshold_deg_per_sec=15.0):
    """
    Detect significant turns in the heading data
    
    Parameters:
    -----------
    headings : array-like
        Array of heading values in degrees
    times : array-like
        Array of corresponding timestamps in seconds
    threshold_deg_per_sec : float
        Threshold for heading change rate (degrees per second) to detect turns
        
    Returns:
    --------
    turn_indices : list
        Indices where turns start
    turn_rates : list
        Corresponding turn rates at those indices
    """
    headings = np.array([normalize_angle(h) for h in headings])
    times = np.array(times)
    
    turn_indices = []
    turn_rates = []
    
    for i in range(1, len(headings)):
        # Handle angle wrapping correctly
        angle_diff = angle_difference(headings[i], headings[i-1])
        
        # Determine direction of change
        if normalize_angle(headings[i] - headings[i-1] + 180) > 180:
            angle_diff = -angle_diff  # Turning clockwise
            
        time_diff = times[i] - times[i-1]
        
        if time_diff > 0:  # Avoid division by zero
            turn_rate = angle_diff / time_diff
            if abs(turn_rate) > threshold_deg_per_sec:
                turn_indices.append(i)
                turn_rates.append(turn_rate)
    
    return turn_indices, turn_rates

def extract_qs_intervals(qs_data):
    """
    Extract QS intervals from the processed QS detection data
    
    Parameters:
    -----------
    qs_data : DataFrame
        DataFrame with QS detection results
    
    Returns:
    --------
    qs_intervals : list of dict
        List of QS intervals with time, heading, and related information
    """
    qs_intervals = []
    
    # Group by QS interval number
    for interval_num in sorted(qs_data[qs_data['Quasi_Static_Interval'] == 1]['Quasi_Static_Interval_Number'].unique()):
        interval_data = qs_data[qs_data['Quasi_Static_Interval_Number'] == interval_num]
        
        if len(interval_data) > 0:
            # Extract key information
            start_time = interval_data['Timestamp_(ms)'].min() / 1000  # Convert to seconds
            end_time = interval_data['Timestamp_(ms)'].max() / 1000
            
            # Choose the column with heading information
            heading_col = 'Compass_Heading' if 'Compass_Heading' in interval_data.columns else 'value_2'
            
            mean_heading = angular_mean(interval_data[heading_col])
            mean_step = interval_data['step'].mean()
            
            # Get any ground truth heading if available
            has_gt = 'true_heading' in interval_data.columns
            gt_heading = None
            
            if has_gt:
                gt_values = interval_data['true_heading'].dropna()
                if len(gt_values) > 0:
                    gt_heading = angular_mean(gt_values)
                
            # Add to list
            qs_intervals.append({
                'interval_number': interval_num,
                'start_time': start_time,
                'end_time': end_time,
                'mean_heading': mean_heading,
                'mean_step': mean_step,
                'gt_heading': gt_heading
            })
    
    return qs_intervals

def extract_ground_truth(ground_truth_data):
    """
    Extract ground truth points
    
    Parameters:
    -----------
    ground_truth_data : DataFrame
        Ground truth data
    
    Returns:
    --------
    gt_points : list of dict
        List of ground truth points with time and heading information
    """
    gt_points = []
    
    for i, row in ground_truth_data.iterrows():
        if row['Type'] in ['Initial_Location', 'Ground_truth_Location']:
            # Check if heading data is available
            if 'GroundTruthHeadingComputed' in ground_truth_data.columns and not pd.isna(row['GroundTruthHeadingComputed']):
                gt_points.append({
                    'time': row['Timestamp_(ms)'] / 1000,  # Convert to seconds
                    'heading': normalize_angle(row['GroundTruthHeadingComputed']),
                    'step': row['step'],
                    'point_name': f"GT{len(gt_points)}"
                })
    
    return gt_points

def generate_correction_points(qs_intervals, gt_points, sensor_data, is_turn_indices):
    """
    Generate correction points from QS intervals and ground truth points
    
    Parameters:
    -----------
    qs_intervals : list of dict
        List of QS intervals
    gt_points : list of dict
        List of ground truth points
    sensor_data : DataFrame
        Sensor data with timestamps
    is_turn_indices : list
        Indices of turn points in sensor data
    
    Returns:
    --------
    correction_points : list of tuple
        Each tuple contains (time, reference_heading, is_turn_point)
    """
    correction_points = []
    
    # First, process QS intervals
    for interval in qs_intervals:
        start_time = interval['start_time']
        end_time = interval['end_time']
        
        # Check if this segment contains any turns
        segment_time_mask = (sensor_data['Timestamp_(ms)'] / 1000 >= start_time) & (sensor_data['Timestamp_(ms)'] / 1000 <= end_time)
        segment_indices = sensor_data[segment_time_mask].index
        contains_turn = any(idx in is_turn_indices for idx in segment_indices)
        
        # Use ground truth heading if available, otherwise use mean compass heading
        heading = interval['gt_heading'] if interval['gt_heading'] is not None else interval['mean_heading']
        
        # Add correction point at the start of the segment
        correction_points.append((start_time, heading, contains_turn))
        
        # For long segments, add additional correction points in the middle
        if end_time - start_time > 5.0:  # If segment is longer than 5 seconds
            mid_time = (start_time + end_time) / 2
            correction_points.append((mid_time, heading, contains_turn))
    
    # Then add ground truth points
    for point in gt_points:
        time = point['time']
        heading = point['heading']
        
        # Check if this point is near a turn
        closest_idx = (sensor_data['Timestamp_(ms)'] / 1000 - time).abs().idxmin()
        is_turn = closest_idx in is_turn_indices
        
        # Add to correction points
        correction_points.append((time, heading, is_turn))
    
    # Sort by time and remove duplicates
    correction_points = sorted(set(correction_points), key=lambda x: x[0])
    
    return correction_points

def apply_correction(sensor_data, correction_points, heading_col):
    """
    Apply heading corrections at specified points, adjusting for turns
    
    Parameters:
    -----------
    sensor_data : DataFrame
        Sensor data with headings
    correction_points : list of tuples
        Each tuple contains (time, reference_heading, is_turn_point)
    heading_col : str
        Column name for the heading data
        
    Returns:
    --------
    corrected_headings : array
        Corrected heading values
    """
    # Start with a copy of the original headings
    corrected_headings = sensor_data[heading_col].copy()
    
    # Sort correction points by time
    correction_points = sorted(correction_points, key=lambda x: x[0])
    
    for i in range(len(correction_points)):
        current_point = correction_points[i]
        current_time, ref_heading, is_turn = current_point
        
        # Find the closest data point
        closest_idx = (sensor_data['Timestamp_(ms)'] / 1000 - current_time).abs().idxmin()
        sensor_heading = sensor_data.loc[closest_idx, heading_col]
        
        # Calculate offset (reference - sensor)
        offset = normalize_angle(ref_heading - sensor_heading + 180) - 180
        
        # Determine range of application
        if i < len(correction_points) - 1:
            next_time = correction_points[i + 1][0]
            # Apply from this point to just before the next point
            time_range = (sensor_data['Timestamp_(ms)'] / 1000 >= current_time) & (sensor_data['Timestamp_(ms)'] / 1000 < next_time)
        else:
            # Apply from this point to the end
            time_range = sensor_data['Timestamp_(ms)'] / 1000 >= current_time
        
        # If this is a turn point, apply more carefully
        if is_turn:
            # Gradually apply the offset over a short distance
            affected_indices = sensor_data.loc[time_range].index
            
            if len(affected_indices) > 0:
                # Apply full correction only to the closest point
                corrected_headings.loc[closest_idx] = normalize_angle(sensor_heading + offset)
                
                # Calculate how quickly to phase out the correction
                fade_length = min(15, len(affected_indices))  # Extended fade for smoother transitions
                fade_weights = np.linspace(1, 0, fade_length)
                
                # Apply faded correction to next few points
                for j, idx in enumerate(affected_indices[:fade_length]):
                    if idx != closest_idx:  # Skip the closest point (already done)
                        weight = fade_weights[j]
                        current_heading = sensor_data.loc[idx, heading_col]
                        corrected_headings.loc[idx] = normalize_angle(current_heading + offset * weight)
        else:
            # For non-turn points, apply the full offset to the range
            sensor_headings_in_range = sensor_data.loc[time_range, heading_col]
            corrected_headings.loc[time_range] = normalize_angle(sensor_headings_in_range + offset)
    
    return corrected_headings

def plot_heading_comparison(sensor_data, gt_data, heading_col, corrected_heading_col, output_folder, title, filename):
    """
    Plot comparison of original vs corrected headings
    
    Parameters:
    -----------
    sensor_data : DataFrame
        Sensor data with headings
    gt_data : DataFrame
        Ground truth data
    heading_col : str
        Column name for original heading
    corrected_heading_col : str
        Column name for corrected heading
    output_folder : str
        Folder to save plots
    title : str
        Plot title
    filename : str
        Filename for saving the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot original heading
    plt.plot(sensor_data['Timestamp_(ms)'] / 1000, sensor_data[heading_col], 
             label='Original Heading', color='gray', alpha=0.6)
    
    # Plot corrected heading
    plt.plot(sensor_data['Timestamp_(ms)'] / 1000, sensor_data[corrected_heading_col], 
             label='Corrected Heading', color='blue', linewidth=2)
    
    # Add ground truth points if available
    if gt_data is not None:
        for i, row in gt_data.iterrows():
            if 'GroundTruthHeadingComputed' in gt_data.columns and not pd.isna(row['GroundTruthHeadingComputed']):
                plt.scatter(row['Timestamp_(ms)'] / 1000, normalize_angle(row['GroundTruthHeadingComputed']), 
                           color='red', s=80, marker='X', label='Ground Truth' if i == 0 else "")
    
    # Add QS intervals if available
    if 'Quasi_Static_Interval' in sensor_data.columns:
        qs_data = sensor_data[sensor_data['Quasi_Static_Interval'] == 1]
        plt.scatter(qs_data['Timestamp_(ms)'] / 1000, qs_data[heading_col], 
                   color='green', s=50, alpha=0.5, marker='o', label='QS Intervals')
    
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Heading (degrees)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=300)
    plt.close()

def calculate_heading_statistics(sensor_data, gt_data, heading_cols):
    """
    Calculate error statistics between sensor and ground truth headings
    
    Parameters:
    -----------
    sensor_data : DataFrame
        Sensor data with headings
    gt_data : DataFrame
        Ground truth data
    heading_cols : list
        Column names for headings to compare
    
    Returns:
    --------
    stats : DataFrame
        Statistics of heading errors
    """
    stats = []
    
    # For each ground truth point
    for i, gt_row in gt_data.iterrows():
        if 'GroundTruthHeadingComputed' not in gt_data.columns or pd.isna(gt_row['GroundTruthHeadingComputed']):
            continue
            
        gt_time = gt_row['Timestamp_(ms)'] / 1000
        gt_heading = normalize_angle(gt_row['GroundTruthHeadingComputed'])
        
        # Find closest sensor point
        closest_idx = (sensor_data['Timestamp_(ms)'] / 1000 - gt_time).abs().idxmin()
        
        row_stats = {'GT_Point': f"GT{i}", 'GT_Heading': gt_heading}
        
        # Calculate error for each heading column
        for col in heading_cols:
            sensor_heading = normalize_angle(sensor_data.loc[closest_idx, col])
            error = angle_difference(gt_heading, sensor_heading)
            row_stats[f"{col}_Error"] = error
        
        stats.append(row_stats)
    
    return pd.DataFrame(stats)

# Main execution
def main():
    print("\n=== Compass and Gyro Heading Correction ===\n")
    
    # 1. Load data
    print("Loading data...")
    compass_data = pd.read_csv(compass_data_path)
    gyro_data = pd.read_csv(gyro_data_path)
    ground_truth_data = pd.read_csv(ground_truth_path)
    qs_data = pd.read_csv(qs_data_path)
    
    print(f"Loaded {len(compass_data)} compass data points")
    print(f"Loaded {len(gyro_data)} gyro data points")
    print(f"Loaded {len(ground_truth_data)} ground truth data points")
    print(f"Loaded {len(qs_data)} QS detection data points")
    
    # 2. Extract QS intervals
    print("\nExtracting QS intervals...")
    qs_intervals = extract_qs_intervals(qs_data)
    print(f"Extracted {len(qs_intervals)} QS intervals")
    
    for i, interval in enumerate(qs_intervals):
        print(f"  QS #{i}: Step {interval['mean_step']:.1f}, "
              f"Time {interval['start_time']:.1f}-{interval['end_time']:.1f} sec, "
              f"Heading {interval['mean_heading']:.1f}°")
    
    # 3. Extract ground truth points
    print("\nExtracting ground truth points...")
    gt_points = extract_ground_truth(ground_truth_data)
    print(f"Extracted {len(gt_points)} ground truth points")
    
    for i, point in enumerate(gt_points):
        print(f"  {point['point_name']}: Step {point['step']}, "
              f"Time {point['time']:.1f} sec, Heading {point['heading']:.1f}°")
    
    # 4. Detect turns in compass and gyro data
    print("\nDetecting turns in compass data...")
    compass_heading_col = 'Compass_Heading' if 'Compass_Heading' in qs_data.columns else 'value_2'
    
    # Ensure columns exist in the original compass data
    if compass_heading_col not in compass_data.columns and compass_heading_col == 'Compass_Heading':
        compass_data['Compass_Heading'] = compass_data['value_2'].apply(normalize_angle)
    
    compass_turn_indices, compass_turn_rates = detect_turns(
        compass_data[compass_heading_col].values, 
        compass_data['Timestamp_(ms)'].values / 1000
    )
    print(f"Detected {len(compass_turn_indices)} turns in compass data")
    
    print("\nDetecting turns in gyro data...")
    gyro_heading_col = 'value_3'  # Heading in gyro data
    gyro_turn_indices, gyro_turn_rates = detect_turns(
        gyro_data[gyro_heading_col].values, 
        gyro_data['Timestamp_(ms)'].values / 1000
    )
    print(f"Detected {len(gyro_turn_indices)} turns in gyro data")
    
    # 5. Generate correction points
    print("\nGenerating correction points for compass data...")
    compass_correction_points = generate_correction_points(
        qs_intervals, gt_points, compass_data, compass_turn_indices
    )
    print(f"Generated {len(compass_correction_points)} correction points for compass data")
    
    print("\nGenerating correction points for gyro data...")
    gyro_correction_points = generate_correction_points(
        qs_intervals, gt_points, gyro_data, gyro_turn_indices
    )
    print(f"Generated {len(gyro_correction_points)} correction points for gyro data")
    
    # 6. Apply corrections
    print("\nApplying corrections to compass data...")
    compass_data['Corrected_Heading'] = apply_correction(
        compass_data, compass_correction_points, compass_heading_col
    )
    
    print("\nApplying corrections to gyro data...")
    gyro_data['Corrected_Heading'] = apply_correction(
        gyro_data, gyro_correction_points, gyro_heading_col
    )
    
    # 7. Plot results
    print("\nGenerating plots...")
    plot_heading_comparison(
        compass_data, ground_truth_data, 
        compass_heading_col, 'Corrected_Heading', 
        output_folder, 'Compass Heading Correction', 'compass_heading_correction.png'
    )
    
    plot_heading_comparison(
        gyro_data, ground_truth_data, 
        gyro_heading_col, 'Corrected_Heading', 
        output_folder, 'Gyro Heading Correction', 'gyro_heading_correction.png'
    )
    
    # 8. Calculate and report statistics
    print("\nCalculating heading error statistics...")
    compass_stats = calculate_heading_statistics(
        compass_data, ground_truth_data, 
        [compass_heading_col, 'Corrected_Heading']
    )
    
    gyro_stats = calculate_heading_statistics(
        gyro_data, ground_truth_data, 
        [gyro_heading_col, 'Corrected_Heading']
    )
    
    print("\nCompass heading error statistics:")
    print(compass_stats)
    
    print("\nGyro heading error statistics:")
    print(gyro_stats)
    
    # 9. Save results
    print("\nSaving corrected data...")
    compass_data.to_csv(os.path.join(output_folder, 'compass_heading_corrected.csv'), index=False)
    gyro_data.to_csv(os.path.join(output_folder, 'gyro_heading_corrected.csv'), index=False)
    
    compass_stats.to_csv(os.path.join(output_folder, 'compass_heading_errors.csv'), index=False)
    gyro_stats.to_csv(os.path.join(output_folder, 'gyro_heading_errors.csv'), index=False)
    
    print("\nHeading correction completed successfully!")
    print(f"Results saved to: {output_folder}")

if __name__ == "__main__":
    main() 