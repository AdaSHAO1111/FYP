import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from scipy.stats import circmean

# Define paths
gyro_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_gyro_data.csv'
qs_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 4/1536_quasi_static_data.csv'
ground_truth_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_ground_truth_data.csv'
output_folder = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 5'

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

def angular_mean(angles):
    """
    Calculate the circular mean of angles in degrees
    This correctly handles the circular nature of angles
    """
    # Convert to radians for numpy's circular functions
    angles_rad = np.radians(angles)
    # Calculate mean of sin and cos components
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
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

def identify_qs_segments(qs_data, gyro_data, gt_points):
    """
    Identify continuous segments of quasi-static field data
    Filter out false QS segments by location (especially the one after GT5)
    
    Parameters:
    -----------
    qs_data : DataFrame
        DataFrame containing QS detection data
    gyro_data : DataFrame
        DataFrame containing gyroscope data
    gt_points : list
        List of ground truth points with time and heading information
    
    Returns:
    --------
    segments : list of tuples
        Each tuple contains (start_time, end_time, mean_heading) for a QS segment
    """
    if len(qs_data) == 0:
        return []
    
    # Sort by time (convert milliseconds to seconds for consistency with gyro data)
    qs_data = qs_data.copy()
    qs_data['Time_Seconds'] = qs_data['Time'] / 1000  # Convert milliseconds to seconds
    qs_data = qs_data.sort_values('Time_Seconds')
    
    # Identify gaps in time that are larger than 0.5 seconds
    qs_data['Time_Diff'] = qs_data['Time_Seconds'].diff()
    segment_starts = [0] + list(qs_data[qs_data['Time_Diff'] > 0.5].index)
    
    # Handle the last segment
    segment_ends = segment_starts[1:] + [len(qs_data)]
    
    # Get GT5 time if available
    gt5_time = None
    if len(gt_points) > 5:  # We have at least 6 points (0-5)
        gt5_time = gt_points[5]['time']
    
    segments = []
    for start, end in zip(segment_starts, segment_ends):
        segment = qs_data.iloc[start:end]
        if len(segment) > 0:
            # Get time range for this segment
            seg_start_time = segment['Time_Seconds'].min()
            seg_end_time = segment['Time_Seconds'].max()
            
            # Skip the false QS segment after GT5
            if gt5_time is not None:
                # If this segment is after GT5 and within 10 seconds, skip it
                if seg_start_time > gt5_time and seg_start_time < gt5_time + 10:
                    print(f"Filtering out false QS segment at time {seg_start_time:.2f}-{seg_end_time:.2f} (after GT5)")
                    continue
            
            # Calculate mean heading for this segment
            mean_heading = angular_mean(segment['True_Heading'])
            
            # Record segment info
            segments.append((
                segment['Time_Seconds'].min(),
                segment['Time_Seconds'].max(),
                mean_heading
            ))
    
    return segments

def apply_correction(gyro_data, correction_points):
    """
    Apply heading corrections at specified points, adjusting for turns
    
    Parameters:
    -----------
    gyro_data : DataFrame
        Gyro data with headings
    correction_points : list of tuples
        Each tuple contains (time, reference_heading, is_turn_point)
        
    Returns:
    --------
    corrected_headings : array
        Corrected heading values
    """
    # Start with a copy of the original headings
    corrected_headings = gyro_data['value_3'].copy()  # Heading is in value_3
    
    # Sort correction points by time
    correction_points = sorted(correction_points, key=lambda x: x[0])
    
    for i in range(len(correction_points)):
        current_point = correction_points[i]
        current_time, ref_heading, is_turn = current_point
        
        # Find the closest gyro data point
        closest_idx = (gyro_data['Timestamp_(ms)'] / 1000 - current_time).abs().idxmin()
        gyro_heading = gyro_data.loc[closest_idx, 'value_3']  # Heading is in value_3
        
        # Calculate offset (reference - gyro)
        offset = normalize_angle(ref_heading - gyro_heading + 180) - 180
        
        # Determine range of application
        if i < len(correction_points) - 1:
            next_time = correction_points[i + 1][0]
            # Apply from this point to just before the next point
            time_range = (gyro_data['Timestamp_(ms)'] / 1000 >= current_time) & (gyro_data['Timestamp_(ms)'] / 1000 < next_time)
        else:
            # Apply from this point to the end
            time_range = gyro_data['Timestamp_(ms)'] / 1000 >= current_time
        
        # If this is a turn point, apply more carefully
        if is_turn:
            # Gradually apply the offset over a short distance
            affected_indices = gyro_data.loc[time_range].index
            
            if len(affected_indices) > 0:
                # Apply full correction only to the closest point
                corrected_headings.loc[closest_idx] = normalize_angle(gyro_heading + offset)
                
                # Calculate how quickly to phase out the correction
                fade_length = min(15, len(affected_indices))  # Extended fade for smoother transitions
                fade_weights = np.linspace(1, 0, fade_length)
                
                # Apply faded correction to next few points
                for j, idx in enumerate(affected_indices[:fade_length]):
                    if idx != closest_idx:  # Skip the closest point (already done)
                        weight = fade_weights[j]
                        current_heading = gyro_data.loc[idx, 'value_3']  # Heading is in value_3
                        corrected_headings.loc[idx] = normalize_angle(current_heading + offset * weight)
        else:
            # For non-turn points, apply the full offset to the range
            gyro_headings_in_range = gyro_data.loc[time_range, 'value_3']  # Heading is in value_3
            corrected_headings.loc[time_range] = normalize_angle(gyro_headings_in_range + offset)
    
    return corrected_headings

def main():
    # Load data
    print("Loading data...")
    gyro_data = pd.read_csv(gyro_data_path)
    qs_data = pd.read_csv(qs_data_path)
    ground_truth_data = pd.read_csv(ground_truth_path)
    
    print(f"Gyro data shape: {gyro_data.shape}")
    print(f"QS data shape: {qs_data.shape}")
    print(f"Ground truth data shape: {ground_truth_data.shape}")
    
    # Convert timestamps to seconds for easier processing
    gyro_data['Time_Seconds'] = gyro_data['Timestamp_(ms)'] / 1000
    
    # Extract ground truth points first (needed for QS filtering)
    gt_points = []
    for i, row in enumerate(ground_truth_data.iterrows()):
        _, row_data = row
        if row_data['Type'] in ['Initial_Location', 'Ground_truth_Location']:
            # Check if heading data is available
            if 'GroundTruthHeadingComputed' in row_data and not pd.isna(row_data['GroundTruthHeadingComputed']):
                gt_points.append({
                    'time': row_data['Timestamp_(ms)'] / 1000,  # Convert to seconds
                    'heading': row_data['GroundTruthHeadingComputed'],
                    'point_name': f"GT{i}"
                })
            else:
                # If no heading data, just record the position for reference
                gt_points.append({
                    'time': row_data['Timestamp_(ms)'] / 1000,
                    'heading': None,
                    'point_name': f"GT{i}"
                })
    
    print(f"Extracted {len(gt_points)} ground truth points")
    
    # Detect turns in the gyro data
    print("\nDetecting turns in gyro data...")
    turn_indices, turn_rates = detect_turns(
        gyro_data['value_3'].values,  # Heading is in value_3
        gyro_data['Time_Seconds'].values
    )
    
    print(f"Detected {len(turn_indices)} significant turns")
    
    # Mark turns in the data
    gyro_data['Is_Turn'] = False
    gyro_data.loc[turn_indices, 'Is_Turn'] = True
    
    # Identify QS segments without the false one after GT5
    print("\nIdentifying quasi-static field segments...")
    qs_segments = identify_qs_segments(qs_data, gyro_data, gt_points)
    print(f"Identified {len(qs_segments)} valid QS segments after filtering")
    
    # Generate correction points
    correction_points = []
    
    # First, process QS segments
    for start_time, end_time, mean_heading in qs_segments:
        # Check if this segment contains any turns
        segment_range = (gyro_data['Time_Seconds'] >= start_time) & (gyro_data['Time_Seconds'] <= end_time)
        contains_turn = gyro_data.loc[segment_range, 'Is_Turn'].any()
        
        # Add correction point at the start of the segment
        correction_points.append((start_time, mean_heading, contains_turn))
    
    # Find and fix the missed turn after GT0
    if len(gt_points) > 0:
        gt0_time = gt_points[0]['time']
        
        # Look within a specific time window after GT0 (5 seconds)
        search_window = (gyro_data['Time_Seconds'] > gt0_time) & (gyro_data['Time_Seconds'] <= gt0_time + 5)
        
        # Enhanced turn detection specifically for this region
        # We'll use a lower threshold to ensure we catch the turn
        window_headings = gyro_data.loc[search_window, 'value_3'].values
        window_times = gyro_data.loc[search_window, 'Time_Seconds'].values
        
        if len(window_headings) > 1:
            # Use a lower threshold just for this region
            window_turn_indices, _ = detect_turns(window_headings, window_times, threshold_deg_per_sec=10.0)
            
            if len(window_turn_indices) > 0:
                # Get the actual index in the original dataframe
                search_window_indices = gyro_data.loc[search_window].index
                turn_idx = search_window_indices[window_turn_indices[0]]
                
                # Get turn time and add a correction point
                turn_time = gyro_data.loc[turn_idx, 'Time_Seconds']
                
                # Use GT0 heading as reference since it's a turn right after GT0
                if gt_points[0]['heading'] is not None:
                    gt0_heading = gt_points[0]['heading']
                    correction_points.append((turn_time, gt0_heading, True))
                    print(f"Added special correction for missed turn after GT0 at time {turn_time:.2f}")
    
    # Add ground truth points as additional correction points
    for gt in gt_points:
        # Skip points without heading information
        if gt['heading'] is None:
            continue
            
        gt_time = gt['time']
        gt_heading = gt['heading']
        
        # Check if this is near a turn
        near_turn_range = (gyro_data['Time_Seconds'] >= gt_time - 1) & (gyro_data['Time_Seconds'] <= gt_time + 1)
        near_turn = gyro_data.loc[near_turn_range, 'Is_Turn'].any()
        
        # Add as correction point
        correction_points.append((gt_time, gt_heading, near_turn))
    
    # Remove duplicate correction points (keeping only the earliest one at each time)
    correction_points = sorted(correction_points, key=lambda x: x[0])
    unique_times = {}
    for time, heading, is_turn in correction_points:
        if time not in unique_times:
            unique_times[time] = (time, heading, is_turn)
    
    # Get final unique correction points
    unique_correction_points = list(unique_times.values())
    print(f"Generated {len(unique_correction_points)} unique correction points")
    
    # Apply corrections to gyro headings
    print("\nApplying heading corrections...")
    corrected_headings = apply_correction(gyro_data, unique_correction_points)
    
    # Update the gyro data with corrected headings
    gyro_data['Corrected_Heading'] = corrected_headings
    
    # Save corrected data
    output_file = os.path.join(output_folder, 'improved_gyro_heading_corrected.csv')
    gyro_data.to_csv(output_file, index=False)
    print(f"Saved corrected heading data to {output_file}")
    
    # Create visualization of the corrections
    plt.figure(figsize=(15, 8))
    
    # Plot original gyro headings
    plt.plot(gyro_data['Time_Seconds'], gyro_data['value_3'], 'b-', alpha=0.6, label='Original Gyro Heading')  # Heading is in value_3
    
    # Plot corrected headings
    plt.plot(gyro_data['Time_Seconds'], gyro_data['Corrected_Heading'], 'g-', label='Corrected Heading')
    
    # Plot turn points
    turn_times = gyro_data.loc[turn_indices, 'Time_Seconds']
    turn_headings = gyro_data.loc[turn_indices, 'value_3']  # Heading is in value_3
    plt.scatter(turn_times, turn_headings, c='r', s=50, marker='^', label='Detected Turns')
    
    # Plot ground truth points
    for gt in gt_points:
        if gt['heading'] is not None:
            plt.scatter(gt['time'], gt['heading'], c='r', s=100, marker='o')
            plt.text(gt['time'], gt['heading'] + 5, gt['point_name'], fontsize=12, weight='bold')
    
    # Plot QS segment points
    for start_time, end_time, mean_heading in qs_segments:
        plt.scatter(start_time, mean_heading, c='purple', s=80, marker='s')
        plt.scatter(end_time, mean_heading, c='purple', s=80, marker='s', alpha=0.5)
    
    plt.title('Gyro Heading Correction with Fixed Turn Detection', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Heading (°)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plot_file = os.path.join(output_folder, 'improved_heading_correction_visualization.png')
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved visualization to {plot_file}")
    
    # Create trajectory comparison visualization
    plot_trajectory_comparison(gyro_data, ground_truth_data, output_folder)
    
    print("Improved heading correction completed!")

def plot_trajectory_comparison(gyro_data, ground_truth_data, output_folder):
    """
    Create a visualization of the trajectory comparison between original and corrected headings
    """
    # Extract ground truth positions
    ground_truth_positions = []
    for _, row in ground_truth_data.iterrows():
        if row['Type'] in ['Initial_Location', 'Ground_truth_Location']:
            east = row['value_4']
            north = row['value_5']
            ground_truth_positions.append((east, north))
    
    # Calculate trajectories if ground truth positions are available
    if len(ground_truth_positions) > 0:
        # Start with the first ground truth position
        start_position = ground_truth_positions[0]
        
        # Compute trajectories
        orig_traj = [start_position]
        corr_traj = [start_position]
        
        curr_orig_pos = start_position
        curr_corr_pos = start_position
        
        # Use adaptive step size
        for i in range(1, len(gyro_data)):
            time_diff = gyro_data.loc[i, 'Time_Seconds'] - gyro_data.loc[i-1, 'Time_Seconds']
            step_size = max(time_diff * 0.5, 0.01)  # Scaled step size based on time difference
            
            # Get headings
            orig_heading = gyro_data.loc[i, 'value_3']  # Heading is in value_3
            corr_heading = gyro_data.loc[i, 'Corrected_Heading']
            
            # Calculate new positions - simple dead reckoning
            # Note: in compass/gyro heading, 0° is North, 90° is East
            orig_north = curr_orig_pos[1] + step_size * np.cos(np.radians(orig_heading))
            orig_east = curr_orig_pos[0] + step_size * np.sin(np.radians(orig_heading))
            
            corr_north = curr_corr_pos[1] + step_size * np.cos(np.radians(corr_heading))
            corr_east = curr_corr_pos[0] + step_size * np.sin(np.radians(corr_heading))
            
            # Update current positions
            curr_orig_pos = (orig_east, orig_north)
            curr_corr_pos = (corr_east, corr_north)
            
            # Add to trajectories
            orig_traj.append(curr_orig_pos)
            corr_traj.append(curr_corr_pos)
        
        # Convert to numpy arrays
        orig_traj = np.array(orig_traj)
        corr_traj = np.array(corr_traj)
        ground_truth_positions = np.array(ground_truth_positions)
        
        # Create plot
        plt.figure(figsize=(14, 12))
        
        # Plot trajectories
        plt.plot(orig_traj[:, 0], orig_traj[:, 1], 'b-', alpha=0.5, label='Traditional Gyro Trajectory')
        plt.plot(corr_traj[:, 0], corr_traj[:, 1], 'g-', label='Corrected Trajectory')
        
        # Plot ground truth
        plt.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 1], c='r', s=120, marker='o', label='Ground Truth Positions')
        plt.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 'r--', alpha=0.7, label='Ground Truth Path')
        
        # Add labels for ground truth points
        for i, (east, north) in enumerate(ground_truth_positions):
            plt.text(east, north + 2, f"GT{i}", color='red', fontsize=12, weight='bold')
        
        plt.title('Trajectory Comparison: Traditional vs. Corrected', fontsize=16)
        plt.xlabel('East', fontsize=14)
        plt.ylabel('North', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.axis('equal')
        
        # Save the plot
        traj_file = os.path.join(output_folder, 'improved_trajectory_comparison.png')
        plt.savefig(traj_file, dpi=300)
        plt.close()
        print(f"Saved trajectory comparison to {traj_file}")

if __name__ == "__main__":
    main() 