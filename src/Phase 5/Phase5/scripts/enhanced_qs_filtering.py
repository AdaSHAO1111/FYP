import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
import math

# Define paths
gyro_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2/gyro_heading_data.csv'
qs_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 4/1536_quasi_static_data.csv'
output_folder = '/Users/shaoxinyi/Downloads/FYP2/src/Phase5/output'

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
    turn_intervals : list of tuples
        Tuple (start_idx, end_idx) for each detected turn
    """
    headings = np.array([normalize_angle(h) for h in headings])
    times = np.array(times)
    
    turn_indices = []
    turn_rates = []
    turn_start_idx = None
    turn_intervals = []
    
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
                
                # Mark the start of a turn if not already in a turn
                if turn_start_idx is None:
                    turn_start_idx = i
            elif turn_start_idx is not None:
                # End of a turn - store the interval
                turn_intervals.append((turn_start_idx, i))
                turn_start_idx = None
    
    # If we ended in a turn, close it
    if turn_start_idx is not None:
        turn_intervals.append((turn_start_idx, len(headings)-1))
    
    return turn_indices, turn_rates, turn_intervals

def validate_qs_segment(qs_segment, gyro_data, heading_change_threshold=5.0, gyro_variance_threshold=8.0):
    """
    Validate if a QS segment is genuine by cross-checking with gyro behavior during the same time interval
    
    Parameters:
    -----------
    qs_segment : tuple
        (start_time, end_time, mean_heading) for the QS segment
    gyro_data : DataFrame
        Gyroscope data with time and heading
    heading_change_threshold : float
        Maximum allowed heading change in gyro during a QS segment
    gyro_variance_threshold : float
        Maximum allowed variance in gyro heading during a QS segment
        
    Returns:
    --------
    valid : bool
        True if the QS segment is valid, False otherwise
    reason : str
        Reason for invalidation if not valid
    """
    start_time, end_time, _ = qs_segment
    
    # Find gyro data points within this time range
    time_range = (gyro_data['Time (s)'] >= start_time) & (gyro_data['Time (s)'] <= end_time)
    gyro_in_range = gyro_data.loc[time_range]
    
    # If no gyro data in this range, consider it invalid
    if len(gyro_in_range) < 2:
        return False, "Insufficient gyro data points"
    
    # Check for significant heading changes in gyro
    gyro_headings = gyro_in_range['Gyro Heading (°)'].values
    max_heading_change = max([angle_difference(gyro_headings[i], gyro_headings[i-1]) 
                              for i in range(1, len(gyro_headings))], default=0)
    
    # Check turn rate
    gyro_times = gyro_in_range['Time (s)'].values
    max_turn_rate = 0
    for i in range(1, len(gyro_headings)):
        time_diff = gyro_times[i] - gyro_times[i-1]
        if time_diff > 0:  # Avoid division by zero
            turn_rate = angle_difference(gyro_headings[i], gyro_headings[i-1]) / time_diff
            max_turn_rate = max(max_turn_rate, abs(turn_rate))
    
    # Calculate variance of gyro heading
    # For circular data, we need a circular variance
    angles_rad = np.radians(gyro_headings)
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    r = np.sqrt(mean_sin**2 + mean_cos**2)  # mean resultant length
    circular_variance = 1 - r  # circular variance
    heading_variance = circular_variance * 180  # Scale to degrees for easier interpretation
    
    # Check if the segment overlaps with detected turns
    overlaps_with_turn = 'Is_Turn' in gyro_in_range.columns and gyro_in_range['Is_Turn'].any()
    
    # Return True only if all validation checks pass
    if max_heading_change > heading_change_threshold:
        return False, f"High heading change: {max_heading_change:.2f}°"
    elif heading_variance > gyro_variance_threshold:
        return False, f"High heading variance: {heading_variance:.2f}"
    elif max_turn_rate > 10.0:  # Additional check for high turn rates
        return False, f"High turn rate: {max_turn_rate:.2f}°/s"
    elif overlaps_with_turn:
        return False, "Overlaps with detected turn"
    
    return True, "Valid"

def enhanced_identify_qs_segments(qs_data, gyro_data):
    """
    Identify and validate continuous segments of quasi-static field data
    
    Returns:
    --------
    valid_segments : list of tuples
        Each tuple contains (start_time, end_time, mean_heading) for a valid QS segment
    invalid_segments : list of tuples
        Each tuple contains (start_time, end_time, mean_heading, reason) for invalid segments
    """
    if len(qs_data) == 0:
        return [], []
    
    # Sort by time (convert milliseconds to seconds for consistency with gyro data)
    qs_data = qs_data.copy()
    qs_data['Time_Seconds'] = qs_data['Time'] / 1000  # Convert milliseconds to seconds
    qs_data = qs_data.sort_values('Time_Seconds')
    
    # Identify gaps in time that are larger than 0.5 seconds
    qs_data['Time_Diff'] = qs_data['Time_Seconds'].diff()
    segment_starts = [0] + list(qs_data[qs_data['Time_Diff'] > 0.5].index)
    
    # Handle the last segment
    segment_ends = segment_starts[1:] + [len(qs_data)]
    
    # Identify turn intervals in gyro data
    turn_indices, turn_rates, turn_intervals = detect_turns(
        gyro_data['Gyro Heading (°)'].values, 
        gyro_data['Time (s)'].values
    )
    
    # Mark turns in the gyro data
    gyro_data['Is_Turn'] = False
    for start_idx, end_idx in turn_intervals:
        gyro_data.loc[start_idx:end_idx, 'Is_Turn'] = True
    
    valid_segments = []
    invalid_segments = []
    
    for start, end in zip(segment_starts, segment_ends):
        segment = qs_data.iloc[start:end]
        if len(segment) > 0:
            # Calculate mean heading for this segment
            mean_heading = angular_mean(segment['True_Heading'])
            
            # Create segment tuple
            qs_segment = (
                segment['Time_Seconds'].min(),
                segment['Time_Seconds'].max(),
                mean_heading
            )
            
            # Validate the segment
            is_valid, reason = validate_qs_segment(qs_segment, gyro_data)
            
            if is_valid:
                valid_segments.append(qs_segment)
            else:
                invalid_segments.append((*qs_segment, reason))
    
    return valid_segments, invalid_segments

def identify_missed_turns(gyro_data, correction_points, min_gap_seconds=3.0):
    """
    Identify turns that are missed in the correction points
    
    Parameters:
    -----------
    gyro_data : DataFrame
        Gyroscope data with time and heading
    correction_points : list of tuples
        Each tuple contains (time, reference_heading, is_turn)
    min_gap_seconds : float
        Minimum time gap between correction points to consider adding a turn
        
    Returns:
    --------
    missed_turns : list of tuples
        Each tuple contains (time, gyro_heading, True) for missed turns
    """
    # Extract turns from gyro data
    turn_indices, turn_rates, turn_intervals = detect_turns(
        gyro_data['Gyro Heading (°)'].values, 
        gyro_data['Time (s)'].values
    )
    
    # Convert correction points to just times
    correction_times = [point[0] for point in correction_points]
    
    # Sort correction times
    correction_times.sort()
    
    # Find significant turns that are far from correction points
    missed_turns = []
    
    # For each turn interval
    for start_idx, end_idx in turn_intervals:
        # Get the middle of the turn
        mid_idx = (start_idx + end_idx) // 2
        turn_time = gyro_data.loc[mid_idx, 'Time (s)']
        
        # Check if this turn is covered by any correction point
        covered = False
        for corr_time in correction_times:
            if abs(turn_time - corr_time) < min_gap_seconds:
                covered = True
                break
        
        # If not covered, it's a missed turn
        if not covered:
            turn_heading = gyro_data.loc[mid_idx, 'Gyro Heading (°)']
            missed_turns.append((turn_time, turn_heading, True))  # Mark as a turn
    
    return missed_turns

def main():
    # Load data
    print("Loading data...")
    gyro_data = pd.read_csv(gyro_data_path)
    qs_data = pd.read_csv(qs_data_path)
    
    print(f"Gyro data shape: {gyro_data.shape}")
    print(f"QS data shape: {qs_data.shape}")
    
    # Detect turns in the gyro data
    print("\nDetecting turns in gyro data...")
    turn_indices, turn_rates, turn_intervals = detect_turns(
        gyro_data['Gyro Heading (°)'].values, 
        gyro_data['Time (s)'].values
    )
    
    print(f"Detected {len(turn_indices)} significant turn points and {len(turn_intervals)} turn intervals")
    
    # Identify and validate QS segments
    print("\nIdentifying and validating quasi-static field segments...")
    valid_segments, invalid_segments = enhanced_identify_qs_segments(qs_data, gyro_data)
    
    print(f"Identified {len(valid_segments)} valid QS segments")
    print(f"Filtered out {len(invalid_segments)} invalid QS segments")
    
    # Generate correction points from valid QS segments
    correction_points = []
    
    # Process QS segments
    for start_time, end_time, mean_heading in valid_segments:
        # Add correction point at the start of the segment
        correction_points.append((start_time, mean_heading, False))  # Not a turn
    
    # Include ground truth points for calibration
    gt_entries = gyro_data[~pd.isna(gyro_data['Ground Truth Heading (°)'])].copy()
    print(f"Found {len(gt_entries)} entries with ground truth heading values.")
    
    # Group by time bins to identify unique ground truth points
    gt_entries['Time_Bin'] = (gt_entries['Time (s)'] // 1).astype(int)  # Bin by seconds
    
    # Get median values for each time bin
    gt_points = gt_entries.groupby('Time_Bin').agg({
        'Time (s)': 'median',
        'Ground Truth Heading (°)': 'median'
    }).reset_index()
    
    # Add ground truth points to correction points
    for _, point in gt_points.iterrows():
        # Check if this point is in a turn
        in_turn = False
        for start_idx, end_idx in turn_intervals:
            if (gyro_data.loc[start_idx, 'Time (s)'] <= point['Time (s)'] <= 
                gyro_data.loc[end_idx, 'Time (s)']):
                in_turn = True
                break
                
        correction_points.append((
            point['Time (s)'], 
            point['Ground Truth Heading (°)'],
            in_turn  # Mark as turn if in a turn interval
        ))
    
    # Check for missed turns
    missed_turns = identify_missed_turns(gyro_data, correction_points)
    print(f"Identified {len(missed_turns)} missed turns")
    
    # Add missed turns to correction points
    correction_points.extend(missed_turns)
    
    # Sort all correction points by time
    correction_points.sort(key=lambda x: x[0])
    
    print(f"Using {len(correction_points)} correction points total.")
    
    # Save filtered QS segments and correction points for analysis
    analysis_data = {
        'valid_qs_segments': valid_segments,
        'invalid_qs_segments': invalid_segments,
        'correction_points': correction_points,
        'turn_intervals': turn_intervals
    }
    
    # Save as CSV for easy viewing
    valid_qs_df = pd.DataFrame(valid_segments, columns=['Start_Time', 'End_Time', 'Mean_Heading'])
    valid_qs_file = os.path.join(output_folder, 'valid_qs_segments.csv')
    valid_qs_df.to_csv(valid_qs_file, index=False)
    print(f"Saved valid QS segments to {valid_qs_file}")
    
    invalid_qs_df = pd.DataFrame(invalid_segments, columns=['Start_Time', 'End_Time', 'Mean_Heading', 'Reason'])
    invalid_qs_file = os.path.join(output_folder, 'invalid_qs_segments.csv')
    invalid_qs_df.to_csv(invalid_qs_file, index=False)
    print(f"Saved invalid QS segments to {invalid_qs_file}")
    
    corr_points_df = pd.DataFrame(correction_points, columns=['Time', 'Reference_Heading', 'Is_Turn'])
    corr_points_file = os.path.join(output_folder, 'enhanced_correction_points.csv')
    corr_points_df.to_csv(corr_points_file, index=False)
    print(f"Saved enhanced correction points to {corr_points_file}")
    
    # Create visualization to show QS segments and gyro data
    plt.figure(figsize=(12, 8))
    
    # Plot gyro heading
    plt.plot(gyro_data['Time (s)'], gyro_data['Gyro Heading (°)'], 'b-', alpha=0.5, label='Gyro Heading')
    
    # Mark turns in gyro data
    for start_idx, end_idx in turn_intervals:
        plt.axvspan(gyro_data.loc[start_idx, 'Time (s)'], gyro_data.loc[end_idx, 'Time (s)'], 
                   alpha=0.2, color='orange', label='_Turn Interval')
    
    # Plot valid QS segments
    for i, (start_time, end_time, mean_heading) in enumerate(valid_segments):
        if i == 0:
            plt.axvspan(start_time, end_time, alpha=0.3, color='green', label='Valid QS Segment')
            plt.plot([start_time, end_time], [mean_heading, mean_heading], 'g-', linewidth=2)
        else:
            plt.axvspan(start_time, end_time, alpha=0.3, color='green')
            plt.plot([start_time, end_time], [mean_heading, mean_heading], 'g-', linewidth=2)
    
    # Plot invalid QS segments
    for i, (start_time, end_time, mean_heading, reason) in enumerate(invalid_segments):
        if i == 0:
            plt.axvspan(start_time, end_time, alpha=0.2, color='red', label='Invalid QS Segment')
            plt.plot([start_time, end_time], [mean_heading, mean_heading], 'r-', linewidth=2, alpha=0.7)
        else:
            plt.axvspan(start_time, end_time, alpha=0.2, color='red')
            plt.plot([start_time, end_time], [mean_heading, mean_heading], 'r-', linewidth=2, alpha=0.7)
    
    # Mark missed turns
    for time, heading, _ in missed_turns:
        plt.plot(time, heading, 'ro', markersize=8, label='_Missed Turn')
        plt.text(time, heading+10, 'Missed Turn', rotation=45, fontsize=8, ha='left')
    
    # Plot ground truth heading
    mask = ~pd.isna(gyro_data['Ground Truth Heading (°)'])
    plt.plot(gyro_data.loc[mask, 'Time (s)'], gyro_data.loc[mask, 'Ground Truth Heading (°)'], 
             'r--', alpha=0.7, label='Ground Truth')
    
    # Create custom legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicate turn interval labels
    seen = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    
    plt.legend(unique_handles, unique_labels)
    plt.title('Enhanced QS Detection and Validation')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading (degrees)')
    plt.grid(True)
    
    # Save plot
    plot_file = os.path.join(output_folder, 'enhanced_qs_validation.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced QS validation visualization to {plot_file}")

if __name__ == "__main__":
    main() 