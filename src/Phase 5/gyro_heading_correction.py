import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from scipy.stats import circmean

# Define paths
gyro_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2/gyro_heading_data.csv'
qs_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 4/1536_quasi_static_data.csv'
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

def identify_qs_segments(qs_data):
    """
    Identify continuous segments of quasi-static field data
    
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
    
    segments = []
    for start, end in zip(segment_starts, segment_ends):
        segment = qs_data.iloc[start:end]
        if len(segment) > 0:
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
    corrected_headings = gyro_data['Gyro Heading (°)'].copy()
    
    # Sort correction points by time
    correction_points = sorted(correction_points, key=lambda x: x[0])
    
    for i in range(len(correction_points)):
        current_point = correction_points[i]
        current_time, ref_heading, is_turn = current_point
        
        # Find the closest gyro data point
        closest_idx = (gyro_data['Time (s)'] - current_time).abs().idxmin()
        gyro_heading = gyro_data.loc[closest_idx, 'Gyro Heading (°)']
        
        # Calculate offset (reference - gyro)
        offset = normalize_angle(ref_heading - gyro_heading + 180) - 180
        
        # Determine range of application
        if i < len(correction_points) - 1:
            next_time = correction_points[i + 1][0]
            # Apply from this point to just before the next point
            time_range = (gyro_data['Time (s)'] >= current_time) & (gyro_data['Time (s)'] < next_time)
        else:
            # Apply from this point to the end
            time_range = gyro_data['Time (s)'] >= current_time
        
        # If this is a turn point, apply more carefully
        if is_turn:
            # Gradually apply the offset over a short distance
            affected_indices = gyro_data.loc[time_range].index
            
            if len(affected_indices) > 0:
                # Apply full correction only to the closest point
                corrected_headings.loc[closest_idx] = normalize_angle(gyro_heading + offset)
                
                # Calculate how quickly to phase out the correction
                fade_length = min(10, len(affected_indices))
                fade_weights = np.linspace(1, 0, fade_length)
                
                # Apply faded correction to next few points
                for j, idx in enumerate(affected_indices[:fade_length]):
                    if idx != closest_idx:  # Skip the closest point (already done)
                        weight = fade_weights[j]
                        current_heading = gyro_data.loc[idx, 'Gyro Heading (°)']
                        corrected_headings.loc[idx] = normalize_angle(current_heading + offset * weight)
        else:
            # For non-turn points, apply the full offset to the range
            gyro_headings_in_range = gyro_data.loc[time_range, 'Gyro Heading (°)']
            corrected_headings.loc[time_range] = normalize_angle(gyro_headings_in_range + offset)
    
    return corrected_headings

def main():
    # Load data
    print("Loading data...")
    gyro_data = pd.read_csv(gyro_data_path)
    qs_data = pd.read_csv(qs_data_path)
    
    print(f"Gyro data shape: {gyro_data.shape}")
    print(f"QS data shape: {qs_data.shape}")
    
    # Detect turns in the gyro data
    print("\nDetecting turns in gyro data...")
    turn_indices, turn_rates = detect_turns(
        gyro_data['Gyro Heading (°)'].values, 
        gyro_data['Time (s)'].values
    )
    
    print(f"Detected {len(turn_indices)} significant turns")
    
    # Mark turns in the data
    gyro_data['Is_Turn'] = False
    gyro_data.loc[turn_indices, 'Is_Turn'] = True
    
    # Identify QS segments
    print("\nIdentifying quasi-static field segments...")
    qs_segments = identify_qs_segments(qs_data)
    print(f"Identified {len(qs_segments)} QS segments")
    
    # Generate correction points from both QS segments and ground truth points
    correction_points = []
    
    # First, process QS segments
    for start_time, end_time, mean_heading in qs_segments:
        # Check if this segment contains any turns
        segment_range = (gyro_data['Time (s)'] >= start_time) & (gyro_data['Time (s)'] <= end_time)
        contains_turn = gyro_data.loc[segment_range, 'Is_Turn'].any()
        
        # Add correction point at the start of the segment
        correction_points.append((start_time, mean_heading, contains_turn))
    
    # Additionally, include ground truth points for calibration
    gt_entries = gyro_data[~pd.isna(gyro_data['Ground Truth Heading (°)'])].copy()
    print(f"Found {len(gt_entries)} entries with ground truth heading values.")
    
    # Group by time bins to identify unique ground truth points
    gt_entries['Time_Bin'] = (gt_entries['Time (s)'] // 1).astype(int)  # Bin by seconds
    
    # Get median values for each time bin
    gt_points = gt_entries.groupby('Time_Bin').agg({
        'Time (s)': 'median',
        'Ground Truth Heading (°)': 'median',
        'Is_Turn': lambda x: any(x)  # True if any point in bin is a turn
    }).reset_index()
    
    # Add ground truth points to correction points
    for _, point in gt_points.iterrows():
        correction_points.append((
            point['Time (s)'], 
            point['Ground Truth Heading (°)'],
            point['Is_Turn']
        ))
    
    # Sort all correction points by time
    correction_points.sort(key=lambda x: x[0])
    
    print(f"Using {len(correction_points)} correction points total.")
    
    # Apply corrections
    print("\nApplying heading corrections...")
    gyro_data['Corrected_Heading'] = apply_correction(gyro_data, correction_points)
    
    # Calculate errors after correction
    gyro_data['Corrected_Error'] = gyro_data.apply(
        lambda row: angle_difference(row['Corrected_Heading'], row['Ground Truth Heading (°)'])
        if not pd.isna(row['Ground Truth Heading (°)']) else np.nan,
        axis=1
    )
    
    # Save the corrected data
    output_file = os.path.join(output_folder, 'gyro_heading_corrected.csv')
    gyro_data.to_csv(output_file, index=False)
    print(f"Saved corrected heading data to {output_file}")
    
    # Display correction points used
    print("\nCorrection Points Used:")
    for i, (time, heading, is_turn) in enumerate(correction_points):
        turn_status = "Turn Area" if is_turn else "Non-Turn"
        print(f"Point {i+1}: Time={time:.2f}s, Reference Heading={heading:.2f}°, {turn_status}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot original gyro heading
    plt.plot(gyro_data['Time (s)'], gyro_data['Gyro Heading (°)'], 'b-', alpha=0.5, label='Original Gyro Heading')
    
    # Plot corrected heading
    plt.plot(gyro_data['Time (s)'], gyro_data['Corrected_Heading'], 'g-', label='Corrected Heading')
    
    # Plot ground truth where available
    mask = ~pd.isna(gyro_data['Ground Truth Heading (°)'])
    plt.plot(gyro_data.loc[mask, 'Time (s)'], gyro_data.loc[mask, 'Ground Truth Heading (°)'], 
             'r--', alpha=0.7, label='Ground Truth')
    
    # Mark turn points
    turn_times = gyro_data.loc[gyro_data['Is_Turn'], 'Time (s)']
    turn_headings = gyro_data.loc[gyro_data['Is_Turn'], 'Gyro Heading (°)']
    plt.scatter(turn_times, turn_headings, c='orange', s=30, alpha=0.5, label='Detected Turns')
    
    # Mark correction points
    for time, heading, is_turn in correction_points:
        marker = '^' if is_turn else 'o'
        color = 'orange' if is_turn else 'red'
        plt.plot(time, heading, marker=marker, color=color, markersize=8)
        
    plt.title('Gyro Heading Correction using QS and Ground Truth References')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading (degrees)')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_file = os.path.join(output_folder, 'gyro_heading_correction_plot.png')
    plt.savefig(plot_file, dpi=300)
    print(f"Saved visualization to {plot_file}")
    
    # Plot error comparison
    plt.figure(figsize=(12, 6))
    
    # Plot original error
    plt.plot(gyro_data['Time (s)'], gyro_data['Heading Error (°)'], 'r-', alpha=0.5, label='Original Error')
    
    # Plot corrected error
    plt.plot(gyro_data['Time (s)'], gyro_data['Corrected_Error'], 'g-', label='Corrected Error')
    
    # Mark correction points
    for time, _, is_turn in correction_points:
        linestyle = '--' if is_turn else '-'
        plt.axvline(x=time, color='b', linestyle=linestyle, alpha=0.3)
        
    plt.title('Heading Error Comparison: Original vs. Corrected')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading Error (degrees)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, max(gyro_data['Heading Error (°)'].max(), gyro_data['Corrected_Error'].max()) * 1.1)
    
    # Save error plot
    error_plot_file = os.path.join(output_folder, 'heading_error_comparison.png')
    plt.savefig(error_plot_file, dpi=300)
    print(f"Saved error comparison to {error_plot_file}")
    
    # Calculate error statistics
    original_error_mean = gyro_data['Heading Error (°)'].mean()
    corrected_error_mean = gyro_data['Corrected_Error'].mean()
    
    original_error_max = gyro_data['Heading Error (°)'].max()
    corrected_error_max = gyro_data['Corrected_Error'].max()
    
    print("\nError Statistics:")
    print(f"Original Mean Error: {original_error_mean:.2f}°")
    print(f"Corrected Mean Error: {corrected_error_mean:.2f}°")
    print(f"Original Max Error: {original_error_max:.2f}°")
    print(f"Corrected Max Error: {corrected_error_max:.2f}°")
    
    if corrected_error_mean < original_error_mean:
        error_reduction = (original_error_mean - corrected_error_mean) / original_error_mean * 100
        print(f"Error Reduction: {error_reduction:.2f}%")
    else:
        error_increase = (corrected_error_mean - original_error_mean) / original_error_mean * 100
        print(f"Error Increase: {error_increase:.2f}%")
    
    # Count how many entries have improved error after correction
    mask = ~pd.isna(gyro_data['Heading Error (°)']) & ~pd.isna(gyro_data['Corrected_Error'])
    improved_entries = (gyro_data.loc[mask, 'Corrected_Error'] < gyro_data.loc[mask, 'Heading Error (°)']).sum()
    total_entries = mask.sum()
    
    if total_entries > 0:
        improvement_percentage = improved_entries / total_entries * 100
        print(f"Entries with reduced error: {improved_entries}/{total_entries} ({improvement_percentage:.2f}%)")

    # Additional visualization: Show heading changes over time to highlight turns
    plt.figure(figsize=(12, 6))
    
    # Calculate heading changes (derivative)
    gyro_data['Heading_Change'] = gyro_data['Gyro Heading (°)'].diff() / gyro_data['Time (s)'].diff()
    
    # Apply smoothing to make turns more visible
    window_size = min(51, len(gyro_data))  # Must be odd and less than data length
    if window_size > 3 and window_size % 2 == 0:
        window_size -= 1  # Ensure it's odd
    
    if window_size >= 3:
        try:
            gyro_data['Smoothed_Change'] = savgol_filter(
                gyro_data['Heading_Change'].fillna(0), 
                window_size, 
                3  # Polynomial order
            )
        except:
            # Fallback if savgol_filter fails
            gyro_data['Smoothed_Change'] = gyro_data['Heading_Change']
    else:
        gyro_data['Smoothed_Change'] = gyro_data['Heading_Change']
    
    # Plot heading change rate
    plt.plot(gyro_data['Time (s)'], gyro_data['Smoothed_Change'], 'b-', label='Heading Change Rate')
    
    # Mark detected turns
    plt.scatter(turn_times, gyro_data.loc[gyro_data['Is_Turn'], 'Smoothed_Change'], 
                c='red', s=30, label='Detected Turns')
    
    # Mark turn threshold
    plt.axhline(y=15, color='r', linestyle='--', alpha=0.5, label='Turn Threshold')
    plt.axhline(y=-15, color='r', linestyle='--', alpha=0.5)
    
    plt.title('Heading Change Rate and Detected Turns')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading Change Rate (degrees/second)')
    plt.legend()
    plt.grid(True)
    
    # Save turn analysis plot
    turn_plot_file = os.path.join(output_folder, 'heading_turn_analysis.png')
    plt.savefig(turn_plot_file, dpi=300)
    print(f"Saved turn analysis to {turn_plot_file}")

if __name__ == "__main__":
    main() 