import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from scipy.signal import savgol_filter

# Define paths
gyro_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2/gyro_heading_data.csv'
correction_points_path = '/Users/shaoxinyi/Downloads/FYP2/src/Phase5/output/enhanced_correction_points.csv'
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

def apply_improved_correction(gyro_data, correction_points, turn_fade_length=15):
    """
    Apply improved heading corrections with better handling of turns
    
    Parameters:
    -----------
    gyro_data : DataFrame
        Gyro data with headings
    correction_points : list of tuples
        Each tuple contains (time, reference_heading, is_turn)
    turn_fade_length : int
        Number of points over which to fade out the correction for turns
        
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
        
        # If this is a turn point, apply more carefully with enhanced fade
        if is_turn:
            # Gradually apply the offset over a short distance with improved fade
            affected_indices = gyro_data.loc[time_range].index
            
            if len(affected_indices) > 0:
                # Apply full correction only to the closest point
                corrected_headings.loc[closest_idx] = normalize_angle(gyro_heading + offset)
                
                # For turn points, use a longer and smoother fade
                fade_length = min(turn_fade_length, len(affected_indices))
                # Use cosine fade for smoother transition (1 at center, 0 at edges)
                fade_weights = np.cos(np.linspace(0, np.pi/2, fade_length))**2
                
                # Apply faded correction to next few points
                for j, idx in enumerate(affected_indices[:fade_length]):
                    if idx != closest_idx:  # Skip the closest point (already done)
                        weight = fade_weights[j]
                        current_heading = gyro_data.loc[idx, 'Gyro Heading (°)']
                        corrected_headings.loc[idx] = normalize_angle(current_heading + offset * weight)
        else:
            # For non-turn points, apply a smoothed offset to the range
            # This creates a more gradual correction that better preserves relative changes
            gyro_headings_in_range = gyro_data.loc[time_range, 'Gyro Heading (°)']
            
            # Apply the full offset to the first point in the range
            if len(time_range) > 0 and any(time_range):
                first_idx = gyro_data.loc[time_range].index[0]
                corrected_headings.loc[first_idx] = normalize_angle(gyro_data.loc[first_idx, 'Gyro Heading (°)'] + offset)
                
                # If there are at least 2 points in the range, apply differential correction
                if len(gyro_data.loc[time_range]) >= 2:
                    # Get consecutive pairs of indices
                    indices = gyro_data.loc[time_range].index
                    
                    for j in range(1, len(indices)):
                        prev_idx = indices[j-1]
                        curr_idx = indices[j]
                        
                        # Get the original heading difference between consecutive points
                        orig_diff = angle_difference(
                            gyro_data.loc[curr_idx, 'Gyro Heading (°)'], 
                            gyro_data.loc[prev_idx, 'Gyro Heading (°)']
                        )
                        
                        # Determine direction of the original change
                        if normalize_angle(gyro_data.loc[curr_idx, 'Gyro Heading (°)'] - 
                                          gyro_data.loc[prev_idx, 'Gyro Heading (°)'] + 180) > 180:
                            orig_diff = -orig_diff  # Negative change
                        
                        # Apply the same relative change to the corrected heading
                        prev_corrected = corrected_headings.loc[prev_idx]
                        corrected_headings.loc[curr_idx] = normalize_angle(prev_corrected + orig_diff)
    
    return corrected_headings

def interpolate_angles(angles, times, target_times):
    """
    Interpolate angles with proper handling of angle wrapping
    
    Parameters:
    -----------
    angles : array-like
        Array of angle values to interpolate
    times : array-like
        Array of time values corresponding to angles
    target_times : array-like
        Array of times at which to interpolate angles
        
    Returns:
    --------
    interpolated_angles : array
        Interpolated angle values at target_times
    """
    # Convert to numpy arrays for processing
    angles = np.array([normalize_angle(a) for a in angles])
    times = np.array(times)
    target_times = np.array(target_times)
    
    # Convert angles to unit vectors on the unit circle
    x = np.cos(np.radians(angles))
    y = np.sin(np.radians(angles))
    
    # Interpolate the x and y components
    x_interp = np.interp(target_times, times, x)
    y_interp = np.interp(target_times, times, y)
    
    # Convert back to angles
    interp_angles = np.degrees(np.arctan2(y_interp, x_interp))
    
    # Normalize to [0, 360)
    return np.array([normalize_angle(a) for a in interp_angles])

def main():
    # Load data
    print("Loading data...")
    gyro_data = pd.read_csv(gyro_data_path)
    
    # Check if the correction points file exists
    if not os.path.exists(correction_points_path):
        print(f"Correction points file {correction_points_path} not found.")
        print("Please run enhanced_qs_filtering.py first.")
        return
    
    correction_points_df = pd.read_csv(correction_points_path)
    
    print(f"Gyro data shape: {gyro_data.shape}")
    print(f"Correction points shape: {correction_points_df.shape}")
    
    # Convert correction points to list of tuples
    correction_points = [
        (row['Time'], row['Reference_Heading'], row['Is_Turn']) 
        for _, row in correction_points_df.iterrows()
    ]
    
    # Apply the improved correction
    print("\nApplying improved heading corrections...")
    gyro_data['Improved_Heading'] = apply_improved_correction(gyro_data, correction_points)
    
    # Calculate errors after correction
    gyro_data['Improved_Error'] = gyro_data.apply(
        lambda row: angle_difference(row['Improved_Heading'], row['Ground Truth Heading (°)'])
        if not pd.isna(row['Ground Truth Heading (°)']) else np.nan,
        axis=1
    )
    
    # Save the corrected data
    output_file = os.path.join(output_folder, 'improved_gyro_heading_corrected.csv')
    gyro_data.to_csv(output_file, index=False)
    print(f"Saved improved heading data to {output_file}")
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Plot original gyro heading
    plt.plot(gyro_data['Time (s)'], gyro_data['Gyro Heading (°)'], 'b-', alpha=0.5, label='Original Gyro Heading')
    
    # Plot improved heading
    plt.plot(gyro_data['Time (s)'], gyro_data['Improved_Heading'], 'g-', label='Improved Heading')
    
    # If the old corrected heading exists, plot it for comparison
    old_corrected_file = '/Users/shaoxinyi/Downloads/FYP2/src/Phase5/output/gyro_heading_corrected.csv'
    if os.path.exists(old_corrected_file):
        old_corrected = pd.read_csv(old_corrected_file)
        plt.plot(old_corrected['Time (s)'], old_corrected['Corrected_Heading'], 
                 'r-', alpha=0.5, label='Previous Corrected Heading')
    
    # Plot ground truth where available
    mask = ~pd.isna(gyro_data['Ground Truth Heading (°)'])
    plt.plot(gyro_data.loc[mask, 'Time (s)'], gyro_data.loc[mask, 'Ground Truth Heading (°)'], 
             'r--', alpha=0.7, label='Ground Truth')
    
    # Mark correction points
    for time, heading, is_turn in correction_points:
        marker = '^' if is_turn else 'o'
        color = 'orange' if is_turn else 'red'
        plt.plot(time, heading, marker=marker, color=color, markersize=8)
        
    plt.title('Improved Gyro Heading Correction', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Heading (degrees)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Save plot
    plot_file = os.path.join(output_folder, 'improved_heading_correction_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {plot_file}")
    
    # Plot error comparison
    plt.figure(figsize=(14, 6))
    
    # Plot original error
    plt.plot(gyro_data['Time (s)'], gyro_data['Heading Error (°)'], 
             'r-', alpha=0.5, label='Original Error')
    
    # Plot improved error
    plt.plot(gyro_data['Time (s)'], gyro_data['Improved_Error'], 
             'g-', label='Improved Error')
    
    # If old error data exists, plot it for comparison
    if os.path.exists(old_corrected_file):
        old_corrected['Corrected_Error'] = old_corrected.apply(
            lambda row: angle_difference(row['Corrected_Heading'], row['Ground Truth Heading (°)'])
            if not pd.isna(row['Ground Truth Heading (°)']) else np.nan,
            axis=1
        )
        plt.plot(old_corrected['Time (s)'], old_corrected['Corrected_Error'], 
                 'm-', alpha=0.5, label='Previous Corrected Error')
    
    # Mark correction points
    for time, _, is_turn in correction_points:
        linestyle = '--' if is_turn else '-'
        plt.axvline(x=time, color='b', linestyle=linestyle, alpha=0.3)
        
    plt.title('Heading Error Comparison', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Heading Error (degrees)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.ylim(0, max(gyro_data['Heading Error (°)'].max(), gyro_data['Improved_Error'].max()) * 1.1)
    
    # Save error plot
    error_plot_file = os.path.join(output_folder, 'improved_heading_error_comparison.png')
    plt.savefig(error_plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved error comparison to {error_plot_file}")
    
    # Calculate error statistics
    original_error_mean = gyro_data['Heading Error (°)'].mean()
    improved_error_mean = gyro_data['Improved_Error'].mean()
    
    original_error_max = gyro_data['Heading Error (°)'].max()
    improved_error_max = gyro_data['Improved_Error'].max()
    
    print("\nError Statistics:")
    print(f"Original Mean Error: {original_error_mean:.2f}°")
    print(f"Improved Mean Error: {improved_error_mean:.2f}°")
    print(f"Original Max Error: {original_error_max:.2f}°")
    print(f"Improved Max Error: {improved_error_max:.2f}°")
    
    if improved_error_mean < original_error_mean:
        error_reduction = (original_error_mean - improved_error_mean) / original_error_mean * 100
        print(f"Error Reduction: {error_reduction:.2f}%")
    else:
        error_increase = (improved_error_mean - original_error_mean) / original_error_mean * 100
        print(f"Error Increase: {error_increase:.2f}%")
    
    # Count how many entries have improved error after correction
    mask = ~pd.isna(gyro_data['Heading Error (°)']) & ~pd.isna(gyro_data['Improved_Error'])
    improved_entries = (gyro_data.loc[mask, 'Improved_Error'] < gyro_data.loc[mask, 'Heading Error (°)']).sum()
    total_entries = mask.sum()
    
    if total_entries > 0:
        improvement_percentage = improved_entries / total_entries * 100
        print(f"Entries with reduced error: {improved_entries}/{total_entries} ({improvement_percentage:.2f}%)")
    
    # Compare with previous correction if available
    if os.path.exists(old_corrected_file):
        previous_error_mean = old_corrected['Corrected_Error'].mean()
        print(f"Previous Corrected Mean Error: {previous_error_mean:.2f}°")
        
        if improved_error_mean < previous_error_mean:
            prev_improvement = (previous_error_mean - improved_error_mean) / previous_error_mean * 100
            print(f"Improvement over previous correction: {prev_improvement:.2f}%")
        else:
            prev_degradation = (improved_error_mean - previous_error_mean) / previous_error_mean * 100
            print(f"Degradation from previous correction: {prev_degradation:.2f}%")

if __name__ == "__main__":
    main() 