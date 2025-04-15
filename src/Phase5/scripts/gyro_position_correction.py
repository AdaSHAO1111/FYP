import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# Define paths
gyro_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_gyro_data.csv'
corrected_heading_path = '/Users/shaoxinyi/Downloads/FYP2/src/Phase5/output/gyro_heading_corrected.csv'
ground_truth_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_ground_truth_data.csv'
output_folder = '/Users/shaoxinyi/Downloads/FYP2/src/Phase5/output'

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def normalize_angle(angle):
    """
    Normalize angle to be in the range [0, 360)
    """
    return angle % 360

def angle_to_radians(angle):
    """
    Convert angle in degrees to radians
    """
    return angle * math.pi / 180.0

def calculate_position(initial_position, step_size, heading_deg):
    """
    Calculate next position given initial position, step size, and heading
    heading_deg is in degrees, measured clockwise from North
    Returns: (east, north) coordinates
    """
    # Convert heading to radians (adjusting for compass heading where 0 is North)
    heading_rad = angle_to_radians(90 - heading_deg)  # Convert from compass to cartesian
    
    # Calculate x (east) and y (north) displacements
    east = initial_position[0] + step_size * math.cos(heading_rad)
    north = initial_position[1] + step_size * math.sin(heading_rad)
    
    return (east, north)

def main():
    # Load data
    print("Loading data...")
    gyro_data = pd.read_csv(gyro_data_path)
    corrected_heading_data = pd.read_csv(corrected_heading_path)
    ground_truth_data = pd.read_csv(ground_truth_path)
    
    print(f"Gyro data shape: {gyro_data.shape}")
    print(f"Corrected heading data shape: {corrected_heading_data.shape}")
    print(f"Ground truth data shape: {ground_truth_data.shape}")
    
    # Extract ground truth positions and steps
    ground_truth_positions = []
    ground_truth_steps = []
    for _, row in ground_truth_data.iterrows():
        if row['Type'] in ['Initial_Location', 'Ground_truth_Location']:
            # Extract east and north coordinates
            east = row['value_4']
            north = row['value_5']
            step = row['step']
            ground_truth_positions.append((east, north))
            ground_truth_steps.append(step)
    
    # Create positions based on original gyro headings
    original_positions = []
    corrected_positions = []
    
    # Start with the initial position from ground truth
    if len(ground_truth_positions) > 0:
        current_original_position = ground_truth_positions[0]
        current_corrected_position = ground_truth_positions[0]
    else:
        current_original_position = (0, 0)
        current_corrected_position = (0, 0)
    
    original_positions.append(current_original_position)
    corrected_positions.append(current_corrected_position)
    
    # Compute average step between ground truth points for better step size estimation
    avg_step_distance = 0
    avg_step_count = 0
    for i in range(1, len(ground_truth_positions)):
        p1 = ground_truth_positions[i-1]
        p2 = ground_truth_positions[i]
        step_diff = ground_truth_steps[i] - ground_truth_steps[i-1]
        
        if step_diff > 0:  # Avoid division by zero
            dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            avg_step_distance += dist / step_diff
            avg_step_count += 1
    
    # Use average distance per step or a default value
    avg_step_size = avg_step_distance / avg_step_count if avg_step_count > 0 else 0.5
    print(f"Calculated average step size: {avg_step_size:.4f}")
    
    # Calibrate step size to match distance between ground truth points
    base_step_size = avg_step_size * 0.05  # Scale down to account for frequency of readings
    print(f"Using base step size: {base_step_size:.4f}")
    
    # Iterate through readings
    previous_time = corrected_heading_data.loc[0, 'Time (s)']
    
    for i in range(1, len(corrected_heading_data)):
        # Get headings
        original_heading = corrected_heading_data.loc[i, 'Gyro Heading (°)']
        corrected_heading = corrected_heading_data.loc[i, 'Corrected_Heading']
        
        # Calculate time difference for adaptive step size
        current_time = corrected_heading_data.loc[i, 'Time (s)']
        time_diff = current_time - previous_time
        previous_time = current_time
        
        # Adjust step size based on time difference (larger time diff = larger step)
        adaptive_step = base_step_size * max(time_diff, 0.01)  # minimum step to avoid zero steps
        
        # Calculate new positions
        new_original_position = calculate_position(current_original_position, adaptive_step, original_heading)
        new_corrected_position = calculate_position(current_corrected_position, adaptive_step, corrected_heading)
        
        # Store positions
        original_positions.append(new_original_position)
        corrected_positions.append(new_corrected_position)
        
        # Update current positions
        current_original_position = new_original_position
        current_corrected_position = new_corrected_position
    
    # Convert to arrays for easier plotting
    original_positions = np.array(original_positions)
    corrected_positions = np.array(corrected_positions)
    ground_truth_positions = np.array(ground_truth_positions)
    
    # Scale positions to match the ground truth scale
    # Find the overall scale difference by comparing the bounding boxes
    original_bbox_size = (np.max(original_positions[:, 0]) - np.min(original_positions[:, 0])) * \
                        (np.max(original_positions[:, 1]) - np.min(original_positions[:, 1]))
    gt_bbox_size = (np.max(ground_truth_positions[:, 0]) - np.min(ground_truth_positions[:, 0])) * \
                  (np.max(ground_truth_positions[:, 1]) - np.min(ground_truth_positions[:, 1]))
    
    # If the bounding box sizes are valid, compute scale factor
    if original_bbox_size > 0 and gt_bbox_size > 0:
        scale_factor = math.sqrt(gt_bbox_size / original_bbox_size)
        print(f"Scaling factor to match ground truth scale: {scale_factor:.4f}")
        
        # Scale the positions using the center of the first ground truth point as reference
        center = ground_truth_positions[0]
        
        # Apply scaling to original positions
        original_positions = ((original_positions - center) * scale_factor) + center
        
        # Apply scaling to corrected positions
        corrected_positions = ((corrected_positions - center) * scale_factor) + center
    
    # Create DataFrame for output
    position_df = pd.DataFrame({
        'Time (s)': corrected_heading_data['Time (s)'],
        'Original_Heading': corrected_heading_data['Gyro Heading (°)'],
        'Corrected_Heading': corrected_heading_data['Corrected_Heading'],
        'Original_East': original_positions[:, 0],
        'Original_North': original_positions[:, 1],
        'Corrected_East': corrected_positions[:, 0],
        'Corrected_North': corrected_positions[:, 1]
    })
    
    # Save positions to CSV
    output_file = os.path.join(output_folder, 'gyro_position_corrected.csv')
    position_df.to_csv(output_file, index=False)
    print(f"Saved corrected position data to {output_file}")
    
    # Create trajectory visualization
    plt.figure(figsize=(14, 12))
    
    # Plot original trajectory
    plt.plot(original_positions[:, 0], original_positions[:, 1], 'b-', alpha=0.5, label='Original Trajectory')
    
    # Plot corrected trajectory
    plt.plot(corrected_positions[:, 0], corrected_positions[:, 1], 'g-', label='Corrected Trajectory')
    
    # Plot ground truth positions
    plt.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 1], c='r', s=120, 
                marker='o', label='Ground Truth Positions')
    
    # Connect ground truth positions with a line
    plt.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 'r--', alpha=0.7, label='Ground Truth Path')
    
    # Add arrows to show direction every N points
    arrow_spacing = 500  # Show an arrow every N points
    for i in range(0, len(original_positions), arrow_spacing):
        if i+1 < len(original_positions):
            # Original trajectory arrows (blue)
            plt.annotate('', xy=(original_positions[i+1, 0], original_positions[i+1, 1]),
                         xytext=(original_positions[i, 0], original_positions[i, 1]),
                         arrowprops=dict(arrowstyle="->", color='blue', alpha=0.6))
            
            # Corrected trajectory arrows (green)
            plt.annotate('', xy=(corrected_positions[i+1, 0], corrected_positions[i+1, 1]),
                         xytext=(corrected_positions[i, 0], corrected_positions[i, 1]),
                         arrowprops=dict(arrowstyle="->", color='green', alpha=0.8))
    
    # Add labels for ground truth points
    for i, (east, north) in enumerate(ground_truth_positions):
        plt.text(east, north + 2, f"GT{i}", color='red', fontsize=12, weight='bold')
    
    plt.title('Trajectory Comparison: Original vs. Corrected', fontsize=16)
    plt.xlabel('East', fontsize=14)
    plt.ylabel('North', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.axis('equal')  # Equal aspect ratio
    
    # Save trajectory plot
    plot_file = os.path.join(output_folder, 'trajectory_comparison.png')
    plt.savefig(plot_file, dpi=300)
    print(f"Saved trajectory comparison to {plot_file}")
    
    # Create zoomed-in view of the trajectory
    plt.figure(figsize=(14, 12))
    
    # Calculate the center of the ground truth points
    gt_center = np.mean(ground_truth_positions, axis=0)
    
    # Plot with a more focused view on the paths
    # Plot original trajectory
    plt.plot(original_positions[:, 0], original_positions[:, 1], 'b-', alpha=0.5, label='Original Trajectory')
    
    # Plot corrected trajectory
    plt.plot(corrected_positions[:, 0], corrected_positions[:, 1], 'g-', label='Corrected Trajectory')
    
    # Plot ground truth points and path
    plt.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 1], c='r', s=120, 
                marker='o', label='Ground Truth Positions')
    plt.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 'r--', alpha=0.7, label='Ground Truth Path')
    
    # Add labels for ground truth points
    for i, (east, north) in enumerate(ground_truth_positions):
        plt.text(east, north + 2, f"GT{i}", color='red', fontsize=12, weight='bold')
    
    # Calculate appropriate zoom level
    margin = 30  # Add some margin around the points
    
    # Calculate bounds considering all points
    min_x = min(np.min(original_positions[:, 0]), np.min(corrected_positions[:, 0]), np.min(ground_truth_positions[:, 0])) - margin
    max_x = max(np.max(original_positions[:, 0]), np.max(corrected_positions[:, 0]), np.max(ground_truth_positions[:, 0])) + margin
    min_y = min(np.min(original_positions[:, 1]), np.min(corrected_positions[:, 1]), np.min(ground_truth_positions[:, 1])) - margin
    max_y = max(np.max(original_positions[:, 1]), np.max(corrected_positions[:, 1]), np.max(ground_truth_positions[:, 1])) + margin
    
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    
    plt.title('Zoomed View of Trajectories', fontsize=16)
    plt.xlabel('East', fontsize=14)
    plt.ylabel('North', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.axis('equal')  # Equal aspect ratio
    
    # Save zoomed trajectory plot
    zoomed_plot_file = os.path.join(output_folder, 'trajectory_comparison_zoomed.png')
    plt.savefig(zoomed_plot_file, dpi=300)
    print(f"Saved zoomed trajectory comparison to {zoomed_plot_file}")
    
    # Create detailed visualization focusing on the ground truth points
    plt.figure(figsize=(14, 12))
    
    # Extract positions near ground truth points for detailed comparison
    for i, gt_pos in enumerate(ground_truth_positions):
        # Find the closest positions in the original and corrected trajectories
        distances_orig = np.sum((original_positions - gt_pos)**2, axis=1)
        distances_corr = np.sum((corrected_positions - gt_pos)**2, axis=1)
        
        # Get the indices of the closest points
        closest_orig_idx = np.argmin(distances_orig)
        closest_corr_idx = np.argmin(distances_corr)
        
        # Plot points in the vicinity of the ground truth point
        window = 200  # Number of points before and after to plot
        
        # Original trajectory points near this ground truth point
        idx_range_orig = range(max(0, closest_orig_idx-window), min(len(original_positions), closest_orig_idx+window))
        orig_segment = original_positions[idx_range_orig]
        plt.plot(orig_segment[:, 0], orig_segment[:, 1], 'b-', alpha=0.5)
        
        # Corrected trajectory points near this ground truth point
        idx_range_corr = range(max(0, closest_corr_idx-window), min(len(corrected_positions), closest_corr_idx+window))
        corr_segment = corrected_positions[idx_range_corr]
        plt.plot(corr_segment[:, 0], corr_segment[:, 1], 'g-')
        
        # Ground truth point
        plt.scatter(gt_pos[0], gt_pos[1], c='r', s=120, marker='o')
        plt.text(gt_pos[0], gt_pos[1] + 2, f"GT{i}", color='red', fontsize=12, weight='bold')
        
        # Mark the closest points on each trajectory
        plt.scatter(original_positions[closest_orig_idx, 0], original_positions[closest_orig_idx, 1], 
                    c='b', s=80, marker='x')
        plt.scatter(corrected_positions[closest_corr_idx, 0], corrected_positions[closest_corr_idx, 1], 
                    c='g', s=80, marker='x')
        
        # Add error measurement lines
        plt.plot([gt_pos[0], original_positions[closest_orig_idx, 0]], 
                 [gt_pos[1], original_positions[closest_orig_idx, 1]], 
                 'b--', alpha=0.5)
        plt.plot([gt_pos[0], corrected_positions[closest_corr_idx, 0]], 
                 [gt_pos[1], corrected_positions[closest_corr_idx, 1]], 
                 'g--', alpha=0.5)
    
    # Connect the ground truth points with a line
    plt.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 'r-', alpha=0.7)
    
    plt.title('Detailed View: Trajectories Near Ground Truth Points', fontsize=16)
    plt.xlabel('East', fontsize=14)
    plt.ylabel('North', fontsize=14)
    plt.legend(['Original Trajectory', 'Corrected Trajectory', 'Ground Truth'], fontsize=12)
    plt.grid(True)
    plt.axis('equal')  # Equal aspect ratio
    
    # Save detailed trajectory plot
    detail_plot_file = os.path.join(output_folder, 'trajectory_detail_comparison.png')
    plt.savefig(detail_plot_file, dpi=300)
    print(f"Saved detailed trajectory comparison to {detail_plot_file}")
    
    # Calculate error statistics
    gt_errors_original = []
    gt_errors_corrected = []
    
    for i, gt_pos in enumerate(ground_truth_positions):
        # Find the closest positions in the original and corrected trajectories
        distances_orig = np.sum((original_positions - gt_pos)**2, axis=1)
        distances_corr = np.sum((corrected_positions - gt_pos)**2, axis=1)
        
        # Get the indices of the closest points
        closest_orig_idx = np.argmin(distances_orig)
        closest_corr_idx = np.argmin(distances_corr)
        
        # Calculate the errors
        error_orig = np.sqrt(distances_orig[closest_orig_idx])
        error_corr = np.sqrt(distances_corr[closest_corr_idx])
        
        gt_errors_original.append(error_orig)
        gt_errors_corrected.append(error_corr)
    
    # Print error statistics
    print("\nError Statistics at Ground Truth Points:")
    for i in range(len(ground_truth_positions)):
        print(f"Ground Truth Point {i}:")
        print(f"  Original Error: {gt_errors_original[i]:.2f}")
        print(f"  Corrected Error: {gt_errors_corrected[i]:.2f}")
        
        if gt_errors_original[i] > 0:  # Avoid division by zero
            if gt_errors_corrected[i] < gt_errors_original[i]:
                improvement = (gt_errors_original[i] - gt_errors_corrected[i]) / gt_errors_original[i] * 100
                print(f"  Improvement: {improvement:.2f}%")
            else:
                degradation = (gt_errors_corrected[i] - gt_errors_original[i]) / gt_errors_original[i] * 100
                print(f"  Degradation: {degradation:.2f}%")
    
    # Calculate improvements for valid points only (avoiding the first point which is zero)
    valid_indices = [i for i in range(len(gt_errors_original)) if gt_errors_original[i] > 0]
    valid_orig_errors = [gt_errors_original[i] for i in valid_indices]
    valid_corr_errors = [gt_errors_corrected[i] for i in valid_indices]
    
    if valid_orig_errors:
        avg_error_orig = np.mean(valid_orig_errors)
        avg_error_corr = np.mean(valid_corr_errors)
        
        print(f"\nAverage Error at Ground Truth Points (excluding starting point):")
        print(f"  Original: {avg_error_orig:.2f}")
        print(f"  Corrected: {avg_error_corr:.2f}")
        
        if avg_error_corr < avg_error_orig:
            improvement = (avg_error_orig - avg_error_corr) / avg_error_orig * 100
            print(f"  Overall Improvement: {improvement:.2f}%")
        else:
            degradation = (avg_error_corr - avg_error_orig) / avg_error_orig * 100
            print(f"  Overall Degradation: {degradation:.2f}%")
    
    # Count improved points and calculate overall effectiveness
    improved_points = sum(1 for i in valid_indices if gt_errors_corrected[i] < gt_errors_original[i])
    if valid_indices:
        improvement_percentage = improved_points / len(valid_indices) * 100
        print(f"\nImproved performance at {improved_points} out of {len(valid_indices)} ground truth points ({improvement_percentage:.2f}%)")

if __name__ == "__main__":
    main() 