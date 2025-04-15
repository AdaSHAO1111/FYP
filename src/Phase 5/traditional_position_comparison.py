import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# Define paths
trad_gyro_pos_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2/traditional_gyro_positions.csv'
corrected_heading_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 5/gyro_heading_corrected.csv'
ground_truth_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_ground_truth_data.csv'
output_folder = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 5'

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
    trad_gyro_pos = pd.read_csv(trad_gyro_pos_path)
    corrected_heading_data = pd.read_csv(corrected_heading_path)
    ground_truth_data = pd.read_csv(ground_truth_path)
    
    print(f"Traditional gyro positions shape: {trad_gyro_pos.shape}")
    print(f"Corrected heading data shape: {corrected_heading_data.shape}")
    print(f"Ground truth data shape: {ground_truth_data.shape}")
    
    # Extract traditional gyro positions
    trad_gyro_positions = []
    for _, row in trad_gyro_pos.iterrows():
        east = row['East_m']
        north = row['North_m']
        trad_gyro_positions.append((east, north))
    
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
    
    # Create positions based on corrected headings
    corrected_positions = []
    
    # Start with the initial position from ground truth
    if len(ground_truth_positions) > 0:
        current_corrected_position = ground_truth_positions[0]
    else:
        current_corrected_position = (0, 0)
    
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
    
    # Iterate through readings for corrected position
    previous_time = corrected_heading_data.loc[0, 'Time (s)']
    
    for i in range(1, len(corrected_heading_data)):
        # Get corrected heading
        corrected_heading = corrected_heading_data.loc[i, 'Corrected_Heading']
        
        # Calculate time difference for adaptive step size
        current_time = corrected_heading_data.loc[i, 'Time (s)']
        time_diff = current_time - previous_time
        previous_time = current_time
        
        # Adjust step size based on time difference (larger time diff = larger step)
        adaptive_step = base_step_size * max(time_diff, 0.01)  # minimum step to avoid zero steps
        
        # Calculate new positions
        new_corrected_position = calculate_position(current_corrected_position, adaptive_step, corrected_heading)
        
        # Store positions
        corrected_positions.append(new_corrected_position)
        
        # Update current positions
        current_corrected_position = new_corrected_position
    
    # Convert to arrays for easier plotting
    trad_gyro_positions = np.array(trad_gyro_positions)
    corrected_positions = np.array(corrected_positions)
    ground_truth_positions = np.array(ground_truth_positions)
    
    # Scale positions to match the ground truth scale
    # Find the overall scale difference by comparing the bounding boxes 
    trad_gyro_bbox_size = (np.max(trad_gyro_positions[:, 0]) - np.min(trad_gyro_positions[:, 0])) * \
                          (np.max(trad_gyro_positions[:, 1]) - np.min(trad_gyro_positions[:, 1]))
    
    corrected_bbox_size = (np.max(corrected_positions[:, 0]) - np.min(corrected_positions[:, 0])) * \
                          (np.max(corrected_positions[:, 1]) - np.min(corrected_positions[:, 1]))
    
    gt_bbox_size = (np.max(ground_truth_positions[:, 0]) - np.min(ground_truth_positions[:, 0])) * \
                   (np.max(ground_truth_positions[:, 1]) - np.min(ground_truth_positions[:, 1]))
    
    # Scale the traditional gyro positions to match ground truth scale
    if trad_gyro_bbox_size > 0 and gt_bbox_size > 0:
        # We need to translate traditional gyro positions to match ground truth starting point
        trad_gyro_start = trad_gyro_positions[0]
        gt_start = ground_truth_positions[0]
        
        # Translate to origin, scale, then translate to ground truth start
        trad_gyro_positions_normalized = trad_gyro_positions - trad_gyro_start
        
        # Apply scaling (using a scale factor that matches the overall dimensions)
        scale_factor_trad = np.sqrt(gt_bbox_size / trad_gyro_bbox_size)
        print(f"Traditional gyro positions scale factor: {scale_factor_trad:.4f}")
        
        trad_gyro_positions_scaled = trad_gyro_positions_normalized * scale_factor_trad
        
        # Translate to ground truth start
        trad_gyro_positions_final = trad_gyro_positions_scaled + gt_start
    else:
        trad_gyro_positions_final = trad_gyro_positions
        
    # Scale the corrected positions to match ground truth
    if corrected_bbox_size > 0 and gt_bbox_size > 0:
        scale_factor_corr = math.sqrt(gt_bbox_size / corrected_bbox_size)
        print(f"Corrected positions scale factor: {scale_factor_corr:.4f}")
        
        # Scale the positions using the center of the first ground truth point as reference
        center = ground_truth_positions[0]
        
        # Apply scaling to corrected positions
        corrected_positions = ((corrected_positions - center) * scale_factor_corr) + center
    
    # Create trajectory visualization
    plt.figure(figsize=(14, 12))
    
    # Plot traditional gyro trajectory
    plt.plot(trad_gyro_positions_final[:, 0], trad_gyro_positions_final[:, 1], 'b-', alpha=0.5, label='Traditional Gyro Trajectory')
    
    # Plot corrected trajectory
    plt.plot(corrected_positions[:, 0], corrected_positions[:, 1], 'g-', label='Corrected Trajectory')
    
    # Plot ground truth positions
    plt.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 1], c='r', s=120, 
                marker='o', label='Ground Truth Positions')
    
    # Connect ground truth positions with a line
    plt.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 'r--', alpha=0.7, label='Ground Truth Path')
    
    # Add arrows to show direction every N points
    arrow_spacing = 20  # Show an arrow every N points (using smaller value for traditional data)
    for i in range(0, len(trad_gyro_positions_final), arrow_spacing):
        if i+1 < len(trad_gyro_positions_final):
            # Traditional gyro trajectory arrows (blue)
            plt.annotate('', xy=(trad_gyro_positions_final[i+1, 0], trad_gyro_positions_final[i+1, 1]),
                         xytext=(trad_gyro_positions_final[i, 0], trad_gyro_positions_final[i, 1]),
                         arrowprops=dict(arrowstyle="->", color='blue', alpha=0.6))
    
    # Add arrows for corrected trajectory
    arrow_spacing_corr = 500  # Show an arrow every N points
    for i in range(0, len(corrected_positions), arrow_spacing_corr):
        if i+1 < len(corrected_positions):  
            # Corrected trajectory arrows (green)
            plt.annotate('', xy=(corrected_positions[i+1, 0], corrected_positions[i+1, 1]),
                         xytext=(corrected_positions[i, 0], corrected_positions[i, 1]),
                         arrowprops=dict(arrowstyle="->", color='green', alpha=0.8))
    
    # Add labels for ground truth points
    for i, (east, north) in enumerate(ground_truth_positions):
        plt.text(east, north + 2, f"GT{i}", color='red', fontsize=12, weight='bold')
    
    plt.title('Trajectory Comparison: Traditional vs. Corrected', fontsize=16)
    plt.xlabel('East', fontsize=14)
    plt.ylabel('North', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.axis('equal')  # Equal aspect ratio
    
    # Save trajectory plot
    plot_file = os.path.join(output_folder, 'traditional_trajectory_comparison.png')
    plt.savefig(plot_file, dpi=300)
    print(f"Saved trajectory comparison to {plot_file}")
    
    # Create detailed visualization focusing on the ground truth points
    plt.figure(figsize=(14, 12))
    
    # Extract positions near ground truth points for detailed comparison
    for i, gt_pos in enumerate(ground_truth_positions):
        # Find the closest positions in the traditional and corrected trajectories
        distances_trad = np.sum((trad_gyro_positions_final - gt_pos)**2, axis=1)
        distances_corr = np.sum((corrected_positions - gt_pos)**2, axis=1)
        
        # Get the indices of the closest points
        closest_trad_idx = np.argmin(distances_trad)
        closest_corr_idx = np.argmin(distances_corr)
        
        # Plot points in the vicinity of the ground truth point
        window_trad = 10  # Number of points before and after to plot (smaller for traditional)
        window_corr = 200  # Number of points before and after to plot
        
        # Traditional gyro trajectory points near this ground truth point
        idx_range_trad = range(max(0, closest_trad_idx-window_trad), min(len(trad_gyro_positions_final), closest_trad_idx+window_trad))
        trad_segment = trad_gyro_positions_final[idx_range_trad]
        plt.plot(trad_segment[:, 0], trad_segment[:, 1], 'b-', alpha=0.5)
        
        # Corrected trajectory points near this ground truth point
        idx_range_corr = range(max(0, closest_corr_idx-window_corr), min(len(corrected_positions), closest_corr_idx+window_corr))
        corr_segment = corrected_positions[idx_range_corr]
        plt.plot(corr_segment[:, 0], corr_segment[:, 1], 'g-')
        
        # Ground truth point
        plt.scatter(gt_pos[0], gt_pos[1], c='r', s=120, marker='o')
        plt.text(gt_pos[0], gt_pos[1] + 2, f"GT{i}", color='red', fontsize=12, weight='bold')
        
        # Mark the closest points on each trajectory
        plt.scatter(trad_gyro_positions_final[closest_trad_idx, 0], trad_gyro_positions_final[closest_trad_idx, 1], 
                    c='b', s=80, marker='x')
        plt.scatter(corrected_positions[closest_corr_idx, 0], corrected_positions[closest_corr_idx, 1], 
                    c='g', s=80, marker='x')
        
        # Add error measurement lines
        plt.plot([gt_pos[0], trad_gyro_positions_final[closest_trad_idx, 0]], 
                 [gt_pos[1], trad_gyro_positions_final[closest_trad_idx, 1]], 
                 'b--', alpha=0.5)
        plt.plot([gt_pos[0], corrected_positions[closest_corr_idx, 0]], 
                 [gt_pos[1], corrected_positions[closest_corr_idx, 1]], 
                 'g--', alpha=0.5)
    
    # Connect the ground truth points with a line
    plt.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 'r-', alpha=0.7)
    
    plt.title('Detailed View: Trajectories Near Ground Truth Points', fontsize=16)
    plt.xlabel('East', fontsize=14)
    plt.ylabel('North', fontsize=14)
    plt.legend(['Traditional Gyro Trajectory', 'Corrected Trajectory', 'Ground Truth'], fontsize=12)
    plt.grid(True)
    plt.axis('equal')  # Equal aspect ratio
    
    # Save detailed trajectory plot
    detail_plot_file = os.path.join(output_folder, 'traditional_trajectory_detail.png')
    plt.savefig(detail_plot_file, dpi=300)
    print(f"Saved detailed trajectory comparison to {detail_plot_file}")
    
    # Calculate error statistics
    gt_errors_trad = []
    gt_errors_corr = []
    
    for i, gt_pos in enumerate(ground_truth_positions):
        # Find the closest positions in the traditional and corrected trajectories
        distances_trad = np.sum((trad_gyro_positions_final - gt_pos)**2, axis=1)
        distances_corr = np.sum((corrected_positions - gt_pos)**2, axis=1)
        
        # Get the indices of the closest points
        closest_trad_idx = np.argmin(distances_trad)
        closest_corr_idx = np.argmin(distances_corr)
        
        # Calculate the errors
        error_trad = np.sqrt(distances_trad[closest_trad_idx])
        error_corr = np.sqrt(distances_corr[closest_corr_idx])
        
        gt_errors_trad.append(error_trad)
        gt_errors_corr.append(error_corr)
    
    # Print error statistics
    print("\nError Statistics at Ground Truth Points:")
    for i in range(len(ground_truth_positions)):
        print(f"Ground Truth Point {i}:")
        print(f"  Traditional Gyro Error: {gt_errors_trad[i]:.2f}")
        print(f"  Corrected Error: {gt_errors_corr[i]:.2f}")
        
        if gt_errors_trad[i] > 0:  # Avoid division by zero
            if gt_errors_corr[i] < gt_errors_trad[i]:
                improvement = (gt_errors_trad[i] - gt_errors_corr[i]) / gt_errors_trad[i] * 100
                print(f"  Improvement: {improvement:.2f}%")
            else:
                degradation = (gt_errors_corr[i] - gt_errors_trad[i]) / gt_errors_trad[i] * 100
                print(f"  Degradation: {degradation:.2f}%")
    
    # Calculate improvements for valid points only (avoiding the first point which is zero)
    valid_indices = [i for i in range(len(gt_errors_trad)) if gt_errors_trad[i] > 0]
    valid_trad_errors = [gt_errors_trad[i] for i in valid_indices]
    valid_corr_errors = [gt_errors_corr[i] for i in valid_indices]
    
    if valid_trad_errors:
        avg_error_trad = np.mean(valid_trad_errors)
        avg_error_corr = np.mean(valid_corr_errors)
        
        print(f"\nAverage Error at Ground Truth Points (excluding starting point):")
        print(f"  Traditional Gyro: {avg_error_trad:.2f}")
        print(f"  Corrected: {avg_error_corr:.2f}")
        
        if avg_error_corr < avg_error_trad:
            improvement = (avg_error_trad - avg_error_corr) / avg_error_trad * 100
            print(f"  Overall Improvement: {improvement:.2f}%")
        else:
            degradation = (avg_error_corr - avg_error_trad) / avg_error_trad * 100
            print(f"  Overall Degradation: {degradation:.2f}%")
    
    # Count improved points and calculate overall effectiveness
    improved_points = sum(1 for i in valid_indices if gt_errors_corr[i] < gt_errors_trad[i])
    if valid_indices:
        improvement_percentage = improved_points / len(valid_indices) * 100
        print(f"\nImproved performance at {improved_points} out of {len(valid_indices)} ground truth points ({improvement_percentage:.2f}%)")

if __name__ == "__main__":
    main() 