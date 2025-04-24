import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from matplotlib.lines import Line2D

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
    
    # Create enhanced detailed visualization with error measurements
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
        plt.plot(trad_segment[:, 0], trad_segment[:, 1], 'b-', alpha=0.6, linewidth=2)
        
        # Corrected trajectory points near this ground truth point
        idx_range_corr = range(max(0, closest_corr_idx-window_corr), min(len(corrected_positions), closest_corr_idx+window_corr))
        corr_segment = corrected_positions[idx_range_corr]
        plt.plot(corr_segment[:, 0], corr_segment[:, 1], 'g-', alpha=0.8, linewidth=2)
        
        # Ground truth point
        plt.scatter(gt_pos[0], gt_pos[1], c='r', s=150, marker='o', zorder=10)
        plt.text(gt_pos[0], gt_pos[1] + 2, f"GT{i}", color='red', fontsize=14, 
                 weight='bold', ha='center', va='bottom', zorder=11)
        
        # Mark the closest points on each trajectory
        plt.scatter(trad_gyro_positions_final[closest_trad_idx, 0], trad_gyro_positions_final[closest_trad_idx, 1], 
                   c='blue', s=100, marker='x', linewidth=2, zorder=9)
        
        plt.scatter(corrected_positions[closest_corr_idx, 0], corrected_positions[closest_corr_idx, 1], 
                   c='green', s=100, marker='x', linewidth=2, zorder=9)
        
        # Add error measurement lines with distance labels
        error_trad = np.sqrt(distances_trad[closest_trad_idx])
        error_corr = np.sqrt(distances_corr[closest_corr_idx])
        
        # Traditional error line
        plt.plot([gt_pos[0], trad_gyro_positions_final[closest_trad_idx, 0]], 
                 [gt_pos[1], trad_gyro_positions_final[closest_trad_idx, 1]], 
                 'b--', alpha=0.7, linewidth=1.5, zorder=8)
        
        # Corrected error line
        plt.plot([gt_pos[0], corrected_positions[closest_corr_idx, 0]], 
                 [gt_pos[1], corrected_positions[closest_corr_idx, 1]], 
                 'g--', alpha=0.7, linewidth=1.5, zorder=8)
    
    # Connect the ground truth points with a line
    plt.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 'r-', alpha=0.8, linewidth=2.5, zorder=7)
    
    # Add error statistics table
    error_table_data = []
    for i in range(1, len(ground_truth_positions)):  # Skip the first point which is zero error
        distances_trad = np.sum((trad_gyro_positions_final - ground_truth_positions[i])**2, axis=1)
        distances_corr = np.sum((corrected_positions - ground_truth_positions[i])**2, axis=1)
        
        closest_trad_idx = np.argmin(distances_trad)
        closest_corr_idx = np.argmin(distances_corr)
        
        error_trad = np.sqrt(distances_trad[closest_trad_idx])
        error_corr = np.sqrt(distances_corr[closest_corr_idx])
        
        percent_change = ((error_corr - error_trad) / error_trad * 100) if error_trad > 0 else 0
        
        error_table_data.append([i, error_trad, error_corr, percent_change])
    
    # Format the table as text
    table_text = "Error Comparison\n\nPoint   Traditional   Corrected    Change\n"
    table_text += "------------------------------------------\n"
    
    for row in error_table_data:
        gt_num, trad_err, corr_err, change = row
        change_sign = "+" if change > 0 else "-"
        table_text += f"GT{gt_num}    {trad_err:.2f}         {corr_err:.2f}        {change_sign}{abs(change):.1f}%\n"
    
    # Add text to the plot
    plt.figtext(0.15, 0.05, table_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8), family='monospace')
    
    # Create a custom legend
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, alpha=0.6, label='Traditional Gyro Trajectory'),
        Line2D([0], [0], color='green', lw=2, alpha=0.8, label='Corrected Trajectory'),
        Line2D([0], [0], color='red', lw=2.5, alpha=0.8, label='Ground Truth Path'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Ground Truth Point'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='blue', markersize=8, markeredgecolor='blue', label='Traditional Closest Point'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='green', markersize=8, markeredgecolor='green', label='Corrected Closest Point')
    ]
    plt.legend(handles=legend_elements, fontsize=12, loc='upper right')
    
    plt.title('Detailed View: Trajectories Near Ground Truth Points', fontsize=18)
    plt.xlabel('East', fontsize=14)
    plt.ylabel('North', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # Equal aspect ratio
    
    # Save enhanced detailed trajectory plot
    enhanced_detail_file = os.path.join(output_folder, 'enhanced_trajectory_detail.png')
    plt.savefig(enhanced_detail_file, dpi=300)
    print(f"Saved enhanced detailed trajectory comparison to {enhanced_detail_file}")
    
    # Save error statistics to CSV
    error_df = pd.DataFrame(error_table_data, columns=['GT_Point', 'Traditional_Error', 'Corrected_Error', 'Percent_Change'])
    error_df.to_csv(os.path.join(output_folder, 'corrected_gt_error_stats.csv'), index=False)
    print("Saved error statistics to CSV file")

if __name__ == "__main__":
    main() 