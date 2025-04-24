import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Define paths
trad_gyro_pos_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2/traditional_gyro_positions.csv'
corrected_heading_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 5/improved_gyro_heading_corrected.csv'
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

def confidence_ellipse(x, y, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

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
    previous_time = corrected_heading_data.loc[0, 'Time_Seconds']
    
    for i in range(1, len(corrected_heading_data)):
        # Get corrected heading
        corrected_heading = corrected_heading_data.loc[i, 'Corrected_Heading']
        
        # Calculate time difference for adaptive step size
        current_time = corrected_heading_data.loc[i, 'Time_Seconds']
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

    # Create error statistics table for the plot
    error_table_data = []
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
        
        # Record for table
        error_table_data.append({
            'GT Point': f'GT{i}',
            'Trad Error': error_trad,
            'Corr Error': error_corr,
            'Improvement': ((error_trad - error_corr) / error_trad * 100) if error_trad > 0 else 0
        })
    
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
    arrow_spacing = 500  # Show an arrow every N points
    for i in range(0, len(trad_gyro_positions_final), arrow_spacing):
        if i+1 < len(trad_gyro_positions_final):
            # Traditional gyro trajectory arrows (blue)
            plt.annotate('', xy=(trad_gyro_positions_final[i+1, 0], trad_gyro_positions_final[i+1, 1]),
                         xytext=(trad_gyro_positions_final[i, 0], trad_gyro_positions_final[i, 1]),
                         arrowprops=dict(arrowstyle="->", color='blue', alpha=0.6))
    
    for i in range(0, len(corrected_positions), arrow_spacing):
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
    plot_file = os.path.join(output_folder, 'improved_trajectory_comparison_full.png')
    plt.savefig(plot_file, dpi=300)
    print(f"Saved trajectory comparison to {plot_file}")
    
    # Create a table of error statistics
    error_df = pd.DataFrame(error_table_data)
    print("\nError Statistics:")
    print(error_df)
    
    # Calculate mean error improvement
    mean_improvement = error_df['Improvement'].mean()
    print(f"\nMean Error Improvement: {mean_improvement:.2f}%")
    
    # Save error statistics
    error_csv = os.path.join(output_folder, 'improved_trajectory_error_stats.csv')
    error_df.to_csv(error_csv, index=False)
    print(f"Saved error statistics to {error_csv}")
    
    # Create a focused view of the GT5-GT7 region (the problematic area)
    plt.figure(figsize=(12, 10))
    
    # Define the region of interest (GT5 to GT7)
    if len(ground_truth_positions) >= 8:  # Make sure we have at least GT0 to GT7
        min_idx = 5  # GT5
        max_idx = 7  # GT7
        
        # Extract the relevant ground truth positions
        roi_gt = ground_truth_positions[min_idx:max_idx+1]
        
        # Calculate the bounds of the region with some padding
        min_x = np.min(roi_gt[:, 0]) - 5
        max_x = np.max(roi_gt[:, 0]) + 5
        min_y = np.min(roi_gt[:, 1]) - 5
        max_y = np.max(roi_gt[:, 1]) + 5
        
        # Plot traditional trajectory
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
        
        # Set the axis limits to focus on the region of interest
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        
        plt.title('Detailed View of GT5-GT7 Region', fontsize=16)
        plt.xlabel('East', fontsize=14)
        plt.ylabel('North', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # Save the detailed view
        detail_file = os.path.join(output_folder, 'improved_trajectory_detail_GT5-GT7.png')
        plt.savefig(detail_file, dpi=300)
        print(f"Saved detailed view to {detail_file}")
    
    # Create a focused view of the GT0-GT1 region (to check for missed turn)
    plt.figure(figsize=(12, 10))
    
    # Define the region of interest (GT0 to GT1)
    if len(ground_truth_positions) >= 2:
        min_idx = 0  # GT0
        max_idx = 1  # GT1
        
        # Extract the relevant ground truth positions
        roi_gt = ground_truth_positions[min_idx:max_idx+1]
        
        # Calculate the bounds of the region with some padding
        min_x = np.min(roi_gt[:, 0]) - 5
        max_x = np.max(roi_gt[:, 0]) + 5
        min_y = np.min(roi_gt[:, 1]) - 5
        max_y = np.max(roi_gt[:, 1]) + 5
        
        # Plot traditional trajectory
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
        
        # Set the axis limits to focus on the region of interest
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        
        plt.title('Detailed View of GT0-GT1 Region', fontsize=16)
        plt.xlabel('East', fontsize=14)
        plt.ylabel('North', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # Save the detailed view
        detail_file = os.path.join(output_folder, 'improved_trajectory_detail_GT0-GT1.png')
        plt.savefig(detail_file, dpi=300)
        print(f"Saved detailed view to {detail_file}")

if __name__ == "__main__":
    main() 