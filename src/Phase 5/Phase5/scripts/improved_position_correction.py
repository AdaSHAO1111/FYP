import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# Define paths
gyro_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_gyro_data.csv'
improved_heading_path = '/Users/shaoxinyi/Downloads/FYP2/src/Phase5/output/improved_gyro_heading_corrected.csv'
ground_truth_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_ground_truth_data.csv'
traditional_gyro_pos_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2/traditional_gyro_positions.csv'
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
    
    # Check if the improved heading file exists
    if not os.path.exists(improved_heading_path):
        print(f"Improved heading file {improved_heading_path} not found.")
        print("Please run improved_heading_correction.py first.")
        return
    
    gyro_data = pd.read_csv(gyro_data_path)
    improved_heading_data = pd.read_csv(improved_heading_path)
    ground_truth_data = pd.read_csv(ground_truth_path)
    traditional_gyro_pos = pd.read_csv(traditional_gyro_pos_path)
    
    print(f"Gyro data shape: {gyro_data.shape}")
    print(f"Improved heading data shape: {improved_heading_data.shape}")
    print(f"Ground truth data shape: {ground_truth_data.shape}")
    print(f"Traditional gyro positions shape: {traditional_gyro_pos.shape}")
    
    # Extract traditional gyro positions
    trad_gyro_positions = []
    for _, row in traditional_gyro_pos.iterrows():
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
    
    # Create positions based on original and improved headings
    original_positions = []
    improved_positions = []
    
    # Start with the initial position from ground truth
    if len(ground_truth_positions) > 0:
        current_original_position = ground_truth_positions[0]
        current_improved_position = ground_truth_positions[0]
    else:
        current_original_position = (0, 0)
        current_improved_position = (0, 0)
    
    original_positions.append(current_original_position)
    improved_positions.append(current_improved_position)
    
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
    # Smaller step size for a more precise trajectory
    base_step_size = avg_step_size * 0.05  # Scale down to account for frequency of readings
    print(f"Using base step size: {base_step_size:.4f}")
    
    # Iterate through readings
    previous_time = improved_heading_data.loc[0, 'Time (s)']
    
    for i in range(1, len(improved_heading_data)):
        # Get headings
        original_heading = improved_heading_data.loc[i, 'Gyro Heading (°)']
        improved_heading = improved_heading_data.loc[i, 'Improved_Heading']
        
        # Calculate time difference for adaptive step size
        current_time = improved_heading_data.loc[i, 'Time (s)']
        time_diff = current_time - previous_time
        previous_time = current_time
        
        # Adjust step size based on time difference (larger time diff = larger step)
        # Minimum step to avoid zero steps
        # For improved accuracy, we can make the minimum step slightly smaller
        adaptive_step = base_step_size * max(time_diff, 0.01)  
        
        # Calculate new positions
        new_original_position = calculate_position(current_original_position, adaptive_step, original_heading)
        new_improved_position = calculate_position(current_improved_position, adaptive_step, improved_heading)
        
        # Store positions
        original_positions.append(new_original_position)
        improved_positions.append(new_improved_position)
        
        # Update current positions
        current_original_position = new_original_position
        current_improved_position = new_improved_position
    
    # Convert to arrays for easier plotting
    original_positions = np.array(original_positions)
    improved_positions = np.array(improved_positions)
    ground_truth_positions = np.array(ground_truth_positions)
    trad_gyro_positions = np.array(trad_gyro_positions)
    
    # Scale positions to match the ground truth scale
    # Find the overall scale difference by comparing the bounding boxes
    original_bbox_size = (np.max(original_positions[:, 0]) - np.min(original_positions[:, 0])) * \
                        (np.max(original_positions[:, 1]) - np.min(original_positions[:, 1]))
    
    improved_bbox_size = (np.max(improved_positions[:, 0]) - np.min(improved_positions[:, 0])) * \
                         (np.max(improved_positions[:, 1]) - np.min(improved_positions[:, 1]))
    
    gt_bbox_size = (np.max(ground_truth_positions[:, 0]) - np.min(ground_truth_positions[:, 0])) * \
                  (np.max(ground_truth_positions[:, 1]) - np.min(ground_truth_positions[:, 1]))
    
    # If the bounding box sizes are valid, compute scale factor
    if original_bbox_size > 0 and gt_bbox_size > 0:
        original_scale_factor = math.sqrt(gt_bbox_size / original_bbox_size)
        print(f"Original positions scaling factor: {original_scale_factor:.4f}")
        
        # Scale the positions using the center of the first ground truth point as reference
        center = ground_truth_positions[0]
        
        # Apply scaling to original positions
        original_positions = ((original_positions - center) * original_scale_factor) + center
    
    if improved_bbox_size > 0 and gt_bbox_size > 0:
        improved_scale_factor = math.sqrt(gt_bbox_size / improved_bbox_size)
        print(f"Improved positions scaling factor: {improved_scale_factor:.4f}")
        
        # Scale the positions using the center of the first ground truth point as reference
        center = ground_truth_positions[0]
        
        # Apply scaling to improved positions
        improved_positions = ((improved_positions - center) * improved_scale_factor) + center
    
    # Scale traditional gyro positions
    trad_gyro_bbox_size = (np.max(trad_gyro_positions[:, 0]) - np.min(trad_gyro_positions[:, 0])) * \
                          (np.max(trad_gyro_positions[:, 1]) - np.min(trad_gyro_positions[:, 1]))
    
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
        trad_gyro_positions = trad_gyro_positions_scaled + gt_start
    
    # Create DataFrame for output
    position_df = pd.DataFrame({
        'Time (s)': improved_heading_data['Time (s)'],
        'Original_Heading': improved_heading_data['Gyro Heading (°)'],
        'Improved_Heading': improved_heading_data['Improved_Heading'],
        'Original_East': original_positions[:, 0],
        'Original_North': original_positions[:, 1],
        'Improved_East': improved_positions[:, 0],
        'Improved_North': improved_positions[:, 1]
    })
    
    # Save positions to CSV
    output_file = os.path.join(output_folder, 'improved_position_corrected.csv')
    position_df.to_csv(output_file, index=False)
    print(f"Saved improved position data to {output_file}")
    
    # Create trajectory visualization
    plt.figure(figsize=(14, 12))
    
    # Plot traditional gyro trajectory
    plt.plot(trad_gyro_positions[:, 0], trad_gyro_positions[:, 1], 
             'b-', alpha=0.5, label='Traditional Gyro Trajectory')
    
    # Plot improved trajectory
    plt.plot(improved_positions[:, 0], improved_positions[:, 1], 
             'g-', linewidth=2, label='Improved Trajectory')
    
    # Plot ground truth positions
    plt.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 
                c='r', s=120, marker='o', label='Ground Truth Positions')
    
    # Connect ground truth positions with a line
    plt.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 
             'r--', alpha=0.7, label='Ground Truth Path')
    
    # Add arrows to show direction
    arrow_spacing_trad = 20  # Show an arrow every N points for traditional trajectory
    for i in range(0, len(trad_gyro_positions), arrow_spacing_trad):
        if i+1 < len(trad_gyro_positions):
            # Traditional gyro trajectory arrows (blue)
            plt.annotate('', xy=(trad_gyro_positions[i+1, 0], trad_gyro_positions[i+1, 1]),
                         xytext=(trad_gyro_positions[i, 0], trad_gyro_positions[i, 1]),
                         arrowprops=dict(arrowstyle="->", color='blue', alpha=0.6))
    
    arrow_spacing_imp = 500  # Show an arrow every N points for improved trajectory
    for i in range(0, len(improved_positions), arrow_spacing_imp):
        if i+1 < len(improved_positions):
            # Improved trajectory arrows (green)
            plt.annotate('', xy=(improved_positions[i+1, 0], improved_positions[i+1, 1]),
                         xytext=(improved_positions[i, 0], improved_positions[i, 1]),
                         arrowprops=dict(arrowstyle="->", color='green', alpha=0.8))
    
    # Add labels for ground truth points
    for i, (east, north) in enumerate(ground_truth_positions):
        plt.text(east, north + 2, f"GT{i}", color='red', fontsize=12, weight='bold')
    
    plt.title('Trajectory Comparison: Traditional vs. Improved', fontsize=16)
    plt.xlabel('East', fontsize=14)
    plt.ylabel('North', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.axis('equal')  # Equal aspect ratio
    
    # Save trajectory plot
    plot_file = os.path.join(output_folder, 'improved_trajectory_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved trajectory comparison to {plot_file}")
    
    # Create zoomed-in view of the trajectory
    plt.figure(figsize=(14, 12))
    
    # Plot zoomed trajectories with focus on problem areas
    # Plot traditional gyro trajectory
    plt.plot(trad_gyro_positions[:, 0], trad_gyro_positions[:, 1], 
             'b-', alpha=0.5, label='Traditional Gyro Trajectory')
    
    # Plot improved trajectory
    plt.plot(improved_positions[:, 0], improved_positions[:, 1], 
             'g-', linewidth=2, label='Improved Trajectory')
    
    # Plot ground truth positions
    plt.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 
                c='r', s=120, marker='o', label='Ground Truth Positions')
    
    # Connect ground truth positions with a line
    plt.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 
             'r--', alpha=0.7, label='Ground Truth Path')
    
    # Add labels for ground truth points
    for i, (east, north) in enumerate(ground_truth_positions):
        plt.text(east, north + 2, f"GT{i}", color='red', fontsize=12, weight='bold')
    
    plt.title('Zoomed Trajectory Comparison', fontsize=16)
    plt.xlabel('East', fontsize=14)
    plt.ylabel('North', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.axis('equal')  # Equal aspect ratio
    
    # Save zoomed trajectory plot
    zoomed_plot_file = os.path.join(output_folder, 'improved_trajectory_comparison_zoomed.png')
    plt.savefig(zoomed_plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved zoomed trajectory comparison to {zoomed_plot_file}")
    
    # Calculate error statistics
    print("\nCalculating error statistics...")
    gt_errors_trad = []
    gt_errors_imp = []
    
    for i, gt_pos in enumerate(ground_truth_positions):
        # Find the closest positions in the trajectories
        distances_trad = np.sum((trad_gyro_positions - gt_pos)**2, axis=1)
        distances_imp = np.sum((improved_positions - gt_pos)**2, axis=1)
        
        # Get the indices of the closest points
        closest_trad_idx = np.argmin(distances_trad)
        closest_imp_idx = np.argmin(distances_imp)
        
        # Calculate the errors
        error_trad = np.sqrt(distances_trad[closest_trad_idx])
        error_imp = np.sqrt(distances_imp[closest_imp_idx])
        
        gt_errors_trad.append(error_trad)
        gt_errors_imp.append(error_imp)
    
    # Print error statistics
    print("\nError Statistics at Ground Truth Points:")
    for i in range(len(ground_truth_positions)):
        print(f"Ground Truth Point {i}:")
        print(f"  Traditional Gyro Error: {gt_errors_trad[i]:.2f}")
        print(f"  Improved Error: {gt_errors_imp[i]:.2f}")
        
        if gt_errors_trad[i] > 0:  # Avoid division by zero
            if gt_errors_imp[i] < gt_errors_trad[i]:
                improvement = (gt_errors_trad[i] - gt_errors_imp[i]) / gt_errors_trad[i] * 100
                print(f"  Improvement: {improvement:.2f}%")
            else:
                degradation = (gt_errors_imp[i] - gt_errors_trad[i]) / gt_errors_trad[i] * 100
                print(f"  Degradation: {degradation:.2f}%")
    
    # Calculate improvements for valid points only (excluding the first point which is the start)
    valid_indices = list(range(1, len(gt_errors_trad)))
    valid_trad_errors = [gt_errors_trad[i] for i in valid_indices]
    valid_imp_errors = [gt_errors_imp[i] for i in valid_indices]
    
    if valid_trad_errors:
        avg_error_trad = np.mean(valid_trad_errors)
        avg_error_imp = np.mean(valid_imp_errors)
        
        print(f"\nAverage Error at Ground Truth Points (excluding starting point):")
        print(f"  Traditional Gyro: {avg_error_trad:.2f}")
        print(f"  Improved: {avg_error_imp:.2f}")
        
        if avg_error_imp < avg_error_trad:
            improvement = (avg_error_trad - avg_error_imp) / avg_error_trad * 100
            print(f"  Overall Improvement: {improvement:.2f}%")
        else:
            degradation = (avg_error_imp - avg_error_trad) / avg_error_trad * 100
            print(f"  Overall Degradation: {degradation:.2f}%")
    
    # Count improved points and calculate overall effectiveness
    improved_points = sum(1 for i in valid_indices if gt_errors_imp[i] < gt_errors_trad[i])
    if valid_indices:
        improvement_percentage = improved_points / len(valid_indices) * 100
        print(f"\nImproved performance at {improved_points} out of {len(valid_indices)} ground truth points ({improvement_percentage:.2f}%)")
    
    # Create a bar chart comparing errors
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(ground_truth_positions))
    width = 0.35
    
    plt.bar(x - width/2, gt_errors_trad, width, label='Traditional Gyro Error')
    plt.bar(x + width/2, gt_errors_imp, width, label='Improved Error')
    
    plt.xlabel('Ground Truth Points', fontsize=14)
    plt.ylabel('Error (meters)', fontsize=14)
    plt.title('Error Comparison at Ground Truth Points', fontsize=16)
    plt.xticks(x, [f'GT{i}' for i in range(len(ground_truth_positions))])
    plt.legend(fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add error values above bars
    for i, v in enumerate(gt_errors_trad):
        plt.text(i - width/2, v + 0.2, f'{v:.2f}', ha='center', fontsize=10)
    
    for i, v in enumerate(gt_errors_imp):
        plt.text(i + width/2, v + 0.2, f'{v:.2f}', ha='center', fontsize=10)
    
    # Save bar chart
    bar_chart_file = os.path.join(output_folder, 'improved_error_comparison_chart.png')
    plt.savefig(bar_chart_file, dpi=300, bbox_inches='tight')
    print(f"Saved error comparison chart to {bar_chart_file}")

if __name__ == "__main__":
    main() 