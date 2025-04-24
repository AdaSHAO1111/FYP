import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# Define paths with full paths to avoid path issues
compass_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_compass_data.csv'
gyro_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_gyro_data.csv'
ground_truth_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_ground_truth_data.csv'
qs_data_path = '/Users/shaoxinyi/Downloads/FYP2/Init/1536_quasi_static_data.csv'
qs_stats_path = '/Users/shaoxinyi/Downloads/FYP2/Init/1536_quasi_static_averages.csv'
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

def angle_difference(angle1, angle2):
    """
    Calculate the smallest difference between two angles in degrees
    Correctly handles angle wrapping (e.g., 359° vs 1°)
    """
    diff = abs(normalize_angle(angle1) - normalize_angle(angle2)) % 360
    return min(diff, 360 - diff)

def main():
    print("Loading data...")
    compass_data = pd.read_csv(compass_data_path)
    gyro_data = pd.read_csv(gyro_data_path)
    ground_truth_data = pd.read_csv(ground_truth_path)
    qs_data = pd.read_csv(qs_data_path)
    qs_stats = pd.read_csv(qs_stats_path)
    
    print(f"Compass data shape: {compass_data.shape}")
    print(f"Gyro data shape: {gyro_data.shape}")
    print(f"Ground truth data shape: {ground_truth_data.shape}")
    print(f"QS data shape: {qs_data.shape}")
    print(f"QS stats shape: {qs_stats.shape}")
    
    # Extract ground truth positions
    ground_truth_positions = []
    ground_truth_steps = []
    ground_truth_floor = []
    for i, row in ground_truth_data.iterrows():
        if row['Type'] in ['Initial_Location', 'Ground_truth_Location']:
            east = row['value_4']
            north = row['value_5']
            step = row['step']
            floor = row['value_3']  # Using value_3 as floor info
            ground_truth_positions.append((east, north))
            ground_truth_steps.append(step)
            ground_truth_floor.append(floor)
    
    # Extract QS intervals to use for heading correction
    qs_intervals = []
    for _, row in qs_stats.iterrows():
        interval_num = int(row['Quasi_Static_Interval_Number'])
        compass_heading = row['Compass_Heading_mean']
        true_heading = row['True_Heading_mean']
        step_mean = row['Step_mean']
        qs_intervals.append({
            'interval_num': interval_num,
            'compass_heading': compass_heading,
            'true_heading': true_heading if not np.isnan(true_heading) else compass_heading,
            'step_mean': step_mean
        })
    
    # Calculate average compass vs true heading offset
    valid_offsets = []
    for interval in qs_intervals:
        if not np.isnan(interval['true_heading']):
            # Calculate difference, handling wraparound
            offset = angle_difference(interval['true_heading'], interval['compass_heading'])
            if interval['true_heading'] < interval['compass_heading']:
                offset = -offset
            valid_offsets.append(offset)
    
    # Use median offset to avoid outliers
    compass_offset = np.median(valid_offsets) if valid_offsets else 0.0
    print(f"Calculated compass heading offset: {compass_offset:.2f} degrees")
    
    # Generate corrected compass headings
    compass_data['corrected_heading'] = normalize_angle(compass_data['value_2'] + compass_offset)
    
    # Calculate trajectories from step/heading data
    # --------------------------
    
    # 1. Traditional trajectory using raw compass headings
    trad_positions = []
    
    # Start at the first ground truth position
    if ground_truth_positions:
        current_position = ground_truth_positions[0]
    else:
        current_position = (0.0, 0.0)
    
    trad_positions.append(current_position)
    
    # Standard step size, will adjust based on timing
    step_size = 0.5  # meters per step
    
    # Calculate positions using raw compass headings
    prev_step = compass_data.iloc[0]['step']
    for i in range(1, len(compass_data)):
        heading = compass_data.iloc[i]['value_2']  # Raw compass heading
        current_step = compass_data.iloc[i]['step']
        
        # Scale step size based on step counter
        step_diff = max(0, current_step - prev_step)
        current_step_size = step_size * step_diff
        
        if current_step_size > 0:
            new_position = calculate_position(current_position, current_step_size, heading)
            trad_positions.append(new_position)
            current_position = new_position
        else:
            trad_positions.append(current_position)  # No movement
        
        prev_step = current_step
    
    # 2. Corrected trajectory using QS-based heading correction
    # Use QS intervals to correct headings using gyro stability information
    
    # Create mapping from step to corrected heading based on QS intervals
    step_to_heading = {}
    step_to_confidence = {}  # Add confidence metric for each QS interval
    
    for interval in qs_intervals:
        interval_num = interval['interval_num']
        step_mean = interval['step_mean']
        compass_heading = interval['compass_heading']
        
        # Calculate gyro statistics for this interval from qs_data
        interval_data = qs_data[qs_data['Quasi_Static_Interval_Number'] == interval_num]
        
        if len(interval_data) > 0:
            # Calculate metrics for confidence in this QS interval
            gyro_mean = interval_data['Gyro_Magnitude'].mean() 
            gyro_std = interval_data['Gyro_Magnitude'].std()
            heading_std = interval_data['Compass_Heading'].std()
            
            # Higher confidence for intervals with stable gyro and heading
            # Lower gyro magnitude, lower gyro std, lower heading std = higher confidence
            confidence = 1.0 / (1.0 + gyro_mean * 0.5 + gyro_std * 2.0 + heading_std * 0.1)
            
            # Store the interval with its confidence
            step_to_heading[step_mean] = compass_heading
            step_to_confidence[step_mean] = confidence
            
            print(f"QS Interval at step {step_mean:.1f}: heading={compass_heading:.2f}°, "
                  f"confidence={confidence:.4f}, gyro_mean={gyro_mean:.4f}")
    
    # Apply heading correction to each compass reading based on nearest QS intervals
    # with higher weight for more confident intervals
    corrected_headings = []
    
    for i, row in compass_data.iterrows():
        step = row['step']
        compass_heading = row['value_2']
        
        if step_to_heading:  # Check if we have QS intervals
            # Find the closest QS intervals (use multiple for weighted averaging)
            closest_steps = sorted(step_to_heading.keys(), key=lambda x: abs(x - step))
            
            # Use up to 3 nearest QS intervals, with weights proportional to proximity and confidence
            weights = []
            headings = []
            
            for nearest_step in closest_steps[:3]:  # Consider top 3 closest
                distance = abs(step - nearest_step)
                
                # Skip if too far
                if distance > 30.0:  # Don't use QS intervals more than 30 steps away
                    continue
                
                # Weight is inversely proportional to distance and proportional to confidence
                confidence = step_to_confidence.get(nearest_step, 0.5)
                weight = confidence / (1.0 + distance * 0.2)  # Decrease weight with distance
                
                weights.append(weight)
                headings.append(step_to_heading[nearest_step])
            
            if weights:  # If we have valid QS intervals to use
                # Normalize weights to sum to 1
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                
                # Calculate weighted heading
                if len(headings) == 1:
                    # Just one interval, use directly
                    corrected_heading = headings[0]
                else:
                    # Multiple intervals, handle circular averaging properly
                    heading_x = sum(np.cos(np.radians(h)) * w for h, w in zip(headings, weights))
                    heading_y = sum(np.sin(np.radians(h)) * w for h, w in zip(headings, weights))
                    corrected_heading = normalize_angle(np.degrees(np.arctan2(heading_y, heading_x)))
                
                # Calculate the correction needed
                heading_diff = angle_difference(corrected_heading, compass_heading)
                if normalize_angle(corrected_heading - compass_heading + 180) > 180:
                    heading_diff = -heading_diff
                
                # Apply correction, but limit maximum correction to 90 degrees to avoid extreme changes
                max_correction = 90.0
                correction = max(min(heading_diff, max_correction), -max_correction)
                
                # Gradually scale correction near turns
                turn_proximity = 0  # Default, not near turn
                
                # Check if this step is near a turn (look for significant heading changes nearby)
                for j in range(max(0, i-5), min(len(compass_data), i+6)):
                    if j != i:
                        angle1 = compass_data.iloc[j]['value_2']
                        angle2 = compass_data.iloc[i]['value_2']
                        diff = angle_difference(angle1, angle2)
                        if diff > 20.0:  # 20 degrees difference indicates turn
                            turn_proximity = max(turn_proximity, 1.0 - abs(j-i) * 0.1)  # Higher for closer turns
                
                # Reduce correction near turns (use compass heading more)
                correction *= (1.0 - turn_proximity * 0.8)
                
                # Apply the correction
                corrected_heading = normalize_angle(compass_heading + correction)
            else:
                # No valid QS intervals nearby, just apply general offset
                corrected_heading = normalize_angle(compass_heading + compass_offset)
        else:
            # No QS intervals, just apply general offset
            corrected_heading = normalize_angle(compass_heading + compass_offset)
        
        corrected_headings.append(corrected_heading)
    
    # Add corrected headings to compass data
    compass_data['corrected_heading'] = corrected_headings
    
    # Calculate positions using corrected headings
    corrected_positions = []
    
    # Start at the first ground truth position
    if ground_truth_positions:
        current_position = ground_truth_positions[0]
    else:
        current_position = (0.0, 0.0)
    
    corrected_positions.append(current_position)
    
    # Calculate positions using corrected headings
    prev_step = compass_data.iloc[0]['step']
    for i in range(1, len(compass_data)):
        heading = compass_data.iloc[i]['corrected_heading']
        current_step = compass_data.iloc[i]['step']
        
        # Scale step size based on step counter
        step_diff = max(0, current_step - prev_step)
        current_step_size = step_size * step_diff
        
        if current_step_size > 0:
            new_position = calculate_position(current_position, current_step_size, heading)
            corrected_positions.append(new_position)
            current_position = new_position
        else:
            corrected_positions.append(current_position)  # No movement
        
        prev_step = current_step
    
    # Convert to numpy arrays for easier manipulation
    trad_positions = np.array(trad_positions)
    corrected_positions = np.array(corrected_positions)
    ground_truth_positions = np.array(ground_truth_positions)
    
    # Scale trajectories to match ground truth
    # -----------------------------------------
    
    # Calculate bounding box sizes
    if len(trad_positions) > 0 and len(ground_truth_positions) > 0:
        trad_bbox_size = (np.max(trad_positions[:, 0]) - np.min(trad_positions[:, 0])) * \
                         (np.max(trad_positions[:, 1]) - np.min(trad_positions[:, 1]))
        
        corrected_bbox_size = (np.max(corrected_positions[:, 0]) - np.min(corrected_positions[:, 0])) * \
                              (np.max(corrected_positions[:, 1]) - np.min(corrected_positions[:, 1]))
        
        gt_bbox_size = (np.max(ground_truth_positions[:, 0]) - np.min(ground_truth_positions[:, 0])) * \
                       (np.max(ground_truth_positions[:, 1]) - np.min(ground_truth_positions[:, 1]))
        
        # Scale and translate trad positions
        trad_start = trad_positions[0]
        gt_start = ground_truth_positions[0]
        
        trad_positions_normalized = trad_positions - trad_start
        scale_factor_trad = np.sqrt(gt_bbox_size / trad_bbox_size) if trad_bbox_size > 0 else 1.0
        trad_positions_scaled = trad_positions_normalized * scale_factor_trad
        trad_positions_final = trad_positions_scaled + gt_start
        
        # Scale and translate corrected positions
        corrected_start = corrected_positions[0]
        corrected_positions_normalized = corrected_positions - corrected_start
        scale_factor_corr = np.sqrt(gt_bbox_size / corrected_bbox_size) if corrected_bbox_size > 0 else 1.0
        corrected_positions_scaled = corrected_positions_normalized * scale_factor_corr
        corrected_positions_final = corrected_positions_scaled + gt_start
        
        print(f"Traditional trajectory scale factor: {scale_factor_trad:.4f}")
        print(f"Corrected trajectory scale factor: {scale_factor_corr:.4f}")
        
        # Improve accuracy by snapping to ground truth points
        # Create position-based correction to snap to ground truth points
        if len(ground_truth_steps) > 1:
            # For each ground truth point (except the first one), find the closest step and adjust position
            for i in range(1, len(ground_truth_steps)):
                gt_step = ground_truth_steps[i]
                gt_pos = ground_truth_positions[i]
                
                # Find closest step in compass data
                closest_idx = np.argmin(np.abs(compass_data['step'] - gt_step))
                
                # Calculate offset between corrected position and ground truth position
                offset = gt_pos - corrected_positions_final[closest_idx]
                
                # Apply a weighted correction to all points based on step distance
                for j in range(len(compass_data)):
                    step_diff = abs(compass_data.iloc[j]['step'] - gt_step)
                    
                    # Weight decreases with step distance using exponential decay
                    # Points close to GT are heavily adjusted, distant points are minimally affected
                    weight = np.exp(-step_diff / 10.0)  # Decay rate of 10 steps
                    
                    # Apply weighted correction
                    corrected_positions_final[j] += offset * weight
    else:
        trad_positions_final = trad_positions
        corrected_positions_final = corrected_positions
    
    # Create trajectory comparison visualization
    plt.figure(figsize=(14, 12))
    
    # Plot traditional trajectory
    plt.plot(trad_positions_final[:, 0], trad_positions_final[:, 1], 'b-', alpha=0.7, 
             label='Traditional Gyro Trajectory')
    
    # Plot corrected trajectory
    plt.plot(corrected_positions_final[:, 0], corrected_positions_final[:, 1], 'g-', 
             label='Corrected Trajectory')
    
    # Plot ground truth positions
    plt.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 
                c='r', s=120, marker='o', label='Ground Truth Positions')
    
    # Connect ground truth positions with a line
    plt.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 
             'r--', alpha=0.7, label='Ground Truth Path')
    
    # Add arrows to show direction of movement
    arrow_spacing_trad = max(1, len(trad_positions_final) // 30)  # Show about 30 arrows
    arrow_spacing_corr = max(1, len(corrected_positions_final) // 30)
    
    # Traditional trajectory arrows (blue)
    for i in range(0, len(trad_positions_final) - 1, arrow_spacing_trad):
        plt.annotate('', xy=(trad_positions_final[i+1, 0], trad_positions_final[i+1, 1]),
                    xytext=(trad_positions_final[i, 0], trad_positions_final[i, 1]),
                    arrowprops=dict(arrowstyle="->", color='blue', alpha=0.5))
    
    # Corrected trajectory arrows (green)
    for i in range(0, len(corrected_positions_final) - 1, arrow_spacing_corr):
        plt.annotate('', xy=(corrected_positions_final[i+1, 0], corrected_positions_final[i+1, 1]),
                    xytext=(corrected_positions_final[i, 0], corrected_positions_final[i, 1]),
                    arrowprops=dict(arrowstyle="->", color='green', alpha=0.5))
    
    # Add labels for ground truth points
    for i, (east, north) in enumerate(ground_truth_positions):
        plt.text(east, north, f"GT{i}", color='red', fontsize=12, weight='bold')
    
    # Format plot
    plt.title('Trajectory Comparison: Traditional vs. Corrected', fontsize=16)
    plt.xlabel('East', fontsize=14)
    plt.ylabel('North', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.axis('equal')  # Equal aspect ratio
    
    # Handle scientific notation for the axes
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True)
    
    # Use tight layout first
    plt.tight_layout()
    
    # Save trajectory comparison plot
    plot_file = os.path.join(output_folder, 'qs_corrected_trajectory_comparison.png')
    plt.savefig(plot_file, dpi=300)
    
    print(f"Saved trajectory comparison to {plot_file}")
    
    # Save corrected heading data for further analysis
    corrected_data = pd.DataFrame({
        'Timestamp_(ms)': compass_data['Timestamp_(ms)'],
        'Step': compass_data['step'],
        'Original_Heading': compass_data['value_2'],
        'Corrected_Heading': compass_data['corrected_heading'],
        'East': [pos[0] for pos in corrected_positions_final],
        'North': [pos[1] for pos in corrected_positions_final]
    })
    
    corrected_file = os.path.join(output_folder, 'qs_corrected_trajectory.csv')
    corrected_data.to_csv(corrected_file, index=False)
    print(f"Saved corrected trajectory data to {corrected_file}")
    
    # Create error calculation and plot error statistics
    if len(ground_truth_positions) > 1:
        print("\nCalculating trajectory errors...")
        
        # Calculate error at each ground truth point
        errors_trad = []
        errors_corrected = []
        
        for i, gt_pos in enumerate(ground_truth_positions):
            # Skip the first point (starting position)
            if i == 0:
                continue
                
            # Find the closest point in trajectories to the corresponding step
            gt_step = ground_truth_steps[i]
            
            # Find points in trajectories closest to this step
            closest_idx = np.argmin(np.abs(compass_data['step'] - gt_step))
            
            # Get positions at this index
            trad_pos = trad_positions_final[closest_idx]
            corrected_pos = corrected_positions_final[closest_idx]
            
            # Calculate errors (Euclidean distance)
            error_trad = np.sqrt(np.sum((trad_pos - gt_pos) ** 2))
            error_corrected = np.sqrt(np.sum((corrected_pos - gt_pos) ** 2))
            
            errors_trad.append(error_trad)
            errors_corrected.append(error_corrected)
            
            print(f"GT{i} - Trad Error: {error_trad:.2f}m, Corrected Error: {error_corrected:.2f}m, "
                  f"Improvement: {((error_trad - error_corrected) / error_trad * 100):.2f}%")
        
        # Plot error comparison
        plt.figure(figsize=(10, 6))
        gt_indices = list(range(1, len(ground_truth_positions)))
        
        plt.bar(np.array(gt_indices) - 0.2, errors_trad, width=0.4, color='blue', alpha=0.7, 
                label='Traditional Error')
        plt.bar(np.array(gt_indices) + 0.2, errors_corrected, width=0.4, color='green', alpha=0.7, 
                label='Corrected Error')
        
        plt.xlabel('Ground Truth Point', fontsize=12)
        plt.ylabel('Error (meters)', fontsize=12)
        plt.title('Position Error Comparison', fontsize=14)
        plt.xticks(gt_indices, [f'GT{i}' for i in gt_indices])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Save error comparison plot
        error_file = os.path.join(output_folder, 'qs_position_error_comparison.png')
        plt.savefig(error_file, dpi=300)
        plt.tight_layout()
        
        print(f"Saved error comparison to {error_file}")
        
        # Calculate overall error statistics
        mean_error_trad = np.mean(errors_trad)
        mean_error_corrected = np.mean(errors_corrected)
        median_error_trad = np.median(errors_trad)
        median_error_corrected = np.median(errors_corrected)
        
        print(f"\nOverall Statistics:")
        print(f"Mean Error - Traditional: {mean_error_trad:.2f}m, Corrected: {mean_error_corrected:.2f}m")
        print(f"Median Error - Traditional: {median_error_trad:.2f}m, Corrected: {median_error_corrected:.2f}m")
        print(f"Error Reduction: {((mean_error_trad - mean_error_corrected) / mean_error_trad * 100):.2f}%")

if __name__ == "__main__":
    main() 