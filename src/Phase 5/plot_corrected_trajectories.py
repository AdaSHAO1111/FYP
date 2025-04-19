import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Define paths
compass_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 5/Compass_QS/compass_heading_corrected.csv'
gyro_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 5/Compass_QS/gyro_heading_corrected.csv'
ground_truth_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_ground_truth_data.csv'
output_folder = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 4/QS_Compass4.19'

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def normalize_angle(angle):
    """Normalize angle to be in the range [0, 360)"""
    return angle % 360

def heading_to_direction(heading_deg):
    """Convert heading in degrees to unit vector direction"""
    # Convert to radians and adjust for coordinate system
    heading_rad = np.radians(heading_deg)
    
    # Calculate x and y components (north = +y, east = +x)
    # In navigation, 0° is North, 90° is East
    x = np.sin(heading_rad)  # East component
    y = np.cos(heading_rad)  # North component
    
    return x, y

def calculate_positions(data, step_column='step', heading_column='Corrected_Heading'):
    """
    Calculate positions based on step and heading data
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame containing step and heading information
    step_column : str
        Column name for step data
    heading_column : str
        Column name for heading data
    
    Returns:
    --------
    positions : DataFrame
        DataFrame with x, y positions
    """
    # Initialize output DataFrame
    positions = pd.DataFrame()
    positions['step'] = data[step_column]
    positions['heading'] = data[heading_column]
    positions['x'] = 0.0
    positions['y'] = 0.0
    
    # Step length in meters (can be adjusted based on actual data)
    step_length = 0.7
    
    # Start from origin (0,0)
    current_x, current_y = 0.0, 0.0
    last_step = 0.0
    
    # For each data point
    for i, row in data.iterrows():
        # Get step and heading
        step = row[step_column]
        heading = row[heading_column]
        
        # Calculate step difference
        step_diff = step - last_step
        last_step = step
        
        if i == 0 or step_diff <= 0:
            # First point or no step change, just store current position
            positions.loc[i, 'x'] = current_x
            positions.loc[i, 'y'] = current_y
            continue
        
        # Get direction from heading
        direction_x, direction_y = heading_to_direction(heading)
        
        # Calculate movement based on step difference
        move_distance = step_diff * step_length
        
        # Update position
        current_x += direction_x * move_distance
        current_y += direction_y * move_distance
        
        # Store position
        positions.loc[i, 'x'] = current_x
        positions.loc[i, 'y'] = current_y
    
    return positions

def extract_ground_truth_positions(ground_truth_data):
    """
    Extract ground truth positions from data
    
    Parameters:
    -----------
    ground_truth_data : DataFrame
        Ground truth data
    
    Returns:
    --------
    gt_positions : DataFrame
        DataFrame with ground truth x, y positions
    """
    # Filter to include only ground truth points
    gt_points = ground_truth_data[ground_truth_data['Type'].isin(['Initial_Location', 'Ground_truth_Location'])]
    
    # Use the easting (value_4) and northing (value_5) columns
    # But normalize to have the first point at origin
    if 'value_4' in gt_points.columns and 'value_5' in gt_points.columns:
        # Extract coordinates
        x_values = gt_points['value_4'].values
        y_values = gt_points['value_5'].values
        
        # Normalize to first point
        x_offset = x_values[0]
        y_offset = y_values[0]
        
        # Adjust scale
        scale = 1.0
        
        # Create positions DataFrame
        gt_positions = pd.DataFrame({
            'step': gt_points['step'].values,
            'x': (x_values - x_offset) * scale,
            'y': (y_values - y_offset) * scale,
            'timestamp': gt_points['Timestamp_(ms)'].values,
            'type': gt_points['Type'].values
        })
        
        return gt_positions
    else:
        return None

def plot_trajectories(compass_positions, gyro_positions, ground_truth_positions, output_folder):
    """
    Plot trajectories from position data
    
    Parameters:
    -----------
    compass_positions : DataFrame
        Compass position data
    gyro_positions : DataFrame
        Gyro position data
    ground_truth_positions : DataFrame
        Ground truth position data
    output_folder : str
        Folder to save plots
    """
    # Plot trajectories
    plt.figure(figsize=(12, 10))
    
    # Plot compass trajectory
    plt.plot(compass_positions['x'], compass_positions['y'], 
             'b-', alpha=0.7, linewidth=2, label='Corrected Compass Trajectory')
    
    # Plot gyro trajectory
    plt.plot(gyro_positions['x'], gyro_positions['y'], 
             'g-', alpha=0.7, linewidth=2, label='Corrected Gyro Trajectory')
    
    # Plot ground truth points
    if ground_truth_positions is not None:
        plt.scatter(ground_truth_positions['x'], ground_truth_positions['y'], 
                   color='red', s=100, marker='X', label='Ground Truth')
        
        # Connect ground truth points
        plt.plot(ground_truth_positions['x'], ground_truth_positions['y'], 
                'r--', alpha=0.7, linewidth=1.5, label='Ground Truth Path')
        
        # Add point labels
        for i, row in ground_truth_positions.iterrows():
            plt.annotate(f"GT{i}", (row['x'], row['y']), 
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=10, fontweight='bold')
    
    # Add grid, legend, and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('Trajectory Comparison')
    plt.legend()
    
    # Make axes equal for proper scaling
    plt.axis('equal')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'trajectory_comparison.png'), dpi=300)
    
    # Create detailed subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot heading comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(compass_positions['step'], compass_positions['heading'], 'b-', label='Compass')
    ax1.plot(gyro_positions['step'], gyro_positions['heading'], 'g-', label='Gyro')
    if ground_truth_positions is not None and 'GroundTruthHeadingComputed' in ground_truth_data.columns:
        headings = ground_truth_data.loc[
            ground_truth_data['Type'].isin(['Initial_Location', 'Ground_truth_Location']), 
            'GroundTruthHeadingComputed'
        ].values
        steps = ground_truth_positions['step'].values
        valid_indices = ~np.isnan(headings)
        ax1.scatter(steps[valid_indices], headings[valid_indices], c='r', marker='X', s=80, label='Ground Truth')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Heading (degrees)')
    ax1.set_title('Corrected Headings vs. Ground Truth')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot position comparison - X (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(compass_positions['step'], compass_positions['x'], 'b-', label='Compass X')
    ax2.plot(gyro_positions['step'], gyro_positions['x'], 'g-', label='Gyro X')
    if ground_truth_positions is not None:
        ax2.scatter(ground_truth_positions['step'], ground_truth_positions['x'], 
                   c='r', marker='X', s=80, label='Ground Truth X')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('East Position (m)')
    ax2.set_title('East Position Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot position comparison - Y (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(compass_positions['step'], compass_positions['y'], 'b-', label='Compass Y')
    ax3.plot(gyro_positions['step'], gyro_positions['y'], 'g-', label='Gyro Y')
    if ground_truth_positions is not None:
        ax3.scatter(ground_truth_positions['step'], ground_truth_positions['y'], 
                   c='r', marker='X', s=80, label='Ground Truth Y')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('North Position (m)')
    ax3.set_title('North Position Comparison')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot full trajectory (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(compass_positions['x'], compass_positions['y'], 'b-', alpha=0.7, linewidth=2, label='Compass')
    ax4.plot(gyro_positions['x'], gyro_positions['y'], 'g-', alpha=0.7, linewidth=2, label='Gyro')
    if ground_truth_positions is not None:
        ax4.scatter(ground_truth_positions['x'], ground_truth_positions['y'], 
                   c='r', marker='X', s=80, label='Ground Truth')
        ax4.plot(ground_truth_positions['x'], ground_truth_positions['y'], 
                'r--', alpha=0.7, linewidth=1.5)
    ax4.set_xlabel('East (m)')
    ax4.set_ylabel('North (m)')
    ax4.set_title('Trajectories')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'trajectory_detailed_comparison.png'), dpi=300)
    
    # Calculate error metrics
    error_metrics = calculate_trajectory_errors(compass_positions, gyro_positions, ground_truth_positions)
    
    # Save error metrics to CSV
    error_df = pd.DataFrame(error_metrics)
    error_df.to_csv(os.path.join(output_folder, 'trajectory_error_metrics.csv'), index=False)
    
    # Create error table plot
    fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
    
    # Hide axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Create table
    table_data = []
    columns = ['GT Point', 'Compass Error (m)', 'Gyro Error (m)']
    
    for point in error_metrics:
        table_data.append([
            point['gt_point'],
            f"{point['compass_error']:.2f}",
            f"{point['gyro_error']:.2f}"
        ])
    
    # Add summary row
    avg_compass_error = np.mean([point['compass_error'] for point in error_metrics])
    avg_gyro_error = np.mean([point['gyro_error'] for point in error_metrics])
    table_data.append(['Average', f"{avg_compass_error:.2f}", f"{avg_gyro_error:.2f}"])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Title
    plt.title('Trajectory Error Metrics (meters)', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'trajectory_error_table.png'), dpi=300)
    
    return error_metrics

def calculate_trajectory_errors(compass_positions, gyro_positions, ground_truth_positions):
    """
    Calculate trajectory error metrics
    
    Parameters:
    -----------
    compass_positions : DataFrame
        Compass position data
    gyro_positions : DataFrame
        Gyro position data
    ground_truth_positions : DataFrame
        Ground truth position data
    
    Returns:
    --------
    error_metrics : list
        List of dictionaries with error metrics
    """
    if ground_truth_positions is None:
        return []
    
    error_metrics = []
    
    for i, gt_row in ground_truth_positions.iterrows():
        gt_step = gt_row['step']
        gt_x = gt_row['x']
        gt_y = gt_row['y']
        
        # Find closest compass and gyro points based on step
        compass_idx = (compass_positions['step'] - gt_step).abs().idxmin()
        gyro_idx = (gyro_positions['step'] - gt_step).abs().idxmin()
        
        compass_x = compass_positions.loc[compass_idx, 'x']
        compass_y = compass_positions.loc[compass_idx, 'y']
        
        gyro_x = gyro_positions.loc[gyro_idx, 'x']
        gyro_y = gyro_positions.loc[gyro_idx, 'y']
        
        # Calculate Euclidean distances (errors)
        compass_error = np.sqrt((compass_x - gt_x)**2 + (compass_y - gt_y)**2)
        gyro_error = np.sqrt((gyro_x - gt_x)**2 + (gyro_y - gt_y)**2)
        
        error_metrics.append({
            'gt_point': f"GT{i}",
            'step': gt_step,
            'compass_error': compass_error,
            'gyro_error': gyro_error
        })
    
    return error_metrics

def main():
    print("\n=== Plotting Corrected Trajectories ===\n")
    
    # 1. Load data
    print("Loading data...")
    compass_data = pd.read_csv(compass_data_path)
    gyro_data = pd.read_csv(gyro_data_path)
    global ground_truth_data
    ground_truth_data = pd.read_csv(ground_truth_path)
    
    print(f"Loaded {len(compass_data)} compass data points")
    print(f"Loaded {len(gyro_data)} gyro data points")
    print(f"Loaded {len(ground_truth_data)} ground truth data points")
    
    # 2. Calculate positions from corrected headings
    print("\nCalculating positions from corrected headings...")
    
    # For compass data
    compass_positions = calculate_positions(compass_data, 'step', 'Corrected_Heading')
    print(f"Calculated positions for {len(compass_positions)} compass data points")
    
    # For gyro data
    gyro_positions = calculate_positions(gyro_data, 'step', 'Corrected_Heading')
    print(f"Calculated positions for {len(gyro_positions)} gyro data points")
    
    # Extract ground truth positions
    ground_truth_positions = extract_ground_truth_positions(ground_truth_data)
    if ground_truth_positions is not None:
        print(f"Extracted {len(ground_truth_positions)} ground truth positions")
    else:
        print("No ground truth position data available")
    
    # 3. Plot trajectories
    print("\nPlotting trajectories...")
    error_metrics = plot_trajectories(compass_positions, gyro_positions, ground_truth_positions, output_folder)
    
    # 4. Print error metrics
    print("\nTrajectory Error Metrics:")
    avg_compass_error = np.mean([point['compass_error'] for point in error_metrics])
    avg_gyro_error = np.mean([point['gyro_error'] for point in error_metrics])
    
    print(f"Average Compass Error: {avg_compass_error:.2f} meters")
    print(f"Average Gyro Error: {avg_gyro_error:.2f} meters")
    
    print("\nDetailed Errors:")
    for point in error_metrics:
        print(f"  {point['gt_point']} (Step {point['step']}): "
              f"Compass Error = {point['compass_error']:.2f}m, "
              f"Gyro Error = {point['gyro_error']:.2f}m")
    
    # 5. Save trajectory data
    print("\nSaving trajectory data...")
    compass_positions.to_csv(os.path.join(output_folder, 'compass_corrected_trajectory.csv'), index=False)
    gyro_positions.to_csv(os.path.join(output_folder, 'gyro_corrected_trajectory.csv'), index=False)
    if ground_truth_positions is not None:
        ground_truth_positions.to_csv(os.path.join(output_folder, 'ground_truth_trajectory.csv'), index=False)
    
    print(f"\nAll results saved to: {output_folder}")
    print("Trajectory plotting completed successfully!")

if __name__ == "__main__":
    main() 