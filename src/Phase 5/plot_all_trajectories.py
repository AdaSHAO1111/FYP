import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths
compass_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_compass_data.csv'
gyro_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_gyro_data.csv'
corrected_compass_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 5/Compass_QS/compass_heading_corrected.csv'
corrected_gyro_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 5/Compass_QS/gyro_heading_corrected.csv'
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

def calculate_positions(data, step_column='step', heading_column='value_2'):
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
    positions['heading'] = data[heading_column].apply(normalize_angle)
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
        heading = normalize_angle(row[heading_column])
        
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

def plot_all_trajectories(original_compass_positions, original_gyro_positions, 
                       corrected_compass_positions, corrected_gyro_positions, 
                       ground_truth_positions, output_folder):
    """
    Plot trajectories from position data
    
    Parameters:
    -----------
    original_compass_positions : DataFrame
        Original compass position data
    original_gyro_positions : DataFrame
        Original gyro position data
    corrected_compass_positions : DataFrame
        Corrected compass position data
    corrected_gyro_positions : DataFrame
        Corrected gyro position data
    ground_truth_positions : DataFrame
        Ground truth position data
    output_folder : str
        Folder to save plots
    """
    # Plot trajectories
    plt.figure(figsize=(14, 12))
    
    # Plot original compass trajectory
    plt.plot(original_compass_positions['x'], original_compass_positions['y'], 
             'b--', alpha=0.4, linewidth=1.5, label='Original Compass Trajectory')
    
    # Plot original gyro trajectory
    plt.plot(original_gyro_positions['x'], original_gyro_positions['y'], 
             'g--', alpha=0.4, linewidth=1.5, label='Original Gyro Trajectory')
    
    # Plot corrected compass trajectory
    plt.plot(corrected_compass_positions['x'], corrected_compass_positions['y'], 
             'b-', alpha=0.8, linewidth=2.5, label='Corrected Compass Trajectory')
    
    # Plot corrected gyro trajectory
    plt.plot(corrected_gyro_positions['x'], corrected_gyro_positions['y'], 
             'g-', alpha=0.8, linewidth=2.5, label='Corrected Gyro Trajectory')
    
    # Plot ground truth points
    if ground_truth_positions is not None:
        plt.scatter(ground_truth_positions['x'], ground_truth_positions['y'], 
                   color='red', s=120, marker='X', label='Ground Truth')
        
        # Connect ground truth points
        plt.plot(ground_truth_positions['x'], ground_truth_positions['y'], 
                'r-', alpha=0.9, linewidth=2, label='Ground Truth Path')
        
        # Add point labels
        for i, row in ground_truth_positions.iterrows():
            plt.annotate(f"GT{i}", (row['x'], row['y']), 
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=12, fontweight='bold')
    
    # Add grid, legend, and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('East (m)', fontsize=14)
    plt.ylabel('North (m)', fontsize=14)
    plt.title('Trajectory Comparison - Original vs. Corrected', fontsize=16)
    plt.legend(fontsize=12)
    
    # Make axes equal for proper scaling
    plt.axis('equal')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'all_trajectories_comparison.png'), dpi=300)
    
    # Save the positions to a CSV file for reference
    all_positions = pd.DataFrame({
        'Step': original_compass_positions['step'],
        'Original_Compass_X': original_compass_positions['x'],
        'Original_Compass_Y': original_compass_positions['y'],
        'Original_Gyro_X': original_gyro_positions['x'],
        'Original_Gyro_Y': original_gyro_positions['y'],
        'Corrected_Compass_X': corrected_compass_positions['x'],
        'Corrected_Compass_Y': corrected_compass_positions['y'],
        'Corrected_Gyro_X': corrected_gyro_positions['x'],
        'Corrected_Gyro_Y': corrected_gyro_positions['y']
    })
    all_positions.to_csv(os.path.join(output_folder, 'position_coordinates.csv'), index=False)
    
    # Print a message with the file path
    print(f"Position coordinates saved to: {os.path.join(output_folder, 'position_coordinates.csv')}")

def main():
    print("\n=== Plotting All Trajectories (Original and Corrected) ===\n")
    
    # 1. Load data
    print("Loading data...")
    original_compass_data = pd.read_csv(compass_data_path)
    original_gyro_data = pd.read_csv(gyro_data_path)
    corrected_compass_data = pd.read_csv(corrected_compass_path)
    corrected_gyro_data = pd.read_csv(corrected_gyro_path)
    ground_truth_data = pd.read_csv(ground_truth_path)
    
    print(f"Loaded original compass data: {len(original_compass_data)} points")
    print(f"Loaded original gyro data: {len(original_gyro_data)} points")
    print(f"Loaded corrected compass data: {len(corrected_compass_data)} points")
    print(f"Loaded corrected gyro data: {len(corrected_gyro_data)} points")
    print(f"Loaded ground truth data: {len(ground_truth_data)} points")
    
    # 2. Calculate positions from original and corrected headings
    print("\nCalculating positions from original and corrected headings...")
    
    # Original compass positions
    original_compass_positions = calculate_positions(original_compass_data, 'step', 'value_2')
    print(f"Calculated positions for {len(original_compass_positions)} original compass data points")
    
    # Original gyro positions
    original_gyro_positions = calculate_positions(original_gyro_data, 'step', 'value_3')
    print(f"Calculated positions for {len(original_gyro_positions)} original gyro data points")
    
    # Corrected compass positions
    corrected_compass_positions = calculate_positions(corrected_compass_data, 'step', 'Corrected_Heading')
    print(f"Calculated positions for {len(corrected_compass_positions)} corrected compass data points")
    
    # Corrected gyro positions
    corrected_gyro_positions = calculate_positions(corrected_gyro_data, 'step', 'Corrected_Heading')
    print(f"Calculated positions for {len(corrected_gyro_positions)} corrected gyro data points")
    
    # Extract ground truth positions
    ground_truth_positions = extract_ground_truth_positions(ground_truth_data)
    if ground_truth_positions is not None:
        print(f"Extracted {len(ground_truth_positions)} ground truth positions")
    else:
        print("No ground truth position data available")
    
    # 3. Plot trajectories
    print("\nPlotting all trajectories...")
    plot_all_trajectories(
        original_compass_positions, 
        original_gyro_positions,
        corrected_compass_positions, 
        corrected_gyro_positions, 
        ground_truth_positions, 
        output_folder
    )
    
    print(f"\nAll results saved to: {output_folder}")
    print("Trajectory plotting completed successfully!")

if __name__ == "__main__":
    main() 