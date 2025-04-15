import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths for input and output
input_file = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_ground_truth_data.csv'

# Create output directory if it doesn't exist
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2'
os.makedirs(output_dir, exist_ok=True)

# Load the Phase 1 cleaned data
try:
    data = pd.read_csv(input_file)
    print(f"Successfully loaded data with {len(data)} rows")
except Exception as e:
    print(f"Error loading data: {e}")
    # Try with different encoding if needed
    try:
        data = pd.read_csv(input_file, encoding='latin1')
        print(f"Successfully loaded data with encoding latin1, {len(data)} rows")
    except Exception as e:
        print(f"Error loading with encoding latin1: {e}")
        exit(1)

# Display the first few rows to understand the structure
print("\nFirst few rows of the data:")
print(data.head())
print("\nColumns in the data:", data.columns.tolist())

# Extract ground truth location data
try:
    # Filter out Initial_Location to focus only on ground truth points
    ground_truth_location_data = data[data['Type'] == 'Ground_truth_Location'].copy()
    
    # Also include the Initial_Location as the starting point
    initial_location = data[data['Type'] == 'Initial_Location'].copy()
    
    # Combine them if both exist
    if len(initial_location) > 0:
        ground_truth_data = pd.concat([initial_location, ground_truth_location_data], ignore_index=True)
        ground_truth_data.sort_values(by='step', inplace=True)
        ground_truth_data.reset_index(drop=True, inplace=True)
    else:
        ground_truth_data = ground_truth_location_data
    
    print(f"\nFound {len(ground_truth_data)} ground truth points (including initial location)")
    
    # Check if there's any ground truth data
    if len(ground_truth_data) == 0:
        print("No ground truth data found.")
        exit(1)
        
    # Display a few ground truth entries to verify
    print("\nSample ground truth entries:")
    print(ground_truth_data.head())
except Exception as e:
    print(f"Error extracting ground truth data: {e}")
    print("Columns in the data:", data.columns.tolist())
    exit(1)

# If ground truth data is successfully extracted, proceed with interpolation
if len(ground_truth_data) > 1:
    # Make sure necessary columns exist
    required_columns = ['step', 'value_4', 'value_5']
    if not all(col in ground_truth_data.columns for col in required_columns):
        print(f"Missing required columns. Available columns: {ground_truth_data.columns.tolist()}")
        exit(1)
    
    # Ensure data types are numeric
    for col in required_columns:
        if ground_truth_data[col].dtype == 'object':
            ground_truth_data[col] = pd.to_numeric(ground_truth_data[col], errors='coerce')
    
    # Initialize an empty list to store the interpolated positions
    interpolated_positions = []

    # Iterate over each pair of successive ground truth locations
    for i in range(len(ground_truth_data) - 1):
        current_row = ground_truth_data.iloc[i]
        next_row = ground_truth_data.iloc[i + 1]
        
        # Calculate the Euclidean distance between points
        distance = np.sqrt(
            (next_row['value_4'] - current_row['value_4'])**2 + 
            (next_row['value_5'] - current_row['value_5'])**2
        )
        
        # Increase the number of points based on the distance
        # Use a higher density to ensure more points between manually labeled ones
        # For every 10 units of distance, add at least 10 points
        density_factor = 1.0  # Adjust this value to control density
        min_points = 20       # Minimum number of points between any two ground truth locations
        num_points = max(min_points, int(distance * density_factor / 10))
        
        # Calculate the step size between successive points
        step_diff = next_row['step'] - current_row['step']
        step_size = step_diff / num_points if num_points > 0 else 0
        
        # Generate interpolated points
        for j in range(num_points):
            # Calculate the current step
            current_step = current_row['step'] + j * step_size
            
            # Calculate the interpolation factor
            interpolation_factor = j / num_points if num_points > 0 else 0
            
            # Calculate the interpolated x and y positions
            x_pos = current_row['value_4'] + interpolation_factor * (next_row['value_4'] - current_row['value_4'])
            y_pos = current_row['value_5'] + interpolation_factor * (next_row['value_5'] - current_row['value_5'])
            
            # Append the interpolated position to the list
            interpolated_positions.append({
                'step': current_step,
                'value_4': x_pos,
                'value_5': y_pos,
            })

    # Append the last ground truth location to the list
    interpolated_positions.append({
        'step': ground_truth_data.iloc[-1]['step'],
        'value_4': ground_truth_data.iloc[-1]['value_4'],
        'value_5': ground_truth_data.iloc[-1]['value_5'],
    })

    # Create a DataFrame from the list of interpolated positions
    interpolated_positions_df = pd.DataFrame(interpolated_positions)

    # Sort the DataFrame by the 'step' column
    interpolated_positions_df.sort_values(by='step', inplace=True)

    # Reset the index of the DataFrame
    interpolated_positions_df.reset_index(drop=True, inplace=True)

    # Select only the 'step' and 'value_4' (x-coordinate) and 'value_5' (y-coordinate) columns
    ground_truth_positions_steps = interpolated_positions_df[['step', 'value_4', 'value_5']]

    # Rename columns for clarity
    ground_truth_positions_steps = ground_truth_positions_steps.rename(columns={
        'value_5': 'ground_y',
        'value_4': 'ground_x'
    })

    # Save the ground truth positions to CSV
    output_csv = os.path.join(output_dir, 'ground_truth_positions_steps.csv')
    ground_truth_positions_steps.to_csv(output_csv, index=False)
    print(f"\nSaved ground truth positions to {output_csv}")

    # Calculate distances between consecutive ground truth positions for speed analysis
    distances = []
    times = []
    speeds = []
    
    # Get ground truth data steps (these are time points)
    time_points = ground_truth_data['step'].values
    
    # Calculate distances between original ground truth points (not interpolated)
    for i in range(len(ground_truth_data) - 1):
        current_row = ground_truth_data.iloc[i]
        next_row = ground_truth_data.iloc[i + 1]
        
        # Calculate the distance
        dist = np.sqrt(
            (next_row['value_4'] - current_row['value_4'])**2 + 
            (next_row['value_5'] - current_row['value_5'])**2
        )
        
        # Calculate the time difference (steps are time units)
        time_diff = next_row['step'] - current_row['step']
        
        # Calculate speed (distance / time)
        speed = dist / time_diff if time_diff > 0 else 0
        
        distances.append(dist)
        times.append(time_diff)
        speeds.append(speed)
    
    # Create a data frame for the speed analysis
    speed_analysis = pd.DataFrame({
        'segment': range(1, len(ground_truth_data)),
        'start_step': time_points[:-1],
        'end_step': time_points[1:],
        'distance': distances,
        'time': times,
        'speed': speeds
    })
    
    # Save the speed analysis to CSV
    speed_csv = os.path.join(output_dir, 'ground_truth_speed_analysis.csv')
    speed_analysis.to_csv(speed_csv, index=False)
    print(f"Saved speed analysis to {speed_csv}")

    # Plot the interpolated positions in the x-y plane with improved styling
    plt.figure(figsize=(15, 8), dpi=300)
    
    # Create color gradient for the path to show progression
    points = np.array([interpolated_positions_df['value_4'], interpolated_positions_df['value_5']]).T
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
    
    # Plot path with line collection for color gradient
    from matplotlib.collections import LineCollection
    norm = plt.Normalize(interpolated_positions_df['step'].min(), interpolated_positions_df['step'].max())
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2, alpha=0.8)
    lc.set_array(interpolated_positions_df['step'][:-1])
    line = plt.gca().add_collection(lc)
    
    # Add colorbar for time reference
    cbar = plt.colorbar(line, ax=plt.gca())
    cbar.set_label('Step Number (Time)', fontsize=12)
    
    # Plot scattered points
    plt.scatter(interpolated_positions_df['value_4'], interpolated_positions_df['value_5'], 
                marker='.', s=20, color='blue', alpha=0.5, label='Interpolated path')
    plt.scatter(ground_truth_data['value_4'], ground_truth_data['value_5'], 
                marker='*', s=150, color='red', label='Ground truth points')
    
    # Mark start and end points
    plt.scatter(ground_truth_data.iloc[0]['value_4'], ground_truth_data.iloc[0]['value_5'],
                marker='o', s=200, color='green', label='Start point')
    plt.scatter(ground_truth_data.iloc[-1]['value_4'], ground_truth_data.iloc[-1]['value_5'],
                marker='s', s=200, color='purple', label='End point')
    
    # Improve plot aesthetics
    plt.xlabel('East Coordinate', fontsize=14)
    plt.ylabel('North Coordinate', fontsize=14)
    plt.title('Ground Truth Path with Time Progression', fontsize=16, fontweight='bold')
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    output_plot = os.path.join(output_dir, 'ground_truth_path_progression.png')
    plt.savefig(output_plot, bbox_inches='tight')
    print(f"Saved ground truth progression plot to {output_plot}")
    
    # Generate a second plot with step annotations
    plt.figure(figsize=(15, 10), dpi=300)
    
    # Plot the base path
    plt.plot(interpolated_positions_df['value_4'], interpolated_positions_df['value_5'], 
             'b-', linewidth=1.5, alpha=0.6)
    
    # Plot the ground truth points
    plt.scatter(ground_truth_data['value_4'], ground_truth_data['value_5'], 
                marker='*', s=180, color='red', label='Ground truth points')
    
    # Mark start and end points
    plt.scatter(ground_truth_data.iloc[0]['value_4'], ground_truth_data.iloc[0]['value_5'],
                marker='o', s=200, color='green', label='Start point')
    plt.scatter(ground_truth_data.iloc[-1]['value_4'], ground_truth_data.iloc[-1]['value_5'],
                marker='s', s=200, color='purple', label='End point')
    
    # Annotate points with step numbers for better reference
    for idx, row in ground_truth_data.iterrows():
        plt.annotate(f"Step {row['step']:.0f}", 
                     (row['value_4'], row['value_5']),
                     textcoords="offset points", 
                     xytext=(5, 10), 
                     ha='center',
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Improve plot aesthetics
    plt.xlabel('East Coordinate', fontsize=14)
    plt.ylabel('North Coordinate', fontsize=14)
    plt.title('Ground Truth Path with Step Annotations', fontsize=16, fontweight='bold')
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    annotated_plot = os.path.join(output_dir, 'ground_truth_path_annotated.png')
    plt.savefig(annotated_plot, bbox_inches='tight')
    print(f"Saved annotated ground truth plot to {annotated_plot}")
    
    # Generate a third visualization - 3D plot with time as the third dimension
    fig = plt.figure(figsize=(16, 12), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D trajectory
    ax.plot3D(
        interpolated_positions_df['value_4'], 
        interpolated_positions_df['value_5'], 
        interpolated_positions_df['step'],
        'b-', linewidth=2, alpha=0.7, label='Path trajectory'
    )
    
    # Plot ground truth points in 3D
    ax.scatter(
        ground_truth_data['value_4'], 
        ground_truth_data['value_5'], 
        ground_truth_data['step'],
        c='red', marker='*', s=150, label='Ground truth points'
    )
    
    # Mark start and end
    ax.scatter(
        ground_truth_data.iloc[0]['value_4'], 
        ground_truth_data.iloc[0]['value_5'], 
        ground_truth_data.iloc[0]['step'],
        c='green', marker='o', s=200, label='Start'
    )
    
    ax.scatter(
        ground_truth_data.iloc[-1]['value_4'], 
        ground_truth_data.iloc[-1]['value_5'], 
        ground_truth_data.iloc[-1]['step'],
        c='purple', marker='s', s=200, label='End'
    )
    
    # Improve 3D plot aesthetics
    ax.set_xlabel('East Coordinate', fontsize=14, labelpad=15)
    ax.set_ylabel('North Coordinate', fontsize=14, labelpad=15)
    ax.set_zlabel('Step (Time)', fontsize=14, labelpad=15)
    ax.set_title('3D Trajectory of Ground Truth Path', fontsize=18, fontweight='bold', pad=20)
    
    # Set better tick parameters
    ax.xaxis.set_tick_params(labelsize=12, pad=5)
    ax.yaxis.set_tick_params(labelsize=12, pad=5)
    ax.zaxis.set_tick_params(labelsize=12, pad=5)
    
    # Set a better viewpoint that shows all axes clearly, including the z-axis
    ax.view_init(elev=25, azim=-35)
    
    # Ensure enough padding around the plot for labels
    ax.autoscale(enable=True, axis='both', tight=False)
    
    # Create improved legend with better positioning
    ax.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1.1, 1.05), frameon=True, 
              framealpha=0.9, facecolor='white', edgecolor='gray')
    
    # Add a bit more margin on the right side of the figure for z-axis visibility
    plt.subplots_adjust(right=0.85)
    
    # Save 3D plot with extra padding on the right
    trajectory_plot = os.path.join(output_dir, 'ground_truth_3d_trajectory.png')
    plt.savefig(trajectory_plot, bbox_inches='tight', pad_inches=0.5)
    print(f"Saved 3D trajectory plot to {trajectory_plot}")
    
    plt.close('all')
    
    print("\nAll ground truth visualizations completed successfully!")
else:
    print("Could not proceed with interpolation due to insufficient ground truth data") 