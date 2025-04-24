import pandas as pd
import numpy as np
import os
import datetime
from math import atan2, degrees, radians, sin, cos
from scipy.interpolate import interp1d

# Set up output directory
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/1578data result/Phase 2'
os.makedirs(output_dir, exist_ok=True)

# Input files
compass_data_file = '/Users/shaoxinyi/Downloads/FYP2/Output/1578data result/Phase 1/1578_cleaned_compass_data.csv'
ground_truth_file = '/Users/shaoxinyi/Downloads/FYP2/Output/1578data result/Phase 1/1578_cleaned_ground_truth_data.csv'

print(f"Loading data from files...")
try:
    # Load compass data
    compass_data = pd.read_csv(compass_data_file)
    # Load ground truth data
    gt_data = pd.read_csv(ground_truth_file)
    
    print(f"Successfully loaded data")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Extract Initial Location and Ground Truth Location data
initial_location_data = gt_data[gt_data['Type'] == 'Initial_Location'].copy()
ground_truth_location_data = gt_data[gt_data['Type'] == 'Ground_truth_Location'].copy()

# Combine initial and ground truth locations and sort
if len(ground_truth_location_data) > 0 and len(initial_location_data) > 0:
    df_gt = pd.concat([initial_location_data, ground_truth_location_data], ignore_index=True)
    df_gt.sort_values(by='Timestamp_(ms)', inplace=True)
    df_gt.reset_index(drop=True, inplace=True)
else:
    print("Missing Ground Truth or Initial Location data")
    exit(1)

# Calculate Ground Truth heading
def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing (azimuth) between two points
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    delta_lon = lon2 - lon1
    x = atan2(
        sin(delta_lon) * cos(lat2),
        cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(delta_lon)
    )
    
    bearing = (degrees(x) + 360) % 360  # Normalize to 0-360 degrees
    return bearing

# Rename compass data columns for clarity
compass_data.rename(columns={'value_1': 'Magnetic_Field_Magnitude'}, inplace=True)
compass_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
compass_data.rename(columns={'value_3': 'compass'}, inplace=True)

# Get initial Ground Truth heading
first_ground_truth = initial_location_data['GroundTruth'].iloc[0] if len(initial_location_data) > 0 else 0

# Calculate Gyro heading from Ground Truth
compass_data['GyroStartByGroundTruth'] = first_ground_truth + compass_data['gyroSumFromstart0'] - compass_data['gyroSumFromstart0'].iloc[0]
compass_data['GyroStartByGroundTruth'] = (compass_data['GyroStartByGroundTruth'] + 360) % 360

print("Data preprocessing complete")

# Calculate positions
print("Calculating position trajectories...")

# Set step length and initial position
step_length = 0.66  # Step length in meters
initial_position = (0, 0)  # Initial position coordinates (x, y)

# Calculate Ground Truth positions
def calculate_gt_positions(df_gt, initial_position=(0, 0)):
    # Ensure Ground Truth data is sorted by step
    df_gt = df_gt.sort_values(by='step')
    
    # Extract position data
    gt_positions = []
    
    # If coordinates are in the data, use them directly
    if 'value_4' in df_gt.columns and 'value_5' in df_gt.columns:
        # Ensure coordinates are converted to numeric
        df_gt['value_4'] = pd.to_numeric(df_gt['value_4'], errors='coerce')
        df_gt['value_5'] = pd.to_numeric(df_gt['value_5'], errors='coerce')
        
        # Use the first Ground Truth point as the origin
        origin_x = df_gt['value_4'].iloc[0]
        origin_y = df_gt['value_5'].iloc[0]
        
        # Extract coordinates relative to origin
        for i in range(len(df_gt)):
            x = df_gt['value_4'].iloc[i] - origin_x
            y = df_gt['value_5'].iloc[i] - origin_y
            gt_positions.append((x, y, df_gt['step'].iloc[i]))
    
    return gt_positions

# Calculate positions based on step count and heading
def calculate_positions(data, heading_column, step_length=0.66, initial_position=(0, 0)):
    positions = [(*initial_position, data['step'].iloc[0])]  # Start from initial position with step count
    current_position = initial_position
    prev_step = data['step'].iloc[0]
    
    for i in range(1, len(data)):
        # Calculate step change
        change_in_step = data['step'].iloc[i] - prev_step
        
        # If steps changed, calculate new position
        if change_in_step != 0:
            # Calculate distance change
            change_in_distance = change_in_step * step_length
            
            # Get heading value (note: 0 degrees is North, 90 degrees is East)
            heading = data[heading_column].iloc[i]
            
            # Calculate new position (East is x-axis, North is y-axis)
            new_x = current_position[0] + change_in_distance * np.sin(np.radians(heading))
            new_y = current_position[1] + change_in_distance * np.cos(np.radians(heading))
            
            # Update current position
            current_position = (new_x, new_y)
            positions.append((*current_position, data['step'].iloc[i]))
            
            # Update previous step count
            prev_step = data['step'].iloc[i]
    
    return positions

# Get Ground Truth positions from the data points we have
gt_positions_with_steps = calculate_gt_positions(df_gt)
print(f"Number of Ground Truth coordinate points: {len(gt_positions_with_steps)}")

# Create a dataframe for the actual ground truth positions
df_gt_actual = pd.DataFrame(gt_positions_with_steps, columns=['GT_X', 'GT_Y', 'step'])

# Calculate positions using traditional methods
print("Calculating positions using traditional methods...")
positions_compass = calculate_positions(compass_data, 'compass', step_length, initial_position)
positions_gyro = calculate_positions(compass_data, 'GyroStartByGroundTruth', step_length, initial_position)
print(f"Number of Compass trajectory points: {len(positions_compass)}")
print(f"Number of Gyro trajectory points: {len(positions_gyro)}")

# Create dataframes for each method
df_compass_positions = pd.DataFrame(positions_compass, columns=['Compass_X', 'Compass_Y', 'step'])
df_gyro_positions = pd.DataFrame(positions_gyro, columns=['Gyro_X', 'Gyro_Y', 'step'])

# Get all unique steps from compass and gyro data
all_steps = sorted(df_compass_positions['step'].unique())

# Create interpolation functions for GT_X and GT_Y based on steps
print("Interpolating ground truth positions for all steps...")
if len(df_gt_actual) >= 2:  # Need at least 2 points for interpolation
    # Create interpolation functions
    interp_x = interp1d(df_gt_actual['step'], df_gt_actual['GT_X'], kind='linear', fill_value='extrapolate')
    interp_y = interp1d(df_gt_actual['step'], df_gt_actual['GT_Y'], kind='linear', fill_value='extrapolate')
    
    # Create a dataframe with interpolated GT positions for all steps
    gt_interpolated = pd.DataFrame({
        'step': all_steps,
        'GT_X': interp_x(all_steps),
        'GT_Y': interp_y(all_steps)
    })
    
    print(f"Interpolated ground truth positions for {len(gt_interpolated)} steps")
else:
    print("Not enough ground truth points for interpolation")
    gt_interpolated = pd.DataFrame(columns=['step', 'GT_X', 'GT_Y'])

# Merge dataframes on step
result_df = df_compass_positions.merge(df_gyro_positions, on='step', how='outer')
result_df = result_df.merge(gt_interpolated, on='step', how='left')

# Sort by step
result_df = result_df.sort_values(by='step')

# Add walked distance column
result_df['Walked_distance'] = result_df['step'] * step_length

# Calculate error metrics for points where we have ground truth data
print("Calculating position error metrics...")

# Initialize error columns
result_df['Gyro_Error_X'] = np.nan
result_df['Gyro_Error_Y'] = np.nan
result_df['Gyro_Distance_Error'] = np.nan
result_df['Compass_Error_X'] = np.nan
result_df['Compass_Error_Y'] = np.nan
result_df['Compass_Distance_Error'] = np.nan

# Calculate errors for all rows that have ground truth data
mask = result_df['GT_X'].notna() & result_df['GT_Y'].notna()
if mask.any():
    # Gyro errors
    result_df.loc[mask, 'Gyro_Error_X'] = np.abs(result_df.loc[mask, 'Gyro_X'] - result_df.loc[mask, 'GT_X'])
    result_df.loc[mask, 'Gyro_Error_Y'] = np.abs(result_df.loc[mask, 'Gyro_Y'] - result_df.loc[mask, 'GT_Y'])
    result_df.loc[mask, 'Gyro_Distance_Error'] = np.sqrt(
        (result_df.loc[mask, 'Gyro_X'] - result_df.loc[mask, 'GT_X'])**2 + 
        (result_df.loc[mask, 'Gyro_Y'] - result_df.loc[mask, 'GT_Y'])**2
    )
    
    # Compass errors
    result_df.loc[mask, 'Compass_Error_X'] = np.abs(result_df.loc[mask, 'Compass_X'] - result_df.loc[mask, 'GT_X'])
    result_df.loc[mask, 'Compass_Error_Y'] = np.abs(result_df.loc[mask, 'Compass_Y'] - result_df.loc[mask, 'GT_Y'])
    result_df.loc[mask, 'Compass_Distance_Error'] = np.sqrt(
        (result_df.loc[mask, 'Compass_X'] - result_df.loc[mask, 'GT_X'])**2 + 
        (result_df.loc[mask, 'Compass_Y'] - result_df.loc[mask, 'GT_Y'])**2
    )

# Save the results to a CSV file
output_file = os.path.join(output_dir, '1578_position_trajectories.csv')
result_df.to_csv(output_file, index=False)
print(f"Saved position trajectories to {output_file}")

# Print summary of error statistics at the end
if mask.any():
    print("\nError Statistics:")
    print(f"Gyro Distance Error - Mean: {result_df['Gyro_Distance_Error'].mean():.2f} m, Max: {result_df['Gyro_Distance_Error'].max():.2f} m")
    print(f"Compass Distance Error - Mean: {result_df['Compass_Distance_Error'].mean():.2f} m, Max: {result_df['Compass_Distance_Error'].max():.2f} m")
else:
    print("\nNo ground truth data available for error calculation")

# Include timestamp in the output file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Processing completed at {timestamp}")
print(f"All results saved to {output_dir}") 