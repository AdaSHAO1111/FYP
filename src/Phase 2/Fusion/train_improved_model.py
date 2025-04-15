import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from advanced_model_v2 import ImprovedPositionModel, calculate_positions_from_heading, interpolate_positions, calculate_bearing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set output directory
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2'
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess data
print("Loading and preprocessing data...")

# Load main data
data_file = '/Users/shaoxinyi/Downloads/FYP2/Data_collected/1536_CompassGyroSumHeadingData.txt'
data = pd.read_csv(data_file, delimiter=';')

# Extract Ground Truth location data
ground_truth_location_data = data[data['Type'] == 'Ground_truth_Location'].copy()
initial_location_data = data[data['Type'] == 'Initial_Location'].copy()

# Create a separate dataframe for Ground Truth heading
if len(ground_truth_location_data) > 0 and len(initial_location_data) > 0:
    df_gt = pd.concat([initial_location_data, ground_truth_location_data], ignore_index=True)
    df_gt.sort_values(by='Timestamp_(ms)', inplace=True)
    df_gt.reset_index(drop=True, inplace=True)

# Calculate Ground Truth heading
print("Calculating Ground Truth heading...")
df_gt["GroundTruthHeadingComputed"] = np.nan

# Calculate the heading between consecutive points
for i in range(1, len(df_gt)):
    df_gt.loc[i, "GroundTruthHeadingComputed"] = calculate_bearing(
        df_gt.loc[i-1, "value_5"], df_gt.loc[i-1, "value_4"],
        df_gt.loc[i, "value_5"], df_gt.loc[i, "value_4"]
    )

# Fill first entry with the second entry's heading
if len(df_gt) > 1:
    df_gt.loc[0, "GroundTruthHeadingComputed"] = df_gt.loc[1, "GroundTruthHeadingComputed"]

# Sort and preprocess data
data.sort_values(by="Timestamp_(ms)", inplace=True)
df_gt.sort_values(by="Timestamp_(ms)", inplace=True)

# Merge Ground Truth heading
data = data.merge(df_gt[["Timestamp_(ms)", "GroundTruthHeadingComputed"]], on="Timestamp_(ms)", how="left")
data["GroundTruthHeadingComputed"] = data["GroundTruthHeadingComputed"].bfill()

# Convert numeric columns to float
for col in ['value_1', 'value_2', 'value_3', 'GroundTruthHeadingComputed']:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Separate data for gyro and compass
gyro_data = data[data['Type'] == 'Gyro'].reset_index(drop=True)
compass_data = data[data['Type'] == 'Compass'].reset_index(drop=True)

# Rename columns
compass_data.rename(columns={'value_1': 'Magnetic_Field_Magnitude'}, inplace=True)
compass_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
compass_data.rename(columns={'value_3': 'compass'}, inplace=True)

gyro_data.rename(columns={'value_1': 'axisZAngle'}, inplace=True)
gyro_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
gyro_data.rename(columns={'value_3': 'compass'}, inplace=True)

# Calculate traditional heading
first_ground_truth = initial_location_data['GroundTruth'].iloc[0] if len(initial_location_data) > 0 else 0

# Traditional heading for compass data
compass_data['GyroStartByGroundTruth'] = first_ground_truth + compass_data['gyroSumFromstart0'] - compass_data['gyroSumFromstart0'].iloc[0]
compass_data['GyroStartByGroundTruth'] = (compass_data['GyroStartByGroundTruth'] + 360) % 360

# Traditional heading for gyro data
gyro_data['GyroStartByGroundTruth'] = first_ground_truth + gyro_data['gyroSumFromstart0'] - gyro_data['gyroSumFromstart0'].iloc[0]
gyro_data['GyroStartByGroundTruth'] = (gyro_data['GyroStartByGroundTruth'] + 360) % 360

# Fill NaN values
gyro_data = gyro_data.fillna(0)
compass_data = compass_data.fillna(0)

# Create step detection data
step_data = pd.DataFrame()
step_data['Timestamp_(ms)'] = compass_data['Timestamp_(ms)']
step_data['step'] = compass_data['step']
step_data['step_diff'] = step_data['step'].diff().fillna(0)  # Step changes

print("Data preprocessing completed")

# Extract ground truth positions from df_gt
print("Preparing ground truth positions...")
gt_positions = []

# Convert to positions relative to the starting point
start_x = df_gt['value_4'].iloc[0]
start_y = df_gt['value_5'].iloc[0]

for i in range(len(df_gt)):
    # Use value_4 (longitude) and value_5 (latitude) for position, shifted to origin
    x = df_gt['value_4'].iloc[i] - start_x
    y = df_gt['value_5'].iloc[i] - start_y
    gt_positions.append((x, y))

# Initialize and train improved position model
print("Initializing and training position-aware model...")
model = ImprovedPositionModel(window_size=20, position_weight=0.7)

# Train the model
training_start_time = time.time()
training_history = model.train(
    gyro_data=gyro_data,
    compass_data=compass_data,
    step_data=step_data,
    ground_truth_heading=gyro_data['GroundTruthHeadingComputed'],
    ground_truth_positions=gt_positions,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)
training_time = time.time() - training_start_time
print(f"Model training completed in {training_time:.2f} seconds")

# Save models
model.save_models(output_dir)
print("Models saved to output directory")

# Generate predictions
print("Generating predictions...")
predictions = model.predict(gyro_data, compass_data, step_data)

# Add heading predictions to data
gyro_data['Improved_Heading'] = predictions['heading']
gyro_data['Integrated_Heading'] = predictions['integrated_heading']

# Calculate position trajectories
print("Calculating position trajectories...")

# Traditional positions
positions_trad = calculate_positions_from_heading(gyro_data, 'GyroStartByGroundTruth')

# Heading-based positions from improved model
positions_improved_heading = calculate_positions_from_heading(gyro_data, 'Improved_Heading')
positions_integrated_heading = calculate_positions_from_heading(gyro_data, 'Integrated_Heading')

# Direct position predictions
positions_direct = interpolate_positions(gyro_data, predictions['position_predictions_full'])
positions_integrated = interpolate_positions(gyro_data, predictions['integrated_position'])

# Calculate position errors
def calc_position_error(positions, gt_positions):
    """Calculate average position error"""
    min_len = min(len(positions), len(gt_positions))
    errors = []
    
    for i in range(min_len):
        error = np.sqrt((positions[i][0] - gt_positions[i][0])**2 + 
                        (positions[i][1] - gt_positions[i][1])**2)
        errors.append(error)
    
    return np.mean(errors) if errors else float('inf')

# Calculate errors
error_trad = calc_position_error(positions_trad, gt_positions)
error_improved_heading = calc_position_error(positions_improved_heading, gt_positions)
error_integrated_heading = calc_position_error(positions_integrated_heading, gt_positions)
error_direct = calc_position_error(positions_direct, gt_positions)
error_integrated = calc_position_error(positions_integrated, gt_positions)

# Calculate heading errors
def calculate_heading_error(true_heading, pred_heading):
    # Convert to numpy arrays
    true_heading = np.array(true_heading)
    pred_heading = np.array(pred_heading)
    
    # Calculate absolute angular difference
    diff = np.abs(true_heading - pred_heading)
    
    # Take the smaller angle (handle 0/360 wrap-around)
    diff = np.minimum(diff, 360 - diff)
    
    # Return mean error
    return np.mean(diff)

# Calculate heading errors
heading_error_trad = calculate_heading_error(gyro_data['GroundTruthHeadingComputed'], gyro_data['GyroStartByGroundTruth'])
heading_error_improved = calculate_heading_error(gyro_data['GroundTruthHeadingComputed'], gyro_data['Improved_Heading'])
heading_error_integrated = calculate_heading_error(gyro_data['GroundTruthHeadingComputed'], gyro_data['Integrated_Heading'])

# Print results
print("\nResults Summary:")
print("----------------")
print("\nHeading Error (degrees):")
print(f"Traditional:  {heading_error_trad:.2f}°")
print(f"Improved:     {heading_error_improved:.2f}°")
print(f"Integrated:   {heading_error_integrated:.2f}°")

print("\nPosition Error (meters):")
print(f"Traditional:        {error_trad:.2f}m")
print(f"Improved Heading:   {error_improved_heading:.2f}m")
print(f"Integrated Heading: {error_integrated_heading:.2f}m")
print(f"Direct Position:    {error_direct:.2f}m")
print(f"Integrated Model:   {error_integrated:.2f}m")

# Calculate improvement percentages
heading_improvement = ((heading_error_trad - heading_error_improved) / heading_error_trad) * 100
position_improvement = ((error_trad - min(error_improved_heading, error_integrated_heading, error_direct, error_integrated)) / error_trad) * 100

print(f"\nImprovement over traditional methods:")
print(f"Heading: {heading_improvement:.2f}%")
print(f"Position: {position_improvement:.2f}%")

# Create a DataFrame with results
results_df = pd.DataFrame({
    'Method': ['Traditional', 'Improved Heading', 'Integrated Heading', 'Direct Position', 'Integrated Model'],
    'Position Error (m)': [error_trad, error_improved_heading, error_integrated_heading, error_direct, error_integrated],
    'Heading Error': [heading_error_trad, heading_error_improved, heading_error_integrated, None, None]
})

# Save results to CSV
results_csv_path = os.path.join(output_dir, 'improved_model_results.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"\nResults saved to {results_csv_path}")

# Visualize position trajectories
print("\nGenerating visualizations...")

# Plot position trajectories
plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)

# Extract coordinates for plotting
x_gt = [pos[0] for pos in gt_positions]
y_gt = [pos[1] for pos in gt_positions]

x_trad = [pos[0] for pos in positions_trad]
y_trad = [pos[1] for pos in positions_trad]

x_improved = [pos[0] for pos in positions_improved_heading]
y_improved = [pos[1] for pos in positions_improved_heading]

x_integrated = [pos[0] for pos in positions_integrated_heading]
y_integrated = [pos[1] for pos in positions_integrated_heading]

x_direct = [pos[0] for pos in positions_direct]
y_direct = [pos[1] for pos in positions_direct]

# Plot trajectories
plt.plot(x_gt, y_gt, 'k-', linewidth=2, label='Ground Truth')
plt.plot(x_trad, y_trad, 'b--', linewidth=1, label=f'Traditional ({error_trad:.2f}m)')
plt.plot(x_improved, y_improved, 'r-.', linewidth=1, label=f'Improved Heading ({error_improved_heading:.2f}m)')
plt.plot(x_integrated, y_integrated, 'g-.', linewidth=1, label=f'Integrated Heading ({error_integrated_heading:.2f}m)')
plt.plot(x_direct, y_direct, 'm:', linewidth=1.5, label=f'Direct Position ({error_direct:.2f}m)')

# Mark start and end points
plt.scatter(x_gt[0], y_gt[0], color='black', marker='o', s=100, label='Start')
plt.scatter(x_gt[-1], y_gt[-1], color='black', marker='x', s=100, label='End')

# Add labels and grid
plt.xlabel('East (m)')
plt.ylabel('North (m)')
plt.title('Position Trajectory Comparison')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best')
plt.axis('equal')

# Save the plot
position_plot_path = os.path.join(output_dir, 'improved_position_comparison.png')
plt.savefig(position_plot_path, dpi=300, bbox_inches='tight')
print(f"Position comparison plot saved to {position_plot_path}")

# Plot heading comparison
plt.figure(figsize=(12, 6))

# Prepare time axis (use relative seconds for better readability)
time_ms = gyro_data['Timestamp_(ms)']
start_time = time_ms.min()
time_s = (time_ms - start_time) / 1000  # Convert to seconds

# Plot headings
plt.plot(time_s, gyro_data['GroundTruthHeadingComputed'], 'k-', label='Ground Truth', linewidth=1.5)
plt.plot(time_s, gyro_data['GyroStartByGroundTruth'], 'b--', label=f'Traditional ({heading_error_trad:.2f}°)', linewidth=1)
plt.plot(time_s, gyro_data['Improved_Heading'], 'r-.', label=f'Improved ({heading_error_improved:.2f}°)', linewidth=1)
plt.plot(time_s, gyro_data['Integrated_Heading'], 'g:', label=f'Integrated ({heading_error_integrated:.2f}°)', linewidth=1)

plt.xlabel('Time (seconds)')
plt.ylabel('Heading (degrees)')
plt.title('Heading Prediction Comparison')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best')

# Save the plot
heading_plot_path = os.path.join(output_dir, 'improved_heading_comparison.png')
plt.savefig(heading_plot_path, dpi=300, bbox_inches='tight')
print(f"Heading comparison plot saved to {heading_plot_path}")

# Create a bar chart for error comparison
plt.figure(figsize=(12, 6))

# Position error comparison
plt.subplot(1, 2, 1)
methods = results_df['Method']
position_errors = results_df['Position Error (m)']
plt.bar(methods, position_errors, color=['blue', 'red', 'green', 'magenta', 'cyan'])
plt.ylabel('Average Error (m)')
plt.title('Position Error Comparison')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle=':', alpha=0.6)

# Heading error comparison
plt.subplot(1, 2, 2)
heading_methods = ['Traditional', 'Improved', 'Integrated']
heading_errors = [heading_error_trad, heading_error_improved, heading_error_integrated]
plt.bar(heading_methods, heading_errors, color=['blue', 'red', 'green'])
plt.ylabel('Average Error (degrees)')
plt.title('Heading Error Comparison')
plt.grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()

# Save the error comparison plot
error_plot_path = os.path.join(output_dir, 'improved_error_comparison.png')
plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
print(f"Error comparison plot saved to {error_plot_path}")

print("\nImproved position model evaluation completed!") 