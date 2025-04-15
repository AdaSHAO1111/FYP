import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from advanced_heading_position_model import AdvancedHeadingModel, calculate_positions, calculate_bearing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set output directory
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2'
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading data...")
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
data["GroundTruthHeadingComputed"] = data["GroundTruthHeadingComputed"].bfill()  # Use bfill instead of fillna(method="bfill")

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

# Fill NaN values to avoid errors in metrics calculation
gyro_data = gyro_data.fillna(0)
compass_data = compass_data.fillna(0)

print("Data preprocessing completed")

# Extract ground truth positions for position-aware training
gt_positions = []
for i in range(len(df_gt)):
    # Use value_4 (longitude) and value_5 (latitude) for position
    gt_positions.append((df_gt['value_4'].iloc[i], df_gt['value_5'].iloc[i]))

# Initialize model
print("Initializing advanced heading model with position awareness...")
model = AdvancedHeadingModel(window_size=20, use_attention=True, use_position_loss=True)

# Train model
print("Training advanced model...")
train_start_time = time.time()
train_history = model.train(
    gyro_data=gyro_data,
    compass_data=compass_data,
    ground_truth_heading=data['GroundTruthHeadingComputed'],
    ground_truth_positions=gt_positions,
    epochs=100
)
train_time = time.time() - train_start_time
print(f"Model training completed in {train_time:.2f} seconds")

# Save models
model.save_models(output_dir)

# Make predictions
print("Generating heading predictions...")
predictions = model.predict_headings(gyro_data, compass_data)

# Add predictions to dataframes
gyro_data['Advanced_Gyro_Heading'] = predictions['gyro']
compass_data['Advanced_Compass_Heading'] = predictions['compass']
if predictions['fusion'] is not None:
    gyro_data['Advanced_Fusion_Heading'] = predictions['fusion']
    compass_data['Advanced_Fusion_Heading'] = predictions['fusion'][:len(compass_data)] if len(predictions['fusion']) > len(compass_data) else np.pad(predictions['fusion'], (0, len(compass_data) - len(predictions['fusion'])), 'constant')

# Safe function to calculate MAE that handles NaN values
def safe_mae(y_true, y_pred):
    # Create a mask for non-NaN values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask) == 0:
        return float('inf')  # Return infinity if no valid pairs
    return mean_absolute_error(y_true[mask], y_pred[mask])

# Calculate traditional heading errors
gyro_trad_mae = safe_mae(gyro_data['GroundTruthHeadingComputed'], gyro_data['GyroStartByGroundTruth'])
compass_trad_mae = safe_mae(compass_data['GroundTruthHeadingComputed'], compass_data['GyroStartByGroundTruth'])

# Calculate advanced model heading errors
gyro_adv_mae = safe_mae(gyro_data['GroundTruthHeadingComputed'], gyro_data['Advanced_Gyro_Heading'])
compass_adv_mae = safe_mae(compass_data['GroundTruthHeadingComputed'], compass_data['Advanced_Compass_Heading'])

# Calculate fusion model heading errors if available
if 'Advanced_Fusion_Heading' in gyro_data.columns:
    fusion_gyro_mae = safe_mae(gyro_data['GroundTruthHeadingComputed'], gyro_data['Advanced_Fusion_Heading'])
    fusion_compass_mae = safe_mae(compass_data['GroundTruthHeadingComputed'], compass_data['Advanced_Fusion_Heading'])
else:
    fusion_gyro_mae = None
    fusion_compass_mae = None

# Print heading error comparison
print("\nHeading Error Comparison (MAE in degrees):")
print(f"Gyro Traditional: {gyro_trad_mae:.2f}°")
print(f"Gyro Advanced: {gyro_adv_mae:.2f}°")
if fusion_gyro_mae:
    print(f"Gyro Fusion: {fusion_gyro_mae:.2f}°")
print(f"Compass Traditional: {compass_trad_mae:.2f}°")
print(f"Compass Advanced: {compass_adv_mae:.2f}°")
if fusion_compass_mae:
    print(f"Compass Fusion: {fusion_compass_mae:.2f}°")

# Calculate improvement percentages
gyro_improvement = ((gyro_trad_mae - gyro_adv_mae) / gyro_trad_mae) * 100
compass_improvement = ((compass_trad_mae - compass_adv_mae) / compass_trad_mae) * 100

print(f"\nImprovement over traditional methods:")
print(f"Gyro Heading: {gyro_improvement:.2f}%")
print(f"Compass Heading: {compass_improvement:.2f}%")

# Calculate positions using heading predictions
print("\nCalculating positions using different heading methods...")

# Get Ground Truth positions
gt_positions = []
for i in range(len(df_gt)):
    # Use value_4 and value_5 but make origin (0,0)
    x = df_gt['value_4'].iloc[i] - df_gt['value_4'].iloc[0]
    y = df_gt['value_5'].iloc[i] - df_gt['value_5'].iloc[0]
    gt_positions.append((x, y))

# Traditional positions
positions_gyro_trad = calculate_positions(gyro_data, 'GyroStartByGroundTruth')
positions_compass_trad = calculate_positions(compass_data, 'GyroStartByGroundTruth')

# Advanced model positions
positions_gyro_adv = calculate_positions(gyro_data, 'Advanced_Gyro_Heading')
positions_compass_adv = calculate_positions(compass_data, 'Advanced_Compass_Heading')

# Fusion model positions if available
if 'Advanced_Fusion_Heading' in gyro_data.columns:
    positions_fusion = calculate_positions(gyro_data, 'Advanced_Fusion_Heading')
else:
    positions_fusion = None

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
gyro_trad_pos_error = calc_position_error(positions_gyro_trad, gt_positions)
compass_trad_pos_error = calc_position_error(positions_compass_trad, gt_positions)
gyro_adv_pos_error = calc_position_error(positions_gyro_adv, gt_positions)
compass_adv_pos_error = calc_position_error(positions_compass_adv, gt_positions)
fusion_pos_error = calc_position_error(positions_fusion, gt_positions) if positions_fusion else None

# Print position error comparison
print("\nPosition Error Comparison (Average error in meters):")
print(f"Gyro Traditional: {gyro_trad_pos_error:.2f}m")
print(f"Gyro Advanced: {gyro_adv_pos_error:.2f}m")
print(f"Compass Traditional: {compass_trad_pos_error:.2f}m")
print(f"Compass Advanced: {compass_adv_pos_error:.2f}m")
if fusion_pos_error:
    print(f"Fusion: {fusion_pos_error:.2f}m")

# Calculate position improvement percentages
gyro_pos_improvement = ((gyro_trad_pos_error - gyro_adv_pos_error) / gyro_trad_pos_error) * 100
compass_pos_improvement = ((compass_trad_pos_error - compass_adv_pos_error) / compass_trad_pos_error) * 100

print(f"\nPosition improvement over traditional methods:")
print(f"Gyro: {gyro_pos_improvement:.2f}%")
print(f"Compass: {compass_pos_improvement:.2f}%")

# Create a DataFrame with all results for easy comparison
results_df = pd.DataFrame({
    'Method': ['Gyro Traditional', 'Gyro Advanced', 'Compass Traditional', 'Compass Advanced'],
    'Heading MAE': [gyro_trad_mae, gyro_adv_mae, compass_trad_mae, compass_adv_mae],
    'Position Error': [gyro_trad_pos_error, gyro_adv_pos_error, compass_trad_pos_error, compass_adv_pos_error]
})

if fusion_gyro_mae and fusion_pos_error:
    fusion_row = pd.DataFrame({
        'Method': ['Fusion'],
        'Heading MAE': [fusion_gyro_mae],
        'Position Error': [fusion_pos_error]
    })
    results_df = pd.concat([results_df, fusion_row], ignore_index=True)

# Save results to CSV
results_df.to_csv(os.path.join(output_dir, 'advanced_model_results.csv'), index=False)
print(f"\nResults saved to {os.path.join(output_dir, 'advanced_model_results.csv')}")

# Plot heading comparison
print("\nGenerating heading comparison plots...")
plt.figure(figsize=(10, 6))
plt.plot(gyro_data['Timestamp_(ms)'], gyro_data['GroundTruthHeadingComputed'], 'k-', label='Ground Truth', linewidth=1.5)
plt.plot(gyro_data['Timestamp_(ms)'], gyro_data['GyroStartByGroundTruth'], 'b--', label='Traditional', linewidth=1)
plt.plot(gyro_data['Timestamp_(ms)'], gyro_data['Advanced_Gyro_Heading'], 'r-', label='Advanced', linewidth=1)
if 'Advanced_Fusion_Heading' in gyro_data.columns:
    plt.plot(gyro_data['Timestamp_(ms)'], gyro_data['Advanced_Fusion_Heading'], 'g-.', label='Fusion', linewidth=1)

plt.xlabel('Time (ms)')
plt.ylabel('Heading (degrees)')
plt.title('Gyro Heading Comparison')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.savefig(os.path.join(output_dir, 'advanced_gyro_heading_comparison.png'), bbox_inches='tight')

# Plot position comparison
print("\nGenerating position trajectory plots...")
plt.figure(figsize=(8, 8))

# Extract coordinates
x_gt = [pos[0] for pos in gt_positions]
y_gt = [pos[1] for pos in gt_positions]

x_gyro_trad = [pos[0] for pos in positions_gyro_trad]
y_gyro_trad = [pos[1] for pos in positions_gyro_trad]

x_compass_trad = [pos[0] for pos in positions_compass_trad]
y_compass_trad = [pos[1] for pos in positions_compass_trad]

x_gyro_adv = [pos[0] for pos in positions_gyro_adv]
y_gyro_adv = [pos[1] for pos in positions_gyro_adv]

x_compass_adv = [pos[0] for pos in positions_compass_adv]
y_compass_adv = [pos[1] for pos in positions_compass_adv]

if positions_fusion:
    x_fusion = [pos[0] for pos in positions_fusion]
    y_fusion = [pos[1] for pos in positions_fusion]

# Plot trajectories
plt.plot(x_gt, y_gt, 'k-', label='Ground Truth', linewidth=2)
plt.plot(x_gyro_trad, y_gyro_trad, 'b--', label='Gyro Traditional', linewidth=1)
plt.plot(x_gyro_adv, y_gyro_adv, 'r-.', label='Gyro Advanced', linewidth=1)
plt.plot(x_compass_trad, y_compass_trad, 'g--', label='Compass Traditional', linewidth=1)
plt.plot(x_compass_adv, y_compass_adv, 'm-.', label='Compass Advanced', linewidth=1)

if positions_fusion:
    plt.plot(x_fusion, y_fusion, 'c-', label='Fusion', linewidth=1.5)

# Mark start and end points
plt.scatter(x_gt[0], y_gt[0], color='k', marker='o', s=100, label='Start')
plt.scatter(x_gt[-1], y_gt[-1], color='k', marker='x', s=100, label='End')

plt.grid(True, linestyle=':', alpha=0.5)
plt.xlabel('East (m)')
plt.ylabel('North (m)')
plt.title('Position Trajectory Comparison')
plt.legend()
plt.axis('equal')
plt.savefig(os.path.join(output_dir, 'advanced_position_comparison.png'), bbox_inches='tight')

# Create a bar chart to compare heading and position errors
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
methods = results_df['Method']
heading_errors = results_df['Heading MAE']
plt.bar(methods, heading_errors, color=['blue', 'red', 'green', 'magenta', 'cyan'][:len(methods)])
plt.title('Heading Error Comparison')
plt.ylabel('MAE (degrees)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle=':', alpha=0.5)

plt.subplot(1, 2, 2)
position_errors = results_df['Position Error']
plt.bar(methods, position_errors, color=['blue', 'red', 'green', 'magenta', 'cyan'][:len(methods)])
plt.title('Position Error Comparison')
plt.ylabel('Average Error (m)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'advanced_error_comparison.png'), bbox_inches='tight')

print("\nAdvanced model training and evaluation completed!") 