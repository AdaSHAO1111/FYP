import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# Set output directory
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2'
os.makedirs(output_dir, exist_ok=True)

# Functions for heading calculation
def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate the bearing (azimuth) between two points"""
    lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    
    delta_lon = lon2 - lon1
    x = math.atan2(
        math.sin(delta_lon) * math.cos(lat2),
        math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    )
    
    bearing = (math.degrees(x) + 360) % 360
    return bearing

# Function for position calculation
def calculate_positions(data, heading_column, step_length=0.66, initial_position=(0, 0)):
    """Calculate positions using step detection and heading"""
    positions = [initial_position]
    current_position = initial_position
    prev_step = data['step'].iloc[0]
    
    for i in range(1, len(data)):
        # Calculate step change
        change_in_step = data['step'].iloc[i] - prev_step
        
        # If step changes, calculate new position
        if change_in_step != 0:
            # Calculate distance change
            change_in_distance = change_in_step * step_length
            
            # Get heading value (0° is North, 90° is East)
            heading = data[heading_column].iloc[i]
            
            # Calculate new position (East is x-axis, North is y-axis)
            new_x = current_position[0] + change_in_distance * np.sin(np.radians(heading))
            new_y = current_position[1] + change_in_distance * np.cos(np.radians(heading))
            
            # Update current position
            current_position = (new_x, new_y)
            positions.append(current_position)
            
            # Update previous step
            prev_step = data['step'].iloc[i]
    
    return positions

# Load and preprocess the data
print("Loading data...")
data_file = '/Users/shaoxinyi/Downloads/FYP2/Data_collected/1536_CompassGyroSumHeadingData.txt'
data = pd.read_csv(data_file, delimiter=';')

# Extract necessary data
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

# Prepare the sensor data
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

print("Data preprocessing completed")

# Extract ground truth positions
gt_positions = []
for i in range(len(df_gt)):
    # Use value_4 and value_5 but make origin (0,0)
    x = df_gt['value_4'].iloc[i] - df_gt['value_4'].iloc[0]
    y = df_gt['value_5'].iloc[i] - df_gt['value_5'].iloc[0]
    gt_positions.append((x, y))

print("Building and training a simpler fusion model...")

# Prepare the model inputs
gyro_input_cols = ['axisZAngle', 'gyroSumFromstart0', 'compass']
gyro_features = gyro_data[gyro_input_cols].values

compass_input_cols = ['Magnetic_Field_Magnitude', 'gyroSumFromstart0', 'compass']
compass_features = compass_data[compass_input_cols].values

# Align gyro and compass data by timestamp
gyro_timestamps = gyro_data['Timestamp_(ms)'].values
compass_timestamps = compass_data['Timestamp_(ms)'].values

# Create a DataFrame for each timestamp
timestamp_df = pd.DataFrame({'timestamp': np.sort(np.unique(np.concatenate([gyro_timestamps, compass_timestamps])))})

# For each timestamp, find the closest gyro and compass readings
timestamp_df['gyro_idx'] = timestamp_df['timestamp'].apply(
    lambda x: np.argmin(np.abs(gyro_timestamps - x)) if len(gyro_timestamps) > 0 else -1
)
timestamp_df['compass_idx'] = timestamp_df['timestamp'].apply(
    lambda x: np.argmin(np.abs(compass_timestamps - x)) if len(compass_timestamps) > 0 else -1
)

# Remove invalid indices
timestamp_df = timestamp_df[
    (timestamp_df['gyro_idx'] != -1) & 
    (timestamp_df['compass_idx'] != -1)
]

# Combine gyro and compass features
X = np.hstack([
    gyro_features[timestamp_df['gyro_idx'].values],
    compass_features[timestamp_df['compass_idx'].values]
])

# Get ground truth heading for each timestamp
y = gyro_data['GroundTruthHeadingComputed'].values[timestamp_df['gyro_idx'].values].reshape(-1, 1)

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split data into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# Build a simpler model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("Model summary:")
model.summary()

# Train the model
print("Training the model...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
]

start_time = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)
training_time = time.time() - start_time
print(f"Model training completed in {training_time:.2f} seconds")

# Save the model
model.save(os.path.join(output_dir, 'simple_fusion_model.keras'))
print(f"Model saved to {os.path.join(output_dir, 'simple_fusion_model.keras')}")

# Make predictions
print("Generating predictions...")
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Create a combined dataframe for evaluation
fusion_df = gyro_data.copy()
fusion_df = fusion_df.iloc[timestamp_df['gyro_idx'].values].reset_index(drop=True)
fusion_df['Fusion_Heading'] = y_pred.flatten()
fusion_df['Fusion_Heading'] = (fusion_df['Fusion_Heading'] + 360) % 360

# Calculate traditional positions
positions_trad = calculate_positions(fusion_df, 'GyroStartByGroundTruth')

# Calculate fusion positions
positions_fusion = calculate_positions(fusion_df, 'Fusion_Heading')

# Calculate heading errors
def calculate_heading_error(true_heading, pred_heading):
    diff = np.abs(true_heading - pred_heading)
    diff = np.minimum(diff, 360 - diff)  # Take smaller angle difference
    return np.mean(diff)

heading_error_trad = calculate_heading_error(
    fusion_df['GroundTruthHeadingComputed'].values, 
    fusion_df['GyroStartByGroundTruth'].values
)

heading_error_fusion = calculate_heading_error(
    fusion_df['GroundTruthHeadingComputed'].values, 
    fusion_df['Fusion_Heading'].values
)

# Calculate position errors
def calc_position_error(positions, gt_positions):
    min_len = min(len(positions), len(gt_positions))
    errors = []
    
    for i in range(min_len):
        error = np.sqrt((positions[i][0] - gt_positions[i][0])**2 + 
                       (positions[i][1] - gt_positions[i][1])**2)
        errors.append(error)
    
    return np.mean(errors) if errors else float('inf')

position_error_trad = calc_position_error(positions_trad, gt_positions)
position_error_fusion = calc_position_error(positions_fusion, gt_positions)

# Print results
print("\nResults:")
print("--------")
print(f"Heading Error - Traditional: {heading_error_trad:.2f}°")
print(f"Heading Error - Fusion:      {heading_error_fusion:.2f}°")
print(f"Position Error - Traditional: {position_error_trad:.2f}m")
print(f"Position Error - Fusion:      {position_error_fusion:.2f}m")

# Calculate improvement
heading_improvement = ((heading_error_trad - heading_error_fusion) / heading_error_trad) * 100
position_improvement = ((position_error_trad - position_error_fusion) / position_error_trad) * 100

print(f"Heading Improvement: {heading_improvement:.2f}%")
print(f"Position Improvement: {position_improvement:.2f}%")

# Save results to CSV
results_df = pd.DataFrame({
    'Method': ['Traditional', 'Fusion Model'],
    'Heading Error (°)': [heading_error_trad, heading_error_fusion],
    'Position Error (m)': [position_error_trad, position_error_fusion]
})

results_df.to_csv(os.path.join(output_dir, 'simple_fusion_results.csv'), index=False)
print(f"Results saved to {os.path.join(output_dir, 'simple_fusion_results.csv')}")

# Visualize predictions
print("Generating visualizations...")

# Plot position trajectories
plt.figure(figsize=(10, 8))

# Extract coordinates for plotting
x_gt = [pos[0] for pos in gt_positions]
y_gt = [pos[1] for pos in gt_positions]

x_trad = [pos[0] for pos in positions_trad]
y_trad = [pos[1] for pos in positions_trad]

x_fusion = [pos[0] for pos in positions_fusion]
y_fusion = [pos[1] for pos in positions_fusion]

# Plot trajectories
plt.plot(x_gt, y_gt, 'k-', linewidth=2, label='Ground Truth')
plt.plot(x_trad, y_trad, 'b--', linewidth=1, label=f'Traditional ({position_error_trad:.2f}m)')
plt.plot(x_fusion, y_fusion, 'r-', linewidth=1.5, label=f'Fusion ({position_error_fusion:.2f}m)')

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
position_plot_path = os.path.join(output_dir, 'simple_fusion_position_comparison.png')
plt.savefig(position_plot_path, dpi=300, bbox_inches='tight')
print(f"Position comparison plot saved to {position_plot_path}")

# Plot heading comparison
plt.figure(figsize=(12, 6))

# Plot only a subset of points for clarity (every 100th point)
step = 100
indices = range(0, len(fusion_df), step)

# Plot headings
plt.plot(indices, fusion_df['GroundTruthHeadingComputed'].iloc[indices], 'k-', label='Ground Truth', linewidth=1.5)
plt.plot(indices, fusion_df['GyroStartByGroundTruth'].iloc[indices], 'b--', label=f'Traditional ({heading_error_trad:.2f}°)', linewidth=1)
plt.plot(indices, fusion_df['Fusion_Heading'].iloc[indices], 'r-', label=f'Fusion ({heading_error_fusion:.2f}°)', linewidth=1)

plt.xlabel('Sample')
plt.ylabel('Heading (degrees)')
plt.title('Heading Prediction Comparison')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best')

# Save the plot
heading_plot_path = os.path.join(output_dir, 'simple_fusion_heading_comparison.png')
plt.savefig(heading_plot_path, dpi=300, bbox_inches='tight')
print(f"Heading comparison plot saved to {heading_plot_path}")

# Plot error comparison
plt.figure(figsize=(10, 5))

# Position error comparison
plt.subplot(1, 2, 1)
plt.bar(['Traditional', 'Fusion'], [position_error_trad, position_error_fusion], color=['blue', 'red'])
plt.ylabel('Average Error (m)')
plt.title('Position Error Comparison')
plt.grid(axis='y', linestyle=':', alpha=0.6)

# Heading error comparison
plt.subplot(1, 2, 2)
plt.bar(['Traditional', 'Fusion'], [heading_error_trad, heading_error_fusion], color=['blue', 'red'])
plt.ylabel('Average Error (degrees)')
plt.title('Heading Error Comparison')
plt.grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()

# Save the error comparison plot
error_plot_path = os.path.join(output_dir, 'simple_fusion_error_comparison.png')
plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
print(f"Error comparison plot saved to {error_plot_path}")

print("Simple fusion model evaluation completed!") 