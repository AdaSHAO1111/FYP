import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GRU, Bidirectional, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set output directory
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2'
os.makedirs(output_dir, exist_ok=True)

# Function to calculate bearing between two points
def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate the bearing (azimuth) between two points"""
    import math
    lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    
    delta_lon = lon2 - lon1
    x = math.atan2(
        math.sin(delta_lon) * math.cos(lat2),
        math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    )
    
    bearing = (math.degrees(x) + 360) % 360
    return bearing

# Function to create sequences for model input
def create_sequences(X, window_size):
    """Create input sequences"""
    X_seq = []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
    return np.array(X_seq)

# Function to calculate positions from headings
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

# Implement direct neural network model
print("Implementing fusion neural network model...")

# Window size for sequence prediction
window_size = 15  # Reduced from 20 to ensure we have sufficient data

# Prepare features
gyro_features = gyro_data[['axisZAngle', 'gyroSumFromstart0', 'compass']].values
compass_features = compass_data[['Magnetic_Field_Magnitude', 'gyroSumFromstart0', 'compass']].values

# Prepare heading target
heading_target = gyro_data['GroundTruthHeadingComputed'].values.reshape(-1, 1)

# Scale features and targets
gyro_scaler = MinMaxScaler()
compass_scaler = MinMaxScaler()
heading_scaler = MinMaxScaler()

gyro_features_scaled = gyro_scaler.fit_transform(gyro_features)
compass_features_scaled = compass_scaler.fit_transform(compass_features)
heading_target_scaled = heading_scaler.fit_transform(heading_target)

# Create sequences
gyro_X_seq = create_sequences(gyro_features_scaled, window_size)
compass_X_seq = create_sequences(compass_features_scaled, window_size)
heading_y_seq = heading_target_scaled[window_size:]

# Split data
indices = np.arange(len(gyro_X_seq))
train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=False)

# Build fusion model for heading prediction
def build_fusion_model(gyro_shape, compass_shape):
    # Gyro input branch
    gyro_input = Input(shape=gyro_shape, name='gyro_input')
    gyro_gru1 = Bidirectional(GRU(64, return_sequences=True))(gyro_input)
    gyro_drop1 = Dropout(0.3)(gyro_gru1)
    gyro_gru2 = Bidirectional(GRU(32, return_sequences=False))(gyro_drop1)
    gyro_drop2 = Dropout(0.3)(gyro_gru2)
    
    # Compass input branch
    compass_input = Input(shape=compass_shape, name='compass_input')
    compass_gru1 = Bidirectional(GRU(64, return_sequences=True))(compass_input)
    compass_drop1 = Dropout(0.3)(compass_gru1)
    compass_gru2 = Bidirectional(GRU(32, return_sequences=False))(compass_drop1)
    compass_drop2 = Dropout(0.3)(compass_gru2)
    
    # Combine branches
    merged = Concatenate()([gyro_drop2, compass_drop2])
    
    # Output layers
    dense1 = Dense(64, activation='relu')(merged)
    drop3 = Dropout(0.2)(dense1)
    dense2 = Dense(32, activation='relu')(drop3)
    heading_output = Dense(1, name='heading_output')(dense2)
    
    # Build model
    model = Model(
        inputs=[gyro_input, compass_input],
        outputs=heading_output,
        name='fusion_heading_model'
    )
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Train fusion model
print("Training fusion model...")
fusion_model = build_fusion_model(gyro_X_seq.shape[1:], compass_X_seq.shape[1:])

# Callbacks for training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001, verbose=1)
]

# Train the model
training_start_time = time.time()
fusion_history = fusion_model.fit(
    [gyro_X_seq[train_idx], compass_X_seq[train_idx]],
    heading_y_seq[train_idx],
    validation_data=([gyro_X_seq[val_idx], compass_X_seq[val_idx]], heading_y_seq[val_idx]),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)
training_time = time.time() - training_start_time
print(f"Model training completed in {training_time:.2f} seconds")

# Save model
fusion_model.save(os.path.join(output_dir, 'fusion_heading_model.keras'))
print("Model saved to output directory")

# Generate predictions
print("Generating predictions...")
heading_pred_scaled = fusion_model.predict([gyro_X_seq, compass_X_seq])
heading_pred = heading_scaler.inverse_transform(heading_pred_scaled)

# Create full predictions array with same length as original data
fusion_heading = np.zeros(len(gyro_data))
fusion_heading[window_size:window_size+len(heading_pred)] = heading_pred.flatten()

# Handle initialization (first window_size elements)
if len(heading_pred) > 0:
    fusion_heading[:window_size] = heading_pred[0]

# Normalize headings to 0-360 degrees
fusion_heading = (fusion_heading + 360) % 360

# Add predictions to data
gyro_data['Fusion_Heading'] = fusion_heading

# Calculate position trajectories
print("Calculating position trajectories...")

# Traditional positions
positions_trad = calculate_positions(gyro_data, 'GyroStartByGroundTruth')

# Fusion model positions
positions_fusion = calculate_positions(gyro_data, 'Fusion_Heading')

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
error_fusion = calc_position_error(positions_fusion, gt_positions)

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
heading_error_fusion = calculate_heading_error(gyro_data['GroundTruthHeadingComputed'], gyro_data['Fusion_Heading'])

# Print results
print("\nResults Summary:")
print("----------------")
print("\nHeading Error (degrees):")
print(f"Traditional: {heading_error_trad:.2f}°")
print(f"Fusion:      {heading_error_fusion:.2f}°")

print("\nPosition Error (meters):")
print(f"Traditional: {error_trad:.2f}m")
print(f"Fusion:      {error_fusion:.2f}m")

# Calculate improvement percentages
heading_improvement = ((heading_error_trad - heading_error_fusion) / heading_error_trad) * 100
position_improvement = ((error_trad - error_fusion) / error_trad) * 100

print(f"\nImprovement over traditional methods:")
print(f"Heading: {heading_improvement:.2f}%")
print(f"Position: {position_improvement:.2f}%")

# Create a DataFrame with results
results_df = pd.DataFrame({
    'Method': ['Traditional', 'Fusion'],
    'Position Error (m)': [error_trad, error_fusion],
    'Heading Error (°)': [heading_error_trad, heading_error_fusion]
})

# Save results to CSV
results_csv_path = os.path.join(output_dir, 'fusion_model_results.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"\nResults saved to {results_csv_path}")

# Visualize position trajectories
print("\nGenerating visualizations...")

# Plot position trajectories
plt.figure(figsize=(10, 10))

# Extract coordinates for plotting
x_gt = [pos[0] for pos in gt_positions]
y_gt = [pos[1] for pos in gt_positions]

x_trad = [pos[0] for pos in positions_trad]
y_trad = [pos[1] for pos in positions_trad]

x_fusion = [pos[0] for pos in positions_fusion]
y_fusion = [pos[1] for pos in positions_fusion]

# Plot trajectories
plt.plot(x_gt, y_gt, 'k-', linewidth=2, label='Ground Truth')
plt.plot(x_trad, y_trad, 'b--', linewidth=1, label=f'Traditional ({error_trad:.2f}m)')
plt.plot(x_fusion, y_fusion, 'r-', linewidth=1.5, label=f'Fusion ({error_fusion:.2f}m)')

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
position_plot_path = os.path.join(output_dir, 'fusion_position_comparison.png')
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
plt.plot(time_s, gyro_data['Fusion_Heading'], 'r-', label=f'Fusion ({heading_error_fusion:.2f}°)', linewidth=1)

plt.xlabel('Time (seconds)')
plt.ylabel('Heading (degrees)')
plt.title('Heading Prediction Comparison')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best')

# Save the plot
heading_plot_path = os.path.join(output_dir, 'fusion_heading_comparison.png')
plt.savefig(heading_plot_path, dpi=300, bbox_inches='tight')
print(f"Heading comparison plot saved to {heading_plot_path}")

# Create a bar chart for error comparison
plt.figure(figsize=(12, 5))

# Position error comparison
plt.subplot(1, 2, 1)
plt.bar(['Traditional', 'Fusion'], [error_trad, error_fusion], color=['blue', 'red'])
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
error_plot_path = os.path.join(output_dir, 'fusion_error_comparison.png')
plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
print(f"Error comparison plot saved to {error_plot_path}")

print("\nFusion model evaluation completed!") 