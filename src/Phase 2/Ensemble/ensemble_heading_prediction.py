import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# Set output directory
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2'
os.makedirs(output_dir, exist_ok=True)

# Function to create sequences for LSTM input
def create_sequences(X, window_size):
    X_seq = []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
    return np.array(X_seq)

# Function to calculate bearing between two points
def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing (azimuth) between two points
    """
    from math import atan2, degrees, radians, sin, cos
    
    lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    
    delta_lon = lon2 - lon1
    x = atan2(
        sin(delta_lon) * cos(lat2),
        cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(delta_lon)
    )
    
    bearing = (degrees(x) + 360) % 360  # Normalize to 0-360 degrees
    return bearing

# Load the data
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
data["GroundTruthHeadingComputed"] = data["GroundTruthHeadingComputed"].fillna(method="bfill")

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

print("Data preprocessing completed")

# Load trained LSTM models
try:
    print("Loading trained LSTM models...")
    gyro_model = load_model(os.path.join(output_dir, 'gyro_heading_lstm_model.keras'))
    compass_model = load_model(os.path.join(output_dir, 'compass_heading_lstm_model.keras'))
    print("LSTM models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Prepare features for LSTM prediction
window_size = 20  # Same as used in training

# For gyro data
gyro_features = gyro_data[['axisZAngle', 'gyroSumFromstart0']].values
gyro_scaler_X = MinMaxScaler()
gyro_features_scaled = gyro_scaler_X.fit_transform(gyro_features)
gyro_X_seq = create_sequences(gyro_features_scaled, window_size)

# For compass data
compass_features = compass_data[['Magnetic_Field_Magnitude', 'gyroSumFromstart0']].values
compass_scaler_X = MinMaxScaler()
compass_features_scaled = compass_scaler_X.fit_transform(compass_features)
compass_X_seq = create_sequences(compass_features_scaled, window_size)

# Prepare for inverse transform
gyro_target = gyro_data['GroundTruthHeadingComputed'].values.reshape(-1, 1)
compass_target = compass_data['GroundTruthHeadingComputed'].values.reshape(-1, 1)
gyro_scaler_y = MinMaxScaler()
compass_scaler_y = MinMaxScaler()
_ = gyro_scaler_y.fit_transform(gyro_target)
_ = compass_scaler_y.fit_transform(compass_target)

# LSTM prediction
print("Generating LSTM heading predictions...")
gyro_pred_scaled = gyro_model.predict(gyro_X_seq)
compass_pred_scaled = compass_model.predict(compass_X_seq)

# Inverse transform predictions
gyro_pred = gyro_scaler_y.inverse_transform(gyro_pred_scaled)
compass_pred = compass_scaler_y.inverse_transform(compass_pred_scaled)

# Process predictions
gyro_lstm_predictions = np.zeros(len(gyro_data))
compass_lstm_predictions = np.zeros(len(compass_data))

# Fill prediction arrays
gyro_lstm_predictions[window_size:window_size+len(gyro_pred)] = gyro_pred.flatten()
compass_lstm_predictions[window_size:window_size+len(compass_pred)] = compass_pred.flatten()

# For the first window_size elements, use traditional predictions
gyro_lstm_predictions[:window_size] = gyro_data['GyroStartByGroundTruth'][:window_size]
compass_lstm_predictions[:window_size] = compass_data['GyroStartByGroundTruth'][:window_size]

# Normalize to 0-360 degrees
gyro_lstm_predictions = (gyro_lstm_predictions + 360) % 360
compass_lstm_predictions = (compass_lstm_predictions + 360) % 360

# Add predictions to dataframes
gyro_data['LSTM_Predicted_Heading'] = gyro_lstm_predictions
compass_data['LSTM_Predicted_Heading'] = compass_lstm_predictions

# Create ensemble predictions with different weighting schemes
print("Creating ensemble heading predictions...")

# Function to calculate heading similarity
def heading_similarity(heading1, heading2):
    """Calculate how similar two headings are (lower value = more similar)"""
    diff = abs(heading1 - heading2)
    return min(diff, 360 - diff)

# Create different ensemble weights based on comparative analysis
weights = [0.3, 0.4, 0.5, 0.6, 0.7]

for weight in weights:
    # Weighted average for gyro data
    trad_weight = weight
    lstm_weight = 1 - weight
    
    # Create dynamic weights for gyro based on heading similarity to ground truth
    gyro_data[f'Ensemble_{int(weight*100)}'] = (
        trad_weight * gyro_data['GyroStartByGroundTruth'] + 
        lstm_weight * gyro_data['LSTM_Predicted_Heading']
    )
    gyro_data[f'Ensemble_{int(weight*100)}'] = (gyro_data[f'Ensemble_{int(weight*100)}'] + 360) % 360
    
    # Create dynamic weights for compass based on heading similarity to ground truth
    compass_data[f'Ensemble_{int(weight*100)}'] = (
        trad_weight * compass_data['GyroStartByGroundTruth'] + 
        lstm_weight * compass_data['LSTM_Predicted_Heading']
    )
    compass_data[f'Ensemble_{int(weight*100)}'] = (compass_data[f'Ensemble_{int(weight*100)}'] + 360) % 360

# Calculate adaptive ensemble based on confidence levels
# For gyro data: higher confidence in traditional method when rotation speed is lower
gyro_data['rotation_speed'] = abs(gyro_data['axisZAngle'])
gyro_max_speed = gyro_data['rotation_speed'].max()
gyro_data['trad_confidence'] = 1 - (gyro_data['rotation_speed'] / gyro_max_speed)
gyro_data['lstm_confidence'] = 1 - gyro_data['trad_confidence'] 

# More adaptive weights
gyro_data['Ensemble_Adaptive'] = (
    gyro_data['trad_confidence'] * gyro_data['GyroStartByGroundTruth'] +
    gyro_data['lstm_confidence'] * gyro_data['LSTM_Predicted_Heading']
) / (gyro_data['trad_confidence'] + gyro_data['lstm_confidence'])
gyro_data['Ensemble_Adaptive'] = (gyro_data['Ensemble_Adaptive'] + 360) % 360

# For compass: higher confidence in LSTM when magnetic field is less stable (more variance)
compass_data['field_stability'] = abs(compass_data['Magnetic_Field_Magnitude'].diff().fillna(0))
compass_max_instability = compass_data['field_stability'].max()
compass_data['lstm_confidence'] = compass_data['field_stability'] / compass_max_instability
compass_data['trad_confidence'] = 1 - compass_data['lstm_confidence']

compass_data['Ensemble_Adaptive'] = (
    compass_data['trad_confidence'] * compass_data['GyroStartByGroundTruth'] +
    compass_data['lstm_confidence'] * compass_data['LSTM_Predicted_Heading']
) / (compass_data['trad_confidence'] + compass_data['lstm_confidence'])
compass_data['Ensemble_Adaptive'] = (compass_data['Ensemble_Adaptive'] + 360) % 360

# Save the predicted data
gyro_data.to_csv(os.path.join(output_dir, 'gyro_ensemble_predictions.csv'), index=False)
compass_data.to_csv(os.path.join(output_dir, 'compass_ensemble_predictions.csv'), index=False)
print(f"Ensemble predictions saved to {output_dir}")

# Plot heading comparisons
print("Generating heading comparison plots...")

# Plot settings
fontSizeAll = 8
plt.rcParams.update({
    'xtick.major.pad': '1',
    'ytick.major.pad': '1',
    'legend.fontsize': fontSizeAll,
    'legend.handlelength': 2,
    'font.size': fontSizeAll,
    'axes.linewidth': 0.2,
    'patch.linewidth': 0.2,
    'font.family': "Times New Roman"
})

# Gyro data heading comparison
plt.figure(figsize=(8, 5), dpi=300)
plt.plot(gyro_data['Timestamp_(ms)'], gyro_data['GroundTruthHeadingComputed'], 'k-', label='Ground Truth', linewidth=1.5)
plt.plot(gyro_data['Timestamp_(ms)'], gyro_data['GyroStartByGroundTruth'], 'b--', label='Traditional', linewidth=1)
plt.plot(gyro_data['Timestamp_(ms)'], gyro_data['LSTM_Predicted_Heading'], 'r-.', label='LSTM', linewidth=1)
plt.plot(gyro_data['Timestamp_(ms)'], gyro_data['Ensemble_Adaptive'], 'g-', label='Adaptive Ensemble', linewidth=1)

plt.xlabel('Time (ms)')
plt.ylabel('Heading (degrees)')
plt.title('Gyro Heading Comparison: Traditional vs LSTM vs Ensemble')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.savefig(os.path.join(output_dir, 'gyro_ensemble_heading_comparison.png'), bbox_inches='tight')

# Compass data heading comparison
plt.figure(figsize=(8, 5), dpi=300)
plt.plot(compass_data['Timestamp_(ms)'], compass_data['GroundTruthHeadingComputed'], 'k-', label='Ground Truth', linewidth=1.5)
plt.plot(compass_data['Timestamp_(ms)'], compass_data['GyroStartByGroundTruth'], 'b--', label='Traditional', linewidth=1)
plt.plot(compass_data['Timestamp_(ms)'], compass_data['LSTM_Predicted_Heading'], 'r-.', label='LSTM', linewidth=1)
plt.plot(compass_data['Timestamp_(ms)'], compass_data['Ensemble_Adaptive'], 'g-', label='Adaptive Ensemble', linewidth=1)

plt.xlabel('Time (ms)')
plt.ylabel('Heading (degrees)')
plt.title('Compass Heading Comparison: Traditional vs LSTM vs Ensemble')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.savefig(os.path.join(output_dir, 'compass_ensemble_heading_comparison.png'), bbox_inches='tight')

print("Ensemble model implementation complete!") 