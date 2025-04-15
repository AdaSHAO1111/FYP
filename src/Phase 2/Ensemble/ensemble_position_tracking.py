import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import os

# Set output directory
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2'
os.makedirs(output_dir, exist_ok=True)

# Load ensemble prediction data
print("Loading ensemble prediction data...")
try:
    gyro_data = pd.read_csv(os.path.join(output_dir, 'gyro_ensemble_predictions.csv'))
    compass_data = pd.read_csv(os.path.join(output_dir, 'compass_ensemble_predictions.csv'))
    print("Ensemble prediction data loaded successfully")
except FileNotFoundError:
    print("Ensemble prediction data not found. Please run the ensemble_heading_prediction.py script first.")
    exit(1)

# Load Ground Truth data
data_file = '/Users/shaoxinyi/Downloads/FYP2/Data_collected/1536_CompassGyroSumHeadingData.txt'
data = pd.read_csv(data_file, delimiter=';')

# Extract Ground Truth location data
ground_truth_location_data = data[data['Type'] == 'Ground_truth_Location'].copy()
initial_location_data = data[data['Type'] == 'Initial_Location'].copy()

# Create separate dataframe for Ground Truth heading
if len(ground_truth_location_data) > 0 and len(initial_location_data) > 0:
    df_gt = pd.concat([initial_location_data, ground_truth_location_data], ignore_index=True)
    df_gt.sort_values(by='Timestamp_(ms)', inplace=True)
    df_gt.reset_index(drop=True, inplace=True)

# Calculate position trajectories
print("Calculating position trajectories...")

# Set step length and initial position
step_length = 0.66  # meters per step
initial_position = (0, 0)

# Calculate Ground Truth positions
def calculate_gt_positions(df_gt, initial_position=(0, 0)):
    # Ensure Ground Truth data is sorted by step
    df_gt = df_gt.sort_values(by='step')
    
    # Extract position data from dataframe
    gt_positions = []
    
    # If coordinates are in the data, use them directly
    if 'value_4' in df_gt.columns and 'value_5' in df_gt.columns:
        # Ensure coordinates are numeric
        df_gt['value_4'] = pd.to_numeric(df_gt['value_4'], errors='coerce')
        df_gt['value_5'] = pd.to_numeric(df_gt['value_5'], errors='coerce')
        
        # Set first Ground Truth point as origin
        origin_x = df_gt['value_4'].iloc[0]
        origin_y = df_gt['value_5'].iloc[0]
        
        # Extract coordinates relative to origin
        for i in range(len(df_gt)):
            x = df_gt['value_4'].iloc[i] - origin_x
            y = df_gt['value_5'].iloc[i] - origin_y
            gt_positions.append((x, y))
    
    return gt_positions

# Calculate positions based on steps and heading
def calculate_positions(data, heading_column, step_length=0.66, initial_position=(0, 0)):
    positions = [initial_position]  # Start from initial position
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

# Get Ground Truth positions
gt_positions = calculate_gt_positions(df_gt)
print(f"Ground Truth position points count: {len(gt_positions)}")

# Calculate positions using traditional methods
print("Calculating positions using traditional methods...")
positions_compass_trad = calculate_positions(compass_data, 'GyroStartByGroundTruth', step_length, initial_position)
positions_gyro_trad = calculate_positions(gyro_data, 'GyroStartByGroundTruth', step_length, initial_position)
print(f"Traditional Compass trajectory points count: {len(positions_compass_trad)}")
print(f"Traditional Gyro trajectory points count: {len(positions_gyro_trad)}")

# Calculate positions using LSTM methods
print("Calculating positions using LSTM methods...")
positions_compass_lstm = calculate_positions(compass_data, 'LSTM_Predicted_Heading', step_length, initial_position)
positions_gyro_lstm = calculate_positions(gyro_data, 'LSTM_Predicted_Heading', step_length, initial_position)
print(f"LSTM Compass trajectory points count: {len(positions_compass_lstm)}")
print(f"LSTM Gyro trajectory points count: {len(positions_gyro_lstm)}")

# Calculate positions using Ensemble methods
print("Calculating positions using Ensemble methods...")

# Using static weight ensembles
ensemble_weights = [0.3, 0.4, 0.5, 0.6, 0.7]
ensemble_positions_compass = {}
ensemble_positions_gyro = {}

for weight in ensemble_weights:
    column_name = f'Ensemble_{int(weight*100)}'
    
    # Calculate positions for compass ensemble
    ensemble_positions_compass[weight] = calculate_positions(
        compass_data, column_name, step_length, initial_position
    )
    
    # Calculate positions for gyro ensemble
    ensemble_positions_gyro[weight] = calculate_positions(
        gyro_data, column_name, step_length, initial_position
    )
    
    print(f"Ensemble {weight:.1f} Compass trajectory points count: {len(ensemble_positions_compass[weight])}")
    print(f"Ensemble {weight:.1f} Gyro trajectory points count: {len(ensemble_positions_gyro[weight])}")

# Calculate positions using adaptive ensemble
positions_compass_adaptive = calculate_positions(compass_data, 'Ensemble_Adaptive', step_length, initial_position)
positions_gyro_adaptive = calculate_positions(gyro_data, 'Ensemble_Adaptive', step_length, initial_position)
print(f"Adaptive Ensemble Compass trajectory points count: {len(positions_compass_adaptive)}")
print(f"Adaptive Ensemble Gyro trajectory points count: {len(positions_gyro_adaptive)}")

# Extract coordinates for plotting
x_gt = [pos[0] for pos in gt_positions]
y_gt = [pos[1] for pos in gt_positions]

x_compass_trad = [pos[0] for pos in positions_compass_trad]
y_compass_trad = [pos[1] for pos in positions_compass_trad]

x_gyro_trad = [pos[0] for pos in positions_gyro_trad]
y_gyro_trad = [pos[1] for pos in positions_gyro_trad]

x_compass_lstm = [pos[0] for pos in positions_compass_lstm]
y_compass_lstm = [pos[1] for pos in positions_compass_lstm]

x_gyro_lstm = [pos[0] for pos in positions_gyro_lstm]
y_gyro_lstm = [pos[1] for pos in positions_gyro_lstm]

x_compass_adaptive = [pos[0] for pos in positions_compass_adaptive]
y_compass_adaptive = [pos[1] for pos in positions_compass_adaptive]

x_gyro_adaptive = [pos[0] for pos in positions_gyro_adaptive]
y_gyro_adaptive = [pos[1] for pos in positions_gyro_adaptive]

# Calculate position error
def calculate_position_error(positions, gt_positions):
    """Calculate average and cumulative error between position sequence and Ground Truth"""
    # Ensure both position sequences have same length, use minimum length for comparison
    min_length = min(len(positions), len(gt_positions))
    
    # Calculate error for each point
    errors = []
    for i in range(min_length):
        # Calculate Euclidean distance
        error = np.sqrt((positions[i][0] - gt_positions[i][0])**2 + 
                         (positions[i][1] - gt_positions[i][1])**2)
        errors.append(error)
    
    # Calculate average and cumulative error
    avg_error = np.mean(errors)
    cumulative_error = np.sum(errors)
    
    return avg_error, cumulative_error, errors

# Calculate errors for all methods
print("Calculating position errors...")
error_results = {}

# Traditional methods
trad_compass_avg, trad_compass_cum, trad_compass_errors = calculate_position_error(positions_compass_trad, gt_positions)
trad_gyro_avg, trad_gyro_cum, trad_gyro_errors = calculate_position_error(positions_gyro_trad, gt_positions)
error_results['Compass (Traditional)'] = trad_compass_avg
error_results['Gyro (Traditional)'] = trad_gyro_avg

# LSTM methods
lstm_compass_avg, lstm_compass_cum, lstm_compass_errors = calculate_position_error(positions_compass_lstm, gt_positions)
lstm_gyro_avg, lstm_gyro_cum, lstm_gyro_errors = calculate_position_error(positions_gyro_lstm, gt_positions)
error_results['Compass (LSTM)'] = lstm_compass_avg
error_results['Gyro (LSTM)'] = lstm_gyro_avg

# Ensemble methods with static weights
for weight in ensemble_weights:
    # Compass ensemble errors
    ens_compass_avg, ens_compass_cum, ens_compass_errors = calculate_position_error(
        ensemble_positions_compass[weight], gt_positions
    )
    error_results[f'Compass (Ensemble {weight:.1f})'] = ens_compass_avg
    
    # Gyro ensemble errors
    ens_gyro_avg, ens_gyro_cum, ens_gyro_errors = calculate_position_error(
        ensemble_positions_gyro[weight], gt_positions
    )
    error_results[f'Gyro (Ensemble {weight:.1f})'] = ens_gyro_avg

# Adaptive ensemble errors
adaptive_compass_avg, adaptive_compass_cum, adaptive_compass_errors = calculate_position_error(positions_compass_adaptive, gt_positions)
adaptive_gyro_avg, adaptive_gyro_cum, adaptive_gyro_errors = calculate_position_error(positions_gyro_adaptive, gt_positions)
error_results['Compass (Adaptive Ensemble)'] = adaptive_compass_avg
error_results['Gyro (Adaptive Ensemble)'] = adaptive_gyro_avg

# Create error comparison table
error_df = pd.DataFrame({
    'Method': list(error_results.keys()),
    'Average Error (m)': list(error_results.values())
})

# Find the best method
best_method = error_df.loc[error_df['Average Error (m)'].idxmin()]['Method']
best_error = error_df['Average Error (m)'].min()
print(f"\nBest position tracking method: {best_method} with {best_error:.2f}m average error")

# Calculate improvement over traditional and LSTM methods
best_ensemble_compass_error = min([error_results[f'Compass (Ensemble {w:.1f})'] for w in ensemble_weights] + [error_results['Compass (Adaptive Ensemble)']])
best_ensemble_gyro_error = min([error_results[f'Gyro (Ensemble {w:.1f})'] for w in ensemble_weights] + [error_results['Gyro (Adaptive Ensemble)']])

compass_improvement_over_trad = (error_results['Compass (Traditional)'] - best_ensemble_compass_error) / error_results['Compass (Traditional)'] * 100
gyro_improvement_over_trad = (error_results['Gyro (Traditional)'] - best_ensemble_gyro_error) / error_results['Gyro (Traditional)'] * 100

compass_improvement_over_lstm = (error_results['Compass (LSTM)'] - best_ensemble_compass_error) / error_results['Compass (LSTM)'] * 100
gyro_improvement_over_lstm = (error_results['Gyro (LSTM)'] - best_ensemble_gyro_error) / error_results['Gyro (LSTM)'] * 100

print(f"\nImprovement over traditional methods:")
print(f"Compass: {compass_improvement_over_trad:.2f}%")
print(f"Gyro: {gyro_improvement_over_trad:.2f}%")

print(f"\nImprovement over LSTM methods:")
print(f"Compass: {compass_improvement_over_lstm:.2f}%")
print(f"Gyro: {gyro_improvement_over_lstm:.2f}%")

# Save error comparison table
error_csv = os.path.join(output_dir, 'ensemble_position_error_comparison.csv')
error_df.to_csv(error_csv, index=False)
print(f"\nPosition error comparison saved to: {error_csv}")

# Visualize position trajectories
print("\nGenerating position trajectory plots...")

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

# Plot comparison of methods (Ground Truth, Traditional, LSTM, Best Ensemble)
fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.92)

# Find the best ensemble for compass and gyro
best_compass_ensemble_weight = min(
    [(w, error_results[f'Compass (Ensemble {w:.1f})']) for w in ensemble_weights] + 
    [(0, error_results['Compass (Adaptive Ensemble)'])], 
    key=lambda x: x[1]
)[0]

best_gyro_ensemble_weight = min(
    [(w, error_results[f'Gyro (Ensemble {w:.1f})']) for w in ensemble_weights] + 
    [(0, error_results['Gyro (Adaptive Ensemble)'])], 
    key=lambda x: x[1]
)[0]

# Plot Ground Truth trajectory
plt.plot(x_gt, y_gt, color='blue', linestyle='-', linewidth=2, label='Ground Truth')

# Plot traditional methods
plt.plot(x_compass_trad, y_compass_trad, color='green', linestyle='--', linewidth=1, label='Compass (Traditional)')
plt.plot(x_gyro_trad, y_gyro_trad, color='red', linestyle='--', linewidth=1, label='Gyro (Traditional)')

# Plot LSTM methods
plt.plot(x_compass_lstm, y_compass_lstm, color='green', linestyle=':', linewidth=1, label='Compass (LSTM)')
plt.plot(x_gyro_lstm, y_gyro_lstm, color='red', linestyle=':', linewidth=1, label='Gyro (LSTM)')

# Plot best compass ensemble
if best_compass_ensemble_weight == 0:  # Adaptive is best
    plt.plot(x_compass_adaptive, y_compass_adaptive, color='green', linestyle='-', linewidth=1.5, label='Compass (Best Ensemble)')
else:  # Static weight is best
    x_best_compass = [pos[0] for pos in ensemble_positions_compass[best_compass_ensemble_weight]]
    y_best_compass = [pos[1] for pos in ensemble_positions_compass[best_compass_ensemble_weight]]
    plt.plot(x_best_compass, y_best_compass, color='green', linestyle='-', linewidth=1.5, 
             label=f'Compass (Ensemble {best_compass_ensemble_weight:.1f})')

# Plot best gyro ensemble
if best_gyro_ensemble_weight == 0:  # Adaptive is best
    plt.plot(x_gyro_adaptive, y_gyro_adaptive, color='red', linestyle='-', linewidth=1.5, label='Gyro (Best Ensemble)')
else:  # Static weight is best
    x_best_gyro = [pos[0] for pos in ensemble_positions_gyro[best_gyro_ensemble_weight]]
    y_best_gyro = [pos[1] for pos in ensemble_positions_gyro[best_gyro_ensemble_weight]]
    plt.plot(x_best_gyro, y_best_gyro, color='red', linestyle='-', linewidth=1.5, 
             label=f'Gyro (Ensemble {best_gyro_ensemble_weight:.1f})')

# Mark start and end points
plt.scatter(x_gt[0], y_gt[0], color='black', marker='o', s=100, label='Start')
plt.scatter(x_gt[-1], y_gt[-1], color='black', marker='x', s=100, label='End')

# Format axes
ax.set_aspect('equal')
ax.set_xlabel('East (m)', labelpad=5)
ax.set_ylabel('North (m)', labelpad=5)
ax.grid(True, linestyle=':', alpha=0.5)

# Add title
plt.title('Position Trajectories Comparison (All Methods)', fontsize=fontSizeAll+2)

# Add legend
plt.legend(loc='best')

# Save image
all_methods_plot_file = os.path.join(output_dir, 'ensemble_position_comparison.png')
plt.savefig(all_methods_plot_file, bbox_inches='tight')
print(f"Position comparison plot saved to: {all_methods_plot_file}")

# Create error bar plot
plt.figure(figsize=(10, 6), dpi=300)
plt.subplots_adjust(left=0.15, bottom=0.25, right=0.95, top=0.92)

# Sort methods by error for better visualization
error_df = error_df.sort_values(by='Average Error (m)')

# Plot errors
bars = plt.barh(error_df['Method'], error_df['Average Error (m)'], color='skyblue')
plt.xlabel('Average Position Error (m)')
plt.title('Position Error Comparison Across Methods')
plt.grid(axis='x', linestyle=':', alpha=0.5)

# Add error values
for i, v in enumerate(error_df['Average Error (m)']):
    plt.text(v + 0.1, i, f'{v:.2f}m', va='center')

# Highlight the best method
bars[error_df['Method'].tolist().index(best_method)].set_color('green')

# Save error bar plot
error_plot_file = os.path.join(output_dir, 'ensemble_position_error_comparison.png')
plt.savefig(error_plot_file, bbox_inches='tight')
print(f"Position error comparison plot saved to: {error_plot_file}")

print("\nEnsemble position tracking analysis complete!") 