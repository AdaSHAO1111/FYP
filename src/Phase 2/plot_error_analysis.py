import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib.ticker import MultipleLocator
import os

# Set up output directory
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2'
os.makedirs(output_dir, exist_ok=True)

# Input file
input_file = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2/position_coordinates.csv'

print(f"Loading data from {input_file}...")
try:
    # Load position coordinates data
    position_data = pd.read_csv(input_file)
    print(f"Successfully loaded data with {len(position_data)} rows")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Set IEEE-style plot parameters
fontSizeAll = 6
plt.rcParams.update({
    'xtick.major.pad': '1',
    'ytick.major.pad': '1',
    'legend.fontsize': 5,
    'legend.handlelength': 2,
    'font.size': 5,
    'axes.linewidth': 0.2,
    'patch.linewidth': 0.2,
    'font.family': "Times New Roman"
})

# Only use rows with valid error data (those with GT data)
valid_data = position_data.dropna(subset=['Gyro_Distance_Error', 'Compass_Distance_Error']).copy()
print(f"Number of data points with valid error metrics: {len(valid_data)}")

### 1. Plot Distance Error of Gyro vs Compass
print("Generating distance error plot...")
fig1, ax1 = plt.subplots(figsize=(3.45, 2.5), dpi=1000)  # IEEE column width

ax1.plot(valid_data['Walked_distance'], valid_data['Gyro_Distance_Error'],
         label='Gyro', linewidth=1.2, color='blue')
ax1.plot(valid_data['Walked_distance'], valid_data['Compass_Distance_Error'],
         label='Compass', linewidth=1.2, color='red')

ax1.set_xlabel('Walked Distance (m)', labelpad=3)
ax1.set_ylabel('Positioning Error (m)', labelpad=3)

ax1.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='k')
ax1.legend()

# Secondary x-axis for step numbers
secx = ax1.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
# Use step values at regular intervals to avoid crowding
step_interval = max(1, len(valid_data) // 10)  # Aim for around 10 labels
secx.set_xticks(valid_data['Walked_distance'][::step_interval])
secx.set_xticklabels(valid_data['step'][::step_interval])
secx.set_xlabel('Number of Walked Steps', labelpad=8)

# Axis formatting
ax1.xaxis.set_major_locator(MultipleLocator(10))  # Adjust spacing as needed
ax1.yaxis.set_major_locator(MultipleLocator(5))   # Major ticks every 5 meters

# Save the figure
error_plot_file = os.path.join(output_dir, 'distance_error_plot.png')
fig1.savefig(error_plot_file, bbox_inches='tight')
print(f"Distance error plot saved to: {error_plot_file}")

### 2. ECDF Plot
print("Generating ECDF plot...")
fig2 = plt.figure(figsize=(3.45, 2.5), dpi=1000)  # IEEE column width

ecdf_gyro = sm.distributions.ECDF(valid_data['Gyro_Distance_Error'])
ecdf_compass = sm.distributions.ECDF(valid_data['Compass_Distance_Error'])

plt.plot(ecdf_gyro.x, ecdf_gyro.y, label='Gyro', color='blue', linewidth=1.2)
plt.plot(ecdf_compass.x, ecdf_compass.y, label='Compass', color='red', linewidth=1.2)

plt.xlabel('Positioning Error (m)', labelpad=3)
plt.ylabel('ECDF', labelpad=3)

plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='k')
plt.legend()

# Save the figure
ecdf_plot_file = os.path.join(output_dir, 'ecdf_plot.png')
fig2.savefig(ecdf_plot_file, bbox_inches='tight')
print(f"ECDF plot saved to: {ecdf_plot_file}")

### 3. Box Plot for Positioning Errors
print("Generating box plot...")
# Create a DataFrame for distance errors
distance_errors_df = pd.DataFrame({
    'Sensor': ['Gyro'] * len(valid_data) + ['Compass'] * len(valid_data),
    'Distance_Error': np.concatenate([valid_data['Gyro_Distance_Error'], valid_data['Compass_Distance_Error']])
})

fig3 = plt.figure(figsize=(3.45, 2.5), dpi=1000)  # IEEE column width

sns.boxplot(x='Sensor', y='Distance_Error', data=distance_errors_df, linewidth=0.6)

plt.xlabel('Sensor', fontsize=5)
plt.ylabel('Positioning Error (m)', fontsize=5)

plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='k')

# Save the figure
box_plot_file = os.path.join(output_dir, 'box_plot.png')
fig3.savefig(box_plot_file, bbox_inches='tight')
print(f"Box plot saved to: {box_plot_file}")

# Calculate and print summary statistics
print("\nSummary statistics:")
print("\nGyro Distance Error (m):")
print(valid_data['Gyro_Distance_Error'].describe())
print("\nCompass Distance Error (m):")
print(valid_data['Compass_Distance_Error'].describe())

print("Error analysis plotting complete") 