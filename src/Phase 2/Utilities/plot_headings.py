import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import os

# Set output directory
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2'
os.makedirs(output_dir, exist_ok=True)

# Input file - use one of the data files from the Data_collected directory
input_file = '/Users/shaoxinyi/Downloads/FYP2/Data_collected/1536_CompassGyroSumHeadingData.txt'

# Loading raw data
print(f"Loading data from {input_file}")
try:
    # First try to read as semicolon-delimited CSV
    data = pd.read_csv(input_file, delimiter=';')
    print(f"Successfully loaded file as semicolon-delimited CSV")
    
    # Check if the dataframe has a header row
    if 'Type' not in data.columns:
        # File doesn't have headers, try to infer them
        column_names = [
            'Timestamp_(ms)', 'Type', 'step', 'value_1', 'value_2', 'value_3', 
            'GroundTruth', 'value_4', 'value_5', 'turns'
        ]
        
        # Try again with column names
        data = pd.read_csv(input_file, delimiter=';', names=column_names)
        
        # If first row contains header values, drop it
        if data.iloc[0]['Type'] == 'Type':
            data = data.iloc[1:].reset_index(drop=True)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Extract ground truth location data
ground_truth_location_data = data[data['Type'] == 'Ground_truth_Location'].copy()
initial_location_data = data[data['Type'] == 'Initial_Location'].copy()

# Create a separate dataframe to store ground truth headings
if len(ground_truth_location_data) > 0 and len(initial_location_data) > 0:
    df_gt = pd.concat([initial_location_data, ground_truth_location_data], ignore_index=True)
    df_gt.sort_values(by='Timestamp_(ms)', inplace=True)
    df_gt.reset_index(drop=True, inplace=True)
else:
    print("Missing ground truth or initial location data")
    exit(1)

# Calculate ground truth heading
def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing (azimuth) between two points
    """
    from math import atan2, degrees, radians, sin, cos
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    delta_lon = lon2 - lon1
    x = atan2(
        sin(delta_lon) * cos(lat2),
        cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(delta_lon)
    )
    
    bearing = (degrees(x) + 360) % 360  # Normalize to 0-360 degrees
    return bearing

# Add a column for ground truth heading
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

# Ensure data and df_gt are sorted by timestamp
data.sort_values(by="Timestamp_(ms)", inplace=True)
df_gt.sort_values(by="Timestamp_(ms)", inplace=True)

# Use backward fill to propagate the GroundTruthHeadingComputed values
data = data.merge(df_gt[["Timestamp_(ms)", "GroundTruthHeadingComputed"]], on="Timestamp_(ms)", how="left")
data["GroundTruthHeadingComputed"] = data["GroundTruthHeadingComputed"].fillna(method="bfill")

# Convert numeric columns to float
for col in ['value_1', 'value_2', 'value_3', 'GroundTruthHeadingComputed']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Separate data for gyro and compass
gyro_data = data[data['Type'] == 'Gyro'].reset_index(drop=True)
compass_data = data[data['Type'] == 'Compass'].reset_index(drop=True)

# Rename columns for clarity
compass_data.rename(columns={'value_1': 'Magnetic_Field_Magnitude'}, inplace=True)
compass_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
compass_data.rename(columns={'value_3': 'compass'}, inplace=True)

gyro_data.rename(columns={'value_1': 'axisZAngle'}, inplace=True)
gyro_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
gyro_data.rename(columns={'value_3': 'compass'}, inplace=True)

# Get the initial ground truth heading
first_ground_truth = initial_location_data['GroundTruth'].iloc[0] if len(initial_location_data) > 0 else 0

# Calculate the gyro heading starting from ground truth
compass_data['GyroStartByGroundTruth'] = first_ground_truth + compass_data['gyroSumFromstart0'] - compass_data['gyroSumFromstart0'].iloc[0]
compass_data['GyroStartByGroundTruth'] = (compass_data['GyroStartByGroundTruth'] + 360) % 360

gyro_data['GyroStartByGroundTruth'] = first_ground_truth + gyro_data['gyroSumFromstart0'] - gyro_data['gyroSumFromstart0'].iloc[0]
gyro_data['GyroStartByGroundTruth'] = (gyro_data['GyroStartByGroundTruth'] + 360) % 360

# Set plot parameters for IEEE format
fontSizeAll = 5  # Keep small font size
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

# Format timestamps for better readability
# Convert timestamps to relative time in seconds from the start
min_timestamp = min(gyro_data['Timestamp_(ms)'].min(), compass_data['Timestamp_(ms)'].min())
gyro_data['Time_relative'] = (gyro_data['Timestamp_(ms)'] - min_timestamp) / 1000  # Convert to seconds
compass_data['Time_relative'] = (compass_data['Timestamp_(ms)'] - min_timestamp) / 1000  # Convert to seconds

# 1. Plot Ground Truth Heading vs Gyro Heading
fig1, ax1 = plt.subplots(figsize=(5, 3), dpi=300)  # Slightly reduce height
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.92)  # More space for labels

# Plot the ground truth heading
plt.plot(gyro_data["Time_relative"], gyro_data["GroundTruthHeadingComputed"], 
         color='red', linestyle='-', linewidth=1.2, label='Ground Truth Heading')

# Plot the gyro heading
plt.plot(gyro_data["Time_relative"], gyro_data["GyroStartByGroundTruth"], 
         color='blue', linestyle='-', linewidth=1, label='Gyro Heading')

# Axis formatting
ax1.yaxis.set_major_locator(MultipleLocator(45))  # Y-axis major tick interval: 45 degrees
ax1.yaxis.set_minor_locator(MultipleLocator(15))  # Y-axis minor tick interval: 15 degrees

# Format x-axis with fewer ticks
max_time = max(gyro_data['Time_relative'].max(), compass_data['Time_relative'].max())
ax1.set_xlim(0, max_time)
ax1.xaxis.set_major_locator(plt.MaxNLocator(8))  # Reduce to 8 ticks for less crowding

# Rotate x-axis labels for better readability
plt.xticks(rotation=30, ha='right')

# Axis labels
plt.xlabel("Time (s)", labelpad=5)
plt.ylabel("Heading (Degrees)", labelpad=5)

# Ticks and grid
plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in', 
               which='major', grid_color='blue', width=0.3, length=2.5)
plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in', 
               which='minor', grid_color='blue', width=0.15, length=1)

# Custom Legend
legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', linewidth=1.2, label='Ground Truth Heading'),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=1.2, label='Gyro Heading')
]

plt.legend(handles=legend_elements, loc='best')

# Grid
plt.grid(linestyle=':', linewidth=0.5, alpha=0.15, color='k')

# Add title
plt.title("Ground Truth Heading vs Gyro Heading (Traditional Method)", fontsize=fontSizeAll+1)

# Save the figure
gyro_plot_file = os.path.join(output_dir, 'ground_truth_vs_gyro_heading.png')
plt.savefig(gyro_plot_file, bbox_inches='tight')
print(f"Saved Ground Truth vs Gyro Heading plot to {gyro_plot_file}")

# 2. Plot Ground Truth Heading vs Compass Heading
fig2, ax2 = plt.subplots(figsize=(5, 3), dpi=300)  # Slightly reduce height
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.92)  # More space for labels

# Plot the ground truth heading
plt.plot(compass_data["Time_relative"], compass_data["GroundTruthHeadingComputed"], 
         color='red', linestyle='-', linewidth=1.2, label='Ground Truth Heading')

# Plot the compass heading
plt.plot(compass_data["Time_relative"], compass_data["compass"], 
         color='green', linestyle='-', linewidth=1, label='Compass Heading')

# Axis formatting
ax2.yaxis.set_major_locator(MultipleLocator(45))  # Y-axis major tick interval: 45 degrees
ax2.yaxis.set_minor_locator(MultipleLocator(15))  # Y-axis minor tick interval: 15 degrees

# Format x-axis with fewer ticks
ax2.set_xlim(0, max_time)
ax2.xaxis.set_major_locator(plt.MaxNLocator(8))  # Reduce to 8 ticks for less crowding

# Rotate x-axis labels for better readability
plt.xticks(rotation=30, ha='right')

# Axis labels
plt.xlabel("Time (s)", labelpad=5)
plt.ylabel("Heading (Degrees)", labelpad=5)

# Ticks and grid
plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in', 
               which='major', grid_color='blue', width=0.3, length=2.5)
plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in', 
               which='minor', grid_color='blue', width=0.15, length=1)

# Custom Legend
legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', linewidth=1.2, label='Ground Truth Heading'),
    Line2D([0], [0], color='green', linestyle='-', linewidth=1.2, label='Compass Heading')
]

plt.legend(handles=legend_elements, loc='best')

# Grid
plt.grid(linestyle=':', linewidth=0.5, alpha=0.15, color='k')

# Add title
plt.title("Ground Truth Heading vs Compass Heading (Traditional Method)", fontsize=fontSizeAll+1)

# Save the figure
compass_plot_file = os.path.join(output_dir, 'ground_truth_vs_compass_heading.png')
plt.savefig(compass_plot_file, bbox_inches='tight')
print(f"Saved Ground Truth vs Compass Heading plot to {compass_plot_file}")

# Display success message
print("Both plots have been generated successfully using the traditional method") 