import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize variables
walked_distance = 0
current_position_gt = initial_position
current_position_compass = initial_position
current_position_gyro = initial_position

# Calculate positions of all steps
positions_compass = []
positions_gt = []
positions_gyro = []
step_numbers = []  # List to track step numbers

# Initialize variables for tracking the previous step number and heading
prev_step = compass_data['step'][0]

for i in range(1, len(compass_data)):
    # Calculate change in step number
    change_in_step = compass_data['step'][i] - prev_step

    # If the step number has changed, compute the new position
    if change_in_step != 0:
        # Calculate walked distance for this step
        change_in_distance = change_in_step * 0.66
        
        # Calculate new positions for each method
        new_x_gt = current_position_gt[0] + change_in_distance * np.sin(np.radians(compass_data['GroundTruth'][i]))
        new_y_gt = current_position_gt[1] + change_in_distance * np.cos(np.radians(compass_data['GroundTruth'][i]))

        new_x_compass = current_position_compass[0] + change_in_distance * np.sin(np.radians(compass_data['compass'][i]))
        new_y_compass = current_position_compass[1] + change_in_distance * np.cos(np.radians(compass_data['compass'][i]))

        new_x_gyro = current_position_gyro[0] + change_in_distance * np.sin(np.radians(compass_data['GyroStartByGroundTruth'][i]))
        new_y_gyro = current_position_gyro[1] + change_in_distance * np.cos(np.radians(compass_data['GyroStartByGroundTruth'][i]))

        # Update current positions
        current_position_gt = (new_x_gt, new_y_gt)
        current_position_compass = (new_x_compass, new_y_compass)
        current_position_gyro = (new_x_gyro, new_y_gyro)

        # Append new positions and step numbers
        positions_gt.append(current_position_gt)
        positions_compass.append(current_position_compass)
        positions_gyro.append(current_position_gyro)
        step_numbers.append(compass_data['step'][i])  # Store step number

        # Update previous step number
        prev_step = compass_data['step'][i]
        
# Extract x and y coordinates from positions
x_positions_compass = [position[0] for position in positions_compass]
y_positions_compass = [position[1] for position in positions_compass]

x_positions_gt = [position[0] for position in positions_gt]
y_positions_gt = [position[1] for position in positions_gt]

x_positions_gyro = [position[0] for position in positions_gyro]
y_positions_gyro = [position[1] for position in positions_gyro]
# Create DataFrames with step numbers
df_positions_compass = pd.DataFrame(positions_compass, columns=['Compass_X', 'Compass_Y'])
df_positions_gt = pd.DataFrame(positions_gt, columns=['GroundTruth_X', 'GroundTruth_Y'])
df_positions_gyro = pd.DataFrame(positions_gyro, columns=['Gyro_X', 'Gyro_Y'])

# Add step number to each DataFrame
df_positions_compass['Step'] = step_numbers
df_positions_gt['Step'] = step_numbers
df_positions_gyro['Step'] = step_numbers

# Merge df_positions_gt with compass_data based on the step number
# Ensure the step column is of the same type in both DataFrames
df_positions_gt['Step'] = df_positions_gt['Step'].astype(float)
compass_data['step'] = compass_data['step'].astype(float)

# Perform the merge
compass_data_merged = compass_data.merge(df_positions_gt, left_on='step', right_on='Step', how='left')

# Drop the redundant 'Step' column from df_positions_gt
compass_data_merged.drop(columns=['Step'], inplace=True)

# Ensure the step column is of the same type in both DataFrames
df_positions_compass['Step'] = df_positions_compass['Step'].astype(float)
compass_data_merged['step'] = compass_data_merged['step'].astype(float)

# Perform the merge
compass_data_final = compass_data_merged.merge(df_positions_compass, left_on='step', right_on='Step', how='left')

# Drop the redundant 'Step' column from df_positions_compass
compass_data_final.drop(columns=['Step'], inplace=True)

# Merge df_positions_gt with compass_data based on the step number
# Ensure the step column is of the same type in both DataFrames
df_positions_gt['Step'] = df_positions_gt['Step'].astype(float)
gyro_data['step'] = gyro_data['step'].astype(float)

# Perform the merge
gyro_data_merged = gyro_data.merge(df_positions_gt, left_on='step', right_on='Step', how='left')

# Drop the redundant 'Step' column from df_positions_gt
gyro_data_merged.drop(columns=['Step'], inplace=True)

# Ensure the step column is of the same type in both DataFrames
df_positions_gyro['Step'] = df_positions_gyro['Step'].astype(float)
gyro_data_merged['step'] = gyro_data_merged['step'].astype(float)

# Perform the merge
gyro_data_final = gyro_data_merged.merge(df_positions_gyro, left_on='step', right_on='Step', how='left')

# Drop the redundant 'Step' column from df_positions_compass
gyro_data_final.drop(columns=['Step'], inplace=True)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Set plot parameters for IEEE format
fontSizeAll = 6
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

# Load custom markers (replace with correct file paths if necessary)
start_img = mpimg.imread("/content/drive/MyDrive/FYP database/data/0225data/start.png")
end_img = mpimg.imread("/content/drive/MyDrive/FYP database/data/0225data/enda.png")

# Function to add image marker at specific coordinates
def add_marker(ax, img, x, y, zoom=0.1):
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=100)
    ax.add_artist(ab)

# Create figure for IEEE column width
fig, ax = plt.subplots(figsize=(3.45, 2.94), dpi=1000)
plt.subplots_adjust(left=0.01, bottom=0.03, right=0.99, top=0.99, wspace=0.00, hspace=0.0)

# Plot Tracks (Swapping axes: East on X, North on Y)
plt.plot(x_positions_compass, y_positions_compass, color='purple', linestyle='--', linewidth=1.2, label='Compass')
plt.plot(x_positions_gyro, y_positions_gyro, color='red', linestyle='-', linewidth=1.2, label='Gyro')

# Ground truth positions 
plt.scatter(interpolated_positions_df['value_4'], interpolated_positions_df['value_5'], 
            c='blue', marker='.', s=30, label='Ground Truth')

# Manually labeled points 
plt.scatter(ground_truth_location_data['value_4'], ground_truth_location_data['value_5'], 
            marker='+', s=50, c='green', label='Manually Labeled')

# Add start and end markers on Ground Truth positions
start_x, start_y = ground_truth_location_data['value_4'].iloc[0], ground_truth_location_data['value_5'].iloc[0]+2
end_x, end_y = ground_truth_location_data['value_4'].iloc[-1], ground_truth_location_data['value_5'].iloc[-1]+2

add_marker(ax, start_img, start_x, start_y, zoom=0.05)
add_marker(ax, end_img, end_x, end_y, zoom=0.013)

# Axis formatting
ax.yaxis.set_major_locator(MultipleLocator(40))  # Y-axis major tick interval: 40m
ax.yaxis.set_minor_locator(MultipleLocator(20))  # Y-axis minor tick interval: 20m
ax.xaxis.set_major_locator(MultipleLocator(40))  # X-axis major tick interval: 40m
ax.xaxis.set_minor_locator(MultipleLocator(10))  # X-axis minor tick interval: 10m

plt.axis('scaled')

# Labels (Swapped: East on X-axis, North on Y-axis)
plt.xlabel('East (m)', labelpad=3)
plt.ylabel('North (m)', labelpad=4)

# Rotate y-tick labels
plt.yticks(rotation=90, va="center")

# Ticks and grid
plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in', 
                which='major', grid_color='blue', width=0.3, length=2.5)
plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in', 
                which='minor', grid_color='blue', width=0.15, length=1)

# Custom Legend
legend_elements = [
    Line2D([0], [0], color='purple', linestyle='--', linewidth=1.2, label='Compass'),
    Line2D([0], [0], color='red', linestyle='-', linewidth=1.2, label='Gyro'),
    Line2D([0], [0], marker='.', color='blue', markersize=5, linestyle='None', label='Ground Truth'),
    Line2D([0], [0], marker='+', color='green', markersize=8, linestyle='None', label='Manually Labeled')
]

plt.legend(handles=legend_elements, loc='best')

# Grid
ax.ticklabel_format(useOffset=False)
plt.grid(linestyle=':', linewidth=0.5, alpha=0.15, color='k')

# Show plot
plt.show()
