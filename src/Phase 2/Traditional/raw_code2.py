# Ensure data and df_gt are sorted by timestamp
data.sort_values(by="Timestamp_(ms)", inplace=True)
df_gt.sort_values(by="Timestamp_(ms)", inplace=True)

# Use backward fill to propagate the GroundTruthHeadingComputed values
data = data.merge(df_gt[["Timestamp_(ms)", "GroundTruthHeadingComputed"]], on="Timestamp_(ms)", how="left")
data["GroundTruthHeadingComputed"] = data["GroundTruthHeadingComputed"].fillna(method="bfill")




# Separate data for gyro and compass
gyro_data = data[data['Type'] == 'Gyro'].reset_index(drop=True)
compass_data = data[data['Type'] == 'Compass'].reset_index(drop=True)

# Rename 'value_1' column to 'Magnetic_Field_Magnitude' in compass data
compass_data.rename(columns={'value_1': 'Magnetic_Field_Magnitude'}, inplace=True)
compass_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
compass_data.rename(columns={'value_3': 'compass'}, inplace=True)

gyro_data.rename(columns={'value_1': 'axisZAngle'}, inplace=True)
gyro_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
gyro_data.rename(columns={'value_3': 'compass'}, inplace=True)


# # # Convert 'Timestamp_(ms)' to datetime format
# compass_data['Timestamp_(ms)'] = pd.to_datetime(compass_data['Timestamp_(ms)'], unit='ms')

# # # Convert datetime to seconds
# compass_data['Time_seconds'] = compass_data['Timestamp_(ms)'].dt.second + compass_data['Timestamp_(ms)'].dt.minute * 60 + compass_data['Timestamp_(ms)'].dt.hour * 3600
# # Find the minimum timestamp and subtract it from all timestamps to start from 0 seconds
# min_timestamp = compass_data['Timestamp_(ms)'].min()
# compass_data['Time_seconds'] = (compass_data['Timestamp_(ms)'] - min_timestamp).dt.total_seconds()


first_ground_truth = initial_location_data['GroundTruth'][0]

# Calculate the value for the new column
compass_data['GyroStartByGroundTruth'] = first_ground_truth + compass_data['gyroSumFromstart0']-compass_data['gyroSumFromstart0'][0]

compass_data['GyroStartByGroundTruth']=(compass_data['GyroStartByGroundTruth'] + 360) % 360


gyro_data['GyroStartByGroundTruth'] = first_ground_truth + gyro_data['gyroSumFromstart0']-gyro_data['gyroSumFromstart0'][0]

gyro_data['GyroStartByGroundTruth']=(gyro_data['GyroStartByGroundTruth'] + 360) % 360

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
plt.rcParams['font.family'] = 'Times New Roman'

# Set plot parameters for IEEE format
fontSizeAll = 4
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

# Create figure for IEEE column width
fig, ax = plt.subplots(figsize=(3.45, 2), dpi=300)
plt.subplots_adjust(left=0.01, bottom=0.03, right=0.99, top=0.99, wspace=0.00, hspace=0.0)

# Ensure gyro_data is sorted by timestamp
# gyro_data = data[data["Type"] == "Gyro"].sort_values(by="Timestamp_(ms)")

# Plot GroundTruthHeadingComputed
plt.plot(gyro_data["Timestamp_(ms)"], gyro_data["GroundTruthHeadingComputed"], 
         color='red', linestyle='--', linewidth=1.2, label='GroundTruth Heading Computed')
plt.plot(gyro_data["Timestamp_(ms)"], gyro_data["GyroStartByGroundTruth"], 
         color='blue', linestyle='--', linewidth=1, label='Gyro Heading')


# Plot GyroStartByGroundTruth (assuming it's stored in 'value_1' for Gyro type rows)
# plt.plot(gyro_data["Timestamp_(ms)"], gyro_data["value_1"], 
#          color='red', linestyle='-', linewidth=1.2, label='Gyro Start By GroundTruth')

# Axis formatting
# ax.yaxis.set_major_locator(MultipleLocator(40))  # Y-axis major tick interval: 40 degrees
# ax.yaxis.set_minor_locator(MultipleLocator(20))  # Y-axis minor tick interval: 20 degrees
# ax.xaxis.set_major_locator(MultipleLocator(5000))  # X-axis major tick interval: 5000ms
# ax.xaxis.set_minor_locator(MultipleLocator(2500))  # X-axis minor tick interval: 2500ms

plt.xlabel("Timestamp (ms)", labelpad=3)
plt.ylabel("Heading (Degrees)", labelpad=4)

# Rotate y-tick labels
plt.yticks(rotation=90, va="center")

# Ticks and grid
plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in', 
                which='major', grid_color='blue', width=0.3, length=2.5)
plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in', 
                which='minor', grid_color='blue', width=0.15, length=1)

# Custom Legend
legend_elements = [
    Line2D([0], [0], color='red', linestyle='--', linewidth=1.2, label='Ground Truth Heading'),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=1.2, label='Gyro Heading')
]

plt.legend(handles=legend_elements, loc='best')

# Grid
ax.ticklabel_format(useOffset=False)
plt.grid(linestyle=':', linewidth=0.5, alpha=0.15, color='k')

# Show plot
plt.show()
