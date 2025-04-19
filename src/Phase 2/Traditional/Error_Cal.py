# Drop duplicate steps
unique_steps_data = compass_data.drop_duplicates(subset=['step']).reset_index(drop=True)

# Concatenate positions DataFrames with unique_steps_data
unique_steps_data = pd.concat([unique_steps_data, df_positions_compass, df_positions_gt, df_positions_gyro], axis=1)

# Display the updated DataFrame
unique_steps_data

# Select only the 'step' and 'value_4' (x-coordinate) and 'value_5' (y-coordinate) columns
compass_gyro_positions_steps = unique_steps_data[['step', 'Gyro_X', 'Gyro_Y', 'Compass_X', 'Compass_Y']]


# Merge the two DataFrames based on the 'step' column
comapre = compass_gyro_positions_steps.merge(ground_truth_positions_steps, on='step')

# Display the combined DataFrame
comapre


# Add error columns to unique_steps_data for gyro
comapre['Gyro_Error_X'] = np.abs(comapre['Gyro_X'] - comapre['ground_x'])
comapre['Gyro_Error_Y'] = np.abs(comapre['Gyro_Y'] - comapre['ground_y'])

# Calculate the distance error between gyro and ground truth for each step
gyro_distance_error = np.sqrt((comapre['Gyro_Error_X'])**2 + (comapre['Gyro_Error_Y'])**2)

# Add distance error column to unique_steps_data for gyro
comapre['Gyro_Distance_Error'] = gyro_distance_error



# Add error columns to unique_steps_data for compass
comapre['Compass_Error_X'] = np.abs(comapre['Compass_X'] - comapre['ground_x'])
comapre['Compass_Error_Y'] = np.abs(comapre['Compass_Y'] - comapre['ground_y'])

# Calculate the distance error between compass and ground truth for each step
compass_distance_error = np.sqrt((comapre['Compass_Error_X'])**2 + (comapre['Compass_Error_Y'])**2)

# Add distance error column to unique_steps_data for compass
comapre['Compass_Distance_Error'] = compass_distance_error

# Display the updated DataFrame
comapre
comapre['Walked_distance']=comapre['step']*0.66
comapre

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.ticker import MultipleLocator

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

### **1. Plot Distance Error of Gyro vs Compass**
fig, ax1 = plt.subplots(figsize=(3.45, 2.5), dpi=1000)  # IEEE column width

ax1.plot(comapre['Walked_distance'], comapre['Gyro_Distance_Error'],
         label='Gyro', linewidth=1.2, color='blue')
ax1.plot(comapre['Walked_distance'], comapre['Compass_Distance_Error'],
         label='Compass', linewidth=1.2, color='red')

ax1.set_xlabel('Walked Distance (m)', labelpad=3)
ax1.set_ylabel('Positioning Error (m)', labelpad=3)

ax1.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='k')
ax1.legend()

# Secondary x-axis for step numbers
secx = ax1.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
secx.set_xticks(comapre['Walked_distance'][::50])
secx.set_xticklabels(comapre['step'][::50])
secx.set_xlabel('Number of Walked Steps', labelpad=8)

# Axis formatting
ax1.xaxis.set_major_locator(MultipleLocator(50))  # Major ticks every 50 meters
ax1.yaxis.set_major_locator(MultipleLocator(5))   # Major ticks every 1 meter

plt.show()


### **2. ECDF Plot**
plt.figure(figsize=(3.45, 2.5), dpi=1000)  # IEEE column width

ecdf_gyro = sm.distributions.ECDF(comapre['Gyro_Distance_Error'])
ecdf_compass = sm.distributions.ECDF(comapre['Compass_Distance_Error'])

plt.plot(ecdf_gyro.x, ecdf_gyro.y, label='Gyro', color='blue', linewidth=1.2)
plt.plot(ecdf_compass.x, ecdf_compass.y, label='Compass', color='red', linewidth=1.2)

plt.xlabel('Positioning Error (m)', labelpad=3)
plt.ylabel('ECDF', labelpad=3)

plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='k')
plt.legend()

plt.show()


### **3. Box Plot for Positioning Errors**
# Create a DataFrame for distance errors
distance_errors_df = pd.DataFrame({
    'Sensor': ['Gyro'] * len(comapre) + ['Compass'] * len(comapre),
    'Distance_Error': np.concatenate([comapre['Gyro_Distance_Error'], comapre['Compass_Distance_Error']])
})

plt.figure(figsize=(3.45, 2.5), dpi=1000)  # IEEE column width

sns.boxplot(x='Sensor', y='Distance_Error', data=distance_errors_df, linewidth=0.6)

plt.xlabel('Sensor', fontsize=5)
plt.ylabel('Positioning Error (m)', fontsize=5)

plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='k')

plt.show()
