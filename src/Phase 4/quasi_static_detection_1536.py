import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.stats import iqr

# Set paths with full paths to avoid path issues
compass_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_compass_data.csv'
gyro_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_gyro_data.csv'
ground_truth_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_ground_truth_data.csv'
output_folder = '/Users/shaoxinyi/Downloads/FYP2/Init'

# Parameters - Updated based on the raw_code4.py settings but adjusted for this dataset
threshold_step_difference = 0  # Step difference threshold
stability_threshold = 100  # Increased threshold to allow more variance in compass readings
window_size = 20  # Reduced window size to make detection more sensitive

# Load data
compass_data = pd.read_csv(compass_data_path)
gyro_data = pd.read_csv(gyro_data_path)
ground_truth_data = pd.read_csv(ground_truth_path)

# Filter data if needed - using all available data for this example
compass_data_final = compass_data.copy()
filtered_data_magnetic = compass_data_final
filtered_data_gyro = gyro_data

# Merge ground truth data with compass data to get true headings
# First, create a merged dataset for analysis
ground_truth_data_with_heading = ground_truth_data.dropna(subset=['GroundTruthHeadingComputed'])
# Map ground truth heading to compass readings based on closest timestamp
true_headings = []

for timestamp in filtered_data_magnetic['Timestamp_(ms)']:
    # Find the closest ground truth timestamp
    if len(ground_truth_data_with_heading) > 0:
        closest_idx = (ground_truth_data_with_heading['Timestamp_(ms)'] - timestamp).abs().idxmin()
        true_headings.append(ground_truth_data_with_heading.loc[closest_idx, 'GroundTruthHeadingComputed'])
    else:
        true_headings.append(np.nan)

# Extract relevant columns for QS detection
timestamps = filtered_data_magnetic['Timestamp_(ms)']
compass_headings = filtered_data_magnetic['value_2']  # Compass heading is in value_2
stepss = filtered_data_magnetic['step']
Floors = filtered_data_magnetic['value_4']
Eastings = filtered_data_magnetic['value_1']  # Using value_1 as Easting
Northings = filtered_data_magnetic['value_3']  # Using value_3 as Northing

# QuasiStaticHeadingTracker class from raw_code4.py
class QuasiStaticHeadingTracker:
    def __init__(self, stability_threshold=5.0, window_size=100):
        self.stability_threshold = stability_threshold
        self.window_size = window_size
        self.compass_heading_window = []

    def add_compass_heading(self, compass_heading):
        self.compass_heading_window.append(compass_heading)
        if len(self.compass_heading_window) > self.window_size:
            self.compass_heading_window.pop(0)  # Remove oldest value to maintain window size

    def is_quasi_static_interval(self):
        if len(self.compass_heading_window) >= self.window_size:
            variance = np.var(self.compass_heading_window)
            return variance < self.stability_threshold
        return False

    def calculate_mean(self):
        if self.compass_heading_window:
            return np.mean(self.compass_heading_window)
        return None

# Create an instance of the QuasiStaticHeadingTracker
tracker = QuasiStaticHeadingTracker(stability_threshold=stability_threshold, window_size=window_size)

# Initialize lists to store data points where the quasi-static interval is detected
quasi_static_intervals = []
quasi_static_headings = []
quasi_static_steps = []
quasi_static_floors = []
quasi_static_east = []
quasi_static_north = []

# Initialize a flag to track the state of the interval
is_quasi_static_interval = False
num_quasi_static_intervals = 0

# Initialize lists to store data for DataFrame
data_QS = {
    'Quasi_Static_Interval_Number': [],
    'Compass_Heading': [],
    'True_Heading': [],
    'Time': [],
    'Step': [],
    'Floor': [],
    'east': [],
    'north': []
}

# Iterate through the compass headings and track quasi-static intervals
start_step = None  # Variable to hold the start step of the interval
temp_data = {'Time': [], 'Compass_Heading': [], 'True_Heading': [], 'Step': [], 'Floor': [], 'east': [], 'north': []}

for timestamp, heading, true, step, floor, east, north in zip(timestamps, compass_headings, true_headings, stepss, Floors, Eastings, Northings):
    tracker.add_compass_heading(heading)

    if tracker.is_quasi_static_interval():
        if start_step is None:
            start_step = step  # Record the start step of the interval

        # Append data to temporary arrays
        temp_data['Time'].append(timestamp)
        temp_data['Compass_Heading'].append(heading)
        temp_data['True_Heading'].append(true)
        temp_data['Step'].append(step)
        temp_data['Floor'].append(floor)
        temp_data['east'].append(east)
        temp_data['north'].append(north)

        # If it's a quasi-static interval and we're not currently in one, increment the counter
        if not is_quasi_static_interval:
            num_quasi_static_intervals += 1
            is_quasi_static_interval = True

    else:
        if is_quasi_static_interval:
            # Calculate the step difference at the end of the interval
            if start_step is not None:
                step_difference = step - start_step

                # Check if the step difference exceeds the threshold
                if step_difference >= threshold_step_difference:
                    # Add the temporary data to data_QS
                    data_QS['Quasi_Static_Interval_Number'].extend([num_quasi_static_intervals] * len(temp_data['Time']))
                    data_QS['Compass_Heading'].extend(temp_data['Compass_Heading'])
                    data_QS['True_Heading'].extend(temp_data['True_Heading'])
                    data_QS['Time'].extend(temp_data['Time'])
                    data_QS['Step'].extend(temp_data['Step'])
                    data_QS['Floor'].extend(temp_data['Floor'])
                    data_QS['east'].extend(temp_data['east'])
                    data_QS['north'].extend(temp_data['north'])
                else:
                    num_quasi_static_intervals -= 1

        is_quasi_static_interval = False
        start_step = None  # Reset the start step for the next interval
        temp_data = {'Time': [], 'Compass_Heading': [], 'True_Heading': [], 'Step': [], 'Floor': [], 'east': [], 'north': []}  # Reset temporary arrays

# Print the total number of detected quasi-static intervals
print("Number of detected quasi-static intervals:", num_quasi_static_intervals)

# Create DataFrame
quasi_static_data = pd.DataFrame(data_QS)

# Define a function to remove outliers using IQR
def remove_outliers(group):
    if len(group) <= 1:
        return group
    q1 = group.quantile(0.25)
    q3 = group.quantile(0.75)
    iqr_val = q3 - q1
    lower_bound = q1 - 1.5 * iqr_val
    upper_bound = q3 + 1.5 * iqr_val
    return group[(group >= lower_bound) & (group <= upper_bound)]

# Apply the function to each group and calculate the mean after removing outliers
def calculate_mean_without_outliers(group):
    if len(group) <= 1:
        return group.mean()
    return remove_outliers(group).mean()

# Group by 'Quasi_Static_Interval_Number' and calculate the required statistics
averages = quasi_static_data.groupby('Quasi_Static_Interval_Number').agg({
    'Compass_Heading': ['mean', calculate_mean_without_outliers],
    'True_Heading': ['mean', calculate_mean_without_outliers],
}).reset_index()

# Rename columns for better understanding
averages.columns = ['Quasi_Static_Interval_Number', 
                    'Compass_Heading', 'Compass_Heading_Mean_No_Outliers',
                    'True_Heading', 'True_Heading_Mean_No_Outliers']

# Calculate absolute difference between Compass_Heading and True_Heading
averages['Abs_Difference'] = abs(averages['Compass_Heading'] - averages['True_Heading'])

# Calculate the average of Abs_Difference
average_abs_difference = averages['Abs_Difference'].mean()
print("Average of Abs_Difference:", average_abs_difference)

# Calculate a new column for Abs_Difference_From_Avg
averages['Abs_Difference_From_Avg'] = abs(averages['Abs_Difference'] - average_abs_difference)

# Save results to CSV
quasi_static_data.to_csv(os.path.join(output_folder, '1536_quasi_static_data.csv'), index=False)
averages.to_csv(os.path.join(output_folder, '1536_quasi_static_averages.csv'), index=False)

# Create visualization of the data
plt.figure(figsize=(12, 6))

plt.plot(timestamps, compass_headings, label='Compass Headings', color='cyan', alpha=0.6)

# Create colormap
if num_quasi_static_intervals > 0:
    cmap = plt.cm.get_cmap('Set1', num_quasi_static_intervals)  # Using Set1 colormap as in raw_code4.py
    plt.scatter(quasi_static_data['Time'], quasi_static_data['Compass_Heading'],
               c=quasi_static_data['Quasi_Static_Interval_Number'], cmap=cmap, 
               s=50, zorder=5, label='Quasi-Static Intervals')

    # Plot true headings for context
    plt.plot(np.array(timestamps), true_headings, 
             marker='.', linestyle='-', markersize=5, color='blue', alpha=0.5, label='True_Heading')

plt.xlabel('Time (ms)')
plt.ylabel('Compass Headings (degrees)')
plt.title('Compass Headings over Time with Quasi-Static Intervals (1536 Data)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(output_folder, '1536_compass_with_QS_intervals.png'), dpi=300)
plt.close()

# Plot step number over time
plt.figure(figsize=(12, 6))
plt.plot(timestamps, stepss, label='Steps', color='cyan', alpha=0.6)

if num_quasi_static_intervals > 0:
    plt.scatter(quasi_static_data['Time'], quasi_static_data['Step'],
                c=quasi_static_data['Quasi_Static_Interval_Number'], cmap=cmap, 
                s=50, zorder=5, label='Quasi-Static Intervals')

plt.xlabel('Time (ms)')
plt.ylabel('Step Number')
plt.title('Step Number over Time with Quasi-Static Intervals (1536 Data)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '1536_steps_with_QS_intervals.png'), dpi=300)
plt.close()

# Plot locations with QS intervals
plt.figure(figsize=(12, 6))
plt.plot(Eastings, Northings, label='All Locations', color='cyan', alpha=0.6)

if num_quasi_static_intervals > 0:
    plt.scatter(quasi_static_data['east'], quasi_static_data['north'],
                c=quasi_static_data['Quasi_Static_Interval_Number'], cmap=cmap, 
                s=50, zorder=5, label='Quasi-Static Intervals')

plt.xlabel('East')
plt.ylabel('North')
plt.title('Locations with Quasi-Static Intervals (1536 Data)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '1536_locations_with_QS_intervals.png'), dpi=300)
plt.close()

# 3D plot of locations with floor info
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory line
ax.plot(Eastings, Northings, Floors, label='All Locations', color='cyan', alpha=0.6)

# Plot QS intervals with more emphasis
if num_quasi_static_intervals > 0:
    ax.scatter(quasi_static_data['east'], quasi_static_data['north'], quasi_static_data['Floor'],
              c=quasi_static_data['Quasi_Static_Interval_Number'], s=50, cmap=cmap, 
              zorder=10, label='Quasi-Static Intervals')

ax.set_xlabel('East')
ax.set_ylabel('North')
ax.set_zlabel('Floor')
ax.set_title('3D Locations with Quasi-Static Intervals (1536 Data)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '1536_3D_locations_with_QS_intervals.png'), dpi=300)
plt.close()

# Floor plot
plt.figure(figsize=(12, 6))
plt.plot(timestamps, Floors, label='Floor', color='cyan', alpha=0.6)

if num_quasi_static_intervals > 0:
    plt.scatter(quasi_static_data['Time'], quasi_static_data['Floor'],
                c=quasi_static_data['Quasi_Static_Interval_Number'], cmap=cmap, 
                s=50, zorder=5, label='Quasi-Static Intervals')

plt.xlabel('Time (ms)')
plt.ylabel('Floor')
plt.title('Floor over Time with Quasi-Static Intervals (1536 Data)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '1536_floor_with_QS_intervals.png'), dpi=300)
plt.close()

# Print the final table of QS intervals
print("\nQuasi-Static Interval Table:")
table_df = averages[['Quasi_Static_Interval_Number', 'Compass_Heading', 'True_Heading', 'Abs_Difference', 'Abs_Difference_From_Avg']]
print(table_df)

# Save table for display
table_df.to_csv(os.path.join(output_folder, '1536_quasi_static_table.csv'), index=False)

# Create a visualization of the table if we have data
if num_quasi_static_intervals > 0:
    plt.figure(figsize=(14, len(table_df) * 0.4 + 2))
    plt.axis('off')
    
    # Create table cell colors with the correct shape
    header_colors = ['#f8f9fa'] * len(table_df.columns)
    cell_colors = [header_colors]
    for _ in range(len(table_df)):
        cell_colors.append(['#FFFFFF'] * len(table_df.columns))
    
    # Format table data for display
    cell_text = [table_df.columns.values.tolist()]
    for _, row in table_df.iterrows():
        # Format numeric values to 2 decimal places
        row_data = [str(row['Quasi_Static_Interval_Number'])]
        for val in [row['Compass_Heading'], row['True_Heading'], row['Abs_Difference'], row['Abs_Difference_From_Avg']]:
            row_data.append(f"{val:.2f}")
        cell_text.append(row_data)
    
    # Draw the table
    plt.table(
        cellText=cell_text,
        cellColours=cell_colors,
        loc='center',
        cellLoc='center',
        bbox=[0.1, 0.1, 0.8, 0.8]
    )
    plt.title('Quasi-Static Interval Summary Table')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '1536_QS_intervals_table.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(f"All results saved to: {output_folder}") 