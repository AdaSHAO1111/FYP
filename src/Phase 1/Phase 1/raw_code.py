import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family'] = 'Times New Roman'

# File path
file_path = "/Users/shaoxinyi/Downloads/FYP2/Data_collected/1536_CompassGyroSumHeadingData.txt" # works

# Read the CSV file into a DataFrame
collected_data = pd.read_csv(file_path, delimiter=';')


# Find the index of the first occurrence of 'Initial_Location'
initial_location_index = collected_data[collected_data['Type'] == 'Initial_Location'].index[0]

# Slice the DataFrame from the first occurrence onwards
data = collected_data.iloc[initial_location_index:].reset_index(drop=True)

# Now 'filtered_data' contains the data recorded after the first 'Initial_Location'
first_rows_unique_step = data.groupby('step').first().reset_index()

initial_location_data = data[data['Type'] == 'Initial_Location'].reset_index(drop=True)

initial_position = (initial_location_data['value_4'][0],initial_location_data['value_5'][0])

ground_truth_location_data = data[(data['Type'] == 'Ground_truth_Location') | (data['Type'] == 'Initial_Location')].reset_index(drop=True)

# Sort the DataFrame by the 'step' column
ground_truth_location_data.sort_values(by='step', inplace=True)

# Drop duplicates based on the 'step' column, keeping the last occurrence
ground_truth_location_data.drop_duplicates(subset='step', keep='last', inplace=True)

# Reset the index after dropping duplicates
ground_truth_location_data.reset_index(drop=True, inplace=True)

import pandas as pd
import numpy as np

# Function to compute azimuth (bearing) between two coordinates
def calculate_initial_compass_bearing(lat1, lon1, lat2, lon2):
    """
    Calculates the bearing between two points on the earth.
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


df_gt = ground_truth_location_data

# Compute azimuth (bearing) using 'east' as longitude and 'north' as latitude
df_gt["GroundTruthHeadingComputed"] = np.nan  # Initialize column

for i in range(1, len(df_gt)):
    df_gt.loc[i, "GroundTruthHeadingComputed"] = calculate_initial_compass_bearing(
        df_gt.loc[i-1, "value_5"], df_gt.loc[i-1, "value_4"],
        df_gt.loc[i, "value_5"], df_gt.loc[i, "value_4"]
    )
df_gt