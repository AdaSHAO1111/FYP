import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from math import atan2, degrees, radians, sin, cos

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

# Rename columns for clarity
gyro_data.rename(columns={'value_1': 'axisZAngle'}, inplace=True)
gyro_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
gyro_data.rename(columns={'value_3': 'compass'}, inplace=True)

# Get the initial ground truth heading
first_ground_truth = initial_location_data['GroundTruth'].iloc[0] if len(initial_location_data) > 0 else 0

# Calculate the gyro heading starting from ground truth
gyro_data['GyroStartByGroundTruth'] = first_ground_truth + gyro_data['gyroSumFromstart0'] - gyro_data['gyroSumFromstart0'].iloc[0]
gyro_data['GyroStartByGroundTruth'] = (gyro_data['GyroStartByGroundTruth'] + 360) % 360

# Convert timestamps to relative time in seconds from the start
min_timestamp = gyro_data['Timestamp_(ms)'].min()
gyro_data['Time_relative'] = (gyro_data['Timestamp_(ms)'] - min_timestamp) / 1000  # Convert to seconds

# Create a dataframe with the necessary columns for the table
gyro_heading_table = gyro_data[['Time_relative', 'GyroStartByGroundTruth', 'GroundTruthHeadingComputed']].copy()
gyro_heading_table.columns = ['Time (s)', 'Gyro Heading (°)', 'Ground Truth Heading (°)']

# Round values for better readability
gyro_heading_table = gyro_heading_table.round({'Time (s)': 2, 'Gyro Heading (°)': 2, 'Ground Truth Heading (°)': 2})

# Calculate heading error
gyro_heading_table['Heading Error (°)'] = abs(gyro_heading_table['Gyro Heading (°)'] - gyro_heading_table['Ground Truth Heading (°)'])
# Handle cases where the error is greater than 180 degrees (shortest angle)
gyro_heading_table['Heading Error (°)'] = gyro_heading_table['Heading Error (°)'].apply(lambda x: min(x, 360-x))
gyro_heading_table['Heading Error (°)'] = gyro_heading_table['Heading Error (°)'].round(2)

# Save as CSV
csv_file = os.path.join(output_dir, 'gyro_heading_data.csv')
gyro_heading_table.to_csv(csv_file, index=False)
print(f"Saved Gyro Heading data as CSV to {csv_file}")

# Create a more readable table with a smaller subset of the data
# Select a subset of rows to keep the table manageable
# For example, take 20 evenly distributed samples
sample_size = min(20, len(gyro_heading_table))
step = len(gyro_heading_table) // sample_size
sampled_table = gyro_heading_table.iloc[::step].head(sample_size).reset_index(drop=True)

# Save as Markdown
md_file = os.path.join(output_dir, 'gyro_heading_data.md')
with open(md_file, 'w') as f:
    f.write('# Gyro Heading Data Analysis\n\n')
    f.write('This table shows a sample of gyro heading measurements compared to ground truth.\n\n')
    
    # Write summary statistics
    f.write('## Summary Statistics\n\n')
    f.write(f"- **Total measurements**: {len(gyro_heading_table)}\n")
    f.write(f"- **Time range**: {gyro_heading_table['Time (s)'].min():.2f}s to {gyro_heading_table['Time (s)'].max():.2f}s\n")
    f.write(f"- **Average error**: {gyro_heading_table['Heading Error (°)'].mean():.2f}°\n")
    f.write(f"- **Maximum error**: {gyro_heading_table['Heading Error (°)'].max():.2f}°\n")
    f.write(f"- **Minimum error**: {gyro_heading_table['Heading Error (°)'].min():.2f}°\n\n")
    
    # Write table header
    f.write('## Sample Data\n\n')
    f.write('| Time (s) | Gyro Heading (°) | Ground Truth Heading (°) | Heading Error (°) |\n')
    f.write('|----------|------------------|-------------------------|------------------|\n')
    
    # Write table rows
    for _, row in sampled_table.iterrows():
        f.write(f"| {row['Time (s)']:.2f} | {row['Gyro Heading (°)']:.2f} | {row['Ground Truth Heading (°)']:.2f} | {row['Heading Error (°)']:.2f} |\n")

print(f"Saved Gyro Heading data as Markdown to {md_file}")

# Also save a version that shows all data points (but with pagination for readability)
md_full_file = os.path.join(output_dir, 'gyro_heading_data_full.md')
with open(md_full_file, 'w') as f:
    f.write('# Complete Gyro Heading Data\n\n')
    
    # Write summary statistics
    f.write('## Summary Statistics\n\n')
    f.write(f"- **Total measurements**: {len(gyro_heading_table)}\n")
    f.write(f"- **Time range**: {gyro_heading_table['Time (s)'].min():.2f}s to {gyro_heading_table['Time (s)'].max():.2f}s\n")
    f.write(f"- **Average error**: {gyro_heading_table['Heading Error (°)'].mean():.2f}°\n")
    f.write(f"- **Maximum error**: {gyro_heading_table['Heading Error (°)'].max():.2f}°\n")
    f.write(f"- **Minimum error**: {gyro_heading_table['Heading Error (°)'].min():.2f}°\n\n")
    
    # Split the data into pages for better readability
    rows_per_page = 50
    num_pages = (len(gyro_heading_table) + rows_per_page - 1) // rows_per_page
    
    for page in range(num_pages):
        start_idx = page * rows_per_page
        end_idx = min((page + 1) * rows_per_page, len(gyro_heading_table))
        page_data = gyro_heading_table.iloc[start_idx:end_idx]
        
        f.write(f'## Page {page+1} (rows {start_idx+1}-{end_idx})\n\n')
        f.write('| Time (s) | Gyro Heading (°) | Ground Truth Heading (°) | Heading Error (°) |\n')
        f.write('|----------|------------------|-------------------------|------------------|\n')
        
        for _, row in page_data.iterrows():
            f.write(f"| {row['Time (s)']:.2f} | {row['Gyro Heading (°)']:.2f} | {row['Ground Truth Heading (°)']:.2f} | {row['Heading Error (°)']:.2f} |\n")
        
        f.write('\n')

print(f"Saved complete Gyro Heading data as Markdown to {md_full_file}")
print("Gyro Heading data extraction and table generation complete!") 