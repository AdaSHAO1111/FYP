import pandas as pd
import os

# Set output directory
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2'
os.makedirs(output_dir, exist_ok=True)

# Input file
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

# Ensure data is sorted by timestamp
data.sort_values(by="Timestamp_(ms)", inplace=True)

# Convert timestamps to relative time in seconds from the start
min_timestamp = data['Timestamp_(ms)'].min()
data['Time_relative'] = (data['Timestamp_(ms)'] - min_timestamp) / 1000  # Convert to seconds

# Extract compass data only
compass_data = data[data['Type'] == 'Compass'].reset_index(drop=True)

# Rename columns for clarity if needed
compass_data.rename(columns={'value_3': 'compass'}, inplace=True)

# Select only the relevant columns for the output
compass_heading_data = compass_data[['Time_relative', 'compass']]

# Save to CSV
csv_output_file = os.path.join(output_dir, 'compass_heading_data.csv')
compass_heading_data.to_csv(csv_output_file, index=False)
print(f"Saved compass heading data to {csv_output_file}")

print("Data extraction completed successfully") 