import pandas as pd
import os

# Define the path to the files
base_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/'

# Process compass data
print("Processing compass data...")
compass_file = os.path.join(base_path, '1536_cleaned_compass_data.csv')
compass_data = pd.read_csv(compass_file)
compass_data.rename(columns={
    'value_1': 'Magnetic_Field_Magnitude',
    'value_2': 'gyroSumFromstart0',
    'value_3': 'compass'
}, inplace=True)
compass_data.to_csv(compass_file, index=False)
print("Compass data processed.")

# Process gyro data
print("Processing gyro data...")
gyro_file = os.path.join(base_path, '1536_cleaned_gyro_data.csv')
gyro_data = pd.read_csv(gyro_file)
gyro_data.rename(columns={
    'value_1': 'axisZAngle',
    'value_2': 'gyroSumFromstart0',
    'value_3': 'compass'
}, inplace=True)
gyro_data.to_csv(gyro_file, index=False)
print("Gyro data processed.")

# Process ground truth data (check if we need to rename anything here)
print("Processing ground truth data...")
ground_truth_file = os.path.join(base_path, '1536_cleaned_ground_truth_data.csv')
ground_truth_data = pd.read_csv(ground_truth_file)
# The user didn't specify any renaming for ground truth data, so we'll keep it as is
ground_truth_data.to_csv(ground_truth_file, index=False)
print("Ground truth data processed.")

print("All files processed successfully.") 