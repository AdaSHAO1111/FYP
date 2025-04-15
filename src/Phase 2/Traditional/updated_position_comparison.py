import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import os

# 设置输出目录
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2'
os.makedirs(output_dir, exist_ok=True)

# 输入文件
input_file = '/Users/shaoxinyi/Downloads/FYP2/Data_collected/1536_CompassGyroSumHeadingData.txt'
ground_truth_file = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2/Sensor_Traditional/ground_truth_positions_steps.csv'

print(f"Loading data from {input_file}...")
try:
    # Load the data with semicolon delimiter
    data = pd.read_csv(input_file, delimiter=';')
    print(f"Successfully loaded file")
    
    # Check if the dataframe has header rows
    if 'Type' not in data.columns:
        # File doesn't have headers, try to infer them
        column_names = [
            'Timestamp_(ms)', 'Type', 'step', 'value_1', 'value_2', 'value_3', 
            'GroundTruth', 'value_4', 'value_5', 'turns'
        ]
        
        # Try again with column names
        data = pd.read_csv(input_file, delimiter=';', names=column_names)
        
        # If first row contains header values, remove it
        if data.iloc[0]['Type'] == 'Type':
            data = data.iloc[1:].reset_index(drop=True)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Load ground truth data from CSV
print(f"Loading ground truth positions from {ground_truth_file}...")
try:
    ground_truth_data = pd.read_csv(ground_truth_file)
    print(f"Successfully loaded ground truth positions with {len(ground_truth_data)} rows")
except Exception as e:
    print(f"Error loading ground truth data: {e}")
    exit(1)

# Extract Ground Truth location data from the original data
ground_truth_location_data = data[data['Type'] == 'Ground_truth_Location'].copy()
initial_location_data = data[data['Type'] == 'Initial_Location'].copy()

# Create a separate dataframe for Ground Truth heading
if len(ground_truth_location_data) > 0 and len(initial_location_data) > 0:
    df_gt = pd.concat([initial_location_data, ground_truth_location_data], ignore_index=True)
    df_gt.sort_values(by='Timestamp_(ms)', inplace=True)
    df_gt.reset_index(drop=True, inplace=True)
else:
    print("Missing Ground Truth or initial location data")
    exit(1)

# Convert numeric columns to float
for col in ['value_1', 'value_2', 'value_3', 'GroundTruth', 'value_4', 'value_5', 'step']:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Separate gyro and compass data
gyro_data = data[data['Type'] == 'Gyro'].reset_index(drop=True)
compass_data = data[data['Type'] == 'Compass'].reset_index(drop=True)

# Rename columns for clarity
compass_data.rename(columns={'value_1': 'Magnetic_Field_Magnitude'}, inplace=True)
compass_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
compass_data.rename(columns={'value_3': 'compass'}, inplace=True)

gyro_data.rename(columns={'value_1': 'axisZAngle'}, inplace=True)
gyro_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
gyro_data.rename(columns={'value_3': 'compass'}, inplace=True)

# Get initial Ground Truth heading
first_ground_truth = initial_location_data['GroundTruth'].iloc[0] if len(initial_location_data) > 0 else 0

# Calculate gyro heading starting from Ground Truth
compass_data['GyroStartByGroundTruth'] = first_ground_truth + compass_data['gyroSumFromstart0'] - compass_data['gyroSumFromstart0'].iloc[0]
compass_data['GyroStartByGroundTruth'] = (compass_data['GyroStartByGroundTruth'] + 360) % 360

gyro_data['GyroStartByGroundTruth'] = first_ground_truth + gyro_data['gyroSumFromstart0'] - gyro_data['gyroSumFromstart0'].iloc[0]
gyro_data['GyroStartByGroundTruth'] = (gyro_data['GyroStartByGroundTruth'] + 360) % 360

print("Data preprocessing complete")

# Calculate positions
print("Calculating position trajectories...")

# Set step length and initial position
step_length = 0.66  # Step length in meters
initial_position = (0, 0)  # Initial position coordinates (x, y)

# Extract new ground truth positions
def extract_ground_truth_from_csv(gt_data):
    """Extract positions from the ground truth CSV file"""
    # Extract coordinates into a list of tuples (x, y) - scaled to match the original plot coordinates
    gt_positions = []
    
    # Scale the coordinate system
    min_east = gt_data['ground_x'].min()
    min_north = gt_data['ground_y'].min()
    
    # Calculate scaled positions relative to the starting point
    for i in range(len(gt_data)):
        x = gt_data['ground_x'].iloc[i] - min_east
        y = gt_data['ground_y'].iloc[i] - min_north
        # Scale down to match the original plot scale (meters)
        x_scaled = (x - gt_data['ground_x'].iloc[0]) / 20
        y_scaled = (y - gt_data['ground_y'].iloc[0]) / 20
        gt_positions.append((x_scaled, y_scaled))
    
    return gt_positions

# Calculate positions based on step count and heading
def calculate_positions(data, heading_column, step_length=0.66, initial_position=(0, 0)):
    positions = [initial_position]  # Start from initial position
    current_position = initial_position
    prev_step = data['step'].iloc[0]
    
    for i in range(1, len(data)):
        # Calculate change in step count
        change_in_step = data['step'].iloc[i] - prev_step
        
        # If step count changes, calculate new position
        if change_in_step != 0:
            # Calculate distance change
            change_in_distance = change_in_step * step_length
            
            # Get heading value (note heading angle: 0 degrees is North, 90 degrees is East)
            heading = data[heading_column].iloc[i]
            
            # Calculate new position (East is x-axis, North is y-axis)
            new_x = current_position[0] + change_in_distance * np.sin(np.radians(heading))
            new_y = current_position[1] + change_in_distance * np.cos(np.radians(heading))
            
            # Update current position
            current_position = (new_x, new_y)
            positions.append(current_position)
            
            # Update previous step count
            prev_step = data['step'].iloc[i]
    
    return positions

# Get new Ground Truth positions from the CSV file
gt_positions = extract_ground_truth_from_csv(ground_truth_data)
print(f"Ground Truth coordinates point count: {len(gt_positions)}")

# Calculate positions using traditional methods
print("Calculating traditional method positions...")
positions_compass_trad = calculate_positions(compass_data, 'compass', step_length, initial_position)
positions_gyro_trad = calculate_positions(compass_data, 'GyroStartByGroundTruth', step_length, initial_position)
print(f"Compass trajectory point count: {len(positions_compass_trad)}")
print(f"Gyro trajectory point count: {len(positions_gyro_trad)}")

# Extract coordinates
x_gt = [pos[0] for pos in gt_positions]
y_gt = [pos[1] for pos in gt_positions]

x_compass_trad = [pos[0] for pos in positions_compass_trad]
y_compass_trad = [pos[1] for pos in positions_compass_trad]

x_gyro_trad = [pos[0] for pos in positions_gyro_trad]
y_gyro_trad = [pos[1] for pos in positions_gyro_trad]

# Visualize position trajectories
print("Generating position trajectory comparison plot...")

# Set plot parameters
fontSizeAll = 12
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

# Position trajectory plot using traditional methods
fig1, ax1 = plt.subplots(figsize=(8, 8), dpi=300)
plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.92)

# Plot Ground Truth trajectory
plt.plot(x_gt, y_gt, color='blue', linestyle='-', linewidth=1.5, label='Ground Truth')

# Plot traditional compass trajectory
plt.plot(x_compass_trad, y_compass_trad, color='green', linestyle='--', linewidth=1, label='Compass (Traditional)')

# Plot traditional gyro trajectory
plt.plot(x_gyro_trad, y_gyro_trad, color='red', linestyle='-.', linewidth=1, label='Gyro (Traditional)')

# Mark start and end points
plt.scatter(x_gt[0], y_gt[0], color='black', marker='o', s=80, label='Start')
plt.scatter(x_gt[-1], y_gt[-1], color='black', marker='x', s=80, label='End')

# Format axes
ax1.set_aspect('equal')
ax1.set_xlabel('East (m)', labelpad=5)
ax1.set_ylabel('North (m)', labelpad=5)
ax1.grid(True, linestyle=':', alpha=0.5)

# Add title
plt.title('Position Trajectories Comparison (Traditional Methods)', fontsize=fontSizeAll+2)

# Add legend
plt.legend(loc='best')

# Save image
updated_position_file = os.path.join(output_dir, 'updated_position_comparison.png')
plt.savefig(updated_position_file, bbox_inches='tight')
print(f"Position comparison plot saved to: {updated_position_file}")

print("Position trajectory comparison analysis complete") 

# 保存Traditional方法计算的Gyro Positioning数据
print("Saving Gyro positioning data from traditional method...")
gyro_positions_df = pd.DataFrame({
    'East_m': x_gyro_trad,
    'North_m': y_gyro_trad
})

# 添加时间戳和步数信息（如果需要的话，从原始数据中提取对应的步数变化点）
steps_used = []
timestamps_used = []
prev_step = compass_data['step'].iloc[0]
steps_used.append(prev_step)
timestamps_used.append(compass_data['Timestamp_(ms)'].iloc[0])

for i in range(1, len(compass_data)):
    if compass_data['step'].iloc[i] != prev_step:
        steps_used.append(compass_data['step'].iloc[i])
        timestamps_used.append(compass_data['Timestamp_(ms)'].iloc[i])
        prev_step = compass_data['step'].iloc[i]

# 确保数据长度匹配
if len(steps_used) > len(gyro_positions_df):
    steps_used = steps_used[:len(gyro_positions_df)]
    timestamps_used = timestamps_used[:len(gyro_positions_df)]
elif len(steps_used) < len(gyro_positions_df):
    # 补足缺少的步数和时间戳（使用最后一个值）
    while len(steps_used) < len(gyro_positions_df):
        steps_used.append(steps_used[-1])
        timestamps_used.append(timestamps_used[-1])

# 添加步数和时间戳到DataFrame
gyro_positions_df['Step'] = steps_used
gyro_positions_df['Timestamp_ms'] = timestamps_used

# 添加航向信息
headings = []
prev_step = compass_data['step'].iloc[0]
for i in range(len(compass_data)):
    if compass_data['step'].iloc[i] != prev_step:
        headings.append(compass_data['GyroStartByGroundTruth'].iloc[i])
        prev_step = compass_data['step'].iloc[i]

# 确保航向数据长度匹配
if len(headings) > len(gyro_positions_df) - 1:  # 减1是因为第一个点没有航向
    headings = headings[:len(gyro_positions_df) - 1]
elif len(headings) < len(gyro_positions_df) - 1:
    # 补足缺少的航向（使用最后一个值）
    while len(headings) < len(gyro_positions_df) - 1:
        headings.append(headings[-1] if headings else 0)

# 第一个点没有航向，设为0或NaN
headings.insert(0, np.nan)
gyro_positions_df['Heading_Degrees'] = headings

# 保存到CSV文件
gyro_positions_file = os.path.join(output_dir, 'traditional_gyro_positions.csv')
gyro_positions_df.to_csv(gyro_positions_file, index=False)
print(f"Traditional Gyro positioning data saved to: {gyro_positions_file}") 