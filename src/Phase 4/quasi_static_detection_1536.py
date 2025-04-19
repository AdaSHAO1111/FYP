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

# Load data
print("Loading data...")
compass_data = pd.read_csv(compass_data_path)
gyro_data = pd.read_csv(gyro_data_path)
ground_truth_data = pd.read_csv(ground_truth_path)

print(f"Loaded {len(compass_data)} compass data points, {len(gyro_data)} gyro data points")

# Print unique step values to understand distribution
unique_steps = sorted(compass_data['step'].unique())
print("Available step values:", unique_steps)
print(f"Min step: {min(unique_steps)}, Max step: {max(unique_steps)}")

# Analyze ground truth data
gt_steps = []
for i, row in ground_truth_data.iterrows():
    if row['Type'] in ['Initial_Location', 'Ground_truth_Location']:
        gt_step = row['step']
        gt_steps.append(gt_step)
        print(f"Ground truth point at step {gt_step}")

# Process gyro data to calculate angular velocity magnitude
gyro_data['gyro_magnitude'] = np.sqrt(
    gyro_data['value_1']**2 + 
    gyro_data['value_2']**2 + 
    gyro_data['value_3']**2
)

# Merge ground truth data with compass data to get true headings
# Map ground truth heading to compass readings based on closest timestamp
true_headings = []
for timestamp in compass_data['Timestamp_(ms)']:
    closest_idx = (ground_truth_data['Timestamp_(ms)'] - timestamp).abs().idxmin()
    true_headings.append(ground_truth_data.loc[closest_idx, 'GroundTruthHeadingComputed'] 
                         if 'GroundTruthHeadingComputed' in ground_truth_data.columns else np.nan)

compass_data['true_heading'] = true_headings

# Extract relevant columns for analysis
timestamps = compass_data['Timestamp_(ms)']
compass_headings = compass_data['value_2']  # Compass heading is in value_2
steps = compass_data['step']
floors = compass_data['value_4']
eastings = compass_data['value_1']  # Using value_1 as Easting
northings = compass_data['value_3']  # Using value_3 as Northing

# Function to get gyro magnitude for a given timestamp
def get_gyro_magnitude(timestamp):
    closest_idx = (gyro_data['Timestamp_(ms)'] - timestamp).abs().idxmin()
    if abs(gyro_data.loc[closest_idx, 'Timestamp_(ms)'] - timestamp) < 100:  # Within 100ms
        return gyro_data.loc[closest_idx, 'gyro_magnitude']
    return None

# Add gyro magnitude to compass data
compass_data['gyro_magnitude'] = compass_data['Timestamp_(ms)'].apply(get_gyro_magnitude)

# Define step ranges for ground truth points based on the actual data
# These are step ranges for each ground truth point adjusted to the actual data
GT_RANGES = {
    'GT0': (0, 5),        # Starting point (step 0)
    'GT1': (10, 20),      # First ground truth (step 16)
    'GT2': (30, 40),      # Second ground truth (step 33.5)
    'GT3': (55, 65),      # Third ground truth (step 60.5)
    'GT4': (60, 70),      # Fourth ground truth (step 64.0)
    'GT5': (75, 85),      # Fifth ground truth (step 79.0)
    'GT6': (110, 120),    # Sixth ground truth (step 112.5)
    'GT7': (150, 155)     # Final ground truth (step 154.5)
}

# Additional specific forced QS points based on the trajectory visualization and actual data
FORCE_QS_POINTS = [
    # Critical turns or positions based on actual data
    {'name': 'GT1_AREA', 'step_range': (15, 25), 'variance_mult': 2.0, 'ignore_gyro': True},
    {'name': 'GT2_AREA', 'step_range': (30, 40), 'variance_mult': 2.0, 'ignore_gyro': True},
    {'name': 'GT3_AREA', 'step_range': (55, 65), 'variance_mult': 2.0, 'ignore_gyro': True},
    {'name': 'GT6_AREA', 'step_range': (110, 120), 'variance_mult': 2.0, 'ignore_gyro': True},
    {'name': 'GT7_AREA', 'step_range': (150, 155), 'variance_mult': 2.0, 'ignore_gyro': True},
]

# Filter out points that don't have data
available_steps = set(compass_data['step'].unique())
valid_gt_ranges = {}
for gt_name, (start, end) in GT_RANGES.items():
    # Find the closest available step values
    steps_in_range = [s for s in available_steps if start <= s <= end]
    if steps_in_range:
        valid_gt_ranges[gt_name] = (min(steps_in_range), max(steps_in_range))
        print(f"Adjusted {gt_name} range to ({min(steps_in_range)}, {max(steps_in_range)})")
    else:
        print(f"Warning: No data points for {gt_name} range ({start}, {end})")

# Replace GT_RANGES with valid ranges
GT_RANGES = valid_gt_ranges

# Same for forced QS points
valid_force_points = []
for point in FORCE_QS_POINTS:
    start, end = point['step_range']
    steps_in_range = [s for s in available_steps if start <= s <= end]
    if steps_in_range:
        valid_point = point.copy()
        valid_point['step_range'] = (min(steps_in_range), max(steps_in_range))
        valid_force_points.append(valid_point)
        print(f"Adjusted {point['name']} range to ({min(steps_in_range)}, {max(steps_in_range)})")
    else:
        print(f"Warning: No data points for {point['name']} range ({start}, {end})")

# Replace FORCE_QS_POINTS with valid points
FORCE_QS_POINTS = valid_force_points

# Count points in each range to verify
print("\nPoints in each GT range:")
for name, (start, end) in GT_RANGES.items():
    count = len(compass_data[(compass_data['step'] >= start) & (compass_data['step'] <= end)])
    print(f"{name}: {count} points (steps {start}-{end})")

print("\nPoints in each forced QS range:")
for point in FORCE_QS_POINTS:
    start, end = point['step_range']
    count = len(compass_data[(compass_data['step'] >= start) & (compass_data['step'] <= end)])
    print(f"{point['name']}: {count} points (steps {start}-{end})")

# A more direct approach to QS detection focused on our specific issues
def detect_quasi_static_intervals(compass_data, window_size=15):
    """
    Detect quasi-static intervals based on gyro stability
    Focus on stable gyro periods rather than ground truth points
    """
    print("Detecting QS intervals based on gyro stability...")
    
    # Calculate moving statistics for gyro magnitude
    compass_data['gyro_roll_mean'] = compass_data['gyro_magnitude'].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    compass_data['gyro_roll_std'] = compass_data['gyro_magnitude'].rolling(window=window_size, center=True).std().fillna(method='bfill').fillna(method='ffill')
    
    # Calculate moving statistics for compass heading
    compass_data['heading_diff'] = (compass_data['value_2'].diff().abs() + compass_data['value_2'].diff(-1).abs()) / 2
    compass_data['heading_diff'] = compass_data['heading_diff'].fillna(0)
    
    # Handle 359° -> 0° transitions in heading differences
    large_diffs = compass_data['heading_diff'] > 180
    compass_data.loc[large_diffs, 'heading_diff'] = 360 - compass_data.loc[large_diffs, 'heading_diff']
    
    compass_data['heading_roll_std'] = compass_data['heading_diff'].rolling(window=window_size, center=True).std().fillna(method='bfill').fillna(method='ffill')
    
    # Sample data to determine appropriate thresholds for gyro stability
    gyro_samples = compass_data['gyro_magnitude'].dropna().sample(min(50, len(compass_data))).tolist()
    mean_gyro = np.mean(gyro_samples) if gyro_samples else 1.0
    std_gyro = np.std(gyro_samples) if len(gyro_samples) > 1 else 0.1
    
    # Sample data to determine appropriate thresholds for heading stability
    heading_std_samples = compass_data['heading_roll_std'].dropna().sample(min(50, len(compass_data))).tolist()
    mean_heading_std = np.mean(heading_std_samples) if heading_std_samples else 1.0
    
    print(f"Gyro data analysis - Mean: {mean_gyro:.2f}, Std: {std_gyro:.2f}")
    print(f"Heading stability analysis - Mean std: {mean_heading_std:.2f}")
    
    # Adaptive thresholds based on data characteristics
    gyro_stability_threshold = mean_gyro * 0.9  # Gyro magnitude should be stable
    heading_stability_threshold = mean_heading_std * 1.2  # Heading change should be minimal
    gyro_change_threshold = std_gyro * 2.0  # Detect significant changes in gyro
    
    print(f"Using adaptive thresholds - Gyro stability: {gyro_stability_threshold:.2f}, "
          f"Heading stability: {heading_stability_threshold:.2f}, "
          f"Gyro change: {gyro_change_threshold:.2f}")
    
    # Detect turns based on heading changes
    compass_data['is_turn'] = False
    for i in range(1, len(compass_data)):
        # Calculate heading change rate per step
        step_diff = compass_data.iloc[i]['step'] - compass_data.iloc[i-1]['step']
        if step_diff > 0:
            # Correctly handle angle wraparound
            angle1 = compass_data.iloc[i-1]['value_2']
            angle2 = compass_data.iloc[i]['value_2']
            angle_diff = abs((angle2 - angle1 + 180) % 360 - 180)
            heading_change_rate = angle_diff / step_diff
            
            # Mark as turn if exceeds threshold
            if heading_change_rate > 20.0:  # 20 degrees per step is a significant turn
                compass_data.loc[compass_data.index[i], 'is_turn'] = True
    
    # Mark potential QS intervals based on gyro and heading stability
    compass_data['is_qs_candidate'] = (
        (compass_data['gyro_roll_std'] < gyro_change_threshold) &  # Stable gyro
        (compass_data['heading_roll_std'] < heading_stability_threshold) &  # Stable heading
        (~compass_data['is_turn'])  # Not in a turn
    )
    
    # Find continuous QS intervals
    qs_intervals = []
    current_interval = None
    interval_number = 0
    
    for i, row in compass_data.iterrows():
        if row['is_qs_candidate']:
            if current_interval is None:
                # Start new interval
                current_interval = {
                    'start_idx': i,
                    'data': compass_data.loc[[i]].copy()
                }
            else:
                # Extend current interval
                current_interval['data'] = pd.concat([current_interval['data'], compass_data.loc[[i]]])
        else:
            if current_interval is not None:
                # End current interval if it's long enough
                if len(current_interval['data']) >= window_size:
                    current_interval['end_idx'] = current_interval['data'].index[-1]
                    current_interval['interval_number'] = interval_number
                    current_interval['region'] = 'Auto'
                    current_interval['forced'] = False
                    
                    # Additional check: ensure this isn't a false QS region
                    # Calculate correlation between heading changes and gyro changes
                    interval_data = current_interval['data']
                    if len(interval_data) > 5:  # Need enough points to calculate correlation
                        heading_changes = interval_data['value_2'].diff().fillna(0)
                        gyro_changes = interval_data['gyro_magnitude'].diff().fillna(0)
                        
                        # Correlation should be low in true QS intervals (heading stable when gyro stable)
                        # High correlation often indicates false QS (gyro changes but heading also changes)
                        correlation = abs(np.corrcoef(heading_changes, gyro_changes)[0, 1]) if len(heading_changes) > 1 else 1.0
                        
                        # Lower correlation indicates a true QS interval
                        if not np.isnan(correlation) and correlation < 0.7:
                            qs_intervals.append(current_interval)
                            interval_number += 1
                current_interval = None
    
    # Add last interval if it exists and is long enough
    if current_interval is not None and len(current_interval['data']) >= window_size:
        current_interval['end_idx'] = current_interval['data'].index[-1]
        current_interval['interval_number'] = interval_number
        current_interval['region'] = 'Auto'
        current_interval['forced'] = False
        
        # Same correlation check as above
        interval_data = current_interval['data']
        if len(interval_data) > 5:
            heading_changes = interval_data['value_2'].diff().fillna(0)
            gyro_changes = interval_data['gyro_magnitude'].diff().fillna(0)
            correlation = abs(np.corrcoef(heading_changes, gyro_changes)[0, 1]) if len(heading_changes) > 1 else 1.0
            
            if not np.isnan(correlation) and correlation < 0.7:
                qs_intervals.append(current_interval)
    
    # Sort intervals by step
    if qs_intervals:
        qs_intervals.sort(key=lambda x: x['data']['step'].mean())
        
        # Renumber intervals after sorting
        for i, interval in enumerate(qs_intervals):
            interval['interval_number'] = i
    
    print(f"Detected {len(qs_intervals)} QS intervals based on gyro stability")
    
    # Find which GT region each QS interval belongs to
    for interval in qs_intervals:
        # Find mean step for this interval
        mean_step = interval['data']['step'].mean()
        # Determine which GT region this falls in
        for gt_name, (start, end) in GT_RANGES.items():
            if start <= mean_step <= end:
                interval['region'] = gt_name
                break
    
    # Print intervals by GT region
    gt_coverage = {}
    for gt_name in GT_RANGES.keys():
        gt_coverage[gt_name] = []
    
    for i, interval in enumerate(qs_intervals):
        gt_coverage.get(interval['region'], []).append(i)
    
    print("\nGround Truth Coverage (for reference):")
    for gt_name, intervals in gt_coverage.items():
        intervals_str = ", ".join([f"#{i}" for i in intervals])
        status = "COVERED" if intervals else "NOT COVERED"
        print(f"{gt_name}: {status} by intervals: {intervals_str if intervals else 'None'}")
    
    # Print detailed interval information
    print("\nDetailed QS Intervals:")
    for interval in qs_intervals:
        interval_data = interval['data']
        start_step = interval_data['step'].iloc[0]
        end_step = interval_data['step'].iloc[-1]
        mean_step = interval_data['step'].mean()
        mean_heading = interval_data['value_2'].mean()
        mean_gyro = interval_data['gyro_magnitude'].mean()
        heading_std = interval_data['value_2'].std()
        
        print(f"QS Interval #{interval['interval_number']} - Steps: {start_step:.1f}-{end_step:.1f} (mean: {mean_step:.1f}), "
              f"Mean Heading: {mean_heading:.2f}°, Heading Std: {heading_std:.2f}, Mean Gyro: {mean_gyro:.4f}, "
              f"Region: {interval['region']}")
    
    return qs_intervals

# Detect QS intervals
qs_intervals = detect_quasi_static_intervals(compass_data)

# Filter auto-detected intervals in the GT5 region (the problem area)
def filter_intervals(qs_intervals, gt5_range, window_size=15):
    """Filter out additional false QS intervals by checking correlation between heading and gyro"""
    filtered_intervals = []
    
    for interval in qs_intervals:
        # Extract interval data
        interval_data = interval['data']
        
        # Calculate correlation between heading changes and gyro changes
        if len(interval_data) > 5:  # Need enough points
            heading_changes = interval_data['value_2'].diff().abs().fillna(0)
            gyro_changes = interval_data['gyro_magnitude'].diff().abs().fillna(0)
            
            # Normalize to compare patterns
            if heading_changes.std() > 0 and gyro_changes.std() > 0:
                heading_changes_norm = (heading_changes - heading_changes.mean()) / heading_changes.std()
                gyro_changes_norm = (gyro_changes - gyro_changes.mean()) / gyro_changes.std()
                
                # Calculate correlation
                correlation = abs(np.corrcoef(heading_changes_norm, gyro_changes_norm)[0, 1])
                
                # High correlation with high gyro values indicates false QS
                is_high_gyro = interval_data['gyro_magnitude'].mean() > interval_data['gyro_magnitude'].median() * 1.2
                
                if np.isnan(correlation) or (correlation > 0.6 and is_high_gyro):
                    # This appears to be a false QS interval (heading follows gyro changes)
                    mean_step = interval_data['step'].mean()
                    print(f"Filtering out QS interval with mean step {mean_step:.1f} (correlation: {correlation:.2f}, high gyro: {is_high_gyro})")
                    continue
        
        # If no problems, keep the interval
        filtered_intervals.append(interval)
    
    # Update interval numbering
    for i, interval in enumerate(filtered_intervals):
        interval['interval_number'] = i
    
    print(f"After filtering: {len(filtered_intervals)} QS intervals remain")
    return filtered_intervals

# Analyzing intervals to see if any cross the GT5 region, which has false QS
gt5_range = GT_RANGES.get('GT5', (75.0, 85.0))
qs_intervals = filter_intervals(qs_intervals, gt5_range, window_size=15)

# Update interval numbering after filtering
for i, interval in enumerate(qs_intervals):
    interval['interval_number'] = i

# Prepare data for analysis and visualization
if qs_intervals:
    # Extract data from QS intervals
    data_QS = {
        'Quasi_Static_Interval_Number': [],
        'Compass_Heading': [],
        'True_Heading': [],
        'Time': [],
        'Step': [],
        'Floor': [],
        'east': [],
        'north': [],
        'Gyro_Magnitude': []
    }
    
    # Process each interval
    for interval in qs_intervals:
        interval_data = interval['data']
        n_points = len(interval_data)
        
        # Add data for each point in the interval
        data_QS['Quasi_Static_Interval_Number'].extend([interval['interval_number']] * n_points)
        data_QS['Compass_Heading'].extend(interval_data['value_2'].tolist())
        data_QS['True_Heading'].extend(interval_data['true_heading'].tolist())
        data_QS['Time'].extend(interval_data['Timestamp_(ms)'].tolist())
        data_QS['Step'].extend(interval_data['step'].tolist())
        data_QS['Floor'].extend(interval_data['value_4'].tolist())
        data_QS['east'].extend(interval_data['value_1'].tolist())
        data_QS['north'].extend(interval_data['value_3'].tolist())
        data_QS['Gyro_Magnitude'].extend(interval_data['gyro_magnitude'].tolist())
        
        # Print interval information
        start_step = interval_data['step'].iloc[0]
        end_step = interval_data['step'].iloc[-1]
        mean_heading = interval_data['value_2'].mean()
        mean_gyro = interval_data['gyro_magnitude'].mean()
        
        # Find which GT region this belongs to
        region = "Unknown"
        mean_step = interval_data['step'].mean()
        for gt, (start, end) in GT_RANGES.items():
            if start <= mean_step <= end:
                region = gt
                break
        
        print(f"QS Interval #{interval['interval_number']} - Steps: {start_step:.1f}-{end_step:.1f}, "
              f"Mean Heading: {mean_heading:.2f}°, Mean Gyro: {mean_gyro:.4f}, Region: {region}")
    
    # Create DataFrame
    quasi_static_data = pd.DataFrame(data_QS)
    
    # Calculate statistics for each interval
    averages = quasi_static_data.groupby('Quasi_Static_Interval_Number').agg({
        'Compass_Heading': ['mean', np.std],
        'True_Heading': ['mean', np.std],
        'Gyro_Magnitude': ['mean', 'max'],
        'Step': ['mean', 'min', 'max']
    }).reset_index()
    
    # Flatten the column names
    averages.columns = ['_'.join(col).strip('_') for col in averages.columns.values]
    
    # Calculate absolute difference between compass and true heading
    averages['Heading_Difference'] = abs(averages['Compass_Heading_mean'] - averages['True_Heading_mean'])
    
    # Save results to CSV
    quasi_static_data.to_csv(os.path.join(output_folder, '1536_quasi_static_data.csv'), index=False)
    averages.to_csv(os.path.join(output_folder, '1536_quasi_static_averages.csv'), index=False)
    
    # Create visualizations
    
    # 1. Compass headings with QS intervals
    plt.figure(figsize=(12, 6))
    plt.plot(compass_data['Timestamp_(ms)'], compass_data['value_2'], 
             label='Compass Headings', color='cyan', alpha=0.6)
    
    # Create colormap for QS intervals
    n_intervals = len(qs_intervals)
    cmap = plt.cm.get_cmap('Set1', n_intervals)
    
    # Plot QS intervals
    plt.scatter(quasi_static_data['Time'], quasi_static_data['Compass_Heading'],
               c=quasi_static_data['Quasi_Static_Interval_Number'], cmap=cmap, 
               s=50, zorder=5, label='Quasi-Static Intervals')
    
    # Add true heading for reference
    plt.plot(compass_data['Timestamp_(ms)'], compass_data['true_heading'], 
             marker='.', linestyle='-', markersize=1, color='blue', alpha=0.5, label='True Heading')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Compass Headings (degrees)')
    plt.title('Compass Headings with Quasi-Static Intervals (1536 Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '1536_compass_with_QS_intervals.png'), dpi=300)
    plt.close()
    
    # 2. Gyro magnitude with QS intervals
    plt.figure(figsize=(12, 6))
    plt.plot(gyro_data['Timestamp_(ms)'], gyro_data['gyro_magnitude'], 
             label='Gyro Magnitude', color='purple', alpha=0.6)
    
    # Plot QS intervals
    plt.scatter(quasi_static_data['Time'], quasi_static_data['Gyro_Magnitude'],
               c=quasi_static_data['Quasi_Static_Interval_Number'], cmap=cmap, 
               s=50, zorder=5, label='Quasi-Static Intervals')
    
    # Add threshold lines
    plt.axhline(y=0.15, color='r', linestyle='--', alpha=0.7, label='General Threshold (0.15)')
    plt.axhline(y=0.10, color='g', linestyle='--', alpha=0.7, label='GT5 Region Threshold (0.10)')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Gyro Magnitude')
    plt.title('Gyro Magnitude with Quasi-Static Intervals (1536 Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '1536_gyro_with_QS_intervals.png'), dpi=300)
    plt.close()
    
    # 3. Steps with QS intervals
    plt.figure(figsize=(12, 6))
    plt.plot(compass_data['Timestamp_(ms)'], compass_data['step'], 
             label='Steps', color='cyan', alpha=0.6)
    
    # Plot QS intervals
    plt.scatter(quasi_static_data['Time'], quasi_static_data['Step'],
               c=quasi_static_data['Quasi_Static_Interval_Number'], cmap=cmap, 
               s=50, zorder=5, label='Quasi-Static Intervals')
    
    # Add GT region markers
    for region, (start, end) in GT_RANGES.items():
        plt.axhspan(start, end, alpha=0.2, color='gray')
        plt.text(compass_data['Timestamp_(ms)'].iloc[0], (start + end) / 2, region, fontsize=12)
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Step Number')
    plt.title('Step Number with Quasi-Static Intervals (1536 Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '1536_steps_with_QS_intervals.png'), dpi=300)
    plt.close()
    
    # 4. Location map with QS intervals
    plt.figure(figsize=(12, 6))
    plt.plot(compass_data['value_1'], compass_data['value_3'], 
             label='All Locations', color='cyan', alpha=0.6)
    
    # Plot QS intervals
    plt.scatter(quasi_static_data['east'], quasi_static_data['north'],
                c=quasi_static_data['Quasi_Static_Interval_Number'], cmap=cmap, 
                s=50, zorder=5, label='Quasi-Static Intervals')
    
    # Mark ground truth points
    gt_data = ground_truth_data[ground_truth_data['Type'].isin(['Initial_Location', 'Ground_truth_Location'])]
    plt.scatter(gt_data['value_4'], gt_data['value_5'], 
                color='red', s=100, marker='o', label='Ground Truth')
    
    # Add labels for ground truth points
    for i, row in gt_data.iterrows():
        plt.text(row['value_4'], row['value_5'], f"GT{i-1}", color='red', fontsize=12, weight='bold')
    
    plt.xlabel('East')
    plt.ylabel('North')
    plt.title('Locations with Quasi-Static Intervals (1536 Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '1536_locations_with_QS_intervals.png'), dpi=300)
    plt.close()
    
    # 5. Create statistics table visualization
    print("\nQS Interval Statistics:")
    table_df = averages[['Quasi_Static_Interval_Number', 
                          'Compass_Heading_mean', 'True_Heading_mean', 
                          'Heading_Difference', 'Gyro_Magnitude_mean', 
                          'Step_mean', 'Step_min', 'Step_max']]
    
    print(table_df)
    table_df.to_csv(os.path.join(output_folder, '1536_quasi_static_table.csv'), index=False)
    
    print(f"All results saved to: {output_folder}")
else:
    print("No QS intervals detected!") 