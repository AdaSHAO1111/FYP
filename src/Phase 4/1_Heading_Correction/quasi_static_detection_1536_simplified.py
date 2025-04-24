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
output_folder = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 4'

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to normalize angles to 0-360 range
def normalize_to_positive(angle):
    """Convert angle to 0-360 range"""
    return (angle + 360) % 360

# Load data
print("Loading data...")
compass_data = pd.read_csv(compass_data_path)
gyro_data = pd.read_csv(gyro_data_path)
ground_truth_data = pd.read_csv(ground_truth_path)

# Normalize compass headings to 0-360 range
compass_data['value_2'] = compass_data['value_2'].apply(normalize_to_positive)
if 'GroundTruthHeadingComputed' in ground_truth_data.columns:
    ground_truth_data['GroundTruthHeadingComputed'] = ground_truth_data['GroundTruthHeadingComputed'].apply(normalize_to_positive)

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
    true_heading = ground_truth_data.loc[closest_idx, 'GroundTruthHeadingComputed'] if 'GroundTruthHeadingComputed' in ground_truth_data.columns else np.nan
    # Ensure true heading is also in 0-360 range
    if not np.isnan(true_heading):
        true_heading = normalize_to_positive(true_heading)
    true_headings.append(true_heading)

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

# Count points in each range to verify
print("\nPoints in each GT range:")
for name, (start, end) in GT_RANGES.items():
    count = len(compass_data[(compass_data['step'] >= start) & (compass_data['step'] <= end)])
    print(f"{name}: {count} points (steps {start}-{end})")

# QS detection function focusing on gyro stability
def detect_quasi_static_intervals(compass_data, window_size=15):
    """
    Detect quasi-static intervals based on gyro stability
    Focus on stable gyro periods rather than ground truth points
    """
    # Ensure compass headings are in 0-360 range
    if 'Compass_Heading' in compass_data.columns:
        compass_data['Compass_Heading'] = compass_data['Compass_Heading'].apply(normalize_to_positive)
    elif 'value_2' in compass_data.columns:
        compass_data['value_2'] = compass_data['value_2'].apply(normalize_to_positive)
    
    print("Detecting QS intervals based on gyro stability...")
    
    # Calculate moving statistics for gyro magnitude
    compass_data['gyro_roll_mean'] = compass_data['gyro_magnitude'].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    compass_data['gyro_roll_std'] = compass_data['gyro_magnitude'].rolling(window=window_size, center=True).std().fillna(method='bfill').fillna(method='ffill')
    
    # Calculate moving statistics for compass heading
    # Use the appropriate heading column based on available data
    heading_col = 'Compass_Heading' if 'Compass_Heading' in compass_data.columns else 'value_2'
    
    # Calculate heading differences using circular difference
    def circular_diff(a, b):
        return min((a - b) % 360, (b - a) % 360)
    
    heading_diffs = []
    for i in range(1, len(compass_data)):
        diff = circular_diff(compass_data[heading_col].iloc[i], compass_data[heading_col].iloc[i-1])
        heading_diffs.append(diff)
    heading_diffs.append(0)  # Add a zero for the last row
    
    compass_data['heading_diff'] = heading_diffs
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
            # Use circular difference for angles
            angle1 = compass_data.iloc[i-1][heading_col]
            angle2 = compass_data.iloc[i][heading_col]
            angle_diff = circular_diff(angle2, angle1)
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
                    
                    # Additional check: ensure this isn't a false QS region
                    # Calculate correlation between heading changes and gyro changes
                    interval_data = current_interval['data']
                    if len(interval_data) > 5:  # Need enough points to calculate correlation
                        heading_changes = interval_data[heading_col].diff().fillna(0)
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
        
        # Same correlation check as above
        interval_data = current_interval['data']
        if len(interval_data) > 5:
            heading_changes = interval_data[heading_col].diff().fillna(0)
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
    
    # Apply QS interval information to the main DataFrame
    compass_data['Quasi_Static_Interval'] = 0
    compass_data['Quasi_Static_Interval_Number'] = -1
    compass_data['QS_Region'] = ''
    
    # Use the heading column name from earlier
    for interval in qs_intervals:
        # Get the interval indices
        interval_indices = interval['data'].index
        
        # Mark the points as QS
        compass_data.loc[interval_indices, 'Quasi_Static_Interval'] = 1
        compass_data.loc[interval_indices, 'Quasi_Static_Interval_Number'] = interval['interval_number']
        compass_data.loc[interval_indices, 'QS_Region'] = interval['region']
    
    # Check GT coverage
    print("\nGround Truth Coverage (for reference):")
    for gt_name, (start, end) in GT_RANGES.items():
        # Find QS intervals that overlap with this GT range
        gt_data = compass_data[(compass_data['step'] >= start) & (compass_data['step'] <= end)]
        overlapping_intervals = gt_data[gt_data['Quasi_Static_Interval'] == 1]['Quasi_Static_Interval_Number'].unique()
        
        if len(overlapping_intervals) > 0:
            print(f"{gt_name}: Covered by intervals: {list(overlapping_intervals)}")
        else:
            print(f"{gt_name}: NOT COVERED by intervals: None")
    
    # Print details for QS intervals
    print("\nDetailed QS Intervals:")
    for interval in qs_intervals:
        interval_data = interval['data']
        interval_number = interval['interval_number']
        heading_mean = interval_data[heading_col].mean()
        heading_std = interval_data[heading_col].std()
        gyro_mean = interval_data['gyro_magnitude'].mean()
        step_min = interval_data['step'].min()
        step_max = interval_data['step'].max()
        step_mean = interval_data['step'].mean()
        region = interval['region']
        
        print(f"QS Interval #{interval_number} - Steps: {step_min}-{step_max} (mean: {step_mean:.1f}), "
              f"Mean Heading: {heading_mean:.2f}°, Heading Std: {heading_std:.2f}, "
              f"Mean Gyro: {gyro_mean:.4f}, Region: {region}")
    
    return compass_data

# Main execution - if run as script
if __name__ == "__main__":
    # Run QS detection
    compass_data_with_qs = detect_quasi_static_intervals(compass_data)
    
    # Extract QS intervals for analysis
    qs_intervals = []
    if 'Quasi_Static_Interval' in compass_data_with_qs.columns:
        for interval_number in sorted(compass_data_with_qs[compass_data_with_qs['Quasi_Static_Interval'] == 1]['Quasi_Static_Interval_Number'].unique()):
            interval_data = compass_data_with_qs[compass_data_with_qs['Quasi_Static_Interval_Number'] == interval_number]
            if len(interval_data) > 0:
                heading_col = 'Compass_Heading' if 'Compass_Heading' in interval_data.columns else 'value_2'
                qs_intervals.append({
                    'number': interval_number,
                    'data': interval_data,
                    'step_min': interval_data['step'].min(),
                    'step_max': interval_data['step'].max(),
                    'step_mean': interval_data['step'].mean(),
                    'heading_mean': interval_data[heading_col].mean(),
                    'heading_std': interval_data[heading_col].std(),
                    'gyro_magnitude_mean': interval_data['gyro_magnitude'].mean()
                })
    
    # Create final QS interval table for visualization
    for interval in qs_intervals:
        region = "Unknown"
        for gt_name, (start, end) in GT_RANGES.items():
            if interval['step_max'] >= start and interval['step_min'] <= end:
                region = gt_name
                break
        
        print(f"QS Interval #{interval['number']} - Steps: {interval['step_min']}-{interval['step_max']}, "
              f"Mean Heading: {interval['heading_mean']:.2f}°, "
              f"Mean Gyro: {interval['gyro_magnitude_mean']:.4f}, "
              f"Region: {region}")
    
    # Save QS interval statistics to CSV
    quasi_static_data = compass_data_with_qs[compass_data_with_qs['Quasi_Static_Interval'] == 1]
    interval_numbers = quasi_static_data['Quasi_Static_Interval_Number'].unique()
    
    # Calculate statistics for each interval
    if len(interval_numbers) > 0:
        heading_col = 'Compass_Heading' if 'Compass_Heading' in quasi_static_data.columns else 'value_2'
        
        averages = quasi_static_data.groupby('Quasi_Static_Interval_Number').agg({
            heading_col: ['mean', 'std'],
            'gyro_magnitude': ['mean', 'std'],
            'step': ['min', 'max', 'mean']
        })
        
        # Flatten the multi-index columns
        averages.columns = ['_'.join(col).strip() for col in averages.columns.values]
        averages = averages.reset_index()
        
        print("\nQS Interval Statistics:")
        print(averages)
        
        # Save statistics to CSV
        averages.to_csv(os.path.join(output_folder, 'qs_interval_statistics.csv'), index=False)
        quasi_static_data.to_csv(os.path.join(output_folder, 'quasi_static_data.csv'), index=False)
        compass_data_with_qs.to_csv(os.path.join(output_folder, 'compass_data_with_qs_intervals.csv'), index=False)
    
    # Visualize the data with QS intervals
    plt.figure(figsize=(14, 8))
    
    heading_col = 'Compass_Heading' if 'Compass_Heading' in compass_data_with_qs.columns else 'value_2'
    plt.plot(compass_data_with_qs['Timestamp_(ms)'], compass_data_with_qs[heading_col], 'c-', alpha=0.8, label='Compass Headings')
    
    # Plot true heading if available
    if 'true_heading' in compass_data_with_qs.columns:
        plt.plot(compass_data_with_qs['Timestamp_(ms)'], compass_data_with_qs['true_heading'], 'b-', alpha=0.8, label='True Heading')
    
    # Plot the QS intervals with different colors
    if len(qs_intervals) > 0:
        n_intervals = len(qs_intervals)
        cmap = plt.cm.get_cmap('Set1', n_intervals)
        
        for i, interval in enumerate(qs_intervals):
            interval_data = interval['data']
            color = cmap(i)
            plt.scatter(interval_data['Timestamp_(ms)'], interval_data[heading_col], 
                        c=[color], s=50, label=f'QS Interval {i}' if i == 0 else "", alpha=0.8)
    
    plt.title('Compass Headings with Quasi-Static Intervals (1536 Data)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Compass Headings (degrees)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'compass_headings_with_qs_intervals.png'), dpi=300)
    
    # Plot gyro magnitude with QS intervals
    plt.figure(figsize=(14, 8))
    plt.plot(compass_data_with_qs['Timestamp_(ms)'], compass_data_with_qs['gyro_magnitude'], 'b-', alpha=0.5, label='Gyro Magnitude')
    
    if len(qs_intervals) > 0:
        for i, interval in enumerate(qs_intervals):
            interval_data = interval['data']
            color = cmap(i)
            plt.scatter(interval_data['Timestamp_(ms)'], interval_data['gyro_magnitude'], 
                        c=[color], s=50, label=f'QS Interval {i}' if i == 0 else "", alpha=0.8)
    
    plt.title('Gyro Magnitude with Quasi-Static Intervals (1536 Data)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Gyro Magnitude')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'gyro_magnitude_with_qs_intervals.png'), dpi=300) 