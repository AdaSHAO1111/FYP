import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.cm as cm

# Set paths to data files
compass_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_compass_data.csv'
gyro_data_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_gyro_data.csv'
ground_truth_path = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_ground_truth_data.csv'

# Set output directory
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 4/QS_Compass4.19'
os.makedirs(output_dir, exist_ok=True)

def normalize_to_positive(angle):
    """Convert angle to 0-360 range"""
    return (angle + 360) % 360

# Function to detect QS intervals
def detect_qs_intervals(compass_data, gyro_data, window_size=15):
    """Detect quasi-static intervals based on gyro stability"""
    # Add gyro magnitude to compass data
    compass_data['gyro_magnitude'] = np.nan
    for i, row in compass_data.iterrows():
        timestamp = row['Timestamp_(ms)']
        closest_idx = (gyro_data['Timestamp_(ms)'] - timestamp).abs().idxmin()
        if abs(gyro_data.loc[closest_idx, 'Timestamp_(ms)'] - timestamp) < 100:  # Within 100ms
            compass_data.loc[i, 'gyro_magnitude'] = gyro_data.loc[closest_idx, 'gyro_magnitude']
    
    # Calculate rolling statistics
    compass_data['gyro_roll_mean'] = compass_data['gyro_magnitude'].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    compass_data['gyro_roll_std'] = compass_data['gyro_magnitude'].rolling(window=window_size, center=True).std().fillna(method='bfill').fillna(method='ffill')
    
    # Calculate heading differences
    compass_data['heading_diff'] = compass_data['Compass_Heading'].diff().abs()
    # Handle circular differences (e.g., 359° to 1°)
    large_diffs = compass_data['heading_diff'] > 180
    compass_data.loc[large_diffs, 'heading_diff'] = 360 - compass_data.loc[large_diffs, 'heading_diff']
    
    compass_data['heading_roll_std'] = compass_data['heading_diff'].rolling(window=window_size, center=True).std().fillna(method='bfill').fillna(method='ffill')
    
    # Determine thresholds
    gyro_mean = compass_data['gyro_magnitude'].mean()
    gyro_std = compass_data['gyro_magnitude'].std()
    heading_std_mean = compass_data['heading_roll_std'].mean()
    
    print(f"Gyro data analysis - Mean: {gyro_mean:.2f}, Std: {gyro_std:.2f}")
    print(f"Heading stability analysis - Mean std: {heading_std_mean:.2f}")
    
    # Adaptive thresholds
    gyro_stability_threshold = gyro_mean * 0.9
    heading_stability_threshold = heading_std_mean * 1.2
    gyro_change_threshold = gyro_std * 2.0
    
    print(f"Using adaptive thresholds - Gyro stability: {gyro_stability_threshold:.2f}, Heading stability: {heading_stability_threshold:.2f}, Gyro change: {gyro_change_threshold:.2f}")
    
    # Mark potential QS intervals
    compass_data['is_qs_candidate'] = (
        (compass_data['gyro_roll_std'] < gyro_change_threshold) & 
        (compass_data['heading_roll_std'] < heading_stability_threshold)
    )
    
    # Find continuous QS intervals
    compass_data['Quasi_Static_Interval'] = 0
    compass_data['Quasi_Static_Interval_Number'] = -1
    
    # Process continuous regions
    in_interval = False
    current_interval = -1
    min_interval_size = window_size
    
    for i in range(len(compass_data)):
        if compass_data.iloc[i]['is_qs_candidate']:
            if not in_interval:
                # Start new interval
                current_interval += 1
                in_interval = True
            compass_data.loc[compass_data.index[i], 'Quasi_Static_Interval'] = 1
            compass_data.loc[compass_data.index[i], 'Quasi_Static_Interval_Number'] = current_interval
        else:
            in_interval = False
    
    # Remove small intervals
    for interval in range(current_interval + 1):
        interval_size = (compass_data['Quasi_Static_Interval_Number'] == interval).sum()
        if interval_size < min_interval_size:
            compass_data.loc[compass_data['Quasi_Static_Interval_Number'] == interval, 'Quasi_Static_Interval'] = 0
            compass_data.loc[compass_data['Quasi_Static_Interval_Number'] == interval, 'Quasi_Static_Interval_Number'] = -1
    
    # Renumber intervals
    valid_intervals = sorted(compass_data[compass_data['Quasi_Static_Interval'] == 1]['Quasi_Static_Interval_Number'].unique())
    new_interval_map = {old: new for new, old in enumerate(valid_intervals)}
    
    for old, new in new_interval_map.items():
        compass_data.loc[compass_data['Quasi_Static_Interval_Number'] == old, 'Quasi_Static_Interval_Number'] = new
    
    num_intervals = len(new_interval_map)
    print(f"Detected {num_intervals} QS intervals based on gyro stability")
    
    return compass_data

try:
    # 1. Load data
    print("Loading data...")
    compass_data = pd.read_csv(compass_data_path)
    gyro_data = pd.read_csv(gyro_data_path)
    ground_truth_data = pd.read_csv(ground_truth_path)
    
    print(f"Loaded {len(compass_data)} compass data points, {len(gyro_data)} gyro data points")
    
    # 2. Process gyro data
    gyro_data['gyro_magnitude'] = np.sqrt(
        gyro_data['value_1']**2 + 
        gyro_data['value_2']**2 + 
        gyro_data['value_3']**2
    )
    
    # 3. Normalize compass headings to 0-360 range
    compass_data['Compass_Heading'] = compass_data['value_2'].apply(normalize_to_positive)
    
    # 4. Normalize ground truth headings
    if 'GroundTruthHeadingComputed' in ground_truth_data.columns:
        ground_truth_data['GroundTruthHeadingComputed'] = ground_truth_data['GroundTruthHeadingComputed'].apply(normalize_to_positive)
    
    # 5. Run QS detection
    compass_data_with_qs = detect_qs_intervals(compass_data, gyro_data)
    
    # 6. Calculate statistics for each interval
    if compass_data_with_qs[compass_data_with_qs['Quasi_Static_Interval'] == 1].empty:
        print("No QS intervals detected!")
    else:
        qs_stats = []
        for interval_number in sorted(compass_data_with_qs[compass_data_with_qs['Quasi_Static_Interval'] == 1]['Quasi_Static_Interval_Number'].unique()):
            interval_data = compass_data_with_qs[compass_data_with_qs['Quasi_Static_Interval_Number'] == interval_number]
            
            # Calculate statistics
            stats = {
                'Quasi_Static_Interval_Number': interval_number,
                'Compass_Heading_mean': interval_data['Compass_Heading'].mean(),
                'Compass_Heading_std': interval_data['Compass_Heading'].std(),
                'gyro_magnitude_mean': interval_data['gyro_magnitude'].mean(),
                'gyro_magnitude_std': interval_data['gyro_magnitude'].std(),
                'Step_mean': interval_data['step'].mean(),
                'Step_min': interval_data['step'].min(),
                'Step_max': interval_data['step'].max(),
                'num_points': len(interval_data)
            }
            qs_stats.append(stats)
            
            print(f"QS Interval #{interval_number} - Steps: {stats['Step_min']}-{stats['Step_max']}, "
                 f"Mean Heading: {stats['Compass_Heading_mean']:.2f}°, "
                 f"Mean Gyro: {stats['gyro_magnitude_mean']:.4f}")
        
        # Create DataFrame with statistics
        stats_df = pd.DataFrame(qs_stats)
        stats_df.to_csv(os.path.join(output_dir, 'qs_interval_statistics.csv'), index=False)
        
        # 7. Save data with QS intervals
        compass_data_with_qs.to_csv(os.path.join(output_dir, 'compass_data_with_qs_intervals.csv'), index=False)
        
        # 8. Create visualizations
        # Compass headings with QS intervals
        plt.figure(figsize=(14, 8))
        plt.plot(compass_data_with_qs['Timestamp_(ms)'], compass_data_with_qs['Compass_Heading'], 'c-', alpha=0.8, label='Compass Headings')
        
        # Add true heading to the plot if available
        true_headings = []
        for timestamp in compass_data['Timestamp_(ms)']:
            closest_idx = (ground_truth_data['Timestamp_(ms)'] - timestamp).abs().idxmin()
            true_heading = ground_truth_data.loc[closest_idx, 'GroundTruthHeadingComputed'] if 'GroundTruthHeadingComputed' in ground_truth_data.columns else np.nan
            if not np.isnan(true_heading):
                true_heading = normalize_to_positive(true_heading)
            true_headings.append(true_heading)
        
        compass_data_with_qs['true_heading'] = true_headings
        plt.plot(compass_data_with_qs['Timestamp_(ms)'], compass_data_with_qs['true_heading'], 'b-', alpha=0.8, label='True Heading')
        
        # Plot QS intervals with different colors
        num_intervals = len(stats_df)
        cmap = plt.cm.get_cmap('Set1', num_intervals)
        
        for i in range(num_intervals):
            interval_data = compass_data_with_qs[compass_data_with_qs['Quasi_Static_Interval_Number'] == i]
            color = cmap(i)
            plt.scatter(interval_data['Timestamp_(ms)'], interval_data['Compass_Heading'], 
                       c=[color], s=50, label=f'QS Interval {i}', alpha=0.8)
        
        plt.title('Compass Headings with Quasi-Static Intervals (1536 Data)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Compass Headings (degrees)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'compass_headings_with_qs_intervals.png'), dpi=300)
        
        # Gyro magnitude with QS intervals
        plt.figure(figsize=(14, 8))
        plt.plot(compass_data_with_qs['Timestamp_(ms)'], compass_data_with_qs['gyro_magnitude'], 'b-', alpha=0.5, label='Gyro Magnitude')
        
        for i in range(num_intervals):
            interval_data = compass_data_with_qs[compass_data_with_qs['Quasi_Static_Interval_Number'] == i]
            color = cmap(i)
            plt.scatter(interval_data['Timestamp_(ms)'], interval_data['gyro_magnitude'], 
                       c=[color], s=50, label=f'QS Interval {i}', alpha=0.8)
        
        plt.title('Gyro Magnitude with Quasi-Static Intervals (1536 Data)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Gyro Magnitude')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gyro_magnitude_with_qs_intervals.png'), dpi=300)
        
        # Also save to the main Phase 4 directory
        main_output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 4'
        os.makedirs(main_output_dir, exist_ok=True)
        stats_df.to_csv(os.path.join(main_output_dir, 'qs_interval_statistics.csv'), index=False)
        
        print(f"All results saved to: {output_dir}")
        print("QS detection completed successfully!")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 