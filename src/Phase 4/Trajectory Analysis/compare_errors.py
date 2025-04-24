import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
from matplotlib.ticker import MultipleLocator

# Set file paths
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 4/QS_Compass4.19'
positions_file = os.path.join(output_dir, 'position_coordinates.csv')
ground_truth_file = os.path.join(output_dir, 'ground_truth_trajectory.csv')
error_output_dir = os.path.join(output_dir, 'error_comparison')
os.makedirs(error_output_dir, exist_ok=True)

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

def load_and_prepare_data():
    # Load position data
    print("Loading position coordinates...")
    positions_df = pd.read_csv(positions_file)
    
    # Load ground truth data
    print("Loading ground truth data...")
    ground_truth_df = pd.read_csv(ground_truth_file)
    
    # Make sure step column is consistent
    ground_truth_df = ground_truth_df.rename(columns={'step': 'Step'})
    
    # Merge position data with ground truth data
    print("Merging data...")
    merged_df = pd.merge_asof(
        positions_df.sort_values('Step'),
        ground_truth_df[['Step', 'x', 'y']].sort_values('Step'),
        on='Step',
        direction='nearest'
    )
    
    print(f"Final merged dataframe has {len(merged_df)} rows")
    
    # Calculate walked distance (0.7m per step)
    merged_df['Walked_distance'] = merged_df['Step'] * 0.7
    
    # Rename ground truth columns
    merged_df = merged_df.rename(columns={'x': 'ground_x', 'y': 'ground_y'})
    
    return merged_df

def calculate_errors(merged_df):
    # Calculate error metrics for each trajectory
    
    # Original Gyro errors
    merged_df['Original_Gyro_Error_X'] = np.abs(merged_df['Original_Gyro_X'] - merged_df['ground_x'])
    merged_df['Original_Gyro_Error_Y'] = np.abs(merged_df['Original_Gyro_Y'] - merged_df['ground_y'])
    merged_df['Original_Gyro_Distance_Error'] = np.sqrt(
        (merged_df['Original_Gyro_X'] - merged_df['ground_x'])**2 + 
        (merged_df['Original_Gyro_Y'] - merged_df['ground_y'])**2
    )
    
    # Original Compass errors
    merged_df['Original_Compass_Error_X'] = np.abs(merged_df['Original_Compass_X'] - merged_df['ground_x'])
    merged_df['Original_Compass_Error_Y'] = np.abs(merged_df['Original_Compass_Y'] - merged_df['ground_y'])
    merged_df['Original_Compass_Distance_Error'] = np.sqrt(
        (merged_df['Original_Compass_X'] - merged_df['ground_x'])**2 + 
        (merged_df['Original_Compass_Y'] - merged_df['ground_y'])**2
    )
    
    # Corrected Gyro errors
    merged_df['Corrected_Gyro_Error_X'] = np.abs(merged_df['Corrected_Gyro_X'] - merged_df['ground_x'])
    merged_df['Corrected_Gyro_Error_Y'] = np.abs(merged_df['Corrected_Gyro_Y'] - merged_df['ground_y'])
    merged_df['Corrected_Gyro_Distance_Error'] = np.sqrt(
        (merged_df['Corrected_Gyro_X'] - merged_df['ground_x'])**2 + 
        (merged_df['Corrected_Gyro_Y'] - merged_df['ground_y'])**2
    )
    
    # Corrected Compass errors
    merged_df['Corrected_Compass_Error_X'] = np.abs(merged_df['Corrected_Compass_X'] - merged_df['ground_x'])
    merged_df['Corrected_Compass_Error_Y'] = np.abs(merged_df['Corrected_Compass_Y'] - merged_df['ground_y'])
    merged_df['Corrected_Compass_Distance_Error'] = np.sqrt(
        (merged_df['Corrected_Compass_X'] - merged_df['ground_x'])**2 + 
        (merged_df['Corrected_Compass_Y'] - merged_df['ground_y'])**2
    )
    
    return merged_df

def plot_distance_error(data):
    # Plot Distance Error over Walked Distance (similar to distance_error_plot.png)
    fig, ax1 = plt.subplots(figsize=(3.45, 2.5), dpi=1000)  # IEEE column width
    
    # Plot original data
    ax1.plot(data['Walked_distance'], data['Original_Gyro_Distance_Error'],
             label='Original Gyro', linewidth=1.2, color='blue')
    ax1.plot(data['Walked_distance'], data['Original_Compass_Distance_Error'],
             label='Original Compass', linewidth=1.2, color='red')
    
    # Plot corrected data
    ax1.plot(data['Walked_distance'], data['Corrected_Gyro_Distance_Error'],
             label='Corrected Gyro', linewidth=1.2, color='blue', linestyle='--')
    ax1.plot(data['Walked_distance'], data['Corrected_Compass_Distance_Error'],
             label='Corrected Compass', linewidth=1.2, color='red', linestyle='--')
    
    ax1.set_xlabel('Walked Distance (m)', labelpad=3)
    ax1.set_ylabel('Positioning Error (m)', labelpad=3)
    
    ax1.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='k')
    ax1.legend()
    
    # Secondary x-axis for step numbers
    secx = ax1.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    secx.set_xticks(data['Walked_distance'][::50])
    secx.set_xticklabels(data['Step'][::50])
    secx.set_xlabel('Number of Walked Steps', labelpad=8)
    
    # Axis formatting
    ax1.xaxis.set_major_locator(MultipleLocator(10))  # Major ticks every 10 meters
    ax1.yaxis.set_major_locator(MultipleLocator(5))   # Major ticks every 5 meters
    
    plt.tight_layout()
    plt.savefig(os.path.join(error_output_dir, 'distance_error_comparison.png'))
    plt.close()

def plot_ecdf(data):
    # Plot ECDF (similar to ecdf_plot.png)
    plt.figure(figsize=(3.45, 2.5), dpi=1000)  # IEEE column width
    
    # Original ECDF
    ecdf_original_gyro = sm.distributions.ECDF(data['Original_Gyro_Distance_Error'])
    ecdf_original_compass = sm.distributions.ECDF(data['Original_Compass_Distance_Error'])
    
    # Corrected ECDF
    ecdf_corrected_gyro = sm.distributions.ECDF(data['Corrected_Gyro_Distance_Error'])
    ecdf_corrected_compass = sm.distributions.ECDF(data['Corrected_Compass_Distance_Error'])
    
    # Plot original
    plt.plot(ecdf_original_gyro.x, ecdf_original_gyro.y, 
             label='Original Gyro', color='blue', linewidth=1.2)
    plt.plot(ecdf_original_compass.x, ecdf_original_compass.y, 
             label='Original Compass', color='red', linewidth=1.2)
    
    # Plot corrected
    plt.plot(ecdf_corrected_gyro.x, ecdf_corrected_gyro.y, 
             label='Corrected Gyro', color='blue', linewidth=1.2, linestyle='--')
    plt.plot(ecdf_corrected_compass.x, ecdf_corrected_compass.y, 
             label='Corrected Compass', color='red', linewidth=1.2, linestyle='--')
    
    plt.xlabel('Positioning Error (m)', labelpad=3)
    plt.ylabel('ECDF', labelpad=3)
    
    plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='k')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(error_output_dir, 'ecdf_comparison.png'))
    plt.close()

def plot_boxplot(data):
    # Create box plot for all four methods
    distance_errors_df = pd.DataFrame({
        'Sensor': ['Original Gyro'] * len(data) + ['Original Compass'] * len(data) + 
                 ['Corrected Gyro'] * len(data) + ['Corrected Compass'] * len(data),
        'Distance_Error': np.concatenate([
            data['Original_Gyro_Distance_Error'],
            data['Original_Compass_Distance_Error'],
            data['Corrected_Gyro_Distance_Error'], 
            data['Corrected_Compass_Distance_Error']
        ])
    })
    
    plt.figure(figsize=(3.45, 2.5), dpi=1000)  # IEEE column width
    
    sns.boxplot(x='Sensor', y='Distance_Error', data=distance_errors_df, linewidth=0.6)
    
    plt.xlabel('Sensor', fontsize=5)
    plt.ylabel('Positioning Error (m)', fontsize=5)
    plt.xticks(rotation=45)
    
    plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='k')
    
    plt.tight_layout()
    plt.savefig(os.path.join(error_output_dir, 'error_boxplot_comparison.png'))
    plt.close()

def calculate_statistics(data):
    # Calculate error statistics for all methods
    stats = pd.DataFrame({
        'Method': ['Original Gyro', 'Original Compass', 'Corrected Gyro', 'Corrected Compass'],
        'Mean_Error': [
            data['Original_Gyro_Distance_Error'].mean(),
            data['Original_Compass_Distance_Error'].mean(),
            data['Corrected_Gyro_Distance_Error'].mean(),
            data['Corrected_Compass_Distance_Error'].mean()
        ],
        'Median_Error': [
            data['Original_Gyro_Distance_Error'].median(),
            data['Original_Compass_Distance_Error'].median(),
            data['Corrected_Gyro_Distance_Error'].median(),
            data['Corrected_Compass_Distance_Error'].median()
        ],
        'Max_Error': [
            data['Original_Gyro_Distance_Error'].max(),
            data['Original_Compass_Distance_Error'].max(),
            data['Corrected_Gyro_Distance_Error'].max(),
            data['Corrected_Compass_Distance_Error'].max()
        ],
        'Std_Dev': [
            data['Original_Gyro_Distance_Error'].std(),
            data['Original_Compass_Distance_Error'].std(),
            data['Corrected_Gyro_Distance_Error'].std(),
            data['Corrected_Compass_Distance_Error'].std()
        ]
    })
    
    # Calculate error reduction percentages
    gyro_improvement = ((data['Original_Gyro_Distance_Error'].mean() - 
                        data['Corrected_Gyro_Distance_Error'].mean()) / 
                        data['Original_Gyro_Distance_Error'].mean() * 100)
    
    compass_improvement = ((data['Original_Compass_Distance_Error'].mean() - 
                           data['Corrected_Compass_Distance_Error'].mean()) / 
                           data['Original_Compass_Distance_Error'].mean() * 100)
    
    stats.loc[stats['Method'] == 'Corrected Gyro', 'Improvement_%'] = gyro_improvement
    stats.loc[stats['Method'] == 'Corrected Compass', 'Improvement_%'] = compass_improvement
    
    # Save statistics to CSV
    stats.to_csv(os.path.join(error_output_dir, 'error_statistics_comparison.csv'), index=False)
    
    # Print summary
    print("\nError Statistics:")
    print(stats)
    print(f"\nGyro Improvement: {gyro_improvement:.2f}%")
    print(f"Compass Improvement: {compass_improvement:.2f}%")
    
    return stats

def main():
    print("\n=== Calculating and Comparing Positioning Errors ===\n")
    
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Calculate errors
    data = calculate_errors(data)
    
    # Generate plots
    print("\nGenerating visualization plots...")
    plot_distance_error(data)
    plot_ecdf(data)
    plot_boxplot(data)
    
    # Calculate and save statistics
    stats = calculate_statistics(data)
    
    print(f"\nResults saved to: {error_output_dir}")
    print("Error comparison completed successfully!")

if __name__ == "__main__":
    main() 