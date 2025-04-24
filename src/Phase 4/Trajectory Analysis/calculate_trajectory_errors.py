import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Set file paths
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 4/QS_Compass4.19'
positions_file = os.path.join(output_dir, 'position_coordinates.csv')
ground_truth_file = os.path.join(output_dir, 'ground_truth_trajectory.csv')

# Create output directory for error analysis if it doesn't exist
error_dir = os.path.join(output_dir, 'error_analysis')
os.makedirs(error_dir, exist_ok=True)

def calculate_errors(positions_df):
    # Extract columns
    steps = positions_df['Step'].values
    
    # Extract trajectory columns
    trajectory_types = [
        ('Original_Compass', 'Original_Compass_X', 'Original_Compass_Y'),
        ('Original_Gyro', 'Original_Gyro_X', 'Original_Gyro_Y'),
        ('Corrected_Compass', 'Corrected_Compass_X', 'Corrected_Compass_Y'),
        ('Corrected_Gyro', 'Corrected_Gyro_X', 'Corrected_Gyro_Y')
    ]
    
    # Get ground truth steps and positions
    ground_truth_steps = []
    ground_truth_x = []
    ground_truth_y = []
    
    for index, row in positions_df.iterrows():
        if not pd.isna(row['Step']) and not (pd.isna(row['Original_Compass_X']) or pd.isna(row['Original_Compass_Y'])):
            step = row['Step']
            
            # Check if this is a ground truth point
            is_ground_truth = False
            for col in positions_df.columns:
                if col.startswith('GT_'):
                    if not pd.isna(row[col]):
                        is_ground_truth = True
                        break
            
            if is_ground_truth:
                gt_x = None
                gt_y = None
                for col in positions_df.columns:
                    if col == 'GT_X':
                        gt_x = row[col]
                    elif col == 'GT_Y':
                        gt_y = row[col]
                
                if gt_x is not None and gt_y is not None:
                    ground_truth_steps.append(step)
                    ground_truth_x.append(gt_x)
                    ground_truth_y.append(gt_y)
    
    if len(ground_truth_steps) < 2:
        print("Not enough ground truth points for interpolation.")
        return None
    
    # Create interpolation function for ground truth
    gt_x_interp = interp1d(ground_truth_steps, ground_truth_x, kind='linear', bounds_error=False, fill_value='extrapolate')
    gt_y_interp = interp1d(ground_truth_steps, ground_truth_y, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Calculate errors for each trajectory type
    error_results = {}
    
    for traj_name, x_col, y_col in trajectory_types:
        errors = []
        steps_with_data = []
        
        for index, row in positions_df.iterrows():
            if pd.isna(row['Step']) or pd.isna(row[x_col]) or pd.isna(row[y_col]):
                continue
                
            step = row['Step']
            x = row[x_col]
            y = row[y_col]
            
            # Only consider steps within ground truth range
            if step < min(ground_truth_steps) or step > max(ground_truth_steps):
                continue
                
            # Get interpolated ground truth position at this step
            gt_x = gt_x_interp(step)
            gt_y = gt_y_interp(step)
            
            # Calculate Euclidean distance error
            error = np.sqrt((x - gt_x)**2 + (y - gt_y)**2)
            errors.append(error)
            steps_with_data.append(step)
        
        if errors:
            error_results[traj_name] = {
                'steps': steps_with_data,
                'errors': errors,
                'mean_error': np.mean(errors),
                'median_error': np.median(errors),
                'max_error': np.max(errors),
                'std_error': np.std(errors)
            }
    
    return error_results

def plot_error_comparison(error_results):
    plt.figure(figsize=(10, 6))
    
    for traj_name, data in error_results.items():
        plt.plot(data['steps'], data['errors'], label=f'{traj_name}')
    
    plt.xlabel('Steps')
    plt.ylabel('Error (m)')
    plt.title('Trajectory Errors Compared to Ground Truth')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(error_dir, 'trajectory_errors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a bar chart of mean errors
    mean_errors = {traj: data['mean_error'] for traj, data in error_results.items()}
    plt.figure(figsize=(10, 6))
    plt.bar(mean_errors.keys(), mean_errors.values())
    plt.ylabel('Mean Error (m)')
    plt.title('Mean Trajectory Errors')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(error_dir, 'mean_errors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return mean_errors

def main():
    # Load position data
    print("Loading position data...")
    positions_df = pd.read_csv(positions_file)
    
    # Load ground truth data from separate file
    print("Loading ground truth data...")
    try:
        ground_truth_df = pd.read_csv(ground_truth_file)
        print(f"Ground truth data loaded successfully. Found {len(ground_truth_df)} points.")
        
        # Check if ground truth file has expected columns
        if 'step' in ground_truth_df.columns and 'x' in ground_truth_df.columns and 'y' in ground_truth_df.columns:
            # Rename columns to match expected format for error calculation
            ground_truth_df = ground_truth_df.rename(columns={'step': 'Step', 'x': 'GT_X', 'y': 'GT_Y'})
            
            # Merge ground truth with position data based on Step column
            merged_df = pd.merge_asof(
                positions_df.sort_values('Step'),
                ground_truth_df[['Step', 'GT_X', 'GT_Y']].sort_values('Step'),
                on='Step',
                direction='nearest'
            )
            
            # Use the merged dataframe for error calculations
            positions_df = merged_df
            print(f"Successfully merged position data with ground truth. Final dataframe has {len(positions_df)} rows.")
        else:
            print("Ground truth file has unexpected column structure. Expected 'step', 'x', and 'y' columns.")
    except Exception as e:
        print(f"Error loading ground truth data: {str(e)}")
    
    # If ground truth columns aren't explicitly named, look for them
    if 'GT_X' not in positions_df.columns or 'GT_Y' not in positions_df.columns:
        for col in positions_df.columns:
            if 'ground' in col.lower() and 'x' in col.lower():
                positions_df.rename(columns={col: 'GT_X'}, inplace=True)
            elif 'ground' in col.lower() and 'y' in col.lower():
                positions_df.rename(columns={col: 'GT_Y'}, inplace=True)
            elif 'truth' in col.lower() and 'x' in col.lower():
                positions_df.rename(columns={col: 'GT_X'}, inplace=True)
            elif 'truth' in col.lower() and 'y' in col.lower():
                positions_df.rename(columns={col: 'GT_Y'}, inplace=True)
    
    # Calculate errors
    print("Calculating trajectory errors...")
    error_results = calculate_errors(positions_df)
    
    if error_results:
        # Plot error comparison
        print("Plotting error comparisons...")
        mean_errors = plot_error_comparison(error_results)
        
        # Print error statistics
        print("\nError Statistics:")
        for traj_name, data in error_results.items():
            print(f"\n{traj_name}:")
            print(f"  Mean Error: {data['mean_error']:.4f} m")
            print(f"  Median Error: {data['median_error']:.4f} m")
            print(f"  Max Error: {data['max_error']:.4f} m")
            print(f"  Standard Deviation: {data['std_error']:.4f} m")
        
        # Save error statistics to CSV
        print("\nSaving error statistics to CSV...")
        error_stats = []
        for traj_name, data in error_results.items():
            error_stats.append({
                'Trajectory': traj_name,
                'Mean_Error': data['mean_error'],
                'Median_Error': data['median_error'],
                'Max_Error': data['max_error'],
                'Std_Error': data['std_error'],
                'Error_Reduction_%': None  # Will fill this in next
            })
        
        # Calculate error reduction percentages relative to original methods
        for i, stat in enumerate(error_stats):
            if 'Corrected_Compass' in stat['Trajectory']:
                for orig_stat in error_stats:
                    if 'Original_Compass' in orig_stat['Trajectory']:
                        reduction = (orig_stat['Mean_Error'] - stat['Mean_Error']) / orig_stat['Mean_Error'] * 100
                        error_stats[i]['Error_Reduction_%'] = reduction
            elif 'Corrected_Gyro' in stat['Trajectory']:
                for orig_stat in error_stats:
                    if 'Original_Gyro' in orig_stat['Trajectory']:
                        reduction = (orig_stat['Mean_Error'] - stat['Mean_Error']) / orig_stat['Mean_Error'] * 100
                        error_stats[i]['Error_Reduction_%'] = reduction
        
        # Save to CSV
        pd.DataFrame(error_stats).to_csv(os.path.join(error_dir, 'error_statistics.csv'), index=False)
        
        print(f"\nResults saved to {error_dir}")
    else:
        print("Could not calculate errors due to insufficient ground truth data.")

if __name__ == "__main__":
    main() 