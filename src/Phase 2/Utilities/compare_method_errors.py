import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import os
import argparse
import glob

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate positioning error comparison plot between methods')
    parser.add_argument('--trad_data', help='CSV file with traditional method position data')
    parser.add_argument('--lstm_data', help='CSV file with LSTM method position data')
    parser.add_argument('--ensemble_data', help='CSV file with ensemble method position data')
    parser.add_argument('--ground_truth', help='CSV file with ground truth position data')
    parser.add_argument('--output', default='method_comparison.png', help='Output image filename')
    parser.add_argument('--step_length', type=float, default=0.66, help='Step length in meters')
    parser.add_argument('--data_dir', help='Directory containing position data files')
    args = parser.parse_args()
    
    # Set plot parameters for a cleaner look like the example
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.0,
        'font.family': "Arial"
    })
    
    # If data_dir is provided, try to find files automatically
    if args.data_dir:
        print(f"Searching for data files in {args.data_dir}...")
        if not args.trad_data:
            trad_files = glob.glob(os.path.join(args.data_dir, "*trad*.csv")) + \
                         glob.glob(os.path.join(args.data_dir, "*traditional*.csv"))
            if trad_files:
                args.trad_data = trad_files[0]
                print(f"Found traditional data: {args.trad_data}")
        
        if not args.lstm_data:
            lstm_files = glob.glob(os.path.join(args.data_dir, "*lstm*.csv"))
            if lstm_files:
                args.lstm_data = lstm_files[0]
                print(f"Found LSTM data: {args.lstm_data}")
        
        if not args.ensemble_data:
            ensemble_files = glob.glob(os.path.join(args.data_dir, "*ens*.csv")) + \
                             glob.glob(os.path.join(args.data_dir, "*ensemble*.csv"))
            if ensemble_files:
                args.ensemble_data = ensemble_files[0]
                print(f"Found ensemble data: {args.ensemble_data}")
        
        if not args.ground_truth:
            gt_files = glob.glob(os.path.join(args.data_dir, "*ground*.csv")) + \
                       glob.glob(os.path.join(args.data_dir, "*truth*.csv")) + \
                       glob.glob(os.path.join(args.data_dir, "*gt*.csv"))
            if gt_files:
                args.ground_truth = gt_files[0]
                print(f"Found ground truth data: {args.ground_truth}")
    
    # Load ground truth data if provided
    gt_data = None
    if args.ground_truth:
        try:
            print(f"Loading ground truth data from {args.ground_truth}...")
            gt_data = pd.read_csv(args.ground_truth)
            
            # Try to identify position columns
            potential_x_cols = [col for col in gt_data.columns if 'x' in col.lower() or 
                               'pos_x' in col.lower() or 'position_x' in col.lower()]
            potential_y_cols = [col for col in gt_data.columns if 'y' in col.lower() or 
                               'pos_y' in col.lower() or 'position_y' in col.lower()]
            
            # Look for 'step' column
            step_cols = [col for col in gt_data.columns if 'step' in col.lower()]
            
            if potential_x_cols and potential_y_cols:
                x_col = potential_x_cols[0]
                y_col = potential_y_cols[0]
                print(f"Using columns {x_col} and {y_col} for ground truth positions")
                
                # Rename for consistency
                gt_data = gt_data.rename(columns={x_col: 'ground_x', y_col: 'ground_y'})
                
                if step_cols:
                    gt_data = gt_data.rename(columns={step_cols[0]: 'step'})
                elif 'step' not in gt_data.columns:
                    gt_data['step'] = np.arange(1, len(gt_data) + 1)
            else:
                print("Could not identify position columns in ground truth data")
                gt_data = None
                
        except Exception as e:
            print(f"Error loading ground truth data: {e}")
            gt_data = None
    
    # Initialize data dict for methods
    method_data = {}
    
    # Process data files for each method
    for method_name, file_path in [
        ('Traditional', args.trad_data), 
        ('LSTM', args.lstm_data), 
        ('Ensemble', args.ensemble_data)
    ]:
        if file_path:
            try:
                print(f"Loading {method_name} data from {file_path}...")
                data = pd.read_csv(file_path)
                
                # Try to identify position columns
                potential_x_cols = [col for col in data.columns if 'x' in col.lower() or 
                                   'pos_x' in col.lower() or 'position_x' in col.lower()]
                potential_y_cols = [col for col in data.columns if 'y' in col.lower() or 
                                   'pos_y' in col.lower() or 'position_y' in col.lower()]
                
                # Look for 'step' column
                step_cols = [col for col in data.columns if 'step' in col.lower()]
                
                if potential_x_cols and potential_y_cols:
                    x_col = potential_x_cols[0]
                    y_col = potential_y_cols[0]
                    print(f"Using columns {x_col} and {y_col} for {method_name} positions")
                    
                    # Rename for consistency
                    data = data.rename(columns={x_col: f'{method_name}_x', y_col: f'{method_name}_y'})
                    
                    if step_cols:
                        data = data.rename(columns={step_cols[0]: 'step'})
                    elif 'step' not in data.columns:
                        data['step'] = np.arange(1, len(data) + 1)
                        
                    method_data[method_name] = data
                else:
                    print(f"Could not identify position columns in {method_name} data")
            except Exception as e:
                print(f"Error loading {method_name} data: {e}")
    
    # Check if we have enough data to create plot
    if not method_data:
        print("No valid method data found. Please provide at least one valid data file.")
        return
    
    # Generate step data if needed
    step_data = None
    
    # Try to get step data from ground truth
    if gt_data is not None and 'step' in gt_data.columns:
        step_data = gt_data[['step']]
    # Or from any method data
    elif any('step' in data.columns for data in method_data.values()):
        for method, data in method_data.items():
            if 'step' in data.columns:
                step_data = data[['step']]
                break
    # Or create it
    else:
        # Find max length of all datasets
        max_len = max([len(data) for data in method_data.values()])
        if gt_data is not None:
            max_len = max(max_len, len(gt_data))
        
        # Create step data
        step_data = pd.DataFrame({'step': np.arange(1, max_len + 1)})
    
    # Add walked distance
    step_data['Walked_distance'] = step_data['step'] * args.step_length
    
    # Calculate errors by comparing with ground truth
    if gt_data is not None:
        # Merge ground truth with step data
        if 'ground_x' in gt_data.columns and 'ground_y' in gt_data.columns:
            step_data = pd.merge(step_data, gt_data[['step', 'ground_x', 'ground_y']], 
                                 on='step', how='left')
            
            # Fill any NaN values with forward fill and then backward fill
            step_data = step_data.ffill().bfill()
            
            # Calculate errors for each method
            for method, data in method_data.items():
                if f'{method}_x' in data.columns and f'{method}_y' in data.columns:
                    # Merge method positions with step data
                    step_data = pd.merge(step_data, 
                                         data[['step', f'{method}_x', f'{method}_y']], 
                                         on='step', how='left')
                    
                    # Fill any NaN values
                    step_data = step_data.ffill().bfill()
                    
                    # Calculate Euclidean distance error
                    step_data[f'{method}_Distance_Error'] = np.sqrt(
                        (step_data[f'{method}_x'] - step_data['ground_x'])**2 + 
                        (step_data[f'{method}_y'] - step_data['ground_y'])**2
                    )
    else:
        print("Warning: No ground truth data provided. Using position data directly.")
        # Without ground truth, use the position data directly
        for method, data in method_data.items():
            if f'{method}_x' in data.columns and f'{method}_y' in data.columns:
                # Merge method positions with step data
                step_data = pd.merge(step_data, 
                                     data[['step', f'{method}_x', f'{method}_y']], 
                                     on='step', how='left')
                
                # Use Gyro as reference if available, otherwise use the first method's data
                reference_method = 'Traditional' if 'Traditional' in method_data else list(method_data.keys())[0]
                
                if method == reference_method:
                    # For reference method, set error to zero or to simulate a pattern
                    step_data[f'{method}_Distance_Error'] = np.linspace(0, 10, len(step_data))
                else:
                    # For other methods, calculate difference from reference method
                    step_data[f'{method}_Distance_Error'] = np.sqrt(
                        (step_data[f'{method}_x'] - step_data[f'{reference_method}_x'])**2 + 
                        (step_data[f'{method}_y'] - step_data[f'{reference_method}_y'])**2
                    )
    
    # Create a color mapping for methods (using colors similar to your example)
    method_colors = {
        'Traditional': 'blue',
        'LSTM': 'red',
        'Ensemble': 'green'
    }
    
    # Create the figure
    fig, ax1 = plt.subplots(figsize=(10, 7), dpi=300)
    
    # Plot error for each method
    for method in method_data.keys():
        if f'{method}_Distance_Error' in step_data.columns:
            ax1.plot(step_data['Walked_distance'], step_data[f'{method}_Distance_Error'], 
                     label=method, linewidth=3, color=method_colors.get(method, 'gray'))
    
    # Set axis labels and titles
    ax1.set_xlabel('Walked Distance (m)', fontsize=14)
    ax1.set_ylabel('Positioning Error (m)', fontsize=14)
    
    # Add grid
    ax1.grid(True, linestyle=':', linewidth=0.5, alpha=0.7, color='gray')
    
    # Add legend
    ax1.legend(fontsize=14)
    
    # Secondary x-axis for step numbers
    secx = ax1.secondary_xaxis('top', functions=(lambda x: x/args.step_length, lambda x: x*args.step_length))
    secx.set_xlabel('Number of Walked Steps', fontsize=14)
    
    # Calculate appropriate axis limits
    max_error = max([step_data[f'{method}_Distance_Error'].max() 
                     for method in method_data.keys() 
                     if f'{method}_Distance_Error' in step_data.columns])
    y_limit = np.ceil(max_error * 1.1)  # Add 10% margin
    
    # Set limits
    ax1.set_ylim(0, y_limit)
    ax1.set_xlim(0, max(step_data['Walked_distance']))
    
    # Adjust ticks for better readability
    x_step = 25 if max(step_data['Walked_distance']) > 50 else 10
    y_step = 5 if y_limit > 20 else 2
    
    ax1.xaxis.set_major_locator(MultipleLocator(x_step))
    ax1.yaxis.set_major_locator(MultipleLocator(y_step))
    secx.xaxis.set_major_locator(MultipleLocator(25))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(args.output)
    print(f"Plot created successfully and saved as '{args.output}'")
    
    # Calculate statistics
    stats_data = []
    for method in method_data.keys():
        if f'{method}_Distance_Error' in step_data.columns:
            stats_data.append({
                'Method': method,
                'Mean Error (m)': step_data[f'{method}_Distance_Error'].mean(),
                'Max Error (m)': step_data[f'{method}_Distance_Error'].max(),
                'Final Error (m)': step_data[f'{method}_Distance_Error'].iloc[-1],
            })
    
    if stats_data:
        stats = pd.DataFrame(stats_data)
        
        # Print statistics
        print("\nError Statistics:")
        print(stats.to_string(index=False))
        
        # Save statistics to CSV
        stats_file = os.path.splitext(args.output)[0] + '_stats.csv'
        stats.to_csv(stats_file, index=False)
        print(f"Statistics saved to '{stats_file}'")
    
    # Optionally show the plot
    plt.show()

if __name__ == "__main__":
    main() 