import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import os
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate positioning error comparison plot')
    parser.add_argument('--input', help='Input CSV file with positioning data')
    parser.add_argument('--output', default='positioning_error_comparison.png', help='Output image filename')
    parser.add_argument('--step_length', type=float, default=0.66, help='Step length in meters')
    parser.add_argument('--simulate', action='store_true', help='Use simulated data instead of input file')
    args = parser.parse_args()
    
    # Set plot parameters
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.0,
        'font.family': "Arial"
    })
    
    # Default parameters
    n_steps = 150
    step_length = args.step_length
    
    # Load data or simulate
    if args.input and not args.simulate:
        try:
            print(f"Loading data from {args.input}...")
            df = pd.read_csv(args.input)
            
            # Check if file has required columns
            required_columns = ['step', 'Walked_distance', 'Gyro_Distance_Error', 'Compass_Distance_Error']
            
            # Rename columns if necessary (common alternative names)
            column_mapping = {
                'steps': 'step',
                'walk_distance': 'Walked_distance',
                'walked_distance': 'Walked_distance',
                'gyro_error': 'Gyro_Distance_Error',
                'compass_error': 'Compass_Distance_Error'
            }
            
            df = df.rename(columns={col: column_mapping[col.lower()] 
                                   for col in df.columns 
                                   if col.lower() in column_mapping})
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Warning: Missing columns in input file: {missing_columns}")
                print("Calculating missing columns if possible...")
                
                # If step is missing but we have a continuous index
                if 'step' in missing_columns and df.index.is_monotonic_increasing:
                    df['step'] = df.index + 1
                
                # If Walked_distance is missing but we have step
                if 'Walked_distance' in missing_columns and 'step' in df.columns:
                    df['Walked_distance'] = df['step'] * step_length
                
                # If either error column is missing, we can't calculate it without ground truth
                if 'Gyro_Distance_Error' in missing_columns or 'Compass_Distance_Error' in missing_columns:
                    print("Cannot calculate positioning errors without ground truth data.")
                    print("Switching to simulated data...")
                    args.simulate = True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Switching to simulated data...")
            args.simulate = True
    else:
        args.simulate = True
    
    if args.simulate:
        print("Using simulated data...")
        # Generate step and distance data
        steps = np.arange(1, n_steps + 1)
        walked_distance = steps * step_length
        
        # Simulate the data
        # Generate gyro error that increases gradually but stays relatively low
        gyro_error = np.zeros(n_steps)
        for i in range(1, n_steps):
            gyro_error[i] = gyro_error[i-1] + np.random.normal(0.05, 0.03)
            if i > 50 and i < 80:  # Add a small bump in the middle
                gyro_error[i] += 0.03
        gyro_error = np.abs(gyro_error)
        gyro_error = gyro_error * 1.2 + 1  # Scale to reasonable range
        
        # Generate compass error that increases more rapidly
        compass_error = np.zeros(n_steps)
        for i in range(1, n_steps):
            compass_error[i] = compass_error[i-1] + np.random.normal(0.15, 0.08)
            if i > 40 and i < 45:  # Add a temporary drop
                compass_error[i] -= 0.1
        compass_error = np.abs(compass_error)
        compass_error = compass_error * 1.7 + 0.5  # Scale to match example
        
        # Create DataFrame
        df = pd.DataFrame({
            'step': steps,
            'Walked_distance': walked_distance,
            'Gyro_Distance_Error': gyro_error,
            'Compass_Distance_Error': compass_error
        })
    
    # Create the figure
    fig, ax1 = plt.subplots(figsize=(10, 7), dpi=300)
    
    # Plot Gyro and Compass errors
    ax1.plot(df['Walked_distance'], df['Gyro_Distance_Error'], 
             label='Gyro', linewidth=3, color='blue')
    ax1.plot(df['Walked_distance'], df['Compass_Distance_Error'], 
             label='Compass', linewidth=3, color='red')
    
    # Set axis labels and titles
    ax1.set_xlabel('Walked Distance (m)', fontsize=14)
    ax1.set_ylabel('Positioning Error (m)', fontsize=14)
    
    # Add grid
    ax1.grid(True, linestyle=':', linewidth=0.5, alpha=0.7, color='gray')
    
    # Add legend
    ax1.legend(fontsize=14)
    
    # Secondary x-axis for step numbers
    secx = ax1.secondary_xaxis('top', functions=(lambda x: x/step_length, lambda x: x*step_length))
    secx.set_xlabel('Number of Walked Steps', fontsize=14)
    
    # Calculate appropriate axis limits
    max_error = max(df['Gyro_Distance_Error'].max(), df['Compass_Distance_Error'].max())
    y_limit = np.ceil(max_error * 1.1)  # Add 10% margin
    
    # Set limits
    ax1.set_ylim(0, y_limit)
    ax1.set_xlim(0, max(df['Walked_distance']))
    
    # Adjust ticks for better readability
    x_step = 25 if max(df['Walked_distance']) > 50 else 10
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
    stats = pd.DataFrame({
        'Method': ['Gyro', 'Compass'],
        'Mean Error (m)': [df['Gyro_Distance_Error'].mean(), df['Compass_Distance_Error'].mean()],
        'Max Error (m)': [df['Gyro_Distance_Error'].max(), df['Compass_Distance_Error'].max()],
        'Final Error (m)': [df['Gyro_Distance_Error'].iloc[-1], df['Compass_Distance_Error'].iloc[-1]],
    })
    
    # Print statistics
    print("\nError Statistics:")
    print(stats.to_string(index=False))
    
    # Save statistics to CSV
    stats_file = os.path.splitext(args.output)[0] + '_stats.csv'
    stats.to_csv(stats_file, index=False)
    print(f"Statistics saved to '{stats_file}'")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main() 