import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_squared_error
from scipy import stats
import math

# Load data from position_analysis output
# In a real scenario, you would load this from your actual position data
# Here we're simulating it similar to position_analysis.py
np.random.seed(42)  # For reproducibility

# Parameters for simulation
n_steps = 300
step_length = 0.66  # meters per step

# Ground truth trajectory (simulated as a reference)
steps = np.arange(1, n_steps + 1)
walked_distance = steps * step_length

# Generate ground truth positions (circular path)
radius = 50
angle = np.linspace(0, 2*np.pi, n_steps)
ground_x = radius * np.cos(angle)
ground_y = radius * np.sin(angle)

# Function to generate positions with controlled error patterns
def generate_positions(error_scale, drift_factor=0.01, noise_level=0.3):
    # Initial error is small
    initial_error_x = np.random.normal(0, error_scale * 0.5, 1)[0]
    initial_error_y = np.random.normal(0, error_scale * 0.5, 1)[0]
    
    # Error grows with steps (drift)
    drift_error_x = np.cumsum(np.random.normal(0, drift_factor, n_steps))
    drift_error_y = np.cumsum(np.random.normal(0, drift_factor, n_steps))
    
    # Random noise component
    noise_x = np.random.normal(0, noise_level, n_steps)
    noise_y = np.random.normal(0, noise_level, n_steps)
    
    # Total error
    error_x = initial_error_x + drift_error_x + noise_x
    error_y = initial_error_y + drift_error_y + noise_y
    
    # Positions with error
    pos_x = ground_x + error_x
    pos_y = ground_y + error_y
    
    return pos_x, pos_y, error_x, error_y

# Generate positions for each method with different error characteristics
# Traditional method (high error)
trad_x, trad_y, trad_error_x, trad_error_y = generate_positions(1.0, 0.02, 0.5)

# LSTM method (medium error)
lstm_x, lstm_y, lstm_error_x, lstm_error_y = generate_positions(0.6, 0.01, 0.3)

# Ensemble method (lowest error)
ens_x, ens_y, ens_error_x, ens_error_y = generate_positions(0.3, 0.005, 0.2)

# Create DataFrame
df = pd.DataFrame({
    'Step': steps,
    'Walked_distance': walked_distance,
    'Ground_X': ground_x,
    'Ground_Y': ground_y,
    'Traditional_X': trad_x,
    'Traditional_Y': trad_y,
    'LSTM_X': lstm_x,
    'LSTM_Y': lstm_y,
    'Ensemble_X': ens_x,
    'Ensemble_Y': ens_y
})

# Calculate Euclidean distance errors
df['Traditional_Error'] = np.sqrt((df['Traditional_X'] - df['Ground_X'])**2 + 
                               (df['Traditional_Y'] - df['Ground_Y'])**2)
df['LSTM_Error'] = np.sqrt((df['LSTM_X'] - df['Ground_X'])**2 + 
                        (df['LSTM_Y'] - df['Ground_Y'])**2)
df['Ensemble_Error'] = np.sqrt((df['Ensemble_X'] - df['Ground_X'])**2 + 
                            (df['Ensemble_Y'] - df['Ground_Y'])**2)

# ============= ERROR COMPONENT ANALYSIS =============

# 1. Analyze error components by segmenting the path
# Divide the walk into segments (e.g., every 50 steps)
segment_size = 50
n_segments = math.ceil(n_steps / segment_size)
segments = []

for i in range(n_segments):
    start_idx = i * segment_size
    end_idx = min((i + 1) * segment_size, n_steps)
    segment_df = df.iloc[start_idx:end_idx].copy()
    
    # Calculate segment metrics for each method
    for method in ['Traditional', 'LSTM', 'Ensemble']:
        segment_df[f'{method}_Seg_Mean_Error'] = segment_df[f'{method}_Error'].mean()
        segment_df[f'{method}_Seg_Max_Error'] = segment_df[f'{method}_Error'].max()
        segment_df[f'{method}_Seg_Std_Error'] = segment_df[f'{method}_Error'].std()
        
        # Calculate error growth rate within segment
        if end_idx - start_idx > 2:  # Need at least 3 points for regression
            slope, _, r_value, _, _ = stats.linregress(
                segment_df['Walked_distance'], 
                segment_df[f'{method}_Error']
            )
            segment_df[f'{method}_Seg_Growth_Rate'] = slope
            segment_df[f'{method}_Seg_R2'] = r_value**2
        else:
            segment_df[f'{method}_Seg_Growth_Rate'] = np.nan
            segment_df[f'{method}_Seg_R2'] = np.nan
    
    segments.append(segment_df)

# Combine segments back into one DataFrame
segmented_df = pd.concat(segments)

# Create segment summary DataFrame
segment_summary = pd.DataFrame({
    'Segment': range(1, n_segments + 1),
    'Start_Step': [i * segment_size + 1 for i in range(n_segments)],
    'End_Step': [min((i + 1) * segment_size, n_steps) for i in range(n_segments)],
    'Start_Distance': [(i * segment_size + 1) * step_length for i in range(n_segments)],
    'End_Distance': [min((i + 1) * segment_size, n_steps) * step_length for i in range(n_segments)]
})

# Add metrics for each method and segment
for method in ['Traditional', 'LSTM', 'Ensemble']:
    segment_summary[f'{method}_Mean_Error'] = [
        df.iloc[i * segment_size:min((i + 1) * segment_size, n_steps)][f'{method}_Error'].mean() 
        for i in range(n_segments)
    ]
    
    segment_summary[f'{method}_Max_Error'] = [
        df.iloc[i * segment_size:min((i + 1) * segment_size, n_steps)][f'{method}_Error'].max() 
        for i in range(n_segments)
    ]
    
    segment_summary[f'{method}_Error_Growth'] = [
        df.iloc[min((i + 1) * segment_size - 1, n_steps - 1)][f'{method}_Error'] - 
        df.iloc[i * segment_size][f'{method}_Error'] 
        for i in range(n_segments)
    ]
    
    # Calculate percentage contribution to total error
    total_error = df[f'{method}_Error'].sum()
    segment_summary[f'{method}_Error_Percent'] = [
        df.iloc[i * segment_size:min((i + 1) * segment_size, n_steps)][f'{method}_Error'].sum() / total_error * 100
        for i in range(n_segments)
    ]

# Print segment summary
print("===== ERROR ANALYSIS BY SEGMENT =====")
pd.set_option('display.max_columns', None)
print(segment_summary[['Segment', 'Start_Step', 'End_Step', 'Start_Distance', 'End_Distance'] + 
                    [f'{m}_{metric}' for m in ['Traditional', 'LSTM', 'Ensemble'] 
                     for metric in ['Mean_Error', 'Error_Percent']]])

# ============= ERROR GROWTH MODELS =============

# Analyze error growth patterns
print("\n===== ERROR GROWTH MODELS =====")

for method in ['Traditional', 'LSTM', 'Ensemble']:
    print(f"\n{method} Method:")
    
    # Linear model (error = a*distance + b)
    linear_slope, linear_intercept, r_value, p_value, std_err = stats.linregress(
        df['Walked_distance'], df[f'{method}_Error']
    )
    
    # Logarithmic model (error = a*log(distance) + b)
    with np.errstate(divide='ignore'):  # Ignore log(0) warning
        log_x = np.log(df['Walked_distance'])
    log_x[np.isneginf(log_x)] = 0  # Replace -inf with 0
    log_model = stats.linregress(log_x[1:], df[f'{method}_Error'][1:])  # Skip first point where log(0) = -inf
    
    # Power model (error = a*distance^b)
    # Transform to log-log space: log(error) = log(a) + b*log(distance)
    valid_mask = (df['Walked_distance'] > 0) & (df[f'{method}_Error'] > 0)
    if sum(valid_mask) > 2:
        log_x = np.log(df['Walked_distance'][valid_mask])
        log_y = np.log(df[f'{method}_Error'][valid_mask])
        power_model = stats.linregress(log_x, log_y)
        power_exponent = power_model.slope
        power_coef = np.exp(power_model.intercept)
    else:
        power_exponent = np.nan
        power_coef = np.nan
    
    # Exponential model (error = a*exp(b*distance))
    # Transform: log(error) = log(a) + b*distance
    valid_mask = df[f'{method}_Error'] > 0
    if sum(valid_mask) > 2:
        exp_model = stats.linregress(
            df['Walked_distance'][valid_mask],
            np.log(df[f'{method}_Error'][valid_mask])
        )
        exp_coef = np.exp(exp_model.intercept)
        exp_rate = exp_model.slope
    else:
        exp_coef = np.nan
        exp_rate = np.nan
    
    print(f"  Linear Model: Error = {linear_slope:.4f} * Distance + {linear_intercept:.4f}  (R² = {r_value**2:.4f})")
    print(f"  Log Model: Error = {log_model.slope:.4f} * log(Distance) + {log_model.intercept:.4f}  (R² = {log_model.rvalue**2:.4f})")
    print(f"  Power Model: Error = {power_coef:.4f} * Distance^{power_exponent:.4f}  (R² = {power_model.rvalue**2 if not np.isnan(power_exponent) else np.nan:.4f})")
    print(f"  Exponential Model: Error = {exp_coef:.4f} * exp({exp_rate:.4f} * Distance)  (R² = {exp_model.rvalue**2 if not np.isnan(exp_rate) else np.nan:.4f})")

# ============= VISUALIZATIONS =============

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

# Create a mapping of methods to colors for consistent plotting
method_colors = {
    'Traditional': 'blue',
    'LSTM': 'green',
    'Ensemble': 'red',
    'Ground Truth': 'black'
}

# 1. Error Growth by Segment
plt.figure(figsize=(3.45, 2.5), dpi=600)

x = segment_summary['Segment']
width = 0.25
x_pos = np.arange(len(x))

plt.bar(x_pos - width, segment_summary['Traditional_Mean_Error'], width, 
        color=method_colors['Traditional'], label='Traditional')
plt.bar(x_pos, segment_summary['LSTM_Mean_Error'], width, 
        color=method_colors['LSTM'], label='LSTM')
plt.bar(x_pos + width, segment_summary['Ensemble_Mean_Error'], width, 
        color=method_colors['Ensemble'], label='Ensemble')

plt.xlabel('Path Segment', labelpad=3)
plt.ylabel('Mean Error (m)', labelpad=3)
plt.xticks(x_pos, segment_summary['Segment'])
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, axis='y')
plt.legend()
plt.tight_layout()
plt.savefig('error_by_segment.png', dpi=600)

# 2. Percentage Error Contribution by Segment
plt.figure(figsize=(3.45, 2.5), dpi=600)

for method in ['Traditional', 'LSTM', 'Ensemble']:
    plt.plot(segment_summary['Segment'], segment_summary[f'{method}_Error_Percent'], 
             'o-', color=method_colors[method], linewidth=0.8, markersize=3, label=method)

plt.xlabel('Path Segment', labelpad=3)
plt.ylabel('Error Contribution (%)', labelpad=3)
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('error_contribution_by_segment.png', dpi=600)

# 3. Error Growth Models Visualization
plt.figure(figsize=(3.45, 2.5), dpi=600)

# Distance range for model prediction
dist_range = np.linspace(step_length, max(df['Walked_distance']), 100)

for method in ['Traditional', 'LSTM', 'Ensemble']:
    # Plot actual error
    plt.scatter(df['Walked_distance'], df[f'{method}_Error'], 
               color=method_colors[method], s=3, alpha=0.3, label=f'{method} (Actual)')
    
    # Fit models
    linear_model = stats.linregress(df['Walked_distance'], df[f'{method}_Error'])
    
    # Plot linear model
    plt.plot(dist_range, linear_model.slope * dist_range + linear_model.intercept, 
             '-', color=method_colors[method], linewidth=0.8, 
             label=f'{method} (Linear, R²={linear_model.rvalue**2:.2f})')

plt.xlabel('Walked Distance (m)', labelpad=3)
plt.ylabel('Positioning Error (m)', labelpad=3)
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
plt.tight_layout()
plt.savefig('error_growth_models.png', dpi=600)

# 4. Error Components Distribution
plt.figure(figsize=(3.45, 2.5), dpi=600)

error_components = pd.DataFrame({
    'Method': np.repeat(['Traditional', 'LSTM', 'Ensemble'], n_steps),
    'Error_X': np.concatenate([trad_error_x, lstm_error_x, ens_error_x]),
    'Error_Y': np.concatenate([trad_error_y, lstm_error_y, ens_error_y])
})

sns.violinplot(x='Method', y='Error_X', data=error_components, 
              palette=method_colors, inner='box', scale='width')
plt.xlabel('Method', labelpad=3)
plt.ylabel('X-axis Error (m)', labelpad=3)
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('x_error_distribution.png', dpi=600)

plt.figure(figsize=(3.45, 2.5), dpi=600)
sns.violinplot(x='Method', y='Error_Y', data=error_components, 
              palette=method_colors, inner='box', scale='width')
plt.xlabel('Method', labelpad=3)
plt.ylabel('Y-axis Error (m)', labelpad=3)
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('y_error_distribution.png', dpi=600)

# 5. Error Rate of Change (Derivative)
# Calculate rate of change of error using central differences
for method in ['Traditional', 'LSTM', 'Ensemble']:
    df[f'{method}_Error_Rate'] = np.gradient(df[f'{method}_Error'], df['Walked_distance'])

plt.figure(figsize=(3.45, 2.5), dpi=600)

for method in ['Traditional', 'LSTM', 'Ensemble']:
    plt.plot(df['Walked_distance'], df[f'{method}_Error_Rate'], 
             color=method_colors[method], linewidth=0.8, label=method)

plt.xlabel('Walked Distance (m)', labelpad=3)
plt.ylabel('Error Rate of Change (m/m)', labelpad=3)
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
plt.legend()

ax = plt.gca()
secx = ax.secondary_xaxis('top', functions=(lambda x: x/step_length, lambda x: x*step_length))
secx.set_xlabel('Number of Steps', labelpad=3)

plt.tight_layout()
plt.savefig('error_rate_of_change.png', dpi=600)

print("\nError analysis complete. Check the output directory for generated plots.") 