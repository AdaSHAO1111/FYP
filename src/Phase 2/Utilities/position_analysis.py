import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

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

# Simulate data for traditional, LSTM, and ensemble methods
# In a real scenario, you would load this data from your model outputs
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

# Calculate cumulative error (error growth over distance)
df['Traditional_Cumulative_Error'] = df['Traditional_Error'].cumsum()
df['LSTM_Cumulative_Error'] = df['LSTM_Error'].cumsum()
df['Ensemble_Cumulative_Error'] = df['Ensemble_Error'].cumsum()

# Calculate error per distance ratio (error/meter)
df['Traditional_Error_Per_Meter'] = df['Traditional_Error'] / df['Walked_distance']
df['LSTM_Error_Per_Meter'] = df['LSTM_Error'] / df['Walked_distance']
df['Ensemble_Error_Per_Meter'] = df['Ensemble_Error'] / df['Walked_distance']

# Calculate relative contribution of each error component to total error
df['Traditional_X_Error_Ratio'] = np.abs(trad_error_x) / (np.abs(trad_error_x) + np.abs(trad_error_y))
df['Traditional_Y_Error_Ratio'] = np.abs(trad_error_y) / (np.abs(trad_error_x) + np.abs(trad_error_y))
df['LSTM_X_Error_Ratio'] = np.abs(lstm_error_x) / (np.abs(lstm_error_x) + np.abs(lstm_error_y))
df['LSTM_Y_Error_Ratio'] = np.abs(lstm_error_y) / (np.abs(lstm_error_x) + np.abs(lstm_error_y))
df['Ensemble_X_Error_Ratio'] = np.abs(ens_error_x) / (np.abs(ens_error_x) + np.abs(ens_error_y))
df['Ensemble_Y_Error_Ratio'] = np.abs(ens_error_y) / (np.abs(ens_error_x) + np.abs(ens_error_y))

# Replace NaN values (in case of division by zero)
df = df.fillna(0)

# Print summary statistics
print("===== ERROR STATISTICS =====")
for method in ['Traditional', 'LSTM', 'Ensemble']:
    print(f"\n{method} Method:")
    
    # Calculate key metrics
    mean_error = df[f'{method}_Error'].mean()
    max_error = df[f'{method}_Error'].max()
    final_error = df[f'{method}_Error'].iloc[-1]
    
    rmse = math.sqrt(mean_squared_error(
        np.column_stack((df['Ground_X'], df['Ground_Y'])), 
        np.column_stack((df[f'{method}_X'], df[f'{method}_Y']))
    ))
    
    mae = mean_absolute_error(
        np.column_stack((df['Ground_X'], df['Ground_Y'])), 
        np.column_stack((df[f'{method}_X'], df[f'{method}_Y']))
    )
    
    print(f"  Mean Error: {mean_error:.2f} m")
    print(f"  Max Error: {max_error:.2f} m")
    print(f"  Final Error: {final_error:.2f} m")
    print(f"  RMSE: {rmse:.2f} m")
    print(f"  MAE: {mae:.2f} m")
    
    # Error growth rate (linear regression)
    from scipy import stats
    slope, _, r_value, _, _ = stats.linregress(df['Walked_distance'], df[f'{method}_Error'])
    print(f"  Error Growth Rate: {slope:.4f} m/m (R²: {r_value**2:.2f})")
    
    # Average error contribution ratios
    x_contrib = df[f'{method}_X_Error_Ratio'].mean() * 100
    y_contrib = df[f'{method}_Y_Error_Ratio'].mean() * 100
    print(f"  Error Contribution: X-axis {x_contrib:.1f}%, Y-axis {y_contrib:.1f}%")

# Create a mapping of methods to colors for consistent plotting
method_colors = {
    'Traditional': 'blue',
    'LSTM': 'green',
    'Ensemble': 'red',
    'Ground Truth': 'black'
}

# 1. Plot Trajectories
plt.figure(figsize=(3.45, 3.0), dpi=600)

plt.plot(ground_x, ground_y, 'k-', linewidth=1.0, label='Ground Truth')
plt.plot(trad_x, trad_y, 'b-', linewidth=0.8, label='Traditional')
plt.plot(lstm_x, lstm_y, 'g-', linewidth=0.8, label='LSTM')
plt.plot(ens_x, ens_y, 'r-', linewidth=0.8, label='Ensemble')

# Mark start and end points
plt.plot(ground_x[0], ground_y[0], 'ko', markersize=3, label='Start')
plt.plot(ground_x[-1], ground_y[-1], 'kx', markersize=3, label='End')

plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
plt.xlabel('X Position (m)', labelpad=3)
plt.ylabel('Y Position (m)', labelpad=3)
plt.legend(loc='best')
plt.title('2D Trajectories Comparison')
plt.axis('equal')  # Equal scaling for x and y
plt.tight_layout()
plt.savefig('trajectories_comparison.png', dpi=600)

# 2. Plot Distance Error vs. Walking Distance
plt.figure(figsize=(3.45, 2.5), dpi=600)

for method in ['Traditional', 'LSTM', 'Ensemble']:
    plt.plot(df['Walked_distance'], df[f'{method}_Error'], 
             color=method_colors[method], linewidth=0.8, label=method)

plt.xlabel('Walked Distance (m)', labelpad=3)
plt.ylabel('Positioning Error (m)', labelpad=3)
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
plt.legend()

# Secondary x-axis for step numbers
ax = plt.gca()
secx = ax.secondary_xaxis('top', functions=(lambda x: x/step_length, lambda x: x*step_length))
secx.set_xlabel('Number of Steps', labelpad=3)

plt.tight_layout()
plt.savefig('error_vs_distance.png', dpi=600)

# 3. Plot Cumulative Error Over Walking Distance
plt.figure(figsize=(3.45, 2.5), dpi=600)

for method in ['Traditional', 'LSTM', 'Ensemble']:
    plt.plot(df['Walked_distance'], df[f'{method}_Cumulative_Error'], 
             color=method_colors[method], linewidth=0.8, label=method)

plt.xlabel('Walked Distance (m)', labelpad=3)
plt.ylabel('Cumulative Error (m)', labelpad=3)
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
plt.legend()

ax = plt.gca()
secx = ax.secondary_xaxis('top', functions=(lambda x: x/step_length, lambda x: x*step_length))
secx.set_xlabel('Number of Steps', labelpad=3)

plt.tight_layout()
plt.savefig('cumulative_error.png', dpi=600)

# 4. Plot Error per Meter Ratio
plt.figure(figsize=(3.45, 2.5), dpi=600)

for method in ['Traditional', 'LSTM', 'Ensemble']:
    plt.plot(df['Walked_distance'], df[f'{method}_Error_Per_Meter'], 
             color=method_colors[method], linewidth=0.8, label=method)

plt.xlabel('Walked Distance (m)', labelpad=3)
plt.ylabel('Error per Meter (m/m)', labelpad=3)
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
plt.legend()

ax = plt.gca()
secx = ax.secondary_xaxis('top', functions=(lambda x: x/step_length, lambda x: x*step_length))
secx.set_xlabel('Number of Steps', labelpad=3)

plt.tight_layout()
plt.savefig('error_per_meter.png', dpi=600)

# 5. Box Plot for Positioning Errors
error_data = pd.DataFrame({
    'Method': np.repeat(['Traditional', 'LSTM', 'Ensemble'], n_steps),
    'Error': np.concatenate([
        df['Traditional_Error'], 
        df['LSTM_Error'], 
        df['Ensemble_Error']
    ])
})

plt.figure(figsize=(3.45, 2.5), dpi=600)
sns.boxplot(x='Method', y='Error', data=error_data, palette=method_colors, linewidth=0.6)
plt.xlabel('Method', labelpad=3)
plt.ylabel('Positioning Error (m)', labelpad=3)
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('error_boxplot.png', dpi=600)

# 6. ECDF Plot
plt.figure(figsize=(3.45, 2.5), dpi=600)

for method in ['Traditional', 'LSTM', 'Ensemble']:
    ecdf = sm.distributions.ECDF(df[f'{method}_Error'])
    plt.plot(ecdf.x, ecdf.y, color=method_colors[method], linewidth=0.8, label=method)

plt.xlabel('Positioning Error (m)', labelpad=3)
plt.ylabel('ECDF', labelpad=3)
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('error_ecdf.png', dpi=600)

# 7. Error Contribution Analysis (Stacked Bar Chart)
methods = ['Traditional', 'LSTM', 'Ensemble']
x_contrib = [df[f'{m}_X_Error_Ratio'].mean() * 100 for m in methods]
y_contrib = [df[f'{m}_Y_Error_Ratio'].mean() * 100 for m in methods]

plt.figure(figsize=(3.45, 2.0), dpi=600)
bar_width = 0.6
indices = np.arange(len(methods))

plt.bar(indices, x_contrib, bar_width, color='skyblue', label='X-axis Error')
plt.bar(indices, y_contrib, bar_width, bottom=x_contrib, color='salmon', label='Y-axis Error')

plt.xlabel('Method', labelpad=3)
plt.ylabel('Error Contribution (%)', labelpad=3)
plt.xticks(indices, methods)
plt.ylim(0, 100)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
plt.tight_layout()
plt.savefig('error_contribution.png', dpi=600)

# 8. Moving Average Error (to smooth out fluctuations)
window_size = 20  # Moving average window

plt.figure(figsize=(3.45, 2.5), dpi=600)

for method in ['Traditional', 'LSTM', 'Ensemble']:
    moving_avg = df[f'{method}_Error'].rolling(window=window_size, min_periods=1).mean()
    plt.plot(df['Walked_distance'], moving_avg, 
             color=method_colors[method], linewidth=0.8, label=f"{method} (MA{window_size})")

plt.xlabel('Walked Distance (m)', labelpad=3)
plt.ylabel('Moving Average Error (m)', labelpad=3)
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
plt.legend()

ax = plt.gca()
secx = ax.secondary_xaxis('top', functions=(lambda x: x/step_length, lambda x: x*step_length))
secx.set_xlabel('Number of Steps', labelpad=3)

plt.tight_layout()
plt.savefig('moving_average_error.png', dpi=600)

print("\nAnalysis complete. Check the output directory for generated plots.") 