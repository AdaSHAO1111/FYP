import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

# Set plot parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.0,
    'font.family': "Arial"
})

# Parameters for simulation
n_steps = 150
step_length = 0.66  # meters per step

# Generate step and distance data
steps = np.arange(1, n_steps + 1)
walked_distance = steps * step_length

# Simulate the data (in a real case, this would be your actual data)
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

# Create the figure with the same style as the example
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

# Set limits to match example
ax1.set_ylim(0, 35)
ax1.set_xlim(0, max(walked_distance))

# Adjust ticks for better readability
ax1.xaxis.set_major_locator(MultipleLocator(25))
ax1.yaxis.set_major_locator(MultipleLocator(5))
secx.xaxis.set_major_locator(MultipleLocator(25))

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig('positioning_error_comparison.png')

# Show the plot
plt.show()

print("Plot created successfully and saved as 'positioning_error_comparison.png'") 