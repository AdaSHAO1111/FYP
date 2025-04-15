import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon
import os

# Output directory
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2'
os.makedirs(output_dir, exist_ok=True)

# Set figure size and style
plt.figure(figsize=(16, 12), dpi=300)
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
})

# Define colors
box_colors = {
    'data': '#E6F2FF',  # Light blue
    'process': '#E6FFE6',  # Light green
    'model': '#FFE6E6',  # Light red
    'evaluation': '#FFF2E6',  # Light orange
    'output': '#F2E6FF'  # Light purple
}

# Define box size and spacing
box_width = 0.25
box_height = 0.05
vertical_gap = 0.08
horizontal_gap = 0.3

# Function to create a box with text
def add_box(ax, x, y, text, box_type='process', width=box_width, height=box_height):
    box = Rectangle((x, y), width, height, facecolor=box_colors[box_type], 
                    edgecolor='black', linewidth=1, alpha=0.8)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, horizontalalignment='center',
            verticalalignment='center', fontsize=9, wrap=True)
    return x, y, width, height

# Function to create an arrow
def add_arrow(ax, start_box, end_box, head_width=0.02, head_length=0.02, fc='black', ec='black'):
    start_x, start_y, start_w, start_h = start_box
    end_x, end_y, end_w, end_h = end_box
    
    # Calculate starting and ending points
    start_point = (start_x + start_w/2, start_y)
    end_point = (end_x + end_w/2, end_y + end_h)
    
    # Create the arrow
    arrow = FancyArrowPatch(start_point, end_point, arrowstyle='->', 
                            linewidth=1, color='black', 
                            connectionstyle='arc3,rad=0.0',
                            shrinkA=0, shrinkB=3)
    ax.add_patch(arrow)

# Create the plot
fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
ax.set_xlim(0, 1.6)
ax.set_ylim(0, 1.4)
ax.axis('off')

# Title
plt.title('LSTM-based Heading Prediction Process Flow', fontsize=16, fontweight='bold', pad=20)

# Add boxes for the main flow
# Data Loading and Preparation
box1 = add_box(ax, 0.1, 1.3, 'Load Raw Sensor Data\n(Gyro & Compass)', 'data')
box2 = add_box(ax, 0.1, 1.22, 'Calculate Ground Truth\nHeading', 'data')
box3 = add_box(ax, 0.1, 1.14, 'Preprocess & Split Data\nby Sensor Type', 'data')
box4 = add_box(ax, 0.1, 1.06, 'Calculate Traditional\nHeading (for comparison)', 'process')

# LSTM Preprocessing - Left Branch (Gyro)
box5a = add_box(ax, 0.4, 1.22, 'Gyro Data\nFeature Extraction', 'process')
box6a = add_box(ax, 0.4, 1.14, 'Scale Features\nMinMaxScaler', 'process')
box7a = add_box(ax, 0.4, 1.06, 'Create Sliding Window\nSequences (20 steps)', 'process')
box8a = add_box(ax, 0.4, 0.98, 'Split Train/Val\n(80/20)', 'process')

# LSTM Preprocessing - Right Branch (Compass)
box5b = add_box(ax, 0.7, 1.22, 'Compass Data\nFeature Extraction', 'process')
box6b = add_box(ax, 0.7, 1.14, 'Scale Features\nMinMaxScaler', 'process')
box7b = add_box(ax, 0.7, 1.06, 'Create Sliding Window\nSequences (20 steps)', 'process')
box8b = add_box(ax, 0.7, 0.98, 'Split Train/Val\n(80/20)', 'process')

# LSTM Model Architecture - Center
box9 = add_box(ax, 0.55, 0.9, 'Bidirectional LSTM Model Architecture', 'model', width=0.4)
box10a = add_box(ax, 0.55, 0.82, 'BiLSTM Layer (64 units, return sequences)', 'model', width=0.4)
box10b = add_box(ax, 0.55, 0.74, 'Dropout (0.2)', 'model', width=0.4)
box10c = add_box(ax, 0.55, 0.66, 'BiLSTM Layer (32 units)', 'model', width=0.4)
box10d = add_box(ax, 0.55, 0.58, 'Dropout (0.2)', 'model', width=0.4)
box10e = add_box(ax, 0.55, 0.50, 'Dense Layer (16 units, ReLU)', 'model', width=0.4)
box10f = add_box(ax, 0.55, 0.42, 'Output Layer (1 unit)', 'model', width=0.4)

# Training and Prediction
box11a = add_box(ax, 0.4, 0.34, 'Train Gyro Model\n(30 epochs, batch=32)', 'process')
box11b = add_box(ax, 0.7, 0.34, 'Train Compass Model\n(30 epochs, batch=32)', 'process')
box12a = add_box(ax, 0.4, 0.26, 'Predict Gyro\nHeadings', 'process')
box12b = add_box(ax, 0.7, 0.26, 'Predict Compass\nHeadings', 'process')

# Evaluation
box13 = add_box(ax, 0.55, 0.18, 'Compare Traditional vs LSTM Methods', 'evaluation', width=0.4)
box14 = add_box(ax, 0.55, 0.10, 'Calculate MAE & RMSE Metrics', 'evaluation', width=0.4)
box15 = add_box(ax, 0.55, 0.02, 'Generate Plots & Save Results', 'output', width=0.4)

# Add arrows for main flow
# Data loading to preprocessing
add_arrow(ax, box1, box2)
add_arrow(ax, box2, box3)
add_arrow(ax, box3, box4)

# Split to gyro and compass branches
add_arrow(ax, box3, box5a)
add_arrow(ax, box3, box5b)

# Gyro preprocessing flow
add_arrow(ax, box5a, box6a)
add_arrow(ax, box6a, box7a)
add_arrow(ax, box7a, box8a)

# Compass preprocessing flow
add_arrow(ax, box5b, box6b)
add_arrow(ax, box6b, box7b)
add_arrow(ax, box7b, box8b)

# Connect to model architecture
add_arrow(ax, box8a, box9)
add_arrow(ax, box8b, box9)

# Model layers flow
add_arrow(ax, box9, box10a)
add_arrow(ax, box10a, box10b)
add_arrow(ax, box10b, box10c)
add_arrow(ax, box10c, box10d)
add_arrow(ax, box10d, box10e)
add_arrow(ax, box10e, box10f)

# Connect to training
add_arrow(ax, box10f, box11a)
add_arrow(ax, box10f, box11b)

# Training to prediction
add_arrow(ax, box11a, box12a)
add_arrow(ax, box11b, box12b)

# Prediction to evaluation
add_arrow(ax, box12a, box13)
add_arrow(ax, box12b, box13)

# Traditional method to evaluation comparison
add_arrow(ax, box4, box13)

# Evaluation flow
add_arrow(ax, box13, box14)
add_arrow(ax, box14, box15)

# Add legend
legend_data = plt.Rectangle((0.05, 0.02), 0.05, 0.05, fc=box_colors['data'], ec='black', alpha=0.8)
legend_process = plt.Rectangle((0.05, 0.09), 0.05, 0.05, fc=box_colors['process'], ec='black', alpha=0.8)
legend_model = plt.Rectangle((0.05, 0.16), 0.05, 0.05, fc=box_colors['model'], ec='black', alpha=0.8)
legend_eval = plt.Rectangle((0.05, 0.23), 0.05, 0.05, fc=box_colors['evaluation'], ec='black', alpha=0.8)
legend_output = plt.Rectangle((0.05, 0.30), 0.05, 0.05, fc=box_colors['output'], ec='black', alpha=0.8)

ax.add_patch(legend_data)
ax.add_patch(legend_process)
ax.add_patch(legend_model)
ax.add_patch(legend_eval)
ax.add_patch(legend_output)

ax.text(0.11, 0.045, 'Data', fontsize=8)
ax.text(0.11, 0.115, 'Process', fontsize=8)
ax.text(0.11, 0.185, 'Model', fontsize=8)
ax.text(0.11, 0.255, 'Evaluation', fontsize=8)
ax.text(0.11, 0.325, 'Output', fontsize=8)

# Annotations to explain key points
ax.text(1.05, 1.25, 'Data Preparation', fontsize=11, fontweight='bold')
ax.text(1.05, 1.2, '• Extract Ground Truth heading from position data', fontsize=8)
ax.text(1.05, 1.17, '• Split data into Gyro and Compass datasets', fontsize=8)
ax.text(1.05, 1.14, '• Calculate traditional headings for comparison', fontsize=8)

ax.text(1.05, 1.05, 'LSTM Model Features', fontsize=11, fontweight='bold')
ax.text(1.05, 1.0, '• Gyro features: axisZAngle, gyroSumFromstart0', fontsize=8)
ax.text(1.05, 0.97, '• Compass features: Magnetic_Field_Magnitude,', fontsize=8)
ax.text(1.05, 0.94, '  gyroSumFromstart0', fontsize=8)
ax.text(1.05, 0.91, '• Window size: 20 time steps', fontsize=8)
ax.text(1.05, 0.88, '• Bidirectional LSTM to capture temporal patterns', fontsize=8)

ax.text(1.05, 0.75, 'Evaluation Results', fontsize=11, fontweight='bold')
ax.text(1.05, 0.7, '• Gyro Traditional: MAE ≈ 98°, RMSE ≈ 162°', fontsize=8)
ax.text(1.05, 0.67, '• Gyro LSTM: MAE ≈ 96°, RMSE ≈ 140°', fontsize=8)
ax.text(1.05, 0.64, '• Compass Traditional: MAE ≈ 217°, RMSE ≈ 252°', fontsize=8)
ax.text(1.05, 0.61, '• Compass LSTM: MAE ≈ 115°, RMSE ≈ 161°', fontsize=8)
ax.text(1.05, 0.58, '• LSTM significantly improves compass accuracy', fontsize=8)
ax.text(1.05, 0.55, '• Modest improvement for gyro accuracy', fontsize=8)

# Save the flowchart
flowchart_file = os.path.join(output_dir, 'lstm_heading_prediction_flowchart.png')
plt.tight_layout()
plt.savefig(flowchart_file, bbox_inches='tight')
print(f"Flowchart saved to: {flowchart_file}")

# Close the plot to free memory
plt.close() 