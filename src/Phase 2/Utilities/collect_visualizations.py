import os
import shutil

# Define paths
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2'
final_report_dir = os.path.join(output_dir, 'Final_Report')

# Create final report directory if it doesn't exist
os.makedirs(final_report_dir, exist_ok=True)

# List of visualization files to collect
visualization_files = [
    # Position comparison plots
    'traditional_position_comparison.png',
    'position_comparison.png',
    'advanced_position_comparison.png',
    'simple_fusion_position_comparison.png',
    
    # Heading comparison plots
    'advanced_gyro_heading_comparison.png',
    'simple_fusion_heading_comparison.png',
    
    # Error comparison plots
    'advanced_error_comparison.png',
    'simple_fusion_error_comparison.png',
    
    # Results files
    'advanced_model_results.csv',
    'simple_fusion_results.csv',
    'final_comparison_summary.md'
]

# Copy files to final report directory
for file in visualization_files:
    source_path = os.path.join(output_dir, file)
    dest_path = os.path.join(final_report_dir, file)
    
    if os.path.exists(source_path):
        shutil.copy2(source_path, dest_path)
        print(f"Copied {file} to {final_report_dir}")
    else:
        print(f"Warning: {file} not found in {output_dir}")

# Create an HTML summary file for easy viewing
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Indoor Position Tracking - Model Comparison Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        .section { margin-bottom: 30px; }
        .image-container { margin: 20px 0; }
        img { max-width: 100%; border: 1px solid #ddd; }
        table { border-collapse: collapse; width: 100%; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Indoor Position Tracking - Model Comparison Report</h1>
    
    <div class="section">
        <h2>Position Tracking Comparison</h2>
        <div class="image-container">
            <h3>Traditional Methods</h3>
            <img src="traditional_position_comparison.png" alt="Traditional Position Tracking">
        </div>
        <div class="image-container">
            <h3>LSTM Methods</h3>
            <img src="position_comparison.png" alt="LSTM Position Tracking">
        </div>
        <div class="image-container">
            <h3>Advanced Model</h3>
            <img src="advanced_position_comparison.png" alt="Advanced Position Tracking">
        </div>
        <div class="image-container">
            <h3>Fusion Model</h3>
            <img src="simple_fusion_position_comparison.png" alt="Fusion Position Tracking">
        </div>
    </div>
    
    <div class="section">
        <h2>Heading Prediction Comparison</h2>
        <div class="image-container">
            <h3>Advanced Model Heading</h3>
            <img src="advanced_gyro_heading_comparison.png" alt="Advanced Heading Comparison">
        </div>
        <div class="image-container">
            <h3>Fusion Model Heading</h3>
            <img src="simple_fusion_heading_comparison.png" alt="Fusion Heading Comparison">
        </div>
    </div>
    
    <div class="section">
        <h2>Error Comparison</h2>
        <div class="image-container">
            <h3>Advanced Model Errors</h3>
            <img src="advanced_error_comparison.png" alt="Advanced Error Comparison">
        </div>
        <div class="image-container">
            <h3>Fusion Model Errors</h3>
            <img src="simple_fusion_error_comparison.png" alt="Fusion Error Comparison">
        </div>
    </div>
    
    <div class="section">
        <h2>Conclusions</h2>
        <p>Please refer to the <a href="final_comparison_summary.md">final comparison summary</a> for detailed analysis and recommendations.</p>
    </div>
</body>
</html>
"""

# Write HTML summary
with open(os.path.join(final_report_dir, 'report.html'), 'w') as f:
    f.write(html_content)
    
print(f"Created HTML report at {os.path.join(final_report_dir, 'report.html')}")
print("All visualizations and results collected successfully!") 