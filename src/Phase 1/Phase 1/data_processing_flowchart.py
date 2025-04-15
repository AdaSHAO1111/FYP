#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Processing Flowchart Generator
This script creates a visual flowchart of the data processing steps in Phase 1.

Author: AI Assistant
Date: 2023
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path

# Define output directory
output_dir = "Output/Phase1_Roadmap"
os.makedirs(output_dir, exist_ok=True)

def create_flowchart():
    """
    Create a flowchart visualization of the data processing pipeline
    for Phase 1 of the Indoor Navigation System project.
    """
    # Set up figure with a light background
    fig, ax = plt.subplots(figsize=(14, 18), facecolor='#f9f9f9')
    ax.set_facecolor('#f9f9f9')
    
    # Remove axes
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Define colors for different types of nodes
    colors = {
        'start_end': '#4CAF50',      # Green
        'process': '#2196F3',        # Blue
        'decision': '#FFC107',       # Amber
        'data': '#9C27B0',           # Purple
        'arrow': '#757575',          # Gray
        'subprocess': '#FF5722',     # Deep Orange
        'connector': '#607D8B'       # Blue Gray
    }
    
    # Create title
    ax.text(50, 98, 'Indoor Navigation System - Phase 1 Data Processing Flowchart',
            fontsize=20, fontweight='bold', ha='center', va='center')
    
    # Add subtitle
    ax.text(50, 95, 'Data Import, Classification, Cleaning, and Visualization Pipeline',
            fontsize=16, ha='center', va='center', color='#555555')
    
    # Define helper functions for drawing nodes
    def draw_rectangle(x, y, width, height, color, label, fontsize=10):
        rect = patches.Rectangle((x-width/2, y-height/2), width, height, 
                                 facecolor=color, alpha=0.7, edgecolor='black',
                                 linewidth=1, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=fontsize, ha='center', va='center', 
                wrap=True, zorder=3, fontweight='bold')
        return rect
    
    def draw_diamond(x, y, width, height, color, label, fontsize=10):
        diamond = patches.Polygon([(x, y+height/2), (x+width/2, y), 
                                  (x, y-height/2), (x-width/2, y)], 
                                  facecolor=color, alpha=0.7, edgecolor='black',
                                  linewidth=1, zorder=2)
        ax.add_patch(diamond)
        ax.text(x, y, label, fontsize=fontsize, ha='center', va='center', 
                wrap=True, zorder=3, fontweight='bold')
        return diamond
    
    def draw_arrow(start_x, start_y, end_x, end_y, label=None, label_pos=0.5, fontsize=8):
        arrow = patches.FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                        arrowstyle='->', color=colors['arrow'],
                                        linewidth=1.5, connectionstyle='arc3,rad=0',
                                        zorder=1)
        ax.add_patch(arrow)
        if label:
            # Calculate the position of the label
            label_x = start_x + label_pos * (end_x - start_x)
            label_y = start_y + label_pos * (end_y - start_y)
            # Small offset to not overlap with the arrow
            offset = 2 if start_y == end_y else 0
            ax.text(label_x, label_y + offset, label, fontsize=fontsize, 
                   ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, 
                                                      boxstyle='round,pad=0.3'))
        return arrow
    
    # Create the flowchart nodes
    # Start 
    draw_rectangle(50, 88, 24, 6, colors['start_end'], "Start", fontsize=12)
    
    # Data Input
    draw_rectangle(50, 80, 24, 6, colors['data'], "Raw Data Input\n(1536_CompassGyroSumHeadingData.txt)", fontsize=11)
    
    # Read Raw Data
    draw_rectangle(50, 72, 24, 5, colors['process'], "1. Read Raw Data\n(read_raw_data)", fontsize=11)
    
    # Parse Data Structure
    draw_rectangle(50, 64, 24, 5, colors['process'], "2. Parse Data Structure\n(parse_data)", fontsize=11)
    
    # Classify Sensor Data
    draw_rectangle(50, 56, 24, 5, colors['process'], "3. Classify Sensor Data\n(classify_sensor_data)", fontsize=11)
    
    # Data Classification Branches
    draw_rectangle(25, 47, 16, 5, colors['data'], "Gyroscope Data", fontsize=10)
    draw_rectangle(50, 47, 16, 5, colors['data'], "Compass Data", fontsize=10)
    draw_rectangle(75, 47, 20, 5, colors['data'], "Ground Truth Data", fontsize=10)
    
    # Cleaning Processes
    draw_rectangle(25, 38, 18, 6, colors['subprocess'], "Gyro Cleaning\n- Duplicate removal\n- Outlier detection (IQR)\n- Savitzky-Golay filtering", fontsize=9)
    draw_rectangle(50, 38, 18, 6, colors['subprocess'], "Compass Cleaning\n- Duplicate removal\n- Circular data handling\n- Median filtering", fontsize=9)
    draw_rectangle(75, 38, 18, 5, colors['subprocess'], "Ground Truth Processing\n- Calculate headings\n- Verify positioning", fontsize=9)
    
    # Merge Cleaned Data
    draw_rectangle(50, 28, 24, 5, colors['process'], "4. Merge & Save Cleaned Data\n(save_cleaned_data)", fontsize=11)
    
    # Visualization Processes
    draw_rectangle(25, 19, 20, 5, colors['subprocess'], "Gyroscope Visualization\n- Raw vs Cleaned\n- Anomaly Detection", fontsize=9)
    draw_rectangle(50, 19, 20, 5, colors['subprocess'], "Compass Visualization\n- Raw vs Cleaned\n- Smoothing Effects", fontsize=9)
    draw_rectangle(75, 19, 20, 5, colors['subprocess'], "Ground Truth Path\n- Spatial Visualization\n- Time Progression", fontsize=9)
    
    # Output Files
    draw_rectangle(50, 10, 24, 5, colors['data'], "Output Files & Visualizations", fontsize=11)
    
    # End
    draw_rectangle(50, 2, 24, 5, colors['start_end'], "End", fontsize=12)
    
    # Draw connecting arrows
    draw_arrow(50, 85, 50, 82.5)
    draw_arrow(50, 77.5, 50, 74.5)
    draw_arrow(50, 69.5, 50, 66.5)
    draw_arrow(50, 61.5, 50, 58.5)
    
    # From classify to branches
    draw_arrow(50, 53.5, 25, 49.5)
    draw_arrow(50, 53.5, 50, 49.5)
    draw_arrow(50, 53.5, 75, 49.5)
    
    # From branches to cleaning processes
    draw_arrow(25, 44.5, 25, 41)
    draw_arrow(50, 44.5, 50, 41)
    draw_arrow(75, 44.5, 75, 41)
    
    # From cleaning to merge
    draw_arrow(25, 35, 25, 30, "Cleaned\nGyro Data", 0.5, 8)
    draw_arrow(50, 35, 50, 30, "Cleaned\nCompass Data", 0.5, 8)
    draw_arrow(75, 35, 75, 30, "Cleaned\nGround Truth", 0.5, 8)
    
    # From merge to visualization
    draw_arrow(50, 25.5, 25, 21.5)
    draw_arrow(50, 25.5, 50, 21.5)
    draw_arrow(50, 25.5, 75, 21.5)
    
    # From visualizations to output
    draw_arrow(25, 16.5, 50, 12.5, "Gyro\nAnalytics", 0.5, 8)
    draw_arrow(50, 16.5, 50, 12.5, "Compass\nAnalytics", 0.5, 8)
    draw_arrow(75, 16.5, 50, 12.5, "Path\nVisualization", 0.5, 8)
    
    # Final arrow to end
    draw_arrow(50, 7.5, 50, 4.5)
    
    # Add legend for node types
    legend_x = 85
    legend_y = 90
    legend_spacing = 4
    
    # Legend title
    ax.text(legend_x-2, legend_y+3, 'Legend:', fontsize=12, fontweight='bold', ha='left')
    
    # Legend items
    rect = patches.Rectangle((legend_x-8, legend_y-1), 4, 2, facecolor=colors['start_end'], 
                            alpha=0.7, edgecolor='black')
    ax.add_patch(rect)
    ax.text(legend_x, legend_y, 'Start/End', fontsize=9, ha='left', va='center')
    
    rect = patches.Rectangle((legend_x-8, legend_y-1-legend_spacing), 4, 2, facecolor=colors['process'], 
                            alpha=0.7, edgecolor='black')
    ax.add_patch(rect)
    ax.text(legend_x, legend_y-legend_spacing, 'Main Process', fontsize=9, ha='left', va='center')
    
    rect = patches.Rectangle((legend_x-8, legend_y-1-2*legend_spacing), 4, 2, facecolor=colors['subprocess'], 
                            alpha=0.7, edgecolor='black')
    ax.add_patch(rect)
    ax.text(legend_x, legend_y-2*legend_spacing, 'Sub-Process', fontsize=9, ha='left', va='center')
    
    rect = patches.Rectangle((legend_x-8, legend_y-1-3*legend_spacing), 4, 2, facecolor=colors['data'], 
                            alpha=0.7, edgecolor='black')
    ax.add_patch(rect)
    ax.text(legend_x, legend_y-3*legend_spacing, 'Data', fontsize=9, ha='left', va='center')
    
    # Add a note about the data processing
    note_text = (
        "Note: This flowchart illustrates the data processing pipeline for Phase 1 of the Indoor Navigation System. "
        "The process begins with raw sensor data, which is parsed, classified, and cleaned using various techniques. "
        "Each sensor type (Gyroscope, Compass, Ground Truth) undergoes specific cleaning procedures appropriate for its "
        "data characteristics. The cleaned data is then saved and visualized to highlight the impact of data preprocessing."
    )
    
    ax.text(50, -3, note_text, fontsize=9, ha='center', va='top', 
           wrap=True, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save the flowchart
    plt.tight_layout()
    plt.savefig(f"{output_dir}/data_processing_flowchart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Flowchart saved to {output_dir}/data_processing_flowchart.png")

if __name__ == "__main__":
    create_flowchart() 