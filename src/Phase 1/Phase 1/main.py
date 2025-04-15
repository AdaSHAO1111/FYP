#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for Phase 1 of Indoor Navigation System
This script runs the complete data processing pipeline for Phase 1 requirements.

Author: AI Assistant
Date: 2023
"""

import os
import argparse
import time
import logging
from data_parser_integrated import DataParser
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_comparison(output_dir, data_id):
    """
    Create comparison visualizations between raw and cleaned data
    """
    try:
        # Paths to data files
        gyro_file = os.path.join(output_dir, f"{data_id}_cleaned_gyro_data.csv")
        compass_file = os.path.join(output_dir, f"{data_id}_cleaned_compass_data.csv")
        ground_truth_file = os.path.join(output_dir, f"{data_id}_cleaned_ground_truth_data.csv")
        
        # Check if files exist
        files_exist = all(os.path.exists(f) for f in [gyro_file, compass_file])
        
        if not files_exist:
            logger.warning("One or more data files missing, skipping comparison visualization")
            return False
        
        # Create detailed visualizations showing data distributions
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Load data
        import pandas as pd
        import numpy as np
        import seaborn as sns
        
        gyro_data = pd.read_csv(gyro_file)
        compass_data = pd.read_csv(compass_file)
        
        # Plot gyro data distributions
        for i, col in enumerate(['value_1', 'value_2', 'value_3']):
            if col in gyro_data.columns:
                sns.kdeplot(gyro_data[col], ax=axes[0, 0], label=f'Gyro {["X", "Y", "Z"][i]}')
        
        axes[0, 0].set_title('Gyroscope Data Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot compass data distribution
        if 'value_1' in compass_data.columns:
            sns.histplot(compass_data['value_1'], kde=True, ax=axes[0, 1])
            axes[0, 1].set_title('Compass Heading Distribution')
            axes[0, 1].grid(True)
        
        # Plot correlation between gyro axes
        gyro_cols = [col for col in ['value_1', 'value_2', 'value_3'] if col in gyro_data.columns]
        if len(gyro_cols) > 1:
            corr = gyro_data[gyro_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[1, 0])
            axes[1, 0].set_title('Correlation Between Gyro Axes')
        
        # Plot ground truth path or step distribution
        if os.path.exists(ground_truth_file):
            gt_data = pd.read_csv(ground_truth_file)
            if 'value_1' in gt_data.columns and 'value_2' in gt_data.columns:
                axes[1, 1].scatter(gt_data['value_1'], gt_data['value_2'], c='k', alpha=0.7)
                axes[1, 1].plot(gt_data['value_1'], gt_data['value_2'], 'k-', alpha=0.5)
                axes[1, 1].set_title('Ground Truth Path Preview')
                axes[1, 1].set_xlabel('X Coordinate')
                axes[1, 1].set_ylabel('Y Coordinate')
                axes[1, 1].grid(True)
                axes[1, 1].set_aspect('equal', 'datalim')
        else:
            # If no ground truth data, show step distribution
            if 'step' in gyro_data.columns:
                sns.histplot(gyro_data['step'], kde=True, ax=axes[1, 1])
                axes[1, 1].set_title('Step Distribution')
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = os.path.join(output_dir, f"{data_id}_data_distributions.png")
        plt.savefig(viz_file, dpi=300)
        plt.close()
        
        logger.info(f"Comparison visualization saved to {viz_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating comparison visualization: {str(e)}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Phase 1 data processing pipeline for indoor navigation system.')
    parser.add_argument('--input', type=str, required=False, 
                        default='../../Data_collected/1536_CompassGyroSumHeadingData.txt',
                        help='Path to the input data file')
    parser.add_argument('--output', type=str, default='../../Output/Phase 1', 
                        help='Path to the output directory')
    
    args = parser.parse_args()
    
    # Extract data ID from filename
    data_id = os.path.basename(args.input).split('_')[0]
    
    # Print header
    print("\n" + "="*80)
    print(f"Phase 1: Data Preprocessing and Classification - Dataset ID: {data_id}")
    print("="*80 + "\n")
    
    # Record start time
    start_time = time.time()
    
    # Step 1: Parse and clean data using the integrated parser
    print("\nStep 1: Parsing and Cleaning Data")
    print("-"*40)
    parser = DataParser(args.input, args.output)
    if not parser.process():
        print("Error in data parsing and cleaning. Exiting.")
        return
    
    # Step 2: Generate additional comparison visualizations
    print("\nStep 2: Generating Additional Visualizations")
    print("-"*40)
    if not visualize_comparison(args.output, data_id):
        print("Warning: Could not generate comparison visualizations")
    
    # Calculate and display elapsed time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    # Print summary
    print("\n" + "="*80)
    print("Phase 1 Processing Complete")
    print(f"Time taken: {int(minutes)} minutes, {seconds:.2f} seconds")
    print(f"Output saved to: {args.output}")
    print("="*80 + "\n")
    
    # List generated files
    print("Generated files:")
    for file in os.listdir(args.output):
        if file.startswith(data_id):
            file_size = os.path.getsize(os.path.join(args.output, file)) / 1024  # Size in KB
            if file_size < 1024:
                print(f"  - {file} ({file_size:.2f} KB)")
            else:
                print(f"  - {file} ({file_size/1024:.2f} MB)")

if __name__ == "__main__":
    main() 