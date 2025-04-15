#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Comparison Visualization
This script creates visualizations comparing raw and cleaned data to illustrate 
the impact of the cleaning process on anomaly amplitude/frequency and data reliability.

Author: AI Assistant
Date: 2023
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
from matplotlib.gridspec import GridSpec

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataComparisonVisualizer:
    """
    Class to create visualizations comparing raw and cleaned data
    """
    
    def __init__(self, raw_data_path, cleaned_data_dir, output_dir='Output/Phase 1'):
        """
        Initialize with paths to raw and cleaned data
        
        Args:
            raw_data_path (str): Path to raw data file
            cleaned_data_dir (str): Directory containing cleaned data files
            output_dir (str): Directory to save output visualizations
        """
        self.raw_data_path = raw_data_path
        self.cleaned_data_dir = cleaned_data_dir
        self.output_dir = output_dir
        
        # Extract data ID from filename
        self.data_id = os.path.basename(raw_data_path).split('_')[0]
        
        # Initialize data containers
        self.raw_data = None
        self.raw_gyro = None
        self.raw_compass = None
        self.raw_ground_truth = None
        
        self.cleaned_gyro = None
        self.cleaned_compass = None
        self.cleaned_ground_truth = None
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self):
        """
        Load raw and cleaned data for comparison
        """
        logger.info("Loading raw and cleaned data...")
        
        try:
            # Load raw data
            raw_data = pd.read_csv(self.raw_data_path, delimiter=';')
            
            # Find the index of the first occurrence of 'Initial_Location'
            initial_location_indices = raw_data[raw_data['Type'] == 'Initial_Location'].index
            
            if len(initial_location_indices) > 0:
                initial_location_index = initial_location_indices[0]
                # Slice the DataFrame from the first occurrence onwards
                self.raw_data = raw_data.iloc[initial_location_index:].reset_index(drop=True)
            else:
                self.raw_data = raw_data
            
            # Convert numeric columns
            numeric_cols = ['step', 'value_1', 'value_2', 'value_3', 'value_4', 'value_5', 'GroundTruth', 'turns']
            for col in numeric_cols:
                if col in self.raw_data.columns:
                    self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')
            
            # Extract raw sensor data
            self.raw_gyro = self.raw_data[self.raw_data['Type'] == 'Gyro'].reset_index(drop=True)
            self.raw_compass = self.raw_data[self.raw_data['Type'] == 'Compass'].reset_index(drop=True)
            self.raw_ground_truth = self.raw_data[(self.raw_data['Type'] == 'Ground_truth_Location') | 
                                               (self.raw_data['Type'] == 'Initial_Location')].reset_index(drop=True)
            
            # Load cleaned data files
            gyro_file = os.path.join(self.cleaned_data_dir, f"{self.data_id}_cleaned_gyro_data.csv")
            compass_file = os.path.join(self.cleaned_data_dir, f"{self.data_id}_cleaned_compass_data.csv")
            ground_truth_file = os.path.join(self.cleaned_data_dir, f"{self.data_id}_cleaned_ground_truth_data.csv")
            
            if os.path.exists(gyro_file):
                self.cleaned_gyro = pd.read_csv(gyro_file)
                logger.info(f"Loaded cleaned gyro data: {len(self.cleaned_gyro)} records")
            
            if os.path.exists(compass_file):
                self.cleaned_compass = pd.read_csv(compass_file)
                logger.info(f"Loaded cleaned compass data: {len(self.cleaned_compass)} records")
            
            if os.path.exists(ground_truth_file):
                self.cleaned_ground_truth = pd.read_csv(ground_truth_file)
                logger.info(f"Loaded cleaned ground truth data: {len(self.cleaned_ground_truth)} records")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def compare_gyro_data(self):
        """
        Compare raw and cleaned gyroscope data
        """
        if self.raw_gyro is None or self.cleaned_gyro is None:
            logger.warning("Gyroscope data not available for comparison")
            return False
        
        logger.info("Creating gyroscope data comparison visualization...")
        
        # Create figure with subplots for each axis
        fig = plt.figure(figsize=(15, 18))
        gs = GridSpec(4, 2, height_ratios=[3, 3, 3, 2])
        
        axes_titles = ['X-axis Angular Velocity', 'Y-axis Angular Velocity', 'Z-axis Angular Velocity']
        value_cols = ['value_1', 'value_2', 'value_3']
        
        # Sample a subset of data for clearer visualization if data is large
        sample_size = min(5000, len(self.raw_gyro))
        sample_indices = np.linspace(0, len(self.raw_gyro)-1, sample_size, dtype=int)
        
        raw_gyro_sample = self.raw_gyro.iloc[sample_indices]
        
        # Make sure we have the same number of samples in cleaned data
        if len(self.cleaned_gyro) > sample_size:
            cleaned_sample_indices = np.linspace(0, len(self.cleaned_gyro)-1, sample_size, dtype=int)
            cleaned_gyro_sample = self.cleaned_gyro.iloc[cleaned_sample_indices]
        else:
            cleaned_gyro_sample = self.cleaned_gyro
        
        # Create subplots for each axis
        for i, (title, col) in enumerate(zip(axes_titles, value_cols)):
            # Left plot: Raw vs Cleaned overlaid
            ax_left = fig.add_subplot(gs[i, 0])
            ax_left.plot(raw_gyro_sample[col], 'r-', alpha=0.5, label='Raw')
            ax_left.plot(cleaned_gyro_sample[col], 'b-', label='Cleaned')
            ax_left.set_title(f'{title} - Raw vs Cleaned')
            ax_left.set_xlabel('Sample Index')
            ax_left.set_ylabel('Value')
            ax_left.legend()
            ax_left.grid(True)
            
            # Right plot: Histogram to show distribution change
            ax_right = fig.add_subplot(gs[i, 1])
            sns.histplot(raw_gyro_sample[col], bins=50, alpha=0.5, label='Raw', ax=ax_right, kde=True, color='r')
            sns.histplot(cleaned_gyro_sample[col], bins=50, alpha=0.5, label='Cleaned', ax=ax_right, kde=True, color='b')
            ax_right.set_title(f'{title} - Distribution')
            ax_right.set_xlabel('Value')
            ax_right.set_ylabel('Frequency')
            ax_right.legend()
        
        # Add statistics at the bottom
        ax_stats = fig.add_subplot(gs[3, :])
        ax_stats.axis('off')
        
        # Calculate statistics for each column
        stats_text = "Statistical Comparison (Raw → Cleaned):\n\n"
        
        for i, (title, col) in enumerate(zip(axes_titles, value_cols)):
            raw_mean = raw_gyro_sample[col].mean()
            raw_std = raw_gyro_sample[col].std()
            raw_min = raw_gyro_sample[col].min()
            raw_max = raw_gyro_sample[col].max()
            
            cleaned_mean = cleaned_gyro_sample[col].mean()
            cleaned_std = cleaned_gyro_sample[col].std()
            cleaned_min = cleaned_gyro_sample[col].min()
            cleaned_max = cleaned_gyro_sample[col].max()
            
            # Count outliers in raw data using IQR method
            Q1 = raw_gyro_sample[col].quantile(0.25)
            Q3 = raw_gyro_sample[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((raw_gyro_sample[col] < (Q1 - 1.5 * IQR)) | 
                             (raw_gyro_sample[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_percentage = outlier_count / len(raw_gyro_sample) * 100
            
            # Calculate noise reduction as percentage decrease in std deviation
            noise_reduction = ((raw_std - cleaned_std) / raw_std) * 100 if raw_std != 0 else 0
            
            stats_text += f"{title}:\n"
            stats_text += f"  Mean: {raw_mean:.4f} → {cleaned_mean:.4f}\n"
            stats_text += f"  Std Dev: {raw_std:.4f} → {cleaned_std:.4f} ({noise_reduction:.1f}% reduction)\n"
            stats_text += f"  Range: [{raw_min:.4f}, {raw_max:.4f}] → [{cleaned_min:.4f}, {cleaned_max:.4f}]\n"
            stats_text += f"  Outliers: {outlier_count} ({outlier_percentage:.1f}% of data)\n\n"
        
        ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12, 
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(self.output_dir, f"{self.data_id}_gyro_data_comparison.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved gyroscope comparison visualization to {output_file}")
        return True
    
    def compare_compass_data(self):
        """
        Compare raw and cleaned compass data
        """
        if self.raw_compass is None or self.cleaned_compass is None:
            logger.warning("Compass data not available for comparison")
            return False
        
        logger.info("Creating compass data comparison visualization...")
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 15))
        
        # Sample a subset of data for clearer visualization if data is large
        sample_size = min(5000, len(self.raw_compass))
        sample_indices = np.linspace(0, len(self.raw_compass)-1, sample_size, dtype=int)
        
        raw_compass_sample = self.raw_compass.iloc[sample_indices]
        
        # Make sure we have the same number of samples in cleaned data
        if len(self.cleaned_compass) > sample_size:
            cleaned_sample_indices = np.linspace(0, len(self.cleaned_compass)-1, sample_size, dtype=int)
            cleaned_compass_sample = self.cleaned_compass.iloc[cleaned_sample_indices]
        else:
            cleaned_compass_sample = self.cleaned_compass
        
        # First subplot: Raw vs Cleaned heading values
        axes[0].plot(raw_compass_sample['value_1'], 'r-', alpha=0.5, label='Raw')
        axes[0].plot(cleaned_compass_sample['value_1'], 'b-', label='Cleaned')
        axes[0].set_title('Compass Heading - Raw vs Cleaned')
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Heading (degrees)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Second subplot: Zoomed section to show smoothing effect
        # Choose a representative section (20% of data)
        zoom_size = sample_size // 5
        start_idx = sample_size // 3  # Start at 1/3 of the data
        
        axes[1].plot(raw_compass_sample['value_1'].iloc[start_idx:start_idx+zoom_size], 
                    'r-', alpha=0.5, label='Raw')
        axes[1].plot(cleaned_compass_sample['value_1'].iloc[start_idx:start_idx+zoom_size], 
                    'b-', label='Cleaned')
        axes[1].set_title('Compass Heading - Zoomed View (Shows Smoothing Effect)')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Heading (degrees)')
        axes[1].legend()
        axes[1].grid(True)
        
        # Third subplot: Distribution histograms
        sns.histplot(raw_compass_sample['value_1'], bins=50, alpha=0.5, label='Raw', 
                    ax=axes[2], kde=True, color='r')
        sns.histplot(cleaned_compass_sample['value_1'], bins=50, alpha=0.5, label='Cleaned', 
                    ax=axes[2], kde=True, color='b')
        axes[2].set_title('Compass Heading - Distribution')
        axes[2].set_xlabel('Heading (degrees)')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        
        # Add statistics text
        raw_mean = raw_compass_sample['value_1'].mean()
        raw_std = raw_compass_sample['value_1'].std()
        raw_min = raw_compass_sample['value_1'].min()
        raw_max = raw_compass_sample['value_1'].max()
        
        cleaned_mean = cleaned_compass_sample['value_1'].mean()
        cleaned_std = cleaned_compass_sample['value_1'].std()
        cleaned_min = cleaned_compass_sample['value_1'].min()
        cleaned_max = cleaned_compass_sample['value_1'].max()
        
        # Calculate smoothing effect as percentage decrease in std deviation
        smoothing_effect = ((raw_std - cleaned_std) / raw_std) * 100 if raw_std != 0 else 0
        
        stats_text = (f"Statistical Comparison (Raw → Cleaned):\n"
                     f"  Mean: {raw_mean:.2f}° → {cleaned_mean:.2f}°\n"
                     f"  Std Dev: {raw_std:.2f}° → {cleaned_std:.2f}° ({smoothing_effect:.1f}% reduction)\n"
                     f"  Range: [{raw_min:.2f}°, {raw_max:.2f}°] → [{cleaned_min:.2f}°, {cleaned_max:.2f}°]")
        
        # Add text box with statistics
        plt.figtext(0.5, 0.01, stats_text, ha='center', va='bottom', fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # Make room for the text box
        
        # Save the figure
        output_file = os.path.join(self.output_dir, f"{self.data_id}_compass_data_comparison.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved compass comparison visualization to {output_file}")
        return True
    
    def visualize_anomaly_detection(self):
        """
        Visualize anomaly detection in raw vs cleaned data
        """
        if self.raw_gyro is None or self.cleaned_gyro is None:
            logger.warning("Gyroscope data not available for anomaly visualization")
            return False
        
        logger.info("Creating anomaly detection visualization...")
        
        # Create figure with 3 subplots for each axis
        fig, axes = plt.subplots(3, 1, figsize=(15, 18))
        
        value_cols = ['value_1', 'value_2', 'value_3']
        axis_names = ['X-axis', 'Y-axis', 'Z-axis']
        
        for i, (col, axis_name) in enumerate(zip(value_cols, axis_names)):
            # Get raw and cleaned values - use all data for better visualization
            raw_values = self.raw_gyro[col].values
            cleaned_values = self.cleaned_gyro[col].values
            
            # Adjust cleaned values array length if needed
            if len(cleaned_values) != len(raw_values):
                # Either truncate or pad the cleaned values
                if len(cleaned_values) > len(raw_values):
                    cleaned_values = cleaned_values[:len(raw_values)]
                else:
                    # Pad with last value
                    padding = np.full(len(raw_values) - len(cleaned_values), cleaned_values[-1])
                    cleaned_values = np.append(cleaned_values, padding)
            
            # Calculate outliers using IQR method
            Q1 = np.percentile(raw_values, 25)
            Q3 = np.percentile(raw_values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers
            outliers_mask = (raw_values < lower_bound) | (raw_values > upper_bound)
            outliers_indices = np.where(outliers_mask)[0]
            
            # Sample indices for clearer visualization
            sample_indices = np.arange(len(raw_values))
            
            # Plot raw data with blue color
            axes[i].plot(sample_indices, raw_values, 'b-', linewidth=1, alpha=0.7, label='Raw Data')
            
            # Plot cleaned data with dark blue color
            axes[i].plot(sample_indices, cleaned_values, 'r-', linewidth=1.5, label='Cleaned Data')
            
            # Highlight outliers with orange dots and black edges
            if len(outliers_indices) > 0:
                axes[i].scatter(
                    outliers_indices, 
                    raw_values[outliers_indices],
                    c='orange', 
                    s=30, 
                    label='Detected Anomalies',
                    zorder=5,
                    edgecolor='black',
                    linewidth=0.5
                )
            
            # Add grid
            axes[i].grid(True, linestyle='--', alpha=0.7)
            
            # Add title and labels
            axes[i].set_title(f'Gyroscope {axis_name} Anomaly Detection', fontsize=14)
            axes[i].set_xlabel('Sample Index', fontsize=12)
            axes[i].set_ylabel('Angular Velocity', fontsize=12)
            axes[i].legend(loc='upper right')
            
            # Add text with anomaly statistics
            outlier_count = len(outliers_indices)
            outlier_percentage = (outlier_count / len(raw_values)) * 100
            
            stats_text = f"Anomalies: {outlier_count} ({outlier_percentage:.1f}% of data)"
            axes[i].annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # Add background shading for areas with anomalies to highlight them better
            for idx in outliers_indices:
                # Set a window around each anomaly
                start_idx = max(0, idx-5)
                end_idx = min(len(raw_values), idx+5)
                axes[i].axvspan(start_idx, end_idx, color='red', alpha=0.1)
        
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(self.output_dir, f"{self.data_id}_gyro_anomalies.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved anomaly detection visualization to {output_file}")
        return True
    
    def create_summary_visualization(self):
        """
        Create a summary visualization showing the overall impact of data cleaning
        """
        logger.info("Creating summary visualization...")
        
        # Create summary statistics
        summary_stats = {}
        
        # Gyroscope data statistics
        if self.raw_gyro is not None and self.cleaned_gyro is not None:
            gyro_stats = {}
            for axis, col in enumerate(['value_1', 'value_2', 'value_3']):
                # Calculate statistics
                raw_mean = self.raw_gyro[col].mean()
                raw_std = self.raw_gyro[col].std()
                
                cleaned_mean = self.cleaned_gyro[col].mean()
                cleaned_std = self.cleaned_gyro[col].std()
                
                # Count outliers in raw data
                Q1 = self.raw_gyro[col].quantile(0.25)
                Q3 = self.raw_gyro[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((self.raw_gyro[col] < (Q1 - 1.5 * IQR)) | 
                                (self.raw_gyro[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_percentage = outlier_count / len(self.raw_gyro) * 100
                
                # Calculate noise reduction
                noise_reduction = ((raw_std - cleaned_std) / raw_std) * 100 if raw_std != 0 else 0
                
                gyro_stats[f'Axis {axis+1}'] = {
                    'raw_mean': raw_mean,
                    'raw_std': raw_std,
                    'cleaned_mean': cleaned_mean,
                    'cleaned_std': cleaned_std,
                    'outlier_pct': outlier_percentage,
                    'noise_reduction': noise_reduction
                }
            
            summary_stats['Gyroscope'] = gyro_stats
        
        # Compass data statistics
        if self.raw_compass is not None and self.cleaned_compass is not None:
            raw_mean = self.raw_compass['value_1'].mean()
            raw_std = self.raw_compass['value_1'].std()
            
            cleaned_mean = self.cleaned_compass['value_1'].mean()
            cleaned_std = self.cleaned_compass['value_1'].std()
            
            # Calculate smoothing effect
            smoothing_effect = ((raw_std - cleaned_std) / raw_std) * 100 if raw_std != 0 else 0
            
            summary_stats['Compass'] = {
                'raw_mean': raw_mean,
                'raw_std': raw_std,
                'cleaned_mean': cleaned_mean,
                'cleaned_std': cleaned_std,
                'smoothing_effect': smoothing_effect
            }
        
        # Create a summary bar chart
        if 'Gyroscope' in summary_stats:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Data for bar chart
            axes = list(summary_stats['Gyroscope'].keys())
            noise_reduction = [summary_stats['Gyroscope'][axis]['noise_reduction'] for axis in axes]
            outlier_pct = [summary_stats['Gyroscope'][axis]['outlier_pct'] for axis in axes]
            
            if 'Compass' in summary_stats:
                axes.append('Compass')
                noise_reduction.append(summary_stats['Compass']['smoothing_effect'])
                outlier_pct.append(0)  # We don't have outlier stats for compass
            
            # Set up bar positions
            x = np.arange(len(axes))
            width = 0.35
            
            # Create bars
            ax.bar(x - width/2, outlier_pct, width, label='Anomalies (%)', color='orange')
            ax.bar(x + width/2, noise_reduction, width, label='Noise Reduction (%)', color='green')
            
            # Add labels and title
            ax.set_xlabel('Sensor Axis')
            ax.set_ylabel('Percentage (%)')
            ax.set_title(f'Data Cleaning Impact Summary - Dataset {self.data_id}')
            ax.set_xticks(x)
            ax.set_xticklabels(axes)
            ax.legend()
            
            # Add a note about what the chart shows
            note = ("This chart shows the percentage of anomalies detected in the raw data\n"
                   "and the percentage reduction in noise/variability after cleaning.")
            plt.figtext(0.5, 0.01, note, ha='center', fontsize=10)
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1)
            
            # Save the figure
            output_file = os.path.join(self.output_dir, f"{self.data_id}_cleaning_impact_summary.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved cleaning impact summary to {output_file}")
        
        return True
    
    def run_all_comparisons(self):
        """
        Run all comparison visualizations
        """
        if not self.load_data():
            logger.error("Failed to load data for comparison")
            return False
        
        self.compare_gyro_data()
        self.compare_compass_data()
        self.visualize_anomaly_detection()
        self.create_summary_visualization()
        
        logger.info("All comparison visualizations completed successfully")
        return True


def main():
    """
    Main entry point for the script
    """
    parser = argparse.ArgumentParser(description="Create visualizations comparing raw and cleaned data")
    parser.add_argument("--raw", required=True, help="Path to raw data file")
    parser.add_argument("--cleaned", required=True, help="Directory containing cleaned data files")
    parser.add_argument("--output", default="Output/Phase 1", help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Create visualizer and run comparisons
    visualizer = DataComparisonVisualizer(args.raw, args.cleaned, args.output)
    success = visualizer.run_all_comparisons()
    
    if success:
        print(f"Data comparison visualizations completed successfully. Results saved to {args.output}")
    else:
        print("Data comparison visualizations failed.")


if __name__ == "__main__":
    main() 