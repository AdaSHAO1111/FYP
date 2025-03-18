import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import os
import logging
from typing import Dict, List, Tuple, Optional, Union
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataVisualizer:
    """A class for visualizing sensor data for indoor navigation systems."""
    
    def __init__(self, output_dir: str = 'output/plots'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set IEEE-style plot parameters
        self.fontSizeAll = 6
        plt.rcParams.update({
            'xtick.major.pad': '1',
            'ytick.major.pad': '1',
            'legend.fontsize': self.fontSizeAll,
            'legend.handlelength': 2,
            'font.size': self.fontSizeAll,
            'axes.linewidth': 0.2,
            'patch.linewidth': 0.2,
            'font.family': "Times New Roman"
        })
        
        # Image markers for start/end points
        self.start_img_path = os.path.join('data', 'start.png')
        self.end_img_path = os.path.join('data', 'enda.png')
        
    def create_all_visualizations(self, 
                                 gyro_data: pd.DataFrame,
                                 compass_data: pd.DataFrame,
                                 ground_truth_data: pd.DataFrame,
                                 interpolated_positions: pd.DataFrame = None,
                                 positions_gyro: pd.DataFrame = None,
                                 positions_compass: pd.DataFrame = None) -> None:
        """
        Create all visualizations for the sensor data.
        
        Args:
            gyro_data: DataFrame with gyroscope data
            compass_data: DataFrame with compass data
            ground_truth_data: DataFrame with ground truth locations
            interpolated_positions: DataFrame with interpolated ground truth positions
            positions_gyro: DataFrame with positions calculated from gyro data
            positions_compass: DataFrame with positions calculated from compass data
        """
        # Create individual visualizations
        self.plot_heading_comparison(gyro_data, compass_data, ground_truth_data)
        
        if interpolated_positions is not None:
            self.plot_interpolated_positions(interpolated_positions, ground_truth_data)
        
        if positions_gyro is not None and positions_compass is not None:
            self.plot_trajectory_comparison(positions_gyro, positions_compass, interpolated_positions)
            self.plot_position_error(positions_gyro, positions_compass, interpolated_positions)
        
        # Plot additional visualizations
        self.plot_sensor_data_over_time(gyro_data, compass_data, ground_truth_data)
        
        logger.info(f"All visualizations created and saved to {self.output_dir}")
    
    def plot_heading_comparison(self, gyro_data: pd.DataFrame, compass_data: pd.DataFrame, 
                               ground_truth_data: pd.DataFrame, save: bool = True) -> None:
        """
        Plot and compare heading values from gyroscope, compass, and ground truth.
        
        Args:
            gyro_data: DataFrame with gyroscope data
            compass_data: DataFrame with compass data
            ground_truth_data: DataFrame with ground truth data
            save: Whether to save the plot to file
        """
        # Create figure for IEEE column width
        fig, ax = plt.subplots(figsize=(3.45, 2), dpi=300)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95, wspace=0.00, hspace=0.0)
        
        # Ensure data is sorted by timestamp
        gyro_data = gyro_data.sort_values(by="Timestamp_(ms)")
        compass_data = compass_data.sort_values(by="Timestamp_(ms)")
        
        # Plot ground truth heading
        if 'GroundTruthHeadingComputed' in gyro_data.columns:
            plt.plot(gyro_data["Timestamp_(ms)"], gyro_data["GroundTruthHeadingComputed"], 
                     color='red', linestyle='--', linewidth=1.2, label='Ground Truth Heading')
        
        # Plot gyro heading
        if 'GyroStartByGroundTruth' in gyro_data.columns:
            plt.plot(gyro_data["Timestamp_(ms)"], gyro_data["GyroStartByGroundTruth"], 
                     color='blue', linestyle='-', linewidth=1, label='Gyro Heading')
        elif 'gyroSumFromstart0' in gyro_data.columns:
            # Need to calculate it if it doesn't exist
            first_ground_truth = gyro_data['GroundTruth'].iloc[0] if 'GroundTruth' in gyro_data.columns else 0
            gyro_heading = (first_ground_truth + gyro_data['gyroSumFromstart0'] - gyro_data['gyroSumFromstart0'].iloc[0]) % 360
            plt.plot(gyro_data["Timestamp_(ms)"], gyro_heading, 
                     color='blue', linestyle='-', linewidth=1, label='Gyro Heading')
        
        # Plot compass heading
        if 'compass' in compass_data.columns:
            plt.plot(compass_data["Timestamp_(ms)"], compass_data["compass"], 
                     color='green', linestyle='-.', linewidth=1, label='Compass Heading')
        
        # Axis formatting
        ax.yaxis.set_major_locator(MultipleLocator(40))  # Y-axis major tick interval
        ax.yaxis.set_minor_locator(MultipleLocator(20))  # Y-axis minor tick interval
        ax.xaxis.set_major_locator(MultipleLocator(50000))  # X-axis major tick interval
        ax.xaxis.set_minor_locator(MultipleLocator(25000))  # X-axis minor tick interval
        
        plt.xlabel("Timestamp (ms)", labelpad=3)
        plt.ylabel("Heading (Degrees)", labelpad=4)
        
        # Rotate y-tick labels
        plt.yticks(rotation=90, va="center")
        
        # Ticks and grid
        plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in', 
                        which='major', width=0.3, length=2.5)
        plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in', 
                        which='minor', width=0.15, length=1)
        
        # Custom Legend
        legend_elements = [
            Line2D([0], [0], color='red', linestyle='--', linewidth=1.2, label='Ground Truth Heading'),
            Line2D([0], [0], color='blue', linestyle='-', linewidth=1.2, label='Gyro Heading'),
            Line2D([0], [0], color='green', linestyle='-.', linewidth=1.2, label='Compass Heading')
        ]
        
        plt.legend(handles=legend_elements, loc='best')
        
        # Grid
        ax.ticklabel_format(useOffset=False)
        plt.grid(linestyle=':', linewidth=0.5, alpha=0.15, color='k')
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'heading_comparison.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved heading comparison plot to {self.output_dir}/heading_comparison.png")
        
        plt.show()
    
    def plot_interpolated_positions(self, interpolated_df: pd.DataFrame, 
                                   ground_truth_df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot interpolated ground truth positions.
        
        Args:
            interpolated_df: DataFrame with interpolated positions
            ground_truth_df: DataFrame with original ground truth positions
            save: Whether to save the plot to file
        """
        plt.figure(figsize=(10, 6), dpi=300)
        
        # Plot interpolated positions
        plt.scatter(interpolated_df['value_4'], interpolated_df['value_5'],
                    marker='.', s=50, label='Interpolated ground truth')
        
        # Plot original ground truth positions
        plt.scatter(ground_truth_df['value_4'], ground_truth_df['value_5'],
                    marker='*', s=100, label='Manually labeled ground truth')
        
        plt.xlabel('East')
        plt.ylabel('North')
        plt.ticklabel_format(useOffset=False)
        plt.legend()
        plt.grid(True)
        plt.title('Ground Truth and Interpolated Positions')
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'interpolated_positions.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved interpolated positions plot to {self.output_dir}/interpolated_positions.png")
        
        plt.show()
    
    def plot_trajectory_comparison(self, positions_gyro: pd.DataFrame, positions_compass: pd.DataFrame,
                                 interpolated_df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot and compare trajectories from gyroscope, compass, and ground truth.
        
        Args:
            positions_gyro: DataFrame with positions calculated from gyro data
            positions_compass: DataFrame with positions calculated from compass data
            interpolated_df: DataFrame with interpolated ground truth positions
            save: Whether to save the plot to file
        """
        # Create figure for IEEE column width
        fig, ax = plt.subplots(figsize=(3.45, 2.94), dpi=300)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95, wspace=0.00, hspace=0.0)
        
        # Function to add image marker at specific coordinates
        def add_marker(ax, img_path, x, y, zoom=0.1):
            try:
                img = mpimg.imread(img_path)
                imagebox = OffsetImage(img, zoom=zoom)
                ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=100)
                ax.add_artist(ab)
            except Exception as e:
                logger.warning(f"Could not add marker image: {str(e)}")
        
        # Check column names in the dataframes
        gyro_x_col = 'Gyro_X' if 'Gyro_X' in positions_gyro.columns else 'x'
        gyro_y_col = 'Gyro_Y' if 'Gyro_Y' in positions_gyro.columns else 'y'
        compass_x_col = 'Compass_X' if 'Compass_X' in positions_compass.columns else 'x'
        compass_y_col = 'Compass_Y' if 'Compass_Y' in positions_compass.columns else 'y'
        gt_x_col = 'value_4' if 'value_4' in interpolated_df.columns else 'x'
        gt_y_col = 'value_5' if 'value_5' in interpolated_df.columns else 'y'
        
        # Plot Tracks
        plt.plot(positions_compass[compass_x_col], positions_compass[compass_y_col], 
                 color='purple', linestyle='--', linewidth=1.2, label='Compass')
        plt.plot(positions_gyro[gyro_x_col], positions_gyro[gyro_y_col], 
                 color='red', linestyle='-', linewidth=1.2, label='Gyro')
        
        # Ground truth positions 
        plt.scatter(interpolated_df[gt_x_col], interpolated_df[gt_y_col], 
                    c='blue', marker='.', s=30, label='Ground Truth')
        
        if gt_x_col in ground_truth_df.columns and gt_y_col in ground_truth_df.columns:
            # Manually labeled points 
            plt.scatter(ground_truth_df[gt_x_col], ground_truth_df[gt_y_col], 
                        marker='+', s=50, c='green', label='Manually Labeled')
            
            # Add start and end markers on Ground Truth positions if images exist
            if os.path.exists(self.start_img_path) and os.path.exists(self.end_img_path):
                start_x, start_y = ground_truth_df[gt_x_col].iloc[0], ground_truth_df[gt_y_col].iloc[0]+2
                end_x, end_y = ground_truth_df[gt_x_col].iloc[-1], ground_truth_df[gt_y_col].iloc[-1]+2
                
                add_marker(ax, self.start_img_path, start_x, start_y, zoom=0.05)
                add_marker(ax, self.end_img_path, end_x, end_y, zoom=0.013)
        
        # Axis formatting
        ax.yaxis.set_major_locator(MultipleLocator(40))  # Y-axis major tick interval: 40m
        ax.yaxis.set_minor_locator(MultipleLocator(20))  # Y-axis minor tick interval: 20m
        ax.xaxis.set_major_locator(MultipleLocator(40))  # X-axis major tick interval: 40m
        ax.xaxis.set_minor_locator(MultipleLocator(10))  # X-axis minor tick interval: 10m
        
        plt.axis('scaled')
        
        # Labels
        plt.xlabel('East (m)', labelpad=3)
        plt.ylabel('North (m)', labelpad=4)
        
        # Rotate y-tick labels
        plt.yticks(rotation=90, va="center")
        
        # Ticks and grid
        plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in', 
                        which='major', grid_color='blue', width=0.3, length=2.5)
        plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in', 
                        which='minor', grid_color='blue', width=0.15, length=1)
        
        # Legend
        plt.legend(loc='best')
        
        # Grid
        ax.ticklabel_format(useOffset=False)
        plt.grid(linestyle=':', linewidth=0.5, alpha=0.15, color='k')
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'trajectory_comparison.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved trajectory comparison plot to {self.output_dir}/trajectory_comparison.png")
        
        plt.show()
    
    def plot_position_error(self, positions_gyro: pd.DataFrame, positions_compass: pd.DataFrame, 
                           ground_truth_df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot and compare position errors between gyroscope and compass against ground truth.
        
        Args:
            positions_gyro: DataFrame with positions calculated from gyro data
            positions_compass: DataFrame with positions calculated from compass data
            ground_truth_df: DataFrame with interpolated ground truth positions
            save: Whether to save the plot to file
        """
        # Prepare data
        # Combine into a single dataframe with step as the common key
        combined_df = pd.merge(
            positions_gyro, 
            positions_compass,
            on='Step',
            how='inner'
        )
        
        ground_truth_positions = ground_truth_df[['step', 'value_4', 'value_5']].rename(
            columns={'step': 'Step', 'value_4': 'ground_x', 'value_5': 'ground_y'}
        )
        
        # Merge with ground truth
        comparison_df = pd.merge(
            combined_df,
            ground_truth_positions,
            on='Step',
            how='inner'
        )
        
        # Calculate errors
        comparison_df['Gyro_Error_X'] = np.abs(comparison_df['Gyro_X'] - comparison_df['ground_x'])
        comparison_df['Gyro_Error_Y'] = np.abs(comparison_df['Gyro_Y'] - comparison_df['ground_y'])
        comparison_df['Compass_Error_X'] = np.abs(comparison_df['Compass_X'] - comparison_df['ground_x'])
        comparison_df['Compass_Error_Y'] = np.abs(comparison_df['Compass_Y'] - comparison_df['ground_y'])
        
        # Calculate Euclidean distance errors
        comparison_df['Gyro_Distance_Error'] = np.sqrt(
            (comparison_df['Gyro_Error_X'])**2 + (comparison_df['Gyro_Error_Y'])**2
        )
        comparison_df['Compass_Distance_Error'] = np.sqrt(
            (comparison_df['Compass_Error_X'])**2 + (comparison_df['Compass_Error_Y'])**2
        )
        
        # Add distance traveled column (assuming step size of 0.66m)
        comparison_df['Walked_distance'] = comparison_df['Step'] * 0.66
        
        # 1. Distance Error Plot
        fig, ax1 = plt.subplots(figsize=(3.45, 2.5), dpi=300)
        
        ax1.plot(comparison_df['Walked_distance'], comparison_df['Gyro_Distance_Error'], 
                 label='Gyro', linewidth=1.2, color='blue')
        ax1.plot(comparison_df['Walked_distance'], comparison_df['Compass_Distance_Error'], 
                 label='Compass', linewidth=1.2, color='red')
        
        ax1.set_xlabel('Walked Distance (m)', labelpad=3)
        ax1.set_ylabel('Positioning Error (m)', labelpad=3)
        
        ax1.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='k')
        ax1.legend()
        
        # Secondary x-axis for step numbers
        secx = ax1.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
        secx.set_xticks(comparison_df['Walked_distance'][::50])  
        secx.set_xticklabels(comparison_df['Step'][::50])  
        secx.set_xlabel('Number of Walked Steps', labelpad=8)
        
        # Axis formatting
        ax1.xaxis.set_major_locator(MultipleLocator(50))  # Major ticks every 50 meters
        ax1.yaxis.set_major_locator(MultipleLocator(5))   # Major ticks every 5 meters
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'distance_error.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved distance error plot to {self.output_dir}/distance_error.png")
        
        plt.show()
        
        # 2. ECDF Plot
        plt.figure(figsize=(3.45, 2.5), dpi=300)
        
        ecdf_gyro = sm.distributions.ECDF(comparison_df['Gyro_Distance_Error'])
        ecdf_compass = sm.distributions.ECDF(comparison_df['Compass_Distance_Error'])
        
        plt.plot(ecdf_gyro.x, ecdf_gyro.y, label='Gyro', color='blue', linewidth=1.2)
        plt.plot(ecdf_compass.x, ecdf_compass.y, label='Compass', color='red', linewidth=1.2)
        
        plt.xlabel('Positioning Error (m)', labelpad=3)
        plt.ylabel('ECDF', labelpad=3)
        
        plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='k')
        plt.legend()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'error_ecdf.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved error ECDF plot to {self.output_dir}/error_ecdf.png")
        
        plt.show()
        
        # 3. Box Plot
        # Create a DataFrame for distance errors
        distance_errors_df = pd.DataFrame({
            'Sensor': ['Gyro'] * len(comparison_df) + ['Compass'] * len(comparison_df),
            'Distance_Error': np.concatenate([
                comparison_df['Gyro_Distance_Error'], 
                comparison_df['Compass_Distance_Error']
            ])
        })
        
        plt.figure(figsize=(3.45, 2.5), dpi=300)
        
        sns.boxplot(x='Sensor', y='Distance_Error', data=distance_errors_df, linewidth=0.6)
        
        plt.xlabel('Sensor', fontsize=5)
        plt.ylabel('Positioning Error (m)', fontsize=5)
        
        plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='k')
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'error_boxplot.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved error boxplot to {self.output_dir}/error_boxplot.png")
        
        plt.show()
    
    def plot_sensor_data_over_time(self, gyro_data: pd.DataFrame, compass_data: pd.DataFrame, 
                                 ground_truth_data: pd.DataFrame, save: bool = True) -> None:
        """
        Plot raw sensor data values over time.
        
        Args:
            gyro_data: DataFrame with gyroscope data
            compass_data: DataFrame with compass data
            ground_truth_data: DataFrame with ground truth data
            save: Whether to save the plot to file
        """
        # Create multi-panel figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot 1: Gyroscope data
        ax1.plot(gyro_data['Timestamp_(ms)'], gyro_data['axisZAngle'], 
                 label='Gyro Angle Rate', color='blue')
        ax1.set_ylabel('Angular Rate (deg/s)')
        ax1.set_title('Gyroscope Data Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Compass data
        ax2.plot(compass_data['Timestamp_(ms)'], compass_data['compass'], 
                 label='Compass Heading', color='red')
        
        # Add ground truth heading if available
        if 'GroundTruthHeadingComputed' in compass_data.columns:
            ax2.plot(compass_data['Timestamp_(ms)'], compass_data['GroundTruthHeadingComputed'], 
                     label='Ground Truth Heading', color='green', linestyle='--')
        
        ax2.set_xlabel('Timestamp (ms)')
        ax2.set_ylabel('Heading (degrees)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'sensor_data_time_series.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved sensor data time series to {self.output_dir}/sensor_data_time_series.png")
        
        plt.show()
        
        # Also plot the steps over time
        plt.figure(figsize=(10, 6))
        plt.plot(compass_data['Timestamp_(ms)'], compass_data['step'], label='Step Count')
        plt.xlabel('Timestamp (ms)')
        plt.ylabel('Step Number')
        plt.title('Step Count Over Time')
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'steps_time_series.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved step count time series to {self.output_dir}/steps_time_series.png")
        
        plt.show()
        
    def plot_magnetic_field_variation(self, compass_data: pd.DataFrame, save: bool = True) -> None:
        """
        Plot magnetic field magnitude variation over time.
        
        Args:
            compass_data: DataFrame with compass data
            save: Whether to save the plot to file
        """
        if 'Magnetic_Field_Magnitude' not in compass_data.columns:
            logger.warning("Magnetic_Field_Magnitude column not found in compass data")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(compass_data['Timestamp_(ms)'], compass_data['Magnetic_Field_Magnitude'], 
                 label='Magnetic Field Magnitude')
        
        plt.xlabel('Timestamp (ms)')
        plt.ylabel('Magnetic Field Magnitude')
        plt.title('Magnetic Field Variation Over Time')
        plt.grid(True, alpha=0.3)
        
        # Add a rolling mean to show the trend
        window = 50  # Window size for rolling mean
        compass_data['Mag_Field_Rolling_Mean'] = compass_data['Magnetic_Field_Magnitude'].rolling(window=window).mean()
        plt.plot(compass_data['Timestamp_(ms)'], compass_data['Mag_Field_Rolling_Mean'], 
                 color='red', label=f'Rolling Mean (window={window})')
        
        plt.legend()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'magnetic_field_variation.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved magnetic field variation to {self.output_dir}/magnetic_field_variation.png")
        
        plt.show()
        
    def plot_quasi_static_detection(self, compass_data: pd.DataFrame, 
                                   quasi_static_data: pd.DataFrame, save: bool = True) -> None:
        """
        Plot compass headings with quasi-static intervals highlighted.
        
        Args:
            compass_data: DataFrame with compass data
            quasi_static_data: DataFrame with quasi-static interval data
            save: Whether to save the plot to file
        """
        plt.figure(figsize=(10, 6))
        
        # Plot compass headings
        plt.plot(compass_data['Timestamp_(ms)'], compass_data['compass'], 
                 label='Compass Headings', color='cyan')
        
        # Scatter plot quasi-static intervals
        plt.scatter(quasi_static_data['Time'], quasi_static_data['Compass_Heading'],
                    c=quasi_static_data['Quasi_Static_Interval_Number'], cmap='Set1',
                    s=20, zorder=5, label='Quasi-Static Intervals')
        
        # Plot ground truth headings if available
        if 'GroundTruthHeadingComputed' in compass_data.columns:
            plt.plot(compass_data['Timestamp_(ms)'], compass_data['GroundTruthHeadingComputed'], 
                     marker='.', linestyle='-', markersize=5, color='blue', label='Ground Truth')
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Compass Headings (degrees)')
        plt.title('Compass Headings over Time with Quasi-Static Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'quasi_static_detection.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved quasi-static detection plot to {self.output_dir}/quasi_static_detection.png")
        
        plt.show()
        
        # Also plot the steps with quasi-static intervals
        plt.figure(figsize=(10, 6))
        plt.plot(compass_data['Timestamp_(ms)'], compass_data['step'], label='Steps', color='cyan')
        plt.scatter(quasi_static_data['Time'], quasi_static_data['Step'],
                    c=quasi_static_data['Quasi_Static_Interval_Number'], cmap='Set1',
                    s=20, zorder=5, label='Quasi-Static Intervals')
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Step Number')
        plt.title('Step Number over Time with Quasi-Static Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'quasi_static_steps.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved quasi-static steps plot to {self.output_dir}/quasi_static_steps.png")
        
        plt.show() 