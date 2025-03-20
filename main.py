#!/usr/bin/env python3
"""
Indoor Navigation System - Phase 1: Data Preprocessing and Classification
Main program to process sensor data, clean it, detect anomalies, and visualize it.
"""

import os
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from src.data_parser import SensorDataParser, list_available_data_files
from src.data_cleaner import SensorDataCleaner
from src.data_visualizer import DataVisualizer
from src.anomaly_detector import SensorAnomalyDetector
from src.sensor_fusion import SensorFusion  # Import the new sensor fusion module
from src.position_tracker import PositionTracker  # Import the position tracker module

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Indoor Navigation System - Phase 1')
    
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the sensor data files')
    parser.add_argument('--file', type=str, default=None,
                        help='Specific data file to process (if not specified, the first file will be used)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output files')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--detect_anomalies', action='store_true',
                        help='Detect anomalies in the data')
    parser.add_argument('--interpolate', action='store_true',
                        help='Interpolate ground truth positions')
    parser.add_argument('--fusion', choices=['ekf', 'ukf', 'lstm', 'adaptive', 'context'], default=None,
                        help='Apply sensor fusion using the specified method')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark comparison of all fusion methods')
    # Add position tracking related arguments
    parser.add_argument('--position_tracking', action='store_true',
                        help='Apply deep learning-based position tracking')
    parser.add_argument('--model_type', choices=['lstm', 'cnn_lstm', 'bidirectional'], default='lstm',
                        help='Type of deep learning model for position tracking')
    parser.add_argument('--seq_length', type=int, default=10,
                        help='Sequence length for position tracking models')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs for position tracking')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training position tracking models')
    parser.add_argument('--benchmark_position', action='store_true',
                        help='Run benchmark comparison of position tracking models')
    
    return parser.parse_args()

def ensure_output_dirs(output_dir):
    """Ensure output directories exist."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'anomalies'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'fusion'), exist_ok=True)  # Add fusion directory
    os.makedirs(os.path.join(output_dir, 'position'), exist_ok=True)  # Add position tracking directory

def main():
    """Main function to process sensor data."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Ensure output directories exist
    ensure_output_dirs(args.output_dir)
    
    # List available data files
    data_files = list_available_data_files(args.data_dir)
    
    if not data_files:
        logger.error(f"No data files found in {args.data_dir}")
        return
    
    # Select the file to process
    file_to_process = args.file
    if file_to_process is None:
        file_to_process = data_files[0]
        logger.info(f"No specific file specified, using {file_to_process}")
    elif file_to_process not in data_files:
        if not os.path.exists(os.path.join(args.data_dir, file_to_process)):
            logger.error(f"Specified file {file_to_process} not found")
            return
    
    # Create full path to the data file
    data_file_path = os.path.join(args.data_dir, file_to_process)
    
    # Initialize the data parser
    parser = SensorDataParser(data_file_path)
    
    # Load and parse the data
    logger.info(f"Processing data file: {data_file_path}")
    raw_data = parser.load_data()
    
    # Classify the data into sensor types
    gyro_data, compass_data, ground_truth_data, initial_location_data = parser.classify_sensor_data()
    
    # Save raw classified data
    raw_data_output_dir = os.path.join(args.output_dir, 'data', 'raw')
    os.makedirs(raw_data_output_dir, exist_ok=True)
    
    gyro_data.to_csv(os.path.join(raw_data_output_dir, 'gyro_data_raw.csv'), index=False)
    compass_data.to_csv(os.path.join(raw_data_output_dir, 'compass_data_raw.csv'), index=False)
    ground_truth_data.to_csv(os.path.join(raw_data_output_dir, 'ground_truth_data_raw.csv'), index=False)
    initial_location_data.to_csv(os.path.join(raw_data_output_dir, 'initial_location_data_raw.csv'), index=False)
    
    # Organize data for cleaning
    data_dict = {
        'gyro': gyro_data,
        'compass': compass_data,
        'ground_truth': ground_truth_data,
        'initial_location': initial_location_data
    }
    
    # Initialize data cleaner and clean the data
    cleaner = SensorDataCleaner()
    cleaned_data = cleaner.clean_all_data(data_dict)
    
    # Save cleaned data
    cleaned_data_output_dir = os.path.join(args.output_dir, 'data', 'cleaned')
    os.makedirs(cleaned_data_output_dir, exist_ok=True)
    
    for sensor_type, data in cleaned_data.items():
        data.to_csv(os.path.join(cleaned_data_output_dir, f'{sensor_type}_data_cleaned.csv'), index=False)
    
    # Detect anomalies if requested
    if args.detect_anomalies:
        anomaly_detector = SensorAnomalyDetector(output_dir=os.path.join(args.output_dir, 'anomalies'))
        
        # Detect anomalies in the data
        anomaly_results = anomaly_detector.detect_all_anomalies(
            cleaned_data['gyro'],
            cleaned_data['compass'],
            cleaned_data['ground_truth'],
            visualize=args.visualize
        )
        
        # Detect unclassifiable data in raw dataset
        unclassifiable_data = anomaly_detector.detect_unclassifiable_data(raw_data)
        
        # Save anomaly detection results
        anomaly_data_output_dir = os.path.join(args.output_dir, 'data', 'anomalies')
        os.makedirs(anomaly_data_output_dir, exist_ok=True)
        
        for sensor_type, data in anomaly_results.items():
            data.to_csv(os.path.join(anomaly_data_output_dir, f'{sensor_type}_anomalies.csv'), index=False)
        
        unclassifiable_data.to_csv(os.path.join(anomaly_data_output_dir, 'unclassifiable_data.csv'), index=False)
    
    # Interpolate ground truth if requested
    interpolated_positions = None
    if args.interpolate:
        logger.info("Interpolating ground truth positions")
        interpolated_positions = parser.interpolate_ground_truth_positions()
        
        # Save interpolated positions
        interpolated_positions.to_csv(os.path.join(cleaned_data_output_dir, 'interpolated_positions.csv'), index=False)
    
    # Apply sensor fusion if requested
    if args.fusion:
        logger.info(f"Applying sensor fusion using {args.fusion.upper()} method")
        
        # Initialize sensor fusion
        fusion = SensorFusion(output_dir=os.path.join(args.output_dir, 'fusion'))
        
        # Select data to use (use anomaly results if available, otherwise cleaned data)
        gyro_data_for_fusion = anomaly_results['gyro'] if args.detect_anomalies else cleaned_data['gyro']
        compass_data_for_fusion = anomaly_results['compass'] if args.detect_anomalies else cleaned_data['compass']
        
        # Apply the fusion method
        fused_data = fusion.fuse_sensors(
            gyro_data_for_fusion,
            compass_data_for_fusion,
            method=args.fusion,
            ground_truth_data=cleaned_data['ground_truth'],
            visualize=args.visualize
        )
        
        logger.info(f"Sensor fusion completed with {len(fused_data)} data points")
    
    # Run benchmark comparison if requested
    if args.benchmark:
        logger.info("Running benchmark comparison of all fusion methods")
        
        # Initialize sensor fusion
        fusion = SensorFusion(output_dir=os.path.join(args.output_dir, 'fusion'))
        
        # Select data to use (use anomaly results if available, otherwise cleaned data)
        gyro_data_for_fusion = anomaly_results['gyro'] if args.detect_anomalies else cleaned_data['gyro']
        compass_data_for_fusion = anomaly_results['compass'] if args.detect_anomalies else cleaned_data['compass']
        
        # Run benchmark
        methods = ['ekf', 'ukf', 'lstm', 'adaptive', 'context']
        benchmark_results = fusion.benchmark_fusion_methods(
            gyro_data_for_fusion,
            compass_data_for_fusion,
            ground_truth_data=cleaned_data['ground_truth'],
            methods=methods,
            visualize=args.visualize
        )
        
        logger.info("Benchmark comparison completed")
    
    # Apply position tracking if requested
    if args.position_tracking:
        logger.info(f"Applying position tracking using {args.model_type} model")
        
        # Initialize position tracker
        position_tracker = PositionTracker(output_dir=os.path.join(args.output_dir, 'position'))
        
        # Prepare sensor data by combining gyro and compass data
        # First ensure we use interpolated positions if available
        if interpolated_positions is not None and len(interpolated_positions) > 0:
            logger.info(f"Using interpolated positions with {len(interpolated_positions)} points for position tracking")
            ground_truth_for_position = interpolated_positions
        else:
            logger.info(f"Using original ground truth with {len(cleaned_data['ground_truth'])} points for position tracking")
            ground_truth_for_position = cleaned_data['ground_truth']
            
        if len(ground_truth_for_position) < 2:
            logger.warning("Not enough ground truth data for position tracking. Augmenting with synthetic points.")
            # Create synthetic ground truth if needed
            if len(ground_truth_for_position) == 0:
                # Create completely synthetic data
                logger.info("Creating synthetic ground truth data")
                synthetic_gt = []
                # Get min and max timestamps from sensor data
                min_ts = cleaned_data['gyro']['Timestamp_(ms)'].min()
                max_ts = cleaned_data['gyro']['Timestamp_(ms)'].max()
                # Create points along a simple trajectory (a line from start to end)
                num_points = 10
                for i in range(num_points):
                    progress = i / (num_points - 1)
                    ts = min_ts + progress * (max_ts - min_ts)
                    # Create a simple line trajectory
                    x = 100 + progress * 100  # 100 to 200
                    y = 100 + progress * 50   # 100 to 150
                    synthetic_gt.append({
                        'Timestamp_(ms)': ts,
                        'step': i * 10,
                        'value_4': x,
                        'value_5': y
                    })
                ground_truth_for_position = pd.DataFrame(synthetic_gt)
            elif len(ground_truth_for_position) == 1:
                # Create more points based on the single point
                logger.info("Extending single ground truth point to create trajectory")
                gt_point = ground_truth_for_position.iloc[0]
                synthetic_gt = []
                for i in range(10):
                    new_point = gt_point.copy()
                    # Add some variation to create a trajectory
                    if 'value_4' in gt_point and 'value_5' in gt_point:
                        new_point['value_4'] = gt_point['value_4'] + (i - 5) * 10
                        new_point['value_5'] = gt_point['value_5'] + (i - 5) * 5
                    else:
                        # If no coordinates exist, create them
                        new_point['value_4'] = 100 + i * 10
                        new_point['value_5'] = 100 + i * 5
                    
                    # Adjust timestamp and step if they exist
                    if 'Timestamp_(ms)' in gt_point:
                        new_point['Timestamp_(ms)'] = gt_point['Timestamp_(ms)'] + i * 1000
                    if 'step' in gt_point:
                        new_point['step'] = i * 10
                    
                    synthetic_gt.append(new_point)
                ground_truth_for_position = pd.DataFrame(synthetic_gt)
                
        logger.info(f"Ground truth data columns: {ground_truth_for_position.columns.tolist()}")
        logger.info(f"Ground truth data shape: {ground_truth_for_position.shape}")
        
        # Create combined sensor data
        sensor_data_for_position = pd.merge_asof(
            cleaned_data['gyro'].sort_values(by='Timestamp_(ms)'),
            cleaned_data['compass'].sort_values(by='Timestamp_(ms)'),
            on='Timestamp_(ms)',
            direction='nearest',
            suffixes=('_gyro', '_compass')
        )
        
        logger.info(f"Combined sensor data shape: {sensor_data_for_position.shape}")
        
        try:
            # Run position prediction
            position_results = position_tracker.run_position_prediction(
                sensor_data=sensor_data_for_position,
                ground_truth_data=ground_truth_for_position,
                model_type=args.model_type,
                seq_length=args.seq_length,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            
            logger.info(f"Position tracking completed, metrics: RMSE={position_results['metrics']['rmse']:.4f}, MAE={position_results['metrics']['mae']:.4f}")
        except Exception as e:
            logger.error(f"Error during position tracking: {str(e)}")
            logger.error("Position tracking could not be completed")
    
    # Run benchmark for position tracking if requested
    if args.benchmark_position:
        logger.info("Running benchmark comparison of position tracking models")
        
        # Initialize position tracker
        position_tracker = PositionTracker(output_dir=os.path.join(args.output_dir, 'position'))
        
        # Prepare sensor data by combining gyro and compass data
        # First ensure we use interpolated positions if available
        if interpolated_positions is not None and len(interpolated_positions) > 0:
            logger.info(f"Using interpolated positions with {len(interpolated_positions)} points for benchmarking")
            ground_truth_for_position = interpolated_positions
        else:
            logger.info(f"Using original ground truth with {len(cleaned_data['ground_truth'])} points for benchmarking")
            ground_truth_for_position = cleaned_data['ground_truth']
            
        if len(ground_truth_for_position) < 2:
            logger.warning("Not enough ground truth data for benchmarking. Augmenting with synthetic points.")
            # Create synthetic ground truth if needed
            if len(ground_truth_for_position) == 0:
                # Create completely synthetic data
                logger.info("Creating synthetic ground truth data")
                synthetic_gt = []
                # Get min and max timestamps from sensor data
                min_ts = cleaned_data['gyro']['Timestamp_(ms)'].min()
                max_ts = cleaned_data['gyro']['Timestamp_(ms)'].max()
                # Create points along a simple trajectory (a line from start to end)
                num_points = 10
                for i in range(num_points):
                    progress = i / (num_points - 1)
                    ts = min_ts + progress * (max_ts - min_ts)
                    # Create a simple line trajectory
                    x = 100 + progress * 100  # 100 to 200
                    y = 100 + progress * 50   # 100 to 150
                    synthetic_gt.append({
                        'Timestamp_(ms)': ts,
                        'step': i * 10,
                        'value_4': x,
                        'value_5': y
                    })
                ground_truth_for_position = pd.DataFrame(synthetic_gt)
            elif len(ground_truth_for_position) == 1:
                # Create more points based on the single point
                logger.info("Extending single ground truth point to create trajectory")
                gt_point = ground_truth_for_position.iloc[0]
                synthetic_gt = []
                for i in range(10):
                    new_point = gt_point.copy()
                    # Add some variation to create a trajectory
                    if 'value_4' in gt_point and 'value_5' in gt_point:
                        new_point['value_4'] = gt_point['value_4'] + (i - 5) * 10
                        new_point['value_5'] = gt_point['value_5'] + (i - 5) * 5
                    else:
                        # If no coordinates exist, create them
                        new_point['value_4'] = 100 + i * 10
                        new_point['value_5'] = 100 + i * 5
                    
                    # Adjust timestamp and step if they exist
                    if 'Timestamp_(ms)' in gt_point:
                        new_point['Timestamp_(ms)'] = gt_point['Timestamp_(ms)'] + i * 1000
                    if 'step' in gt_point:
                        new_point['step'] = i * 10
                    
                    synthetic_gt.append(new_point)
                ground_truth_for_position = pd.DataFrame(synthetic_gt)
        
        # Create combined sensor data
        sensor_data_for_position = pd.merge_asof(
            cleaned_data['gyro'].sort_values(by='Timestamp_(ms)'),
            cleaned_data['compass'].sort_values(by='Timestamp_(ms)'),
            on='Timestamp_(ms)',
            direction='nearest',
            suffixes=('_gyro', '_compass')
        )
        
        try:
            # Run benchmark for position tracking
            benchmark_results = position_tracker.benchmark_models(
                sensor_data=sensor_data_for_position,
                ground_truth_data=ground_truth_for_position,
                seq_lengths=[5, 10, 15],
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            
            logger.info("Position tracking benchmark completed")
        except Exception as e:
            logger.error(f"Error during position tracking benchmark: {str(e)}")
            logger.error("Position tracking benchmark could not be completed")
    
    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating visualizations")
        visualizer = DataVisualizer(output_dir=os.path.join(args.output_dir, 'plots'))
        
        # Create visualizations
        visualizer.create_all_visualizations(
            cleaned_data['gyro'],
            cleaned_data['compass'],
            cleaned_data['ground_truth'],
            interpolated_positions
        )
    
    logger.info("Processing completed successfully")

if __name__ == "__main__":
    main() 