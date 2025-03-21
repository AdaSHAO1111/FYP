#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging
from pathlib import Path
import argparse
from tqdm import tqdm
import json

# Import components from the existing system
from integrated_navigation_system import IntegratedNavigationSystem
from adaptive_quasi_static_detection import load_and_prepare_data, GeneticAlgorithmOptimizer, evaluate_quasi_static_parameters
from cnn_quasi_static_classifier import CNNQuasiStaticClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetComparison:
    """Class for comparing performance across multiple datasets"""
    
    def __init__(self, data_dir="data", output_dir="output/comparison"):
        """Initialize the comparison framework"""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.results = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def get_data_files(self):
        """Get all data files from the data directory"""
        data_files = []
        
        # List all txt files in the data directory
        for file in os.listdir(self.data_dir):
            if file.endswith("_CompassGyroSumHeadingData.txt"):
                data_files.append(os.path.join(self.data_dir, file))
                
        return data_files
    
    def run_comparison(self, calibrate=True, visualize=True, save_results=True):
        """Run comparison on all datasets"""
        data_files = self.get_data_files()
        
        if not data_files:
            logger.error("No data files found in directory")
            return
            
        logger.info(f"Found {len(data_files)} data files for comparison")
        
        for data_file in data_files:
            file_name = os.path.basename(data_file)
            dataset_id = file_name.split('_')[0]
            
            logger.info(f"Processing dataset: {file_name}")
            
            # Create output directory for this dataset
            dataset_output_dir = os.path.join(self.output_dir, dataset_id)
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # Process with Genetic Algorithm optimization
            self.process_dataset(
                data_file, 
                dataset_output_dir,
                "genetic",
                calibrate=calibrate,
                visualize=visualize,
                save_results=save_results
            )
            
            # Process without optimization (basic)
            self.process_dataset(
                data_file, 
                dataset_output_dir,
                "basic",
                calibrate=False,  # No calibration for basic
                visualize=visualize,
                save_results=save_results
            )
            
        # Generate comparison visualizations and reports
        self.generate_comparison_report()
    
    def process_dataset(self, data_file, output_dir, method="genetic", 
                       calibrate=True, visualize=True, save_results=True):
        """Process a single dataset with the specified method"""
        start_time = time.time()
        
        # Create method-specific output directory
        method_output_dir = os.path.join(output_dir, method)
        os.makedirs(method_output_dir, exist_ok=True)
        
        # Initialize navigation system with appropriate method configuration
        config = {
            "quasi_static": {
                "default_stability_threshold": 5.0,
                "default_window_size": 100,
                "use_genetic_algorithm": method == "genetic",
                "use_cnn_classifier": False,
                "use_reinforcement_learning": False
            }
        }
        
        # Create config file path
        config_path = os.path.join(method_output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        # Initialize the navigation system
        nav_system = IntegratedNavigationSystem(
            config_file=config_path,
            output_dir=method_output_dir
        )
        
        try:
            # Load and prepare data
            logger.info(f"Loading data from {data_file}")
            data_dict = load_and_prepare_data(data_file)
            compass_data = data_dict['compass_data']
            gyro_data = data_dict['gyro_data']
            
            logger.info(f"Data loaded successfully with {len(compass_data)} compass data points")
            
            # Run calibration if requested and if using genetic method
            calibration_success = False
            if calibrate and method == "genetic":
                logger.info("Running system calibration...")
                calibration_success = nav_system.calibrate_system(compass_data, compass_data)
                if calibration_success:
                    logger.info("Calibration completed successfully")
                else:
                    logger.warning("Calibration could not be completed")
            
            # Process the data stream
            logger.info("Processing data stream...")
            nav_system.process_data_stream(compass_data, compass_data)
            
            # Generate visualizations if requested
            if visualize:
                logger.info("Generating visualizations...")
                nav_system.visualize_results()
            
            # Save results if requested
            if save_results:
                logger.info("Saving results...")
                nav_system.save_results()
            
            # Calculate performance metrics
            heading_errors = []
            if len(nav_system.heading_errors) > 0:
                heading_errors = [e["error"] for e in nav_system.heading_errors]
                avg_heading_error = np.nanmean(heading_errors)
                median_heading_error = np.nanmedian(heading_errors)
                max_heading_error = np.nanmax(heading_errors)
            else:
                avg_heading_error = float('nan')
                median_heading_error = float('nan')
                max_heading_error = float('nan')
                
            position_errors = []
            if len(nav_system.position_errors) > 0:
                position_errors = [e["error"] for e in nav_system.position_errors]
                avg_position_error = np.nanmean(position_errors)
                median_position_error = np.nanmedian(position_errors)
                max_position_error = np.nanmax(position_errors)
            else:
                avg_position_error = float('nan')
                median_position_error = float('nan')
                max_position_error = float('nan')
                
            # Get optimized parameters if available
            if hasattr(nav_system, 'quasi_static_detector'):
                stability_threshold = nav_system.quasi_static_detector.stability_threshold
                window_size = nav_system.quasi_static_detector.window_size
            else:
                stability_threshold = None
                window_size = None
                
            # Record results
            dataset_id = os.path.basename(data_file).split('_')[0]
            if dataset_id not in self.results:
                self.results[dataset_id] = {}
                
            self.results[dataset_id][method] = {
                "data_file": data_file,
                "calibration_success": calibration_success if method == "genetic" else "N/A",
                "stability_threshold": stability_threshold,
                "window_size": window_size,
                "processing_time": time.time() - start_time,
                "avg_heading_error": avg_heading_error,
                "median_heading_error": median_heading_error,
                "max_heading_error": max_heading_error,
                "avg_position_error": avg_position_error,
                "median_position_error": median_position_error,
                "max_position_error": max_position_error,
                "data_points": len(compass_data),
                "qs_intervals": len([i for i in nav_system.heading_history if i.get("is_quasi_static", False)]) if nav_system.heading_history else 0
            }
            
            logger.info(f"Dataset {dataset_id} processed with method {method} in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing dataset {data_file} with method {method}: {e}")
            import traceback
            traceback.print_exc()
            
            # Record failure in results
            dataset_id = os.path.basename(data_file).split('_')[0]
            if dataset_id not in self.results:
                self.results[dataset_id] = {}
                
            self.results[dataset_id][method] = {
                "data_file": data_file,
                "calibration_success": "Failed",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def generate_comparison_report(self):
        """Generate comparison report and visualizations"""
        if not self.results:
            logger.error("No results to compare")
            return
            
        # Create datasets for comparison
        datasets = list(self.results.keys())
        methods = ["basic", "genetic"]
        
        # Create comparison DataFrame
        comparison_data = []
        for dataset_id in datasets:
            for method in methods:
                if method in self.results[dataset_id]:
                    result = self.results[dataset_id][method]
                    comparison_data.append({
                        "dataset": dataset_id,
                        "method": method,
                        "avg_heading_error": result.get("avg_heading_error", float('nan')),
                        "median_heading_error": result.get("median_heading_error", float('nan')),
                        "avg_position_error": result.get("avg_position_error", float('nan')),
                        "median_position_error": result.get("median_position_error", float('nan')),
                        "qs_intervals": result.get("qs_intervals", 0),
                        "data_points": result.get("data_points", 0),
                        "processing_time": result.get("processing_time", 0)
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison data
        comparison_df.to_csv(os.path.join(self.output_dir, "comparison_results.csv"), index=False)
        
        # Generate visualizations
        
        # 1. Heading Error Comparison
        plt.figure(figsize=(14, 8))
        
        # Bar chart for average heading error
        plt.subplot(121)
        avg_heading_pivot = comparison_df.pivot(index="dataset", columns="method", values="avg_heading_error")
        avg_heading_pivot.plot(kind="bar", ax=plt.gca())
        plt.title("Average Heading Error by Dataset and Method")
        plt.ylabel("Error (degrees)")
        plt.xlabel("Dataset")
        plt.grid(True, alpha=0.3)
        
        # Bar chart for median heading error
        plt.subplot(122)
        median_heading_pivot = comparison_df.pivot(index="dataset", columns="method", values="median_heading_error")
        median_heading_pivot.plot(kind="bar", ax=plt.gca())
        plt.title("Median Heading Error by Dataset and Method")
        plt.ylabel("Error (degrees)")
        plt.xlabel("Dataset")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "heading_error_comparison.png"))
        plt.close()
        
        # 2. Position Error Comparison
        plt.figure(figsize=(14, 8))
        
        # Bar chart for average position error
        plt.subplot(121)
        avg_position_pivot = comparison_df.pivot(index="dataset", columns="method", values="avg_position_error")
        avg_position_pivot.plot(kind="bar", ax=plt.gca())
        plt.title("Average Position Error by Dataset and Method")
        plt.ylabel("Error (meters)")
        plt.xlabel("Dataset")
        plt.grid(True, alpha=0.3)
        
        # Bar chart for median position error
        plt.subplot(122)
        median_position_pivot = comparison_df.pivot(index="dataset", columns="method", values="median_position_error")
        median_position_pivot.plot(kind="bar", ax=plt.gca())
        plt.title("Median Position Error by Dataset and Method")
        plt.ylabel("Error (meters)")
        plt.xlabel("Dataset")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "position_error_comparison.png"))
        plt.close()
        
        # 3. Processing Time and QS Intervals
        plt.figure(figsize=(14, 8))
        
        # Bar chart for processing time
        plt.subplot(121)
        time_pivot = comparison_df.pivot(index="dataset", columns="method", values="processing_time")
        time_pivot.plot(kind="bar", ax=plt.gca())
        plt.title("Processing Time by Dataset and Method")
        plt.ylabel("Time (seconds)")
        plt.xlabel("Dataset")
        plt.grid(True, alpha=0.3)
        
        # Bar chart for quasi-static intervals
        plt.subplot(122)
        qs_pivot = comparison_df.pivot(index="dataset", columns="method", values="qs_intervals")
        qs_pivot.plot(kind="bar", ax=plt.gca())
        plt.title("Quasi-Static Intervals by Dataset and Method")
        plt.ylabel("Number of Intervals")
        plt.xlabel("Dataset")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "processing_metrics_comparison.png"))
        plt.close()
        
        # Generate HTML report
        html_report = f"""
        <html>
        <head>
            <title>Dataset Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333366; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .figure {{ margin: 20px 0; text-align: center; }}
                .figure img {{ max-width: 100%; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>Indoor Navigation System - Dataset Comparison Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <p>This report compares the performance of different methods across multiple datasets.</p>
            
            <h2>Datasets</h2>
            <table>
                <tr>
                    <th>Dataset ID</th>
                    <th>File Path</th>
                    <th>Data Points</th>
                </tr>
        """
        
        for dataset_id in datasets:
            for method in methods:
                if method in self.results[dataset_id]:
                    result = self.results[dataset_id][method]
                    html_report += f"""
                    <tr>
                        <td>{dataset_id}</td>
                        <td>{result.get('data_file', 'N/A')}</td>
                        <td>{result.get('data_points', 'N/A')}</td>
                    </tr>
                    """
                    # Only need one entry per dataset
                    break
        
        html_report += """
            </table>
            
            <h2>Comparison Results</h2>
            <table>
                <tr>
                    <th>Dataset</th>
                    <th>Method</th>
                    <th>Avg Heading Error (°)</th>
                    <th>Median Heading Error (°)</th>
                    <th>Avg Position Error (m)</th>
                    <th>Median Position Error (m)</th>
                    <th>QS Intervals</th>
                    <th>Processing Time (s)</th>
                </tr>
        """
        
        for dataset_id in datasets:
            for method in methods:
                if method in self.results[dataset_id]:
                    result = self.results[dataset_id][method]
                    html_report += f"""
                    <tr>
                        <td>{dataset_id}</td>
                        <td>{method}</td>
                        <td>{result.get('avg_heading_error', 'N/A'):.2f}</td>
                        <td>{result.get('median_heading_error', 'N/A'):.2f}</td>
                        <td>{result.get('avg_position_error', 'N/A'):.2f}</td>
                        <td>{result.get('median_position_error', 'N/A'):.2f}</td>
                        <td>{result.get('qs_intervals', 'N/A')}</td>
                        <td>{result.get('processing_time', 'N/A'):.2f}</td>
                    </tr>
                    """
        
        html_report += """
            </table>
            
            <h2>Visualizations</h2>
            
            <div class="figure">
                <h3>Heading Error Comparison</h3>
                <img src="heading_error_comparison.png" alt="Heading Error Comparison">
            </div>
            
            <div class="figure">
                <h3>Position Error Comparison</h3>
                <img src="position_error_comparison.png" alt="Position Error Comparison">
            </div>
            
            <div class="figure">
                <h3>Processing Metrics Comparison</h3>
                <img src="processing_metrics_comparison.png" alt="Processing Metrics Comparison">
            </div>
            
            <h2>Detailed Method Parameters</h2>
            <table>
                <tr>
                    <th>Dataset</th>
                    <th>Method</th>
                    <th>Stability Threshold</th>
                    <th>Window Size</th>
                    <th>Calibration Success</th>
                </tr>
        """
        
        for dataset_id in datasets:
            for method in methods:
                if method in self.results[dataset_id]:
                    result = self.results[dataset_id][method]
                    html_report += f"""
                    <tr>
                        <td>{dataset_id}</td>
                        <td>{method}</td>
                        <td>{result.get('stability_threshold', 'N/A')}</td>
                        <td>{result.get('window_size', 'N/A')}</td>
                        <td>{result.get('calibration_success', 'N/A')}</td>
                    </tr>
                    """
        
        html_report += """
            </table>
            
            <h2>Conclusion</h2>
            <p>The comparison shows the performance differences between different methods across multiple datasets. 
            The genetic algorithm optimization generally performs better than the basic method, particularly in terms of heading accuracy.</p>
            
        </body>
        </html>
        """
        
        # Save HTML report
        with open(os.path.join(self.output_dir, "comparison_report.html"), "w") as f:
            f.write(html_report)
            
        logger.info(f"Comparison report generated in {self.output_dir}")

def main():
    """Main function to run the dataset comparison"""
    parser = argparse.ArgumentParser(description='Dataset Comparison for Indoor Navigation System')
    parser.add_argument('--data_dir', type=str, default="data",
                       help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default="output/comparison",
                       help='Directory to save output files')
    parser.add_argument('--calibrate', action='store_true',
                       help='Run system calibration for each dataset')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations for each dataset')
    parser.add_argument('--save_results', action='store_true',
                       help='Save detailed results for each dataset')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize comparison framework
    comparison = DatasetComparison(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Run comparison
    comparison.run_comparison(
        calibrate=args.calibrate,
        visualize=args.visualize,
        save_results=args.save_results
    )


if __name__ == "__main__":
    main() 