#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import logging
import time
from typing import Dict, List, Tuple, Union, Optional, Any
import json
from pathlib import Path

# Import components from the existing system
from adaptive_quasi_static_detection import QuasiStaticDetector, GeneticAlgorithmOptimizer, load_and_prepare_data, evaluate_quasi_static_parameters
from cnn_quasi_static_classifier import CNNQuasiStaticClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("navigation_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NavigationContext:
    """Class to track and manage the context of the navigation system"""
    
    def __init__(self):
        self.current_environment = "unknown"  # stable, moderate, unstable
        self.current_motion_state = "moving"  # moving, quasi_static
        self.current_floor = 0
        self.magnetic_disturbance_level = "low"  # low, medium, high
        self.position_confidence = 1.0  # 0.0 to 1.0
        self.heading_confidence = 1.0  # 0.0 to 1.0
        self.available_sensors = []
        self.battery_level = 100  # 0-100%
        self.device_orientation = "portrait"  # portrait, landscape
        self.step_frequency = 0.0  # steps per second
        
    def update_context(self, 
                       environment: Optional[str] = None,
                       motion_state: Optional[str] = None, 
                       floor: Optional[int] = None,
                       magnetic_disturbance: Optional[str] = None,
                       position_confidence: Optional[float] = None,
                       heading_confidence: Optional[float] = None,
                       sensors: Optional[List[str]] = None,
                       battery: Optional[float] = None,
                       orientation: Optional[str] = None,
                       step_freq: Optional[float] = None):
        """Update context with provided parameters"""
        if environment is not None:
            self.current_environment = environment
        if motion_state is not None:
            self.current_motion_state = motion_state
        if floor is not None:
            self.current_floor = floor
        if magnetic_disturbance is not None:
            self.magnetic_disturbance_level = magnetic_disturbance
        if position_confidence is not None:
            self.position_confidence = position_confidence
        if heading_confidence is not None:
            self.heading_confidence = heading_confidence
        if sensors is not None:
            self.available_sensors = sensors
        if battery is not None:
            self.battery_level = battery
        if orientation is not None:
            self.device_orientation = orientation
        if step_freq is not None:
            self.step_frequency = step_freq
    
    def to_dict(self) -> Dict:
        """Convert context to dictionary for logging/storage"""
        return {
            "environment": self.current_environment,
            "motion_state": self.current_motion_state,
            "floor": self.current_floor,
            "magnetic_disturbance": self.magnetic_disturbance_level,
            "position_confidence": self.position_confidence,
            "heading_confidence": self.heading_confidence,
            "available_sensors": self.available_sensors,
            "battery_level": self.battery_level,
            "device_orientation": self.device_orientation,
            "step_frequency": self.step_frequency
        }
        
    def determine_best_fusion_method(self) -> str:
        """Determine the best fusion method based on context"""
        if self.current_motion_state == "quasi_static":
            # During quasi-static periods, trust compass more
            return "compass_calibration"
        
        if self.magnetic_disturbance_level == "high":
            # With high magnetic disturbance, rely more on gyro
            return "gyro_only"
        
        if self.heading_confidence < 0.4:
            # When confidence is low, use the most reliable method
            if self.position_confidence > 0.7:
                # If we're confident about position, use map-matching
                return "map_matching"
            # Otherwise use the most robust method
            return "cnn_lstm"
        
        # Default method when everything is working well
        return "adaptive"


class IntegratedNavigationSystem:
    """Main class for the integrated navigation system that combines all components"""
    
    def __init__(self, config_file: Optional[str] = None, output_dir: str = "output"):
        """Initialize the integrated navigation system"""
        self.config = self._load_config(config_file)
        self.output_dir = output_dir
        self.context = NavigationContext()
        
        # Initialize detectors and models
        self._initialize_components()
        
        # Current state variables
        self.current_heading = 0.0
        self.current_position = (0.0, 0.0)  # (east, north)
        self.current_step_count = 0
        self.current_floor = 0
        
        # History tracking
        self.heading_history = []
        self.position_history = []
        self.context_history = []
        
        # Performance metrics
        self.heading_errors = []
        self.position_errors = []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "quasi_static": {
                "default_stability_threshold": 5.0,
                "default_window_size": 100,
                "use_genetic_algorithm": True,
                "use_cnn_classifier": False,
                "use_reinforcement_learning": False
            },
            "sensor_fusion": {
                "default_method": "adaptive",
                "available_methods": ["ekf", "ukf", "adaptive", "cnn_lstm"],
                "adaptive_switching": True
            },
            "position_tracking": {
                "model_type": "cnn_lstm",
                "step_length": 0.66,  # meters
                "use_map_matching": False
            },
            "system": {
                "real_time_processing": True,
                "data_collection_interval": 60,  # seconds
                "model_update_interval": 86400,  # seconds (daily)
                "battery_optimization": True
            }
        }
        
        if config_file is not None and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Update default config with user config
                self._update_nested_dict(default_config, user_config)
        
        return default_config
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Update nested dictionary with another dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _initialize_components(self):
        """Initialize all system components based on configuration"""
        # Initialize Quasi-Static Detection
        qs_config = self.config["quasi_static"]
        self.quasi_static_detector = QuasiStaticDetector(
            stability_threshold=qs_config["default_stability_threshold"],
            window_size=qs_config["default_window_size"]
        )
        
        # Initialize CNN classifier if configured
        self.cnn_classifier = None
        if qs_config["use_cnn_classifier"]:
            try:
                self.cnn_classifier = CNNQuasiStaticClassifier(window_size=50)
                # Try to load pre-trained model if exists
                if os.path.exists("best_cnn_model.h5"):
                    self.cnn_classifier.model = tf.keras.models.load_model("best_cnn_model.h5")
                    logger.info("Loaded pre-trained CNN model for quasi-static detection")
            except Exception as e:
                logger.error(f"Failed to initialize CNN classifier: {e}")
                
        # Initialize sensor fusion methods
        self.current_fusion_method = self.config["sensor_fusion"]["default_method"]
        
        # Initialize available methods dictionary to store model instances
        self.fusion_methods = {}
        
        # Initialize position tracking model
        self.position_tracking_model = None
        # In a real implementation, we would load the appropriate model here
        
    def update_context_from_sensor_data(self, data: pd.DataFrame):
        """Update context based on latest sensor data"""
        if len(data) == 0:
            return
        
        # Extract the latest row of data
        latest = data.iloc[-1]
        
        # Update floor information if available
        if 'value_4' in latest:
            try:
                floor_val = int(float(latest['value_4']))
                self.context.update_context(floor=floor_val)
            except (ValueError, TypeError):
                pass
            
        # Update step frequency if step data available
        if 'step' in latest and len(data) > 10:
            try:
                # Calculate steps per second based on last 10 readings
                times = data['Timestamp_(ms)'].iloc[-10:].values.astype(float)
                steps = data['step'].iloc[-10:].values.astype(float)
                if len(times) > 1 and steps[-1] != steps[0]:
                    time_diff = (times[-1] - times[0]) / 1000  # convert to seconds
                    step_diff = steps[-1] - steps[0]
                    if time_diff > 0:
                        step_freq = step_diff / time_diff
                        self.context.update_context(step_freq=step_freq)
            except (ValueError, TypeError):
                pass
                
        # Determine magnetic disturbance level
        if 'Magnetic_Field_Magnitude' in latest:
            # Simple example - in a real system this would be more sophisticated
            mag_field = latest['Magnetic_Field_Magnitude']
            # Convert mag_field to float before comparison
            try:
                mag_field = float(mag_field)
                if mag_field < 20:
                    disturbance = "low"
                elif mag_field < 50:
                    disturbance = "medium"
                else:
                    disturbance = "high"
                self.context.update_context(magnetic_disturbance=disturbance)
            except (TypeError, ValueError):
                # If conversion fails, use default value
                self.context.update_context(magnetic_disturbance="medium")
            
        # Add more context updates based on available sensor data
    
    def detect_quasi_static_state(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """
        Detect if the device is in a quasi-static state using configured detection methods.
        
        Args:
            data: DataFrame containing sensor data
            
        Returns:
            Tuple of (is_quasi_static, confidence_score)
        """
        if len(data) < self.quasi_static_detector.window_size:
            logger.warning(f"Not enough data points for quasi-static detection. "
                          f"Have {len(data)}, need {self.quasi_static_detector.window_size}")
            return False, 0.0
        
        qs_config = self.config["quasi_static"]
        
        # Reset detector state to ensure fresh detection
        self.quasi_static_detector.reset()
        
        # Extract latest window of data
        window_data = data.tail(self.quasi_static_detector.window_size)
        
        # Process each data point through multiple detection methods
        cnn_prediction = 0.0
        threshold_prediction = 0.0
        
        # 1. Traditional threshold-based method
        for _, row in window_data.iterrows():
            if 'compass' in row:
                heading = row['compass']
                
                # Add all available sensor data
                gyro_value = row.get('gyroSumFromstart0') if 'gyroSumFromstart0' in row else None
                mag_value = row.get('Magnetic_Field_Magnitude') if 'Magnetic_Field_Magnitude' in row else None
                
                self.quasi_static_detector.add_sensor_data(heading, gyro_value, mag_value)
        
        # Get result and stability score from the threshold-based detector
        is_quasi_static_threshold = self.quasi_static_detector.is_quasi_static_interval()
        threshold_score = self.quasi_static_detector.get_stability_score()
        threshold_prediction = 1.0 if is_quasi_static_threshold else 0.0
        
        # 2. Use CNN classifier if configured
        cnn_score = 0.0
        if qs_config["use_cnn_classifier"] and self.cnn_classifier and self.cnn_classifier.model:
            try:
                # Prepare data in the format expected by the CNN
                predictions, probabilities = self.cnn_classifier.predict(window_data)
                cnn_prediction = probabilities[-1]  # Last value as we're interested in current state
                cnn_score = float(cnn_prediction)
            except Exception as e:
                logger.error(f"Error in CNN-based quasi-static detection: {e}")
                # Fall back to threshold-based method
                cnn_score = 0.0
        
        # Combine predictions with weighting
        if qs_config["use_cnn_classifier"] and self.cnn_classifier and self.cnn_classifier.model:
            # Use weighted combination
            cnn_weight = 0.6  # Give more weight to CNN when available
            threshold_weight = 0.4
            combined_score = (cnn_weight * cnn_score) + (threshold_weight * threshold_score)
            is_quasi_static = combined_score > 0.5
        else:
            # Use threshold method only
            combined_score = threshold_score
            is_quasi_static = is_quasi_static_threshold
        
        # Log detection results
        logger.debug(f"Quasi-static detection: {is_quasi_static} (score: {combined_score:.3f}, "
                    f"threshold: {threshold_score:.3f}, CNN: {cnn_score:.3f})")
        
        # Update navigation context
        self.context.update_context(
            motion_state="quasi_static" if is_quasi_static else "moving",
            heading_confidence=min(combined_score * 1.5, 1.0)  # Scale confidence but cap at 1.0
        )
        
        return is_quasi_static, combined_score

    def update_heading(self, data: pd.DataFrame, ground_truth: Optional[pd.DataFrame] = None):
        """
        Update the current heading using sensor fusion and quasi-static detection.
        
        Args:
            data: DataFrame containing sensor data
            ground_truth: Optional DataFrame containing ground truth data
        """
        if len(data) < 5:  # Need at least a few points to calculate headings
            logger.warning("Not enough data points to update heading")
            return
        
        # Extract relevant values from the latest data point
        latest = data.iloc[-1]
        
        # Get gyro and compass readings
        gyro_heading = None
        compass_heading = None
        
        if 'gyroSumFromstart0' in latest:
            gyro_heading = latest['gyroSumFromstart0'] % 360
        
        if 'compass' in latest:
            compass_heading = latest['compass']
        
        if gyro_heading is None and compass_heading is None:
            logger.error("No heading information available in data")
            return
        
        # Detect if device is in quasi-static state
        is_quasi_static, quasi_static_confidence = self.detect_quasi_static_state(data)
        
        # Determine fusion method based on context
        fusion_method = self.context.determine_best_fusion_method()
        logger.debug(f"Using fusion method: {fusion_method}")
        
        # Calculate fused heading based on selected method
        previous_heading = self.current_heading
        
        if fusion_method == "compass_calibration" and is_quasi_static and compass_heading is not None:
            # During quasi-static periods, use compass for recalibration
            # But weight it based on confidence
            if quasi_static_confidence > 0.8:
                # High confidence - use compass directly
                self.current_heading = compass_heading
                logger.info(f"Recalibrating heading using compass: {compass_heading:.2f}° (confidence: {quasi_static_confidence:.2f})")
            else:
                # Medium confidence - weighted average
                weight = quasi_static_confidence 
                self.current_heading = (weight * compass_heading) + ((1 - weight) * previous_heading)
                logger.info(f"Partially recalibrating heading: {self.current_heading:.2f}° (weight: {weight:.2f})")
                
        elif fusion_method == "gyro_only" and gyro_heading is not None:
            # In high magnetic disturbance, rely on gyro
            self.current_heading = gyro_heading
            
        elif fusion_method == "cnn_lstm":
            # Implement more sophisticated fusion using deep learning models
            if "cnn_lstm" in self.fusion_methods:
                # Use pre-trained model if available
                pass
            else:
                # Fall back to adaptive fusion
                if compass_heading is not None and gyro_heading is not None:
                    # Simple weighted average with adaptive weights based on context
                    mag_disturbance = self.context.magnetic_disturbance_level
                    
                    if mag_disturbance == "high":
                        compass_weight = 0.1
                    elif mag_disturbance == "medium":
                        compass_weight = 0.3
                    else:
                        compass_weight = 0.5
                    
                    if is_quasi_static:
                        # Increase compass weight in quasi-static periods
                        compass_weight = min(compass_weight + 0.3, 0.9)
                    
                    gyro_weight = 1.0 - compass_weight
                    
                    # Calculate heading using weighted average
                    self.current_heading = (compass_weight * compass_heading) + (gyro_weight * gyro_heading)
                    
                    # Normalize to 0-360 range
                    self.current_heading = self.current_heading % 360
                elif compass_heading is not None:
                    self.current_heading = compass_heading
                elif gyro_heading is not None:
                    self.current_heading = gyro_heading
        
        else:  # Default adaptive fusion
            if compass_heading is not None and gyro_heading is not None:
                # Calculate heading difference (considering circular nature)
                diff = min(abs(compass_heading - gyro_heading), 360 - abs(compass_heading - gyro_heading))
                
                # Determine weights based on difference and quasi-static state
                compass_weight = 0.5  # Default equal weighting
                
                if diff > 45:
                    # Large difference - likely magnetic disturbance
                    compass_weight = 0.2
                    if is_quasi_static:
                        # If quasi-static, give a bit more weight to compass, but still be cautious
                        compass_weight = 0.4
                else:
                    # Small difference - sensors agree
                    if is_quasi_static:
                        # Strongly favor compass when quasi-static and sensors agree
                        compass_weight = 0.8
                    else:
                        # Slightly favor gyro when moving
                        compass_weight = 0.4
                
                gyro_weight = 1.0 - compass_weight
                
                # Calculate heading using weighted average
                self.current_heading = (compass_weight * compass_heading) + (gyro_weight * gyro_heading)
                
                # Normalize to 0-360 range
                self.current_heading = self.current_heading % 360
                
                logger.debug(f"Fused heading: {self.current_heading:.2f}° (compass: {compass_heading:.2f}° × {compass_weight:.2f}, "
                            f"gyro: {gyro_heading:.2f}° × {gyro_weight:.2f})")
            elif compass_heading is not None:
                self.current_heading = compass_heading
            elif gyro_heading is not None:
                self.current_heading = gyro_heading
        
        # Store heading in history
        self.heading_history.append({
            'timestamp': latest.get('Timestamp_(ms)', len(self.heading_history)),
            'heading': self.current_heading,
            'compass': compass_heading,
            'gyro': gyro_heading,
            'quasi_static': is_quasi_static,
            'fusion_method': fusion_method
        })
        
        # Calculate heading error if ground truth available
        if ground_truth is not None and len(ground_truth) > 0:
            try:
                # Find closest ground truth point by timestamp or index
                gt_heading = None
                
                if 'Timestamp_(ms)' in latest and 'Timestamp_(ms)' in ground_truth.columns:
                    # Find by timestamp
                    latest_ts = latest['Timestamp_(ms)']
                    closest_gt = ground_truth.iloc[(ground_truth['Timestamp_(ms)'] - latest_ts).abs().argsort()[0]]
                    if 'GroundTruthHeadingComputed' in closest_gt:
                        gt_heading = closest_gt['GroundTruthHeadingComputed']
                elif 'step' in latest and 'step' in ground_truth.columns:
                    # Find by step
                    if latest['step'] in ground_truth['step'].values:
                        closest_gt = ground_truth[ground_truth['step'] == latest['step']].iloc[0]
                        if 'GroundTruthHeadingComputed' in closest_gt:
                            gt_heading = closest_gt['GroundTruthHeadingComputed']
                
                if gt_heading is not None:
                    # Calculate error (considering circular nature of headings)
                    error = min(abs(self.current_heading - gt_heading), 360 - abs(self.current_heading - gt_heading))
                    self.heading_errors.append(error)
                    logger.debug(f"Heading error: {error:.2f}° (true: {gt_heading:.2f}°, estimated: {self.current_heading:.2f}°)")
            except Exception as e:
                logger.error(f"Error calculating heading error: {e}")
        
        return self.current_heading
    
    def update_position(self, data: pd.DataFrame, ground_truth: Optional[pd.DataFrame] = None):
        """
        Update position based on heading and step count
        
        Args:
            data: DataFrame containing sensor data
            ground_truth: Optional DataFrame containing ground truth data
        """
        if len(data) == 0:
            return
        
        # Check if we have direct position data (from synthetic data)
        if 'east_pos' in data.columns and 'north_pos' in data.columns:
            # Use provided position data directly
            try:
                # For each row in the data, add a position point to the history
                for _, row in data.iterrows():
                    timestamp = row['Timestamp_(ms)'] if 'Timestamp_(ms)' in row else len(self.position_history)
                    east = row['east_pos']
                    north = row['north_pos']
                    heading = self.current_heading
                    step = row['step'] if 'step' in row else len(self.position_history)
                    floor = 0.0  # Default floor
                    
                    # Update current position
                    self.current_position = (east, north)
                    
                    self.position_history.append({
                        'timestamp': timestamp,
                        'east': east,
                        'north': north,
                        'heading': heading,
                        'step': step,
                        'floor': floor
                    })
                return
            except (IndexError, KeyError) as e:
                logger.warning(f"Error using direct position data: {e}")
                # Fall through to calculated position
        
        # If we don't have direct position data, calculate it
        try:
            # Get step information
            if 'step' in data.columns:
                # If we have step information directly
                current_step = data['step'].iloc[-1]
                
                # Check if we already have a position for this step
                if self.position_history and self.position_history[-1]['step'] == current_step:
                    # Skip duplicate step
                    return
                    
                # Calculate step size - simplified assumption
                step_size = 0.7  # meters per step
                
                # Calculate distance moved
                distance = step_size * step_size
                
                # Convert heading to radians for trigonometric calculations
                heading_rad = np.radians(self.current_heading)
                
                # Calculate new position
                new_east = self.current_position[0] + distance * np.sin(heading_rad)
                new_north = self.current_position[1] + distance * np.cos(heading_rad)
                
                # Update current position
                self.current_position = (new_east, new_north)
                
                # Update current step count
                self.current_step_count = current_step
                
                # Record position
                try:
                    timestamp = float(data['Timestamp_(ms)'].iloc[-1])
                except (ValueError, TypeError):
                    timestamp = len(self.position_history)  # Use sequential index if timestamp conversion fails
                    
                self.position_history.append({
                    "timestamp": timestamp,
                    "east": float(self.current_position[0]),
                    "north": float(self.current_position[1]),
                    "heading": float(self.current_heading),
                    "step": float(self.current_step_count),
                    "floor": float(self.current_floor)
                })
                
                # Calculate error if ground truth is available
                if ground_truth is not None:
                    try:
                        # Find the closest ground truth point by step number
                        matching_gt = ground_truth[ground_truth["step"] == current_step]
                        
                        if len(matching_gt) > 0:
                            gt_east = float(matching_gt["GroundTruth_X"].values[0])
                            gt_north = float(matching_gt["GroundTruth_Y"].values[0])
                            
                            # Calculate Euclidean distance error
                            position_error = np.sqrt(
                                (self.current_position[0] - gt_east)**2 + 
                                (self.current_position[1] - gt_north)**2
                            )
                            
                            self.position_errors.append({
                                "timestamp": timestamp,
                                "error": float(position_error),
                                "step": float(self.current_step_count)
                            })
                    except (ValueError, TypeError, KeyError):
                        pass  # Skip error calculation if any conversion fails
        except (ValueError, TypeError, KeyError):
            pass  # Skip position update if step conversion fails
    
    def process_data_stream(self, data: pd.DataFrame, ground_truth: Optional[pd.DataFrame] = None):
        """
        Process a stream of sensor data to update heading and position.
        This is the main entry point for real-time processing.
        
        Args:
            data: DataFrame containing sensor data
            ground_truth: Optional DataFrame containing ground truth data
        """
        if len(data) == 0:
            logger.warning("No data to process")
            return
        
        # First update the context based on sensor data
        self.update_context_from_sensor_data(data)
        
        # Update heading using sensor fusion
        self.update_heading(data, ground_truth)
        
        # Update position using the latest heading
        self.update_position(data, ground_truth)
        
        # Save intermediate results periodically
        if len(self.heading_history) % 100 == 0:
            self.save_results()
            
    def create_hybrid_model_selector(self):
        """
        Create a hybrid model selector that switches between different algorithms
        based on context, as specified in Phase 5 of the roadmap.
        """
        logger.info("Creating hybrid model selector for context-based algorithm switching")
        
        # Define model switching logic based on context
        def model_selector(context: NavigationContext) -> str:
            """Select the best model based on context"""
            
            # 1. Check if we're in a quasi-static period
            if context.current_motion_state == "quasi_static":
                return "compass_calibration"
                
            # 2. Check for magnetic disturbance
            if context.magnetic_disturbance_level == "high":
                return "gyro_only"
                
            # 3. Check for low confidence
            if context.heading_confidence < 0.4:
                if context.position_confidence > 0.7:
                    return "map_matching"
                else:
                    return "cnn_lstm"
                    
            # 4. Choose efficient model based on battery level
            if context.battery_level < 20:
                return "efficient_mode"
                
            # 5. Default to adaptive for balanced approach
            return "adaptive"
            
        return model_selector
        
    def initialize_unified_pipeline(self):
        """
        Initialize the unified pipeline for real-time heading correction and navigation,
        as specified in Phase 5 of the roadmap.
        """
        logger.info("Initializing unified navigation pipeline")
        
        # 1. Set up the hybrid model selector
        self.model_selector = self.create_hybrid_model_selector()
        
        # 2. Initialize all available fusion methods
        self.fusion_methods = {
            "ekf": self._initialize_ekf_fusion(),
            "ukf": self._initialize_ukf_fusion(),
            "adaptive": self._initialize_adaptive_fusion(),
            "gyro_only": None,  # Simple method that doesn't need initialization
            "compass_calibration": None,  # Simple method that doesn't need initialization
            "efficient_mode": self._initialize_efficient_fusion(),
        }
        
        # 3. Initialize CNN-LSTM models if available
        try:
            if os.path.exists(os.path.join(self.output_dir, "models", "cnn_lstm_fusion_model.h5")):
                self.fusion_methods["cnn_lstm"] = tf.keras.models.load_model(
                    os.path.join(self.output_dir, "models", "cnn_lstm_fusion_model.h5")
                )
                logger.info("Loaded CNN-LSTM fusion model")
        except Exception as e:
            logger.error(f"Failed to load CNN-LSTM fusion model: {e}")
            
        # 4. Set up floor detection if available (Phase 3 requirement)
        self.floor_detection_model = None
        try:
            if os.path.exists(os.path.join(self.output_dir, "models", "floor_detection_model.h5")):
                self.floor_detection_model = tf.keras.models.load_model(
                    os.path.join(self.output_dir, "models", "floor_detection_model.h5")
                )
                logger.info("Loaded floor detection model")
        except Exception as e:
            logger.error(f"Failed to load floor detection model: {e}")
            
        # 5. Set up error correction based on quasi-static detection
        self.enable_real_time_error_correction = self.config["sensor_fusion"]["adaptive_switching"]
        
        logger.info("Unified navigation pipeline initialized successfully")
        
    def _initialize_ekf_fusion(self):
        """Initialize Extended Kalman Filter fusion"""
        # Simplified placeholder - in a real implementation, this would create and return an EKF model
        from filterpy.kalman import ExtendedKalmanFilter
        import numpy as np
        
        # Create a simple EKF for heading estimation
        ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)  # State: [heading, gyro_bias], Measurement: [compass]
        
        # Initialize state
        ekf.x = np.array([0.0, 0.0])  # Initial heading and gyro bias
        
        # State transition matrix (updated dynamically during predict)
        ekf.F = np.array([[1.0, 1.0], [0.0, 1.0]])  # Simple model: heading += gyro + bias
        
        # Measurement function (updated dynamically during update)
        ekf.H = np.array([[1.0, 0.0]])  # Measure heading directly
        
        # Covariance matrices
        ekf.P = np.array([[10.0, 0.0], [0.0, 1.0]])  # Initial uncertainty
        ekf.R = np.array([[10.0]])  # Measurement uncertainty (compass noise)
        ekf.Q = np.array([[1.0, 0.0], [0.0, 0.1]])  # Process noise
        
        return ekf
        
    def _initialize_ukf_fusion(self):
        """Initialize Unscented Kalman Filter fusion"""
        # Simplified placeholder - in a real implementation, this would create and return a UKF model
        return None
        
    def _initialize_adaptive_fusion(self):
        """Initialize adaptive fusion model"""
        # This could be a wrapper around multiple models with a switching mechanism
        return None
        
    def _initialize_efficient_fusion(self):
        """Initialize energy-efficient fusion model for low battery situations"""
        # This would be a simpler model optimized for energy efficiency
        return None
    
    def calibrate_system(self, data: pd.DataFrame, ground_truth: Optional[pd.DataFrame] = None):
        """
        Calibrate the system using genetic algorithm optimization for quasi-static detection.
        """
        qs_config = self.config["quasi_static"]
        
        if qs_config["use_genetic_algorithm"] and ground_truth is not None:
            logger.info("Starting genetic algorithm optimization for quasi-static parameters...")
            
            try:
                # Create optimizer instance
                ga_optimizer = GeneticAlgorithmOptimizer(
                    data,
                    ground_truth_data=data,
                    population_size=20,
                    generations=10
                )
                
                # Run optimization
                stability_threshold, window_size = ga_optimizer.optimize()
                
                # Update detector with optimized parameters
                self.quasi_static_detector = QuasiStaticDetector(
                    stability_threshold=stability_threshold,
                    window_size=window_size
                )
                
                logger.info(f"Genetic algorithm optimization complete. New parameters: "
                          f"stability_threshold={stability_threshold:.2f}, window_size={window_size}")
                
                # Evaluate parameters
                evaluation = evaluate_quasi_static_parameters(
                    data, 
                    stability_threshold,
                    window_size,
                    plot=False
                )
                
                logger.info(f"Evaluation results: "
                          f"num_intervals={evaluation['num_intervals']}, "
                          f"average_difference={evaluation['average_difference']:.2f}°, "
                          f"mse={evaluation['mse']:.2f}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error during system calibration: {e}")
                return False
        
        return False
    
    def reset(self):
        """Reset the navigation system to initial state"""
        self.current_heading = 0.0
        self.current_position = (0.0, 0.0)
        self.current_step_count = 0
        self.current_floor = 0
        
        self.heading_history = []
        self.position_history = []
        self.context_history = []
        
        self.heading_errors = []
        self.position_errors = []
        
        # Reset detectors
        self._initialize_components()
        
        logger.info("Navigation system reset to initial state")
    
    def visualize_results(self, output_dir="output"):
        """
        Visualize the navigation results
        
        Args:
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        heading_history = pd.DataFrame(self.heading_history)
        position_history = pd.DataFrame(self.position_history)
        
        if len(heading_history) <= 1 or len(position_history) <= 1:
            logging.warning("Not enough data to visualize (need at least 2 data points)")
            return
        
        # 1. Plot heading over time
        if 'timestamp' in heading_history.columns and 'heading' in heading_history.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(heading_history['timestamp'], heading_history['heading'], 
                     label='Estimated Heading')
            
            if 'compass' in heading_history.columns:
                plt.plot(heading_history['timestamp'], heading_history['compass'], 
                         alpha=0.5, label='Compass')
                
            if 'gyro_heading' in heading_history.columns:
                plt.plot(heading_history['timestamp'], heading_history['gyro_heading'], 
                         alpha=0.5, label='Gyro Heading')
            
            # Mark quasi-static periods if available
            if 'is_quasi_static' in heading_history.columns:
                qs_indices = heading_history[heading_history['is_quasi_static']].index
                if len(qs_indices) > 0:
                    plt.scatter(
                        heading_history.loc[qs_indices, 'timestamp'], 
                        heading_history.loc[qs_indices, 'heading'],
                        color='red', s=20, alpha=0.7, label='Quasi-Static Points'
                    )
            
            plt.xlabel('Time (ms)')
            plt.ylabel('Heading (degrees)')
            plt.title('Heading Estimation Over Time')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'heading_over_time.png'))
            plt.close()
            logging.info(f"Saved heading visualization to {output_dir}/heading_over_time.png")
        
        # 2. Plot trajectory in 2D
        if 'east' in position_history.columns and 'north' in position_history.columns:
            plt.figure(figsize=(12, 12))
            plt.plot(position_history['east'], position_history['north'], 'b-', 
                    label='Estimated Path')
            
            # Mark start and end points
            plt.scatter(position_history['east'].iloc[0], position_history['north'].iloc[0], 
                       color='green', s=100, label='Start')
            plt.scatter(position_history['east'].iloc[-1], position_history['north'].iloc[-1], 
                       color='red', s=100, label='End')
            
            # If we have ground truth for comparison
            if ('true_east' in position_history.columns and 
                'true_north' in position_history.columns):
                plt.plot(position_history['true_east'], position_history['true_north'], 
                        'g--', label='Ground Truth')
            
            plt.xlabel('East (m)')
            plt.ylabel('North (m)')
            plt.title('Estimated Trajectory')
            plt.axis('equal')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'estimated_trajectory.png'))
            plt.close()
            logging.info(f"Saved trajectory visualization to {output_dir}/estimated_trajectory.png")
        
        # Save data for potential further analysis
        heading_history.to_csv(os.path.join(output_dir, 'heading_history.csv'), index=False)
        position_history.to_csv(os.path.join(output_dir, 'position_history.csv'), index=False)
        logging.info(f"Saved result data to {output_dir}/heading_history.csv and {output_dir}/position_history.csv")
    
    def save_results(self, output_dir: Optional[str] = None):
        """Save the results of the navigation system to CSV files"""
        if output_dir is None:
            output_dir = self.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save heading history
        if len(self.heading_history) > 0:
            heading_df = pd.DataFrame(self.heading_history)
            heading_df.to_csv(os.path.join(output_dir, 'heading_history.csv'), index=False)
            
        # Save position history
        if len(self.position_history) > 0:
            position_df = pd.DataFrame(self.position_history)
            position_df.to_csv(os.path.join(output_dir, 'position_history.csv'), index=False)
            
        # Save heading errors
        if len(self.heading_errors) > 0:
            error_df = pd.DataFrame(self.heading_errors)
            error_df.to_csv(os.path.join(output_dir, 'heading_errors.csv'), index=False)
            
        # Save position errors
        if len(self.position_errors) > 0:
            pos_error_df = pd.DataFrame(self.position_errors)
            pos_error_df.to_csv(os.path.join(output_dir, 'position_errors.csv'), index=False)
            
        logger.info(f"Results saved to {output_dir}")


def main():
    """Main function to run the integrated navigation system"""
    logger.info("Starting Integrated Navigation System")
    
    # Initialize the system
    nav_system = IntegratedNavigationSystem()
    
    # Initialize the unified pipeline
    nav_system.initialize_unified_pipeline()
    
    # Load sample data
    try:
        # Try to load from the data directory if available
        data_file = os.path.join("data", "sample_navigation_data.csv")
        if not os.path.exists(data_file):
            # Fallback to a different location
            data_file = "sample_navigation_data.csv"
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Cannot find data file: {data_file}")
        
        logger.info(f"Loading data from {data_file}")
        data = load_and_prepare_data(data_file)
        
        # Check if ground truth is available
        ground_truth_file = os.path.join("data", "ground_truth_data.csv")
        ground_truth = None
        if os.path.exists(ground_truth_file):
            logger.info(f"Loading ground truth from {ground_truth_file}")
            ground_truth = load_and_prepare_data(ground_truth_file)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.info("Generating synthetic test data instead")
        
        # Generate synthetic data for testing with a more realistic trajectory
        num_points = 1000
        timestamps = np.arange(num_points)
        
        # Generate a more complex compass heading pattern with turns
        # Start with a straight line, then make a few turns
        compass_base = np.ones(num_points) * 180
        
        # Add some turns at specific points
        turn_points = [200, 400, 600, 800]
        turn_angles = [90, -45, 30, -60]
        
        for i, (point, angle) in enumerate(zip(turn_points, turn_angles)):
            compass_base[point:] += angle
        
        # Add some noise to the compass readings
        compass = compass_base + np.random.normal(0, 5, num_points)
        compass = compass % 360
        
        # Generate gyro readings with gradual drift
        gyro_increments = np.random.normal(0, 0.1, num_points)
        # Add some bias to simulate gyro drift
        gyro_increments += 0.01  
        
        # At turn points, add actual turn angles to gyro
        for point, angle in zip(turn_points, turn_angles):
            gyro_increments[point] += angle
            
        gyro_sum = np.cumsum(gyro_increments) % 360
        
        # Create quasi-static periods
        is_quasi_static = np.zeros(num_points, dtype=bool)
        quasi_static_periods = [(50, 100), (250, 300), (450, 500), (650, 700), (850, 900)]
        
        for start, end in quasi_static_periods:
            is_quasi_static[start:end] = True
            
        # Make gyro and compass more consistent during quasi-static periods
        for start, end in quasi_static_periods:
            # Reduce noise in quasi-static periods
            compass[start:end] = compass_base[start:end] + np.random.normal(0, 1, end-start)
            # Almost no increments in gyro during quasi-static
            gyro_increments[start:end] = np.random.normal(0, 0.01, end-start)
        
        # Generate position data based on heading
        east = np.zeros(num_points)
        north = np.zeros(num_points)
        
        # Use the average of compass and gyro for heading to calculate positions
        heading = (compass + gyro_sum) / 2
        step_length = 0.5  # meters
        
        for i in range(1, num_points):
            # Convert heading to radians (adjusting for heading direction convention)
            heading_rad = np.radians(90 - heading[i])
            # Calculate position increment
            east[i] = east[i-1] + step_length * np.cos(heading_rad)
            north[i] = north[i-1] + step_length * np.sin(heading_rad)

        # Create the DataFrame with all the synthetic data
        data = pd.DataFrame({
            'Timestamp_(ms)': timestamps,
            'compass': compass,
            'gyroSumFromstart0': gyro_sum,
            'step': timestamps,
            'is_quasi_static': is_quasi_static,
            'east_pos': east,
            'north_pos': north
        })
        
        ground_truth = None
    
    # First, calibrate the system using a portion of the data
    if len(data) > 100:
        logger.info("Calibrating system...")
        calibration_data = data.iloc[:min(500, len(data)//2)]
        nav_system.calibrate_system(calibration_data, ground_truth)
    
    # Process the data
    logger.info("Processing data...")
    
    # Track performance metrics
    start_time = time.time()
    
    # Decide processing method based on size
    if len(data) > 1000:
        # For large datasets, process in chunks to simulate real-time
        chunk_size = 100
        chunks = [data.iloc[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} records)")
            
            # Get corresponding ground truth if available
            gt_chunk = None
            if ground_truth is not None:
                # Match ground truth based on timestamp or step
                if 'Timestamp_(ms)' in ground_truth.columns and 'Timestamp_(ms)' in chunk.columns:
                    min_ts = chunk['Timestamp_(ms)'].min()
                    max_ts = chunk['Timestamp_(ms)'].max()
                    gt_chunk = ground_truth[(ground_truth['Timestamp_(ms)'] >= min_ts) & 
                                          (ground_truth['Timestamp_(ms)'] <= max_ts)]
                elif 'step' in ground_truth.columns and 'step' in chunk.columns:
                    min_step = chunk['step'].min()
                    max_step = chunk['step'].max()
                    gt_chunk = ground_truth[(ground_truth['step'] >= min_step) & 
                                          (ground_truth['step'] <= max_step)]
            
            # Process the chunk
            nav_system.process_data_stream(chunk, gt_chunk)
    else:
        # For small datasets, process row by row to generate points at each timestamp
        for i in range(len(data)):
            # Create a single-row DataFrame for this timestamp
            row_data = data.iloc[[i]]
            
            # Process this individual data point
            nav_system.process_data_stream(row_data, ground_truth)
            
            # Log progress periodically
            if i % 100 == 0:
                logger.debug(f"Processed {i}/{len(data)} records")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info(f"Data processing completed in {processing_time:.2f} seconds")
    logger.info(f"Processed {len(data)} records at {len(data)/processing_time:.2f} records/second")
    
    # Calculate and display error metrics if ground truth is available
    if ground_truth is not None and nav_system.heading_errors:
        mean_heading_error = np.mean(nav_system.heading_errors)
        median_heading_error = np.median(nav_system.heading_errors)
        max_heading_error = np.max(nav_system.heading_errors)
        
        logger.info("Heading error metrics:")
        logger.info(f"Mean error: {mean_heading_error:.2f} degrees")
        logger.info(f"Median error: {median_heading_error:.2f} degrees")
        logger.info(f"Max error: {max_heading_error:.2f} degrees")
    
    # Visualize results
    logger.info("Generating visualization...")
    nav_system.visualize_results()
    
    # Save results
    logger.info("Saving results...")
    nav_system.save_results()
    
    logger.info("Integrated Navigation System completed successfully")
    
if __name__ == "__main__":
    main() 