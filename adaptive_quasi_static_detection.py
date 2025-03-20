#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
from deap import base, creator, tools, algorithms
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
from scipy.stats import iqr
import os

class QuasiStaticDetector:
    """Base class for quasi-static detection"""
    def __init__(self, stability_threshold=5.0, window_size=100, use_multi_features=True, 
                 gyro_threshold=2.0, mag_threshold=5.0, combined_weights=None):
        self.stability_threshold = stability_threshold
        self.window_size = window_size
        self.compass_heading_window = []
        self.gyro_values_window = []
        self.mag_values_window = []
        self.use_multi_features = use_multi_features
        self.gyro_threshold = gyro_threshold
        self.mag_threshold = mag_threshold
        
        # Default weights for combined score if not provided
        self.combined_weights = combined_weights or {
            'compass': 0.4,
            'gyro': 0.5,  # Give more weight to gyro as it's more responsive
            'mag': 0.1
        }
        
    def add_sensor_data(self, compass_heading, gyro_value=None, mag_value=None):
        """Add sensor data to the windows"""
        self.compass_heading_window.append(compass_heading)
        if len(self.compass_heading_window) > self.window_size:
            self.compass_heading_window.pop(0)  # Remove oldest value
        
        if gyro_value is not None:
            self.gyro_values_window.append(gyro_value)
            if len(self.gyro_values_window) > self.window_size:
                self.gyro_values_window.pop(0)
                
        if mag_value is not None:
            self.mag_values_window.append(mag_value)
            if len(self.mag_values_window) > self.window_size:
                self.mag_values_window.pop(0)
            
    def add_compass_heading(self, compass_heading):
        """Legacy method for backward compatibility"""
        self.add_sensor_data(compass_heading)
            
    def is_quasi_static_interval(self):
        """Determine if the current interval is quasi-static"""
        if len(self.compass_heading_window) < self.window_size:
            return False
            
        if not self.use_multi_features:
            # Traditional approach: just check compass heading variance
            variance = np.var(self.compass_heading_window)
            return variance < self.stability_threshold
        
        # Multi-feature approach
        compass_variance = np.var(self.compass_heading_window)
        compass_score = 1.0 if compass_variance < self.stability_threshold else 0.0
        
        # Initialize other scores
        gyro_score, mag_score = 0.0, 0.0
        
        # Check gyroscope data if available
        if len(self.gyro_values_window) >= self.window_size:
            # For gyro, we look at both variance and the absolute mean
            # (to detect slight but consistent rotation)
            gyro_variance = np.var(self.gyro_values_window)
            gyro_abs_mean = np.abs(np.mean(self.gyro_values_window))
            
            gyro_var_score = 1.0 if gyro_variance < self.gyro_threshold else 0.0
            gyro_mean_score = 1.0 if gyro_abs_mean < self.gyro_threshold/2 else 0.0
            
            # Combined gyro score with more weight on variance
            gyro_score = (0.7 * gyro_var_score) + (0.3 * gyro_mean_score)
        
        # Check magnetometer data if available
        if len(self.mag_values_window) >= self.window_size:
            mag_variance = np.var(self.mag_values_window)
            mag_score = 1.0 if mag_variance < self.mag_threshold else 0.0
        
        # Calculate combined score
        combined_score = (
            self.combined_weights['compass'] * compass_score +
            self.combined_weights['gyro'] * gyro_score +
            self.combined_weights['mag'] * mag_score
        )
        
        # Normalize weights if not all sensors are available
        total_weight = 0
        for sensor, weight in self.combined_weights.items():
            if sensor == 'compass' or (sensor == 'gyro' and len(self.gyro_values_window) >= self.window_size) or \
               (sensor == 'mag' and len(self.mag_values_window) >= self.window_size):
                total_weight += weight
        
        if total_weight > 0:
            combined_score /= total_weight
            
        # Consider quasi-static if the combined score exceeds 0.7 (configurable threshold)
        return combined_score > 0.7
    
    def get_stability_score(self):
        """Return a continuous stability score between 0 and 1"""
        if len(self.compass_heading_window) < self.window_size:
            return 0.0
            
        # Calculate basic compass score
        compass_variance = np.var(self.compass_heading_window)
        compass_score = max(0, 1.0 - (compass_variance / self.stability_threshold))
        
        if not self.use_multi_features:
            return compass_score
            
        # Multi-feature approach with continuous scores
        gyro_score, mag_score = 0.0, 0.0
        
        # Check gyroscope data if available
        if len(self.gyro_values_window) >= self.window_size:
            gyro_variance = np.var(self.gyro_values_window)
            gyro_abs_mean = np.abs(np.mean(self.gyro_values_window))
            
            gyro_var_score = max(0, 1.0 - (gyro_variance / self.gyro_threshold))
            gyro_mean_score = max(0, 1.0 - (gyro_abs_mean / (self.gyro_threshold/2)))
            
            gyro_score = (0.7 * gyro_var_score) + (0.3 * gyro_mean_score)
        
        # Check magnetometer data if available
        if len(self.mag_values_window) >= self.window_size:
            mag_variance = np.var(self.mag_values_window)
            mag_score = max(0, 1.0 - (mag_variance / self.mag_threshold))
        
        # Calculate weighted score
        combined_score = (
            self.combined_weights['compass'] * compass_score +
            self.combined_weights['gyro'] * gyro_score +
            self.combined_weights['mag'] * mag_score
        )
        
        # Normalize weights if not all sensors are available
        total_weight = 0
        for sensor, weight in self.combined_weights.items():
            if sensor == 'compass' or (sensor == 'gyro' and len(self.gyro_values_window) >= self.window_size) or \
               (sensor == 'mag' and len(self.mag_values_window) >= self.window_size):
                total_weight += weight
        
        if total_weight > 0:
            combined_score /= total_weight
            
        return min(1.0, max(0.0, combined_score))
        
    def calculate_mean(self):
        if self.compass_heading_window:
            return np.mean(self.compass_heading_window)
        return None
    
    def reset(self):
        self.compass_heading_window = []
        self.gyro_values_window = []
        self.mag_values_window = []


class GeneticAlgorithmOptimizer:
    """Class for genetic algorithm optimization of quasi-static detection parameters"""
    def __init__(self, compass_data, ground_truth_data=None, population_size=50, generations=20):
        self.compass_data = compass_data
        self.ground_truth_data = ground_truth_data
        self.population_size = population_size
        self.generations = generations
        
        # Initialize genetic algorithm parameters
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        # Attributes: stability_threshold (0.1-20), window_size (10-500)
        self.toolbox.register("attr_stability", random.uniform, 0.1, 20.0)
        self.toolbox.register("attr_window", random.randint, 10, 500)
        
        # Initialize structures
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                             (self.toolbox.attr_stability, self.toolbox.attr_window), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual, 
                             n=self.population_size)
        
        # Register operators
        self.toolbox.register("evaluate", self.evaluate_parameters)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self.mutate_params)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def mutate_params(self, individual):
        """Custom mutation operator for our parameters"""
        if random.random() < 0.5:
            individual[0] += random.gauss(0, 2)  # Mutate stability threshold
            individual[0] = max(0.1, min(20.0, individual[0]))  # Clamp to bounds
        else:
            individual[1] += random.randint(-50, 50)  # Mutate window size
            individual[1] = max(10, min(500, individual[1]))  # Clamp to bounds
        return individual,
    
    def evaluate_parameters(self, individual):
        """Evaluate the fitness of a parameter set"""
        stability_threshold = individual[0]
        window_size = int(individual[1])
        
        # Create detector with the parameters
        detector = QuasiStaticDetector(stability_threshold=stability_threshold, window_size=window_size)
        
        # Parameters to track
        num_intervals = 0
        mean_variance = 0
        quasi_static_headings = []
        ground_truth_headings = []
        
        # Track intervals
        is_quasi_static = False
        current_interval_headings = []
        current_interval_ground_truths = []
        
        # Process data
        for idx in range(len(self.compass_data)):
            heading = self.compass_data.iloc[idx]['compass']
            detector.add_sensor_data(heading)
            
            if detector.is_quasi_static_interval():
                current_interval_headings.append(heading)
                
                # If we have ground truth, add it
                if self.ground_truth_data is not None:
                    ground_truth = self.compass_data.iloc[idx]['GroundTruthHeadingComputed']
                    current_interval_ground_truths.append(ground_truth)
                
                if not is_quasi_static:
                    is_quasi_static = True
            else:
                if is_quasi_static:
                    # End of an interval
                    if len(current_interval_headings) > 0:
                        # Calculate mean and variance for this interval
                        mean_heading = np.mean(current_interval_headings)
                        var_heading = np.var(current_interval_headings)
                        mean_variance += var_heading
                        
                        quasi_static_headings.append(mean_heading)
                        
                        if len(current_interval_ground_truths) > 0:
                            ground_truth_headings.append(np.mean(current_interval_ground_truths))
                    
                    # Reset for next interval
                    num_intervals += 1
                    current_interval_headings = []
                    current_interval_ground_truths = []
                    is_quasi_static = False
        
        # Calculate fitness based on:
        # 1. Number of intervals found (want a reasonable number, not too few or too many)
        # 2. Mean variance within intervals (lower is better)
        # 3. If ground truth available, accuracy of the compass heading vs. ground truth
        
        if num_intervals == 0:
            return (0.0,)  # Penalize if no intervals found
        
        interval_score = min(num_intervals / 10.0, 1.0)  # Normalize, want ~10 intervals
        
        variance_score = 1.0 / (1.0 + mean_variance / num_intervals)  # Lower variance is better
        
        accuracy_score = 0.0
        if self.ground_truth_data is not None and len(quasi_static_headings) == len(ground_truth_headings) and len(quasi_static_headings) > 0:
            # Calculate heading error
            heading_errors = [min(abs(q - g), 360 - abs(q - g)) for q, g in zip(quasi_static_headings, ground_truth_headings)]
            mean_error = np.mean(heading_errors)
            accuracy_score = 1.0 / (1.0 + mean_error / 10.0)  # Lower error is better
        
        # Combine scores
        final_score = (interval_score * 0.2) + (variance_score * 0.3) + (accuracy_score * 0.5)
        
        return (final_score,)
    
    def optimize(self, verbose=True):
        """Run the genetic algorithm optimization"""
        start_time = time.time()
        
        if verbose:
            print("Starting genetic algorithm optimization...")
            print(f"Population size: {self.population_size}, Generations: {self.generations}")
        
        # Initialize population
        pop = self.toolbox.population()
        
        # Track stats
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run algorithm
        if verbose:
            pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=0.7, mutpb=0.3,
                                           ngen=self.generations, stats=stats, verbose=True)
        else:
            pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=0.7, mutpb=0.3,
                                           ngen=self.generations, stats=stats, verbose=False)
        
        # Get best individual
        best_ind = tools.selBest(pop, 1)[0]
        best_stability = best_ind[0]
        best_window = int(best_ind[1])
        
        end_time = time.time()
        
        if verbose:
            print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
            print(f"Best parameters:")
            print(f"  Stability Threshold: {best_stability:.2f}")
            print(f"  Window Size: {best_window}")
            print(f"  Fitness: {best_ind.fitness.values[0]:.4f}")
        
        return best_stability, best_window


class ReinforcementLearningOptimizer:
    """Class for reinforcement learning optimization of quasi-static detection parameters"""
    def __init__(self, compass_data, ground_truth_data=None, learning_rate=0.1, discount_factor=0.9):
        self.compass_data = compass_data
        self.ground_truth_data = ground_truth_data
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Define the state space
        # State: (current_variance, recent_interval_count)
        # Actions: (adjust_stability, adjust_window)
        self.stability_actions = [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]
        self.window_actions = [-20, -10, -5, 0, 5, 10, 20]
        
        # Initialize Q-table
        self.q_table = {}
        
    def get_state(self, detector, recent_intervals):
        """Convert current detector state to a discrete state"""
        if len(detector.compass_heading_window) < detector.window_size:
            current_variance = 0
        else:
            current_variance = min(20, int(np.var(detector.compass_heading_window)))
        
        interval_count = min(5, recent_intervals)
        
        return (current_variance, interval_count)
    
    def get_action(self, state, epsilon=0.1):
        """Select an action using epsilon-greedy policy"""
        if random.random() < epsilon:
            # Exploration
            stability_idx = random.randint(0, len(self.stability_actions) - 1)
            window_idx = random.randint(0, len(self.window_actions) - 1)
        else:
            # Exploitation
            if state not in self.q_table:
                self.q_table[state] = np.zeros((len(self.stability_actions), len(self.window_actions)))
            
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            # Find all actions with max Q-value
            max_actions = [(i, j) for i in range(len(self.stability_actions)) 
                           for j in range(len(self.window_actions)) 
                           if q_values[i, j] == max_q]
            
            # Randomly select from max actions
            stability_idx, window_idx = random.choice(max_actions)
        
        return self.stability_actions[stability_idx], self.window_actions[window_idx]
    
    def apply_action(self, detector, action):
        """Apply selected action to the detector"""
        stability_change, window_change = action
        
        # Update parameters
        detector.stability_threshold = max(0.1, detector.stability_threshold + stability_change)
        detector.window_size = max(10, detector.window_size + window_change)
        
        return detector
    
    def calculate_reward(self, detector, is_quasi_static, ground_truth=None, compass_heading=None):
        """Calculate reward based on detector performance"""
        # Base reward
        reward = 0
        
        # If we're in a quasi-static interval
        if is_quasi_static:
            # Good if variance is low
            current_variance = np.var(detector.compass_heading_window)
            variance_reward = 1.0 / (1.0 + current_variance)
            reward += variance_reward
            
            # If ground truth available, check accuracy
            if ground_truth is not None and compass_heading is not None:
                # Calculate heading error
                heading_error = min(abs(compass_heading - ground_truth), 360 - abs(compass_heading - ground_truth))
                accuracy_reward = 1.0 / (1.0 + heading_error / 10.0)
                reward += 2.0 * accuracy_reward  # Weight accuracy higher
        
        return reward
    
    def train(self, episodes=10, max_steps=1000, epsilon=0.1, verbose=True):
        """Train the RL agent"""
        if verbose:
            print("Starting reinforcement learning training...")
        
        best_stability = 5.0  # Default
        best_window = 100  # Default
        best_reward = -float('inf')
        
        for episode in range(episodes):
            if verbose:
                print(f"Episode {episode+1}/{episodes}")
            
            # Initialize detector with random parameters
            stability = random.uniform(0.1, 10.0)
            window = random.randint(20, 200)
            detector = QuasiStaticDetector(stability_threshold=stability, window_size=window)
            
            total_reward = 0
            recent_intervals = 0
            is_quasi_static = False
            
            # Random starting point in the data
            if len(self.compass_data) > max_steps:
                start_idx = random.randint(0, len(self.compass_data) - max_steps - 1)
            else:
                start_idx = 0
                max_steps = len(self.compass_data)
            
            for step in range(max_steps):
                idx = start_idx + step
                if idx >= len(self.compass_data):
                    break
                
                # Get current compass heading
                heading = self.compass_data.iloc[idx]['compass']
                detector.add_sensor_data(heading)
                
                # Get current state
                state = self.get_state(detector, recent_intervals)
                
                # Select action
                action = self.get_action(state, epsilon)
                
                # Apply action
                detector = self.apply_action(detector, action)
                
                # Check if we're in a quasi-static interval
                was_quasi_static = is_quasi_static
                is_quasi_static = detector.is_quasi_static_interval()
                
                # If state changed, count it
                if is_quasi_static and not was_quasi_static:
                    recent_intervals += 1
                
                # Calculate reward
                ground_truth = None
                if self.ground_truth_data is not None:
                    ground_truth = self.compass_data.iloc[idx]['GroundTruthHeadingComputed']
                
                reward = self.calculate_reward(detector, is_quasi_static, ground_truth, heading)
                total_reward += reward
                
                # Get next state
                next_state = self.get_state(detector, recent_intervals)
                
                # Update Q-table
                if state not in self.q_table:
                    self.q_table[state] = np.zeros((len(self.stability_actions), len(self.window_actions)))
                
                if next_state not in self.q_table:
                    self.q_table[next_state] = np.zeros((len(self.stability_actions), len(self.window_actions)))
                
                # Find indices of the applied action
                stability_idx = self.stability_actions.index(action[0])
                window_idx = self.window_actions.index(action[1])
                
                # Q-learning update
                best_next_q = np.max(self.q_table[next_state])
                self.q_table[state][stability_idx, window_idx] += self.learning_rate * (
                    reward + self.discount_factor * best_next_q - self.q_table[state][stability_idx, window_idx]
                )
            
            if verbose:
                print(f"  Episode reward: {total_reward}")
                print(f"  Final parameters: Stability={detector.stability_threshold:.2f}, Window={detector.window_size}")
            
            # Track best parameters
            if total_reward > best_reward:
                best_reward = total_reward
                best_stability = detector.stability_threshold
                best_window = detector.window_size
        
        if verbose:
            print("\nTraining completed")
            print(f"Best parameters:")
            print(f"  Stability Threshold: {best_stability:.2f}")
            print(f"  Window Size: {best_window}")
        
        return best_stability, best_window


def evaluate_quasi_static_parameters(data, stability_threshold, window_size, threshold_step_difference=0, plot=True):
    """Evaluate quasi-static detection with given parameters"""
    # Create an instance of the QuasiStaticHeadingTracker
    tracker = QuasiStaticDetector(stability_threshold=stability_threshold, window_size=window_size)
    
    # Extract data
    timestamps = data['Timestamp_(ms)']
    compass_headings = data['compass']
    true_headings = data['GroundTruthHeadingComputed']
    steps = data['step']
    
    # Try to get floor, east, north if available
    try:
        floors = data['value_4']
        easts = data['GroundTruth_X']
        norths = data['GroundTruth_Y']
    except:
        floors = None
        easts = None
        norths = None
    
    # Initialize lists to store data points where the quasi-static interval is detected
    quasi_static_intervals = []
    quasi_static_headings = []
    quasi_static_steps = []
    
    # Initialize a flag to track the state of the interval
    is_quasi_static_interval = False
    num_quasi_static_intervals = 0
    
    # Initialize data dictionary
    data_QS = {
        'Quasi_Static_Interval_Number': [],
        'Compass_Heading': [],
        'True_Heading': [],
        'Time': [],
        'Step': []
    }
    
    if floors is not None:
        data_QS['Floor'] = []
        data_QS['east'] = []
        data_QS['north'] = []
    
    # Iterate through the compass headings and track quasi-static intervals
    start_step = None  # Variable to hold the start step of the interval
    temp_data = {
        'Time': [], 
        'Compass_Heading': [], 
        'True_Heading': [], 
        'Step': []
    }
    
    if floors is not None:
        temp_data['Floor'] = []
        temp_data['east'] = []
        temp_data['north'] = []
    
    for i in range(len(timestamps)):
        timestamp = timestamps.iloc[i]
        heading = compass_headings.iloc[i]
        true = true_headings.iloc[i]
        step = steps.iloc[i]
        
        if floors is not None:
            floor = floors.iloc[i]
            east = easts.iloc[i]
            north = norths.iloc[i]
        
        tracker.add_sensor_data(heading)
        
        if tracker.is_quasi_static_interval():
            if start_step is None:
                start_step = step  # Record the start step of the interval
            
            # Append data to temporary arrays
            temp_data['Time'].append(timestamp)
            temp_data['Compass_Heading'].append(heading)
            temp_data['True_Heading'].append(true)
            temp_data['Step'].append(step)
            
            if floors is not None:
                temp_data['Floor'].append(floor)
                temp_data['east'].append(east)
                temp_data['north'].append(north)
            
            # If it's a quasi-static interval and we're not currently in one, increment the counter
            if not is_quasi_static_interval:
                num_quasi_static_intervals += 1
                is_quasi_static_interval = True
        
        else:
            if is_quasi_static_interval:
                # Calculate the step difference at the end of the interval
                if start_step is not None:
                    step_difference = step - start_step
                    
                    # Check if the step difference exceeds the threshold
                    if step_difference >= threshold_step_difference:
                        # Add the temporary data to data_QS
                        data_QS['Quasi_Static_Interval_Number'].extend([num_quasi_static_intervals] * len(temp_data['Time']))
                        data_QS['Compass_Heading'].extend(temp_data['Compass_Heading'])
                        data_QS['True_Heading'].extend(temp_data['True_Heading'])
                        data_QS['Time'].extend(temp_data['Time'])
                        data_QS['Step'].extend(temp_data['Step'])
                        
                        if floors is not None:
                            data_QS['Floor'].extend(temp_data['Floor'])
                            data_QS['east'].extend(temp_data['east'])
                            data_QS['north'].extend(temp_data['north'])
                    else:
                        num_quasi_static_intervals -= 1
            
            is_quasi_static_interval = False
            start_step = None  # Reset the start step for the next interval
            temp_data = {
                'Time': [], 
                'Compass_Heading': [], 
                'True_Heading': [], 
                'Step': []
            }
            
            if floors is not None:
                temp_data['Floor'] = []
                temp_data['east'] = []
                temp_data['north'] = []
    
    # Create DataFrame from the data
    quasi_static_data = pd.DataFrame(data_QS)
    
    # Calculate metrics
    if len(quasi_static_data) > 0:
        # Group by interval and calculate statistics
        averages = quasi_static_data.groupby('Quasi_Static_Interval_Number').agg({
            'Compass_Heading': 'mean',
            'True_Heading': 'mean'
        }).reset_index()
        
        # Calculate absolute difference between Compass_Heading and True_Heading
        averages['Abs_Difference'] = abs(averages['Compass_Heading'] - averages['True_Heading'])
        
        # Calculate the average absolute difference across all intervals
        average_abs_difference = averages['Abs_Difference'].mean()
        
        # Calculate mean squared error between compass and true headings for evaluation
        if len(averages) > 0:
            mse = mean_squared_error(averages['True_Heading'], averages['Compass_Heading'])
        else:
            mse = float('inf')
    else:
        average_abs_difference = float('inf')
        mse = float('inf')
    
    if plot:
        # Create visualizations
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, compass_headings, label='Compass Headings', color='cyan')
        
        if len(quasi_static_data) > 0:
            plt.scatter(quasi_static_data['Time'], quasi_static_data['Compass_Heading'],
                    c=quasi_static_data['Quasi_Static_Interval_Number'], cmap='Set1', zorder=1, 
                    label='Quasi-Static Intervals')
        
        plt.plot(timestamps, true_headings, marker='.', linestyle='-', 
                markersize=5, color='blue', label='True Heading')
        
        plt.xlabel('Time')
        plt.ylabel('Heading (degrees)')
        plt.title(f'Quasi-Static Detection (Stability: {stability_threshold:.2f}, Window: {window_size})')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot step numbers
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, steps, label='Steps', color='cyan')
        
        if len(quasi_static_data) > 0:
            plt.scatter(quasi_static_data['Time'], quasi_static_data['Step'],
                    c=quasi_static_data['Quasi_Static_Interval_Number'], cmap='Set1', zorder=1,
                    label='Quasi-Static Intervals')
        
        plt.xlabel('Time')
        plt.ylabel('Step Number')
        plt.title(f'Steps with Quasi-Static Intervals (Intervals: {num_quasi_static_intervals})')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # If we have position data, plot it
        if floors is not None and len(quasi_static_data) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(easts, norths, label='All Locations', color='cyan')
            plt.scatter(quasi_static_data['east'], quasi_static_data['north'],
                    c=quasi_static_data['Quasi_Static_Interval_Number'], cmap='Set1', zorder=5,
                    label='Quasi-Static Intervals')
            
            plt.xlabel('East')
            plt.ylabel('North')
            plt.title('Locations with Quasi-Static Intervals')
            plt.legend()
            plt.tight_layout()
            plt.show()
    
    return {
        'num_intervals': num_quasi_static_intervals,
        'average_difference': average_abs_difference,
        'mse': mse,
        'quasi_static_data': quasi_static_data
    }


def load_and_prepare_data(file_path):
    """Load and prepare data from a file"""
    # Read the data file
    collected_data = pd.read_csv(file_path, delimiter=';')
    
    # Find the index of the first occurrence of 'Initial_Location'
    initial_location_index = collected_data[collected_data['Type'] == 'Initial_Location'].index[0]
    
    # Slice the DataFrame from the first occurrence onwards
    data = collected_data.iloc[initial_location_index:].reset_index(drop=True)
    
    # Get initial position
    initial_location_data = data[data['Type'] == 'Initial_Location'].reset_index(drop=True)
    initial_position = (initial_location_data['value_4'][0], initial_location_data['value_5'][0])
    
    # Process ground truth data
    ground_truth_location_data = data[(data['Type'] == 'Ground_truth_Location') | 
                                     (data['Type'] == 'Initial_Location')].reset_index(drop=True)
    
    # Sort and deduplicate
    ground_truth_location_data.sort_values(by='step', inplace=True)
    ground_truth_location_data.drop_duplicates(subset='step', keep='last', inplace=True)
    ground_truth_location_data.reset_index(drop=True, inplace=True)
    
    # Compute azimuth between ground truth points
    df_gt = ground_truth_location_data.copy()
    df_gt["GroundTruthHeadingComputed"] = np.nan
    
    # Function to compute azimuth (bearing) between two coordinates
    def calculate_initial_compass_bearing(lat1, lon1, lat2, lon2):
        from math import atan2, degrees, radians, sin, cos
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        delta_lon = lon2 - lon1
        x = atan2(
            sin(delta_lon) * cos(lat2),
            cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(delta_lon)
        )
        
        bearing = (degrees(x) + 360) % 360
        return bearing
    
    # Calculate headings between ground truth points
    for i in range(1, len(df_gt)):
        df_gt.loc[i, "GroundTruthHeadingComputed"] = calculate_initial_compass_bearing(
            df_gt.loc[i-1, "value_5"], df_gt.loc[i-1, "value_4"],
            df_gt.loc[i, "value_5"], df_gt.loc[i, "value_4"]
        )
    
    # Sort data by timestamp and merge with ground truth headings
    data.sort_values(by="Timestamp_(ms)", inplace=True)
    df_gt.sort_values(by="Timestamp_(ms)", inplace=True)
    
    # Merge and fill ground truth headings
    data = data.merge(df_gt[["Timestamp_(ms)", "GroundTruthHeadingComputed"]], 
                     on="Timestamp_(ms)", how="left")
    data["GroundTruthHeadingComputed"] = data["GroundTruthHeadingComputed"].fillna(method="bfill")
    
    # Separate gyro and compass data
    gyro_data = data[data['Type'] == 'Gyro'].reset_index(drop=True)
    compass_data = data[data['Type'] == 'Compass'].reset_index(drop=True)
    
    # Rename columns for clarity
    compass_data.rename(columns={'value_1': 'Magnetic_Field_Magnitude'}, inplace=True)
    compass_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
    compass_data.rename(columns={'value_3': 'compass'}, inplace=True)
    
    gyro_data.rename(columns={'value_1': 'axisZAngle'}, inplace=True)
    gyro_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
    gyro_data.rename(columns={'value_3': 'compass'}, inplace=True)
    
    # Calculate gyro heading starting from ground truth
    first_ground_truth = initial_location_data['GroundTruth'][0]
    
    compass_data['GyroStartByGroundTruth'] = first_ground_truth + compass_data['gyroSumFromstart0'] - compass_data['gyroSumFromstart0'][0]
    compass_data['GyroStartByGroundTruth'] = (compass_data['GyroStartByGroundTruth'] + 360) % 360
    
    gyro_data['GyroStartByGroundTruth'] = first_ground_truth + gyro_data['gyroSumFromstart0'] - gyro_data['gyroSumFromstart0'][0]
    gyro_data['GyroStartByGroundTruth'] = (gyro_data['GyroStartByGroundTruth'] + 360) % 360
    
    return {
        'data': data,
        'compass_data': compass_data,
        'gyro_data': gyro_data,
        'ground_truth_data': df_gt,
        'initial_position': initial_position
    }


def main():
    """Main function to run adaptive quasi-static detection"""
    # Load sample data
    file_path = "1740735201447_CompassGyroSumHeadingData.TXT"
    
    try:
        # Check if file exists, otherwise create synthetic data
        if not os.path.exists(file_path):
            print(f"File {file_path} not found, creating synthetic data for demonstration...")
            # Create synthetic data with similar structure as the real data
            import numpy as np
            import pandas as pd
            
            # Number of data points
            n_points = 1000
            
            # Create timestamps
            timestamps = np.arange(0, n_points * 100, 100)
            
            # Create compass headings with some noise
            ground_truth_headings = np.zeros(n_points)
            # Add some turns
            ground_truth_headings[200:400] = 90
            ground_truth_headings[400:600] = 180
            ground_truth_headings[600:800] = 270
            ground_truth_headings[800:] = 0
            
            # Add noise to compass readings
            compass_headings = ground_truth_headings + np.random.normal(0, 10, n_points)
            compass_headings = compass_headings % 360
            
            # Create quasi-static periods (lower noise)
            is_static = np.zeros(n_points, dtype=bool)
            is_static[50:100] = True
            is_static[300:350] = True
            is_static[550:600] = True
            is_static[800:850] = True
            
            # Reduce noise in static periods
            compass_headings[is_static] = ground_truth_headings[is_static] + np.random.normal(0, 2, np.sum(is_static))
            
            # Create steps
            steps = np.floor(np.arange(0, n_points) / 5)  # 5 data points per step
            
            # Create gyro data (cumulative sum with drift)
            gyro_readings = np.zeros(n_points)
            gyro_cum = np.zeros(n_points)
            
            # Simulate gyro drift
            drift = np.cumsum(np.random.normal(0, 0.05, n_points))
            
            # Create compass data DataFrame
            compass_data = pd.DataFrame({
                'Timestamp_(ms)': timestamps,
                'Type': 'Compass',
                'step': steps,
                'compass': compass_headings,
                'GroundTruthHeadingComputed': ground_truth_headings,
                'Magnetic_Field_Magnitude': np.random.uniform(20, 50, n_points),
                'gyroSumFromstart0': drift,
                'GyroStartByGroundTruth': (ground_truth_headings[0] + drift) % 360,
                'value_4': steps * 0.1,  # Floor
                'GroundTruth_X': np.cumsum(np.sin(np.radians(ground_truth_headings)) * 0.66),  # East
                'GroundTruth_Y': np.cumsum(np.cos(np.radians(ground_truth_headings)) * 0.66)   # North
            })
            
            print("Synthetic data created successfully!")
            filtered_data_magnetic = compass_data
        else:
            data_dict = load_and_prepare_data(file_path)
            compass_data = data_dict['compass_data']
            
            print("Data loaded successfully!")
            print(f"Total data points: {len(compass_data)}")
            
            # Filter data for faster processing if needed
            filtered_data_magnetic = compass_data[(compass_data['step'] >= 0) & (compass_data['step'] <= 5000)]
        
        print("\n1. Evaluating default parameters...")
        # Default parameters from reference code
        default_stability = 5.0
        default_window = 300
        
        default_results = evaluate_quasi_static_parameters(
            filtered_data_magnetic, 
            default_stability, 
            default_window
        )
        
        print(f"Default parameters (Stability: {default_stability}, Window: {default_window}):")
        print(f"  Number of intervals: {default_results['num_intervals']}")
        print(f"  Average heading difference: {default_results['average_difference']:.2f} degrees")
        print(f"  MSE: {default_results['mse']:.2f}")
        
        print("\n2. Running genetic algorithm optimization...")
        ga_optimizer = GeneticAlgorithmOptimizer(
            filtered_data_magnetic, 
            ground_truth_data=filtered_data_magnetic,
            population_size=20,  # Smaller for demo
            generations=5  # Fewer for demo
        )
        
        ga_stability, ga_window = ga_optimizer.optimize()
        
        print("\n3. Evaluating GA-optimized parameters...")
        ga_results = evaluate_quasi_static_parameters(
            filtered_data_magnetic, 
            ga_stability, 
            ga_window
        )
        
        print(f"GA-optimized parameters (Stability: {ga_stability:.2f}, Window: {ga_window}):")
        print(f"  Number of intervals: {ga_results['num_intervals']}")
        print(f"  Average heading difference: {ga_results['average_difference']:.2f} degrees")
        print(f"  MSE: {ga_results['mse']:.2f}")
        
        print("\n4. Running reinforcement learning optimization...")
        rl_optimizer = ReinforcementLearningOptimizer(
            filtered_data_magnetic,
            ground_truth_data=filtered_data_magnetic
        )
        
        rl_stability, rl_window = rl_optimizer.train(
            episodes=3,  # Fewer for demo
            max_steps=1000
        )
        
        print("\n5. Evaluating RL-optimized parameters...")
        rl_results = evaluate_quasi_static_parameters(
            filtered_data_magnetic, 
            rl_stability, 
            rl_window
        )
        
        print(f"RL-optimized parameters (Stability: {rl_stability:.2f}, Window: {rl_window}):")
        print(f"  Number of intervals: {rl_results['num_intervals']}")
        print(f"  Average heading difference: {rl_results['average_difference']:.2f} degrees")
        print(f"  MSE: {rl_results['mse']:.2f}")
        
        # Compare all methods
        print("\n6. Comparison of all methods:")
        print("-" * 60)
        print(f"{'Method':<20} {'Stability':<10} {'Window':<10} {'Intervals':<10} {'Avg Diff':<10} {'MSE':<10}")
        print("-" * 60)
        print(f"{'Default':<20} {default_stability:<10.2f} {default_window:<10d} {default_results['num_intervals']:<10d} {default_results['average_difference']:<10.2f} {default_results['mse']:<10.2f}")
        print(f"{'Genetic Algorithm':<20} {ga_stability:<10.2f} {ga_window:<10d} {ga_results['num_intervals']:<10d} {ga_results['average_difference']:<10.2f} {ga_results['mse']:<10.2f}")
        print(f"{'Reinforcement Learning':<20} {rl_stability:<10.2f} {rl_window:<10d} {rl_results['num_intervals']:<10d} {rl_results['average_difference']:<10.2f} {rl_results['mse']:<10.2f}")
        print("-" * 60)
        
        # Determine the best method based on MSE
        best_mse = min(default_results['mse'], ga_results['mse'], rl_results['mse'])
        if best_mse == default_results['mse']:
            best_method = "Default"
            best_stability = default_stability
            best_window = default_window
        elif best_mse == ga_results['mse']:
            best_method = "Genetic Algorithm"
            best_stability = ga_stability
            best_window = ga_window
        else:
            best_method = "Reinforcement Learning"
            best_stability = rl_stability
            best_window = rl_window
        
        print(f"\nBest method: {best_method}")
        print(f"Best parameters: Stability={best_stability:.2f}, Window={best_window}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 