#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_synthetic_data():
    """
    Generate synthetic data for visualization
    """
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
        'timestamp': timestamps,
        'compass': compass,
        'gyro_sum': gyro_sum,
        'heading': heading,
        'is_quasi_static': is_quasi_static,
        'east': east,
        'north': north
    })
    
    return data

def visualize_results(data, output_dir="output"):
    """
    Visualize the navigation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot heading over time
    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'], data['heading'], label='Fused Heading')
    plt.plot(data['timestamp'], data['compass'], alpha=0.5, label='Compass')
    plt.plot(data['timestamp'], data['gyro_sum'], alpha=0.5, label='Gyro Sum')
    
    # Mark quasi-static periods
    qs_indices = data[data['is_quasi_static']].index
    if len(qs_indices) > 0:
        plt.scatter(data.loc[qs_indices, 'timestamp'], 
                   data.loc[qs_indices, 'heading'],
                   color='red', s=20, alpha=0.7, label='Quasi-Static Points')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Heading (degrees)')
    plt.title('Heading Estimation Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'heading_over_time.png'))
    plt.close()
    
    # 2. Plot trajectory in 2D
    plt.figure(figsize=(12, 12))
    plt.plot(data['east'], data['north'], 'b-', label='Estimated Path')
    plt.scatter(data['east'].iloc[0], data['north'].iloc[0], 
               color='green', s=100, label='Start')
    plt.scatter(data['east'].iloc[-1], data['north'].iloc[-1], 
               color='red', s=100, label='End')
    
    # Mark quasi-static points on the trajectory
    if len(qs_indices) > 0:
        plt.scatter(
            data.loc[qs_indices, 'east'], 
            data.loc[qs_indices, 'north'],
            color='orange', s=50, label='Quasi-Static Points'
        )
    
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('Estimated Trajectory')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'estimated_trajectory.png'))
    plt.close()
    
    # 3. Save CSV files for potential further analysis
    data[['timestamp', 'heading', 'compass', 'gyro_sum', 'is_quasi_static']].to_csv(
        os.path.join(output_dir, 'heading_history.csv'), index=False
    )
    
    data[['timestamp', 'east', 'north', 'heading']].to_csv(
        os.path.join(output_dir, 'position_history.csv'), index=False
    )
    
    print(f"Visualizations saved to {output_dir}")

def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_data()
    
    # Visualize results
    print("Generating visualizations...")
    visualize_results(data)
    
    print("Demo completed successfully!")

if __name__ == "__main__":
    main() 