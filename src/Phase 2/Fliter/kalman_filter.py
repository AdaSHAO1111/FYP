"""
Kalman Filter implementation for sensor fusion
Primarily used for gyroscope and compass data fusion to get improved heading estimates
"""

import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    """
    A generic Kalman Filter implementation for sensor fusion
    """
    
    def __init__(self, state_dim, measurement_dim, dt=0.1):
        """
        Initialize Kalman Filter parameters
        
        Parameters:
        -----------
        state_dim : int
            Dimension of the state vector
        measurement_dim : int
            Dimension of the measurement vector
        dt : float
            Time step between measurements (in seconds)
        """
        # Dimensions
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.dt = dt
        
        # State vector [x1, x2, ..., xn]
        self.x = np.zeros((state_dim, 1))
        
        # State transition matrix
        self.F = np.eye(state_dim)
        
        # Measurement matrix
        self.H = np.zeros((measurement_dim, state_dim))
        
        # State covariance matrix
        self.P = np.eye(state_dim)
        
        # Process noise covariance matrix
        self.Q = np.eye(state_dim) * 0.01
        
        # Measurement noise covariance matrix
        self.R = np.eye(measurement_dim) * 0.1
    
    def predict(self):
        """
        Prediction step
        
        Returns:
        --------
        x_pred : numpy.ndarray
            Predicted state vector
        """
        # State prediction
        self.x = self.F @ self.x
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x
    
    def update(self, z):
        """
        Update step
        
        Parameters:
        -----------
        z : numpy.ndarray
            Measurement vector
            
        Returns:
        --------
        x_updated : numpy.ndarray
            Updated state vector
        """
        # Convert z to numpy array if it's not
        if not isinstance(z, np.ndarray):
            z = np.array(z).reshape(-1, 1)
        
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update (using Joseph form for numerical stability)
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        
        return self.x
    
    def set_state(self, x):
        """Set the current state vector"""
        self.x = np.array(x).reshape(-1, 1)
    
    def set_state_transition_matrix(self, F):
        """Set the state transition matrix"""
        self.F = F
    
    def set_measurement_matrix(self, H):
        """Set the measurement matrix"""
        self.H = H
    
    def set_process_noise(self, Q):
        """Set the process noise covariance matrix"""
        self.Q = Q
    
    def set_measurement_noise(self, R):
        """Set the measurement noise covariance matrix"""
        self.R = R
    
    def set_error_covariance(self, P):
        """Set the state error covariance matrix"""
        self.P = P


class HeadingKalmanFilter(KalmanFilter):
    """
    Specialized Kalman Filter for heading estimation using gyroscope and compass data
    
    State vector: [heading, gyro_bias]
    - heading: The estimated heading angle in degrees
    - gyro_bias: The estimated gyroscope bias in degrees/second
    
    Measurement: compass heading in degrees
    """
    
    def __init__(self, initial_heading=0.0, initial_bias=0.0, dt=0.1):
        """
        Initialize the Heading Kalman Filter
        
        Parameters:
        -----------
        initial_heading : float
            Initial heading estimate in degrees
        initial_bias : float
            Initial gyroscope bias estimate in degrees/second
        dt : float
            Time step between measurements (in seconds)
        """
        # Call parent constructor: state=[heading, bias], measurement=[compass_heading]
        super().__init__(state_dim=2, measurement_dim=1, dt=dt)
        
        # Initial state: [heading, gyro_bias]
        self.set_state([initial_heading, initial_bias])
        
        # State transition model: heading = heading + (gyro - bias) * dt
        # F = [[1, -dt], [0, 1]]
        self.F = np.array([[1, -dt], [0, 1]])
        
        # Measurement model: z = [1, 0] * [heading, bias]
        self.H = np.array([[1, 0]])
        
        # Initial estimate uncertainty
        self.P = np.array([[100.0, 0.0], [0.0, 1.0]])
        
        # Process noise (adjusted for heading and bias)
        # Q represents how much we trust our motion model
        # Higher values for heading component mean less trust in gyro
        self.Q = np.array([[1.0, 0.0], [0.0, 0.01]])
        
        # Measurement noise (how much we trust the compass)
        # Higher value means less trust in compass
        self.R = np.array([[10.0]])
        
        # Store the time step
        self.dt = dt
    
    def update_with_gyro(self, gyro_rate):
        """
        Update the state prediction based on gyroscope reading
        
        Parameters:
        -----------
        gyro_rate : float
            Gyroscope angular rate in degrees/second
            
        Returns:
        --------
        heading : float
            Predicted heading in degrees
        """
        # Update the state transition model for the current gyro reading
        # F = [[1, -dt], [0, 1]]
        self.F = np.array([[1, -self.dt], [0, 1]])
        
        # Add gyro measurement to state prediction
        gyro_effect = np.array([[gyro_rate * self.dt], [0]])
        
        # Predict next state
        self.x = self.F @ self.x + gyro_effect
        
        # Normalize heading to [0, 360)
        self.x[0, 0] = self.x[0, 0] % 360
        
        # Update error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[0, 0]
    
    def update_with_compass(self, compass_heading):
        """
        Update the state estimate based on compass reading
        
        Parameters:
        -----------
        compass_heading : float
            Compass heading in degrees
            
        Returns:
        --------
        heading : float
            Updated heading estimate in degrees
        """
        # Handle the 0/360 degree discontinuity
        # If the difference between compass and predicted heading is more than 180,
        # adjust compass reading by adding/subtracting 360
        heading_pred = self.x[0, 0]
        diff = compass_heading - heading_pred
        
        if diff > 180:
            compass_heading -= 360
        elif diff < -180:
            compass_heading += 360
        
        # Call the standard Kalman update
        z = np.array([[compass_heading]])
        self.update(z)
        
        # Normalize heading to [0, 360)
        self.x[0, 0] = self.x[0, 0] % 360
        
        return self.x[0, 0]
    
    def get_heading(self):
        """Return the current heading estimate"""
        return self.x[0, 0]
    
    def get_bias(self):
        """Return the current gyro bias estimate"""
        return self.x[1, 0]


# Usage example:
if __name__ == "__main__":
    # Create a simple test with simulated data
    np.random.seed(42)  # For reproducibility
    
    # Simulation parameters
    dt = 0.1  # 10 Hz
    sim_time = 60  # 60 seconds
    num_steps = int(sim_time / dt)
    
    # True values (ground truth)
    true_headings = np.zeros(num_steps)
    true_bias = 0.5  # 0.5 deg/s bias
    
    # Initial heading is 0, increases at 6 deg/s
    for i in range(1, num_steps):
        true_headings[i] = true_headings[i-1] + 6 * dt
    
    # Simulate sensor readings
    gyro_noise_std = 0.2  # deg/s
    compass_noise_std = 5.0  # deg
    
    gyro_readings = np.zeros(num_steps)
    compass_readings = np.zeros(num_steps)
    
    for i in range(num_steps):
        # Gyro reads angular rate with bias and noise
        gyro_readings[i] = 6 + true_bias + np.random.normal(0, gyro_noise_std)
        
        # Compass reads absolute heading with noise
        compass_readings[i] = true_headings[i] + np.random.normal(0, compass_noise_std)
        
        # Normalize compass readings to [0, 360)
        compass_readings[i] = compass_readings[i] % 360
    
    # Initialize Kalman filter
    kf = HeadingKalmanFilter(initial_heading=0.0, initial_bias=0.0, dt=dt)
    
    # Run filter
    estimated_headings = np.zeros(num_steps)
    estimated_biases = np.zeros(num_steps)
    
    for i in range(num_steps):
        # First update with gyro measurement
        kf.update_with_gyro(gyro_readings[i])
        
        # Then update with compass measurement
        kf.update_with_compass(compass_readings[i])
        
        # Store the estimates
        estimated_headings[i] = kf.get_heading()
        estimated_biases[i] = kf.get_bias()
    
    # Calculate error metrics
    heading_rmse = np.sqrt(np.mean((estimated_headings - true_headings) ** 2))
    compass_rmse = np.sqrt(np.mean((compass_readings - true_headings) ** 2))
    
    print(f"Heading RMSE: {heading_rmse:.2f} degrees")
    print(f"Compass RMSE: {compass_rmse:.2f} degrees")
    print(f"Final bias estimate: {estimated_biases[-1]:.2f} deg/s (true: {true_bias:.2f} deg/s)")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    # Plot headings
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(num_steps) * dt, true_headings, 'k-', label='True Heading')
    plt.plot(np.arange(num_steps) * dt, compass_readings, 'g.', alpha=0.3, label='Compass Readings')
    plt.plot(np.arange(num_steps) * dt, estimated_headings, 'r-', label='Kalman Filter')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading (degrees)')
    plt.legend()
    plt.title('Heading Estimation')
    plt.grid(True)
    
    # Plot bias estimate
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(num_steps) * dt, np.ones(num_steps) * true_bias, 'k-', label='True Bias')
    plt.plot(np.arange(num_steps) * dt, estimated_biases, 'r-', label='Estimated Bias')
    plt.xlabel('Time (s)')
    plt.ylabel('Gyro Bias (deg/s)')
    plt.legend()
    plt.title('Gyro Bias Estimation')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('kalman_filter_simulation.png')
    plt.close()
    
    print("Simulation results saved to 'kalman_filter_simulation.png'") 