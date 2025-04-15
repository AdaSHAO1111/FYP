"""
Extended Kalman Filter (EKF) implementation for nonlinear sensor fusion
Handles the nonlinearities in heading estimation especially with gyroscope and compass data
"""

import numpy as np
import matplotlib.pyplot as plt
import math

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) for nonlinear systems
    """
    
    def __init__(self, state_dim, measurement_dim, dt=0.1):
        """
        Initialize Extended Kalman Filter parameters
        
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
        
        # State vector x
        self.x = np.zeros((state_dim, 1))
        
        # State covariance matrix P
        self.P = np.eye(state_dim)
        
        # Process noise covariance matrix Q
        self.Q = np.eye(state_dim) * 0.01
        
        # Measurement noise covariance matrix R
        self.R = np.eye(measurement_dim) * 0.1
    
    def state_transition_function(self, x, u=None):
        """
        Nonlinear state transition function f(x, u)
        To be overridden by derived classes
        
        Parameters:
        -----------
        x : numpy.ndarray
            Current state vector
        u : numpy.ndarray
            Control input (optional)
            
        Returns:
        --------
        x_next : numpy.ndarray
            Next state vector
        """
        return x
    
    def measurement_function(self, x):
        """
        Nonlinear measurement function h(x)
        To be overridden by derived classes
        
        Parameters:
        -----------
        x : numpy.ndarray
            Current state vector
            
        Returns:
        --------
        z_pred : numpy.ndarray
            Predicted measurement vector
        """
        return np.zeros((self.measurement_dim, 1))
    
    def state_transition_jacobian(self, x, u=None):
        """
        Jacobian of the state transition function F = df/dx
        To be overridden by derived classes
        
        Parameters:
        -----------
        x : numpy.ndarray
            Current state vector
        u : numpy.ndarray
            Control input (optional)
            
        Returns:
        --------
        F : numpy.ndarray
            Jacobian matrix of state transition function
        """
        return np.eye(self.state_dim)
    
    def measurement_jacobian(self, x):
        """
        Jacobian of the measurement function H = dh/dx
        To be overridden by derived classes
        
        Parameters:
        -----------
        x : numpy.ndarray
            Current state vector
            
        Returns:
        --------
        H : numpy.ndarray
            Jacobian matrix of measurement function
        """
        return np.zeros((self.measurement_dim, self.state_dim))
    
    def predict(self, u=None):
        """
        EKF prediction step
        
        Parameters:
        -----------
        u : numpy.ndarray
            Control input (optional)
            
        Returns:
        --------
        x_pred : numpy.ndarray
            Predicted state vector
        """
        # State prediction using nonlinear state transition function
        self.x = self.state_transition_function(self.x, u)
        
        # Compute state transition Jacobian at current state
        F = self.state_transition_jacobian(self.x, u)
        
        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q
        
        return self.x
    
    def update(self, z):
        """
        EKF update step
        
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
        
        # Predict measurement using nonlinear measurement function
        z_pred = self.measurement_function(self.x)
        
        # Compute measurement Jacobian at current state
        H = self.measurement_jacobian(self.x)
        
        # Innovation (measurement residual)
        y = z - z_pred
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update (using Joseph form for numerical stability)
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        
        return self.x


class HeadingEKF(ExtendedKalmanFilter):
    """
    Extended Kalman Filter for heading estimation with gyroscope and compass
    
    State vector: [heading, gyro_bias]
    - heading: The estimated heading angle in degrees (0-360)
    - gyro_bias: The estimated gyroscope bias in degrees/second
    
    Measurement: compass heading in degrees (0-360)
    """
    
    def __init__(self, initial_heading=0.0, initial_bias=0.0, dt=0.1):
        """
        Initialize the Heading EKF
        
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
        self.x = np.array([[initial_heading], [initial_bias]])
        
        # Initial estimate uncertainty
        self.P = np.array([[100.0, 0.0], [0.0, 1.0]])
        
        # Process noise (adjusted for heading and bias)
        # Higher values mean more uncertainty in the model
        self.Q = np.array([[2.0, 0.0], [0.0, 0.02]])
        
        # Measurement noise (how much we trust the compass)
        # Higher value means less trust in compass
        self.R = np.array([[5.0]])
        
        # Store gyro measurement for state transition
        self.gyro_rate = 0.0
        
        # Store the time step
        self.dt = dt
    
    def set_gyro_rate(self, gyro_rate):
        """Set the current gyroscope angular rate for state transition"""
        self.gyro_rate = gyro_rate
    
    def state_transition_function(self, x, u=None):
        """
        Nonlinear state transition function
        
        heading_next = heading + (gyro_rate - bias) * dt
        bias_next = bias
        
        Parameters:
        -----------
        x : numpy.ndarray
            Current state [heading, bias]
        u : numpy.ndarray
            Control input (not used here)
            
        Returns:
        --------
        x_next : numpy.ndarray
            Next state [heading_next, bias_next]
        """
        heading = x[0, 0]
        bias = x[1, 0]
        
        # Update heading using gyro rate (corrected by estimated bias)
        heading_next = heading + (self.gyro_rate - bias) * self.dt
        
        # Normalize heading to [0, 360) degrees
        heading_next = heading_next % 360
        
        # Bias remains constant in the model
        bias_next = bias
        
        return np.array([[heading_next], [bias_next]])
    
    def measurement_function(self, x):
        """
        Nonlinear measurement function
        The compass directly measures the heading
        
        Parameters:
        -----------
        x : numpy.ndarray
            Current state [heading, bias]
            
        Returns:
        --------
        z_pred : numpy.ndarray
            Predicted measurement [compass_heading]
        """
        heading = x[0, 0]
        
        # Compass directly measures heading
        compass_heading = heading
        
        return np.array([[compass_heading]])
    
    def state_transition_jacobian(self, x, u=None):
        """
        Jacobian of state transition function
        
        Parameters:
        -----------
        x : numpy.ndarray
            Current state [heading, bias]
        u : numpy.ndarray
            Control input (not used here)
            
        Returns:
        --------
        F : numpy.ndarray
            Jacobian matrix of state transition function
        """
        # F = [[df1/dx1, df1/dx2], [df2/dx1, df2/dx2]]
        # where f1 = heading_next, f2 = bias_next
        # x1 = heading, x2 = bias
        
        # df1/dx1 = 1 (heading depends on previous heading)
        # df1/dx2 = -dt (heading depends negatively on bias)
        # df2/dx1 = 0 (bias doesn't depend on heading)
        # df2/dx2 = 1 (bias depends on previous bias)
        
        return np.array([[1.0, -self.dt], [0.0, 1.0]])
    
    def measurement_jacobian(self, x):
        """
        Jacobian of measurement function
        
        Parameters:
        -----------
        x : numpy.ndarray
            Current state [heading, bias]
            
        Returns:
        --------
        H : numpy.ndarray
            Jacobian matrix of measurement function
        """
        # H = [[dh1/dx1, dh1/dx2]]
        # where h1 = compass_heading
        # x1 = heading, x2 = bias
        
        # dh1/dx1 = 1 (compass heading depends directly on state heading)
        # dh1/dx2 = 0 (compass heading doesn't depend on bias)
        
        return np.array([[1.0, 0.0]])
    
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
        # Store gyro rate for state transition
        self.set_gyro_rate(gyro_rate)
        
        # Predict next state
        self.predict()
        
        return self.get_heading()
    
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
        
        # Call the standard EKF update
        z = np.array([[compass_heading]])
        self.update(z)
        
        # Normalize heading to [0, 360) degrees
        self.x[0, 0] = self.x[0, 0] % 360
        
        return self.get_heading()
    
    def get_heading(self):
        """Return the current heading estimate in degrees"""
        return self.x[0, 0]
    
    def get_bias(self):
        """Return the current gyro bias estimate in degrees/second"""
        return self.x[1, 0]


# More advanced EKF for a full 3D orientation (for future extension)
class OrientationEKF(ExtendedKalmanFilter):
    """
    Extended Kalman Filter for full 3D orientation estimation
    Using quaternions for attitude representation
    
    State vector: [q0, q1, q2, q3, bx, by, bz]
    - q0, q1, q2, q3: Quaternion representing 3D orientation
    - bx, by, bz: Gyroscope bias in x, y, z axes
    
    Measurements: Accelerometer and magnetometer readings
    """
    
    def __init__(self, dt=0.01):
        """
        Initialize the Orientation EKF
        
        Parameters:
        -----------
        dt : float
            Time step between measurements (in seconds)
        """
        # Call parent constructor
        # State is 7-dimensional: 4 for quaternion, 3 for gyro bias
        # Measurement is 6-dimensional: 3 for accelerometer, 3 for magnetometer
        super().__init__(state_dim=7, measurement_dim=6, dt=dt)
        
        # Initial state: Identity quaternion [1,0,0,0] and zero bias
        self.x = np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        
        # Initial estimate uncertainty
        self.P = np.eye(7) * 0.1
        self.P[4:7, 4:7] *= 0.01  # Lower initial uncertainty for bias
        
        # Process noise
        self.Q = np.eye(7) * 0.001
        self.Q[0:4, 0:4] *= 0.01  # Lower process noise for quaternion
        
        # Measurement noise
        self.R = np.eye(6) * 0.1
        self.R[0:3, 0:3] *= 0.2  # Higher noise for accelerometer
        self.R[3:6, 3:6] *= 0.1  # Lower noise for magnetometer
        
        # Store gyro measurements
        self.gyro = np.zeros(3)
        
    def set_gyro(self, gyro_x, gyro_y, gyro_z):
        """Set the current gyroscope angular rates"""
        self.gyro = np.array([gyro_x, gyro_y, gyro_z])
    
    def quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions
        
        Parameters:
        -----------
        q1, q2 : numpy.ndarray
            Quaternions to multiply
            
        Returns:
        --------
        q : numpy.ndarray
            Resulting quaternion
        """
        w1, x1, y1, z1 = q1.flatten()
        w2, x2, y2, z2 = q2.flatten()
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ]).reshape(-1, 1)
    
    def quaternion_from_angular_velocity(self, w, dt):
        """
        Create a quaternion representing rotation during time dt with angular velocity w
        
        Parameters:
        -----------
        w : numpy.ndarray
            Angular velocity vector [wx, wy, wz]
        dt : float
            Time step
            
        Returns:
        --------
        q : numpy.ndarray
            Quaternion representing the rotation
        """
        wx, wy, wz = w
        theta = np.linalg.norm(w) * dt
        
        if theta < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
        
        vec = np.array([wx, wy, wz]) / np.linalg.norm(w)
        
        return np.array([
            math.cos(theta/2),
            vec[0] * math.sin(theta/2),
            vec[1] * math.sin(theta/2),
            vec[2] * math.sin(theta/2)
        ]).reshape(-1, 1)
    
    def normalize_quaternion(self, q):
        """Normalize a quaternion to unit length"""
        mag = np.linalg.norm(q)
        return q / mag if mag > 0 else q
    
    def quaternion_to_euler(self, q):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw)
        
        Parameters:
        -----------
        q : numpy.ndarray
            Quaternion [w, x, y, z]
            
        Returns:
        --------
        euler : tuple
            Euler angles (roll, pitch, yaw) in degrees
        """
        q0, q1, q2, q3 = q.flatten()
        
        # Roll (rotation around X axis)
        roll = math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
        
        # Pitch (rotation around Y axis)
        sinp = 2 * (q0 * q2 - q3 * q1)
        pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
        
        # Yaw (rotation around Z axis)
        yaw = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
        
        # Convert to degrees
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        
        # Normalize yaw to [0, 360)
        yaw_deg = yaw_deg % 360
        
        return (roll_deg, pitch_deg, yaw_deg)
    
    def state_transition_function(self, x, u=None):
        """
        Nonlinear state transition function for orientation EKF
        
        Parameters:
        -----------
        x : numpy.ndarray
            Current state [q0, q1, q2, q3, bx, by, bz]
        u : numpy.ndarray
            Control input (not used here)
            
        Returns:
        --------
        x_next : numpy.ndarray
            Next state
        """
        # Extract state components
        q = x[0:4]  # Quaternion
        b = x[4:7]  # Bias
        
        # Correct gyro measurements with estimated bias
        corrected_gyro = self.gyro - b.flatten()
        
        # Create quaternion representing rotation during dt
        dq = self.quaternion_from_angular_velocity(corrected_gyro, self.dt)
        
        # Update quaternion by multiplying with rotation quaternion
        q_next = self.quaternion_multiply(q, dq)
        
        # Normalize quaternion
        q_next = self.normalize_quaternion(q_next)
        
        # Bias remains constant in the model
        b_next = b
        
        # Combine updated quaternion and bias
        x_next = np.vstack([q_next, b_next])
        
        return x_next
    
    def measurement_function(self, x):
        """
        Nonlinear measurement function for orientation EKF
        
        Parameters:
        -----------
        x : numpy.ndarray
            Current state [q0, q1, q2, q3, bx, by, bz]
            
        Returns:
        --------
        z_pred : numpy.ndarray
            Predicted measurement [ax, ay, az, mx, my, mz]
        """
        # Implement the complex measurement function
        # This would transform the quaternion to predict accelerometer
        # and magnetometer readings based on gravity and Earth's magnetic field
        
        # For now, just return a placeholder
        return np.zeros((self.measurement_dim, 1))
    
    def get_heading(self):
        """Return the current heading (yaw) in degrees"""
        q = self.x[0:4]
        _, _, yaw = self.quaternion_to_euler(q)
        return yaw


# Example usage
if __name__ == "__main__":
    # Simple test for HeadingEKF
    np.random.seed(42)  # For reproducibility
    
    # Simulation parameters
    dt = 0.1  # 10 Hz
    sim_time = 30  # 30 seconds
    num_steps = int(sim_time / dt)
    
    # True values (ground truth)
    true_headings = np.zeros(num_steps)
    true_bias = 1.0  # 1.0 deg/s bias
    
    # Initial heading is 0, then varies
    for i in range(1, num_steps):
        # Simulate a varying heading rate
        rate = 5 * math.sin(i * dt / 2)
        true_headings[i] = true_headings[i-1] + rate * dt
    
    # Normalize headings to [0, 360)
    true_headings = true_headings % 360
    
    # Simulate sensor readings
    gyro_noise_std = 0.5  # deg/s
    compass_noise_std = 8.0  # deg
    
    # Generate true gyro rates from heading changes
    true_gyro_rates = np.zeros(num_steps)
    for i in range(1, num_steps):
        # Calculate rate from heading change
        diff = true_headings[i] - true_headings[i-1]
        # Handle wrap-around
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        true_gyro_rates[i] = diff / dt
    
    # Add bias and noise to gyro readings
    gyro_readings = true_gyro_rates + true_bias + np.random.normal(0, gyro_noise_std, num_steps)
    
    # Add noise to compass readings
    compass_readings = true_headings + np.random.normal(0, compass_noise_std, num_steps)
    
    # Normalize compass readings to [0, 360)
    compass_readings = compass_readings % 360
    
    # Initialize EKF
    ekf = HeadingEKF(initial_heading=0.0, initial_bias=0.0, dt=dt)
    
    # Adjust filter parameters
    ekf.Q = np.array([[2.0, 0.0], [0.0, 0.1]])  # Process noise
    ekf.R = np.array([[10.0]])  # Measurement noise
    
    # Run filter
    ekf_headings = np.zeros(num_steps)
    ekf_biases = np.zeros(num_steps)
    
    for i in range(num_steps):
        # First update with gyro
        ekf.update_with_gyro(gyro_readings[i])
        
        # Then update with compass
        ekf.update_with_compass(compass_readings[i])
        
        # Store results
        ekf_headings[i] = ekf.get_heading()
        ekf_biases[i] = ekf.get_bias()
    
    # Calculate error metrics
    heading_rmse = np.sqrt(np.mean((ekf_headings - true_headings) ** 2))
    compass_rmse = np.sqrt(np.mean((compass_readings - true_headings) ** 2))
    
    # Output results
    print(f"Heading RMSE: {heading_rmse:.2f} degrees")
    print(f"Compass RMSE: {compass_rmse:.2f} degrees")
    print(f"Final bias estimate: {ekf_biases[-1]:.2f} deg/s (true: {true_bias:.2f} deg/s)")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    # Plot headings
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(num_steps) * dt, true_headings, 'k-', label='True Heading')
    plt.plot(np.arange(num_steps) * dt, compass_readings, 'g.', alpha=0.3, label='Compass Readings')
    plt.plot(np.arange(num_steps) * dt, ekf_headings, 'r-', label='EKF Heading')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading (degrees)')
    plt.legend()
    plt.title('Heading Estimation with EKF')
    plt.grid(True)
    
    # Plot bias
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(num_steps) * dt, np.ones(num_steps) * true_bias, 'k-', label='True Bias')
    plt.plot(np.arange(num_steps) * dt, ekf_biases, 'r-', label='EKF Bias Estimate')
    plt.xlabel('Time (s)')
    plt.ylabel('Gyro Bias (deg/s)')
    plt.legend()
    plt.title('Gyro Bias Estimation with EKF')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ekf_heading_simulation.png')
    plt.close()
    
    print("Simulation results saved to 'ekf_heading_simulation.png'") 