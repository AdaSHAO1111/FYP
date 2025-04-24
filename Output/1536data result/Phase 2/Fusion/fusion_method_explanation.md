# Neural Network-Based Sensor Fusion Method

## Overview

The script `train_improved_model_v3.py` implements a neural network-based sensor fusion approach to improve positioning accuracy by combining data from two primary sensors:

- 📱 **Gyroscope data** (measuring angular velocity)
- 🧭 **Compass/Magnetometer data** (measuring magnetic field and orientation)

This approach leverages machine learning to learn the optimal way to combine these sensor inputs for more accurate heading estimation, which in turn leads to better position tracking.

## Fusion Architecture

![Fusion Architecture](https://mermaid.ink/img/pako:eNptkU1PwzAMhv9K5BMSUksfHRzagQPaxdLQ7sAlcmO3C-R78pYyVf3v-IM0TLk5fp74tZ0jVFYj5HAnlkKDOXqM0dVjuWiWa-lkIw5WL-eNj6PW9V5bOHthIw42eAwRH9a5oRLF1o3C0aBUOLPbdmEeZvVhp-3yjOKzEyHYhZPmMUQnrB_JA3qmw5W9JaScUhZvudoItwpvlCJiO1Ufx1ftZ4KpEDfDaI1SmmN2O_TXD33CsT74YZrU9wdQ1t9M-AZnKD1V0VoWlJkc0dW8Gf4z5xTykE85NmIw82-j9BrMlZCvE6UxPSbJhdjx-Nrg80h-PemLMfrJGKSqoN4rVKqgHcllBQYbalyztqv4f6hZUtBoY4xrOqZYlXRVwF04zGpI3WPCv9OmUFVPwQcl9L1WkG9_AM8Ukto?type=png)

### Key Components:

1. **Input Layer**: 
   - Takes combined features from both gyroscope and compass sensors
   - Features include: axisZAngle, gyroSumFromstart0, compass (from gyro), and Magnetic_Field_Magnitude, gyroSumFromstart0, compass (from compass)

2. **Dense Neural Network Architecture**:
   - Input Layer: 6 features (3 from gyro, 3 from compass)
   - Hidden Layer 1: 128 neurons with ReLU activation and 30% dropout
   - Hidden Layer 2: 64 neurons with ReLU activation and 30% dropout
   - Hidden Layer 3: 32 neurons with ReLU activation
   - Output Layer: Single neuron predicting the heading angle

3. **Training Process**:
   - Loss Function: Mean Squared Error (MSE)
   - Optimizer: Adam with learning rate of 0.001
   - Early Stopping and Learning Rate Reduction to prevent overfitting
   - Ground Truth Heading as supervision signal

## How the Fusion Works

1. **Data Preprocessing**:
   - Gyroscope and compass readings are collected and synchronized by timestamp
   - Features are normalized using MinMaxScaler to bring all inputs to a similar scale
   - Data is split into training (80%) and validation (20%) sets

2. **Fusion Learning**:
   - The neural network learns to weight the importance of each sensor input
   - It implicitly learns when to trust gyroscope data more (e.g., during magnetic interference)
   - It learns when compass data is more reliable (e.g., when gyroscope drift accumulates)

3. **Heading Prediction**:
   - The trained model outputs a fused heading estimate
   - This heading is used with step detection to calculate position

## Performance Results

| Method      | Heading Error (°) | Position Error (m) | 
|-------------|------------------:|-------------------:|
| Traditional | Higher            | Higher             |
| Fusion      | Lower             | Lower              |

The fusion approach demonstrates significant improvements:
- **Heading Accuracy**: Reduces heading estimation errors
- **Position Accuracy**: Results in more accurate position tracking
- **Robustness**: Better handles the weaknesses of individual sensors

## Advantages of This Fusion Approach

1. **Data-Driven**: Learns optimal sensor fusion from actual data rather than using fixed rules
2. **Adaptive**: Can implicitly learn to handle various environmental conditions
3. **Lightweight**: Simple feed-forward neural network can run efficiently on mobile devices
4. **End-to-End**: Directly optimizes for heading accuracy
5. **Complementary**: Leverages strengths of both sensors while mitigating their individual weaknesses

## Conclusion

This neural network fusion approach improves indoor positioning accuracy by intelligently combining gyroscope and compass data. The model learns the complex relationships between sensor inputs and the ground truth heading, resulting in more reliable position tracking compared to using either sensor alone.

The simple feed-forward architecture makes this approach computationally efficient while still capturing the complementary nature of the sensors, making it suitable for real-time indoor positioning applications. 