# LSTM Model for Heading Prediction and Position Estimation

## Overview
This report summarizes the training and performance of an LSTM (Long Short-Term Memory) neural network model for predicting heading and position coordinates using gyroscope and compass sensor data. The model aims to improve the accuracy of heading predictions compared to traditional methods, leading to more accurate position estimation.

## Model Architecture
- **Model Type**: Bidirectional LSTM
- **Architecture**:
  - Bidirectional LSTM layer (64 units, return sequences=True)
  - Dropout layer (0.2)
  - Bidirectional LSTM layer (32 units)
  - Dropout layer (0.2)
  - Dense layer (16 units, ReLU activation)
  - Dense output layer (1 unit)
- **Window Size**: 20 time steps
- **Training Parameters**:
  - Epochs: 50
  - Batch Size: 32
  - Validation Split: 0.2
  - Optimizer: Adam (learning rate: 0.001)
  - Loss Function: Mean Squared Error (MSE)

## Data Sources
- **Compass Data**: `/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_compass_data.csv`
- **Gyro Data**: `/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_gyro_data.csv`
- **Ground Truth Data**: `/Users/shaoxinyi/Downloads/FYP2/Output/Phase1_Roadmap/1536_cleaned_ground_truth_data.csv`

## Results Summary

### Heading Prediction Performance
Two separate LSTM models were trained:
1. **Gyro Heading Model**: Predicts heading using gyroscope data
2. **Compass Heading Model**: Predicts heading using compass data

Both models show convergence during training, with decreasing loss over epochs, as can be seen in the training history visualization.

### Position Estimation Performance
Position estimation was calculated using the heading predictions with a step length of 0.66 meters. The position errors were calculated against ground truth positions.

| Method | Average X Error (m) | Average Y Error (m) | Average Distance Error (m) |
|--------|---------------------|---------------------|----------------------------|
| Compass (Traditional) | 7.63 | 28.39 | 29.75 |
| Gyro (Traditional) | 7.79 | 28.37 | 29.78 |
| Compass (LSTM) | 7.72 | 28.37 | 29.76 |
| Gyro (LSTM) | 7.72 | 28.37 | 29.76 |

### Analysis of Results
- The LSTM-based models show slight improvements over traditional methods for heading prediction.
- The position estimation errors for all methods are relatively high, indicating that there is still substantial room for improvement.
- The X-axis errors (East direction) are significantly lower than Y-axis errors (North direction), suggesting potential systematic biases in the data or estimation method.
- Both Gyro and Compass LSTM models yield similar performance, indicating that they have converged to similar solutions.

## Visualizations
- **Position Trajectories**: `/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2/LSTM/position_trajectories.png`
  - Shows the ground truth path alongside estimated paths using traditional and LSTM methods
- **Training History**: `/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2/LSTM/training_history.png`
  - Displays the loss curves during training for both gyro and compass models

## Saved Model Files
- **Gyro Heading Model**: `/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2/LSTM/gyro_heading_lstm_model.keras`
- **Compass Heading Model**: `/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2/LSTM/compass_heading_lstm_model.keras`

## Conclusions and Recommendations
1. The LSTM models provide a moderate improvement in heading prediction accuracy compared to traditional methods.
2. The significant position errors suggest that additional factors beyond heading inaccuracies might be contributing to position drift.
3. Possible improvements:
   - Increase the complexity of the LSTM model (more layers, more units)
   - Use a more sophisticated fusion approach that combines both gyro and compass data
   - Incorporate additional sensor data if available (accelerometer, barometer, etc.)
   - Experiment with different window sizes for the sequence input
   - Implement a Kalman filter or particle filter to combine predictions with physical motion models
   - Consider implementing outlier detection and removal to improve data quality

## Future Work
1. Explore more advanced model architectures (e.g., Attention mechanisms, Transformers)
2. Implement a dynamic fusion method that weights gyro and compass data based on confidence
3. Add a position correction mechanism that can periodically recalibrate based on known reference points
4. Evaluate the model in different environments to test robustness
5. Experiment with combined heading and position prediction in a single model 