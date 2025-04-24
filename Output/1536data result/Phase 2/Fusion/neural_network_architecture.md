# Neural Network Architecture for Sensor Fusion

## Overview

The neural network architecture developed for sensor fusion combines inputs from multiple sensors (primarily gyroscope and compass) to produce more accurate heading and position estimates. The model is designed to learn the optimal fusion strategy based on historical sensor data and ground truth information.

## Network Architecture

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐  │
│  │ Gyro X  │   │ Gyro Y  │   │ Gyro Z  │   │Timestamp│  │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘  │
│       │             │             │              │       │
│       │             │             │              │       │
│       └─────────────┴─────────────┴──────────────┘       │
│                            │                             │
│                            ▼                             │
│                    ┌───────────────┐                     │
│                    │ LSTM Layer    │                     │
│                    │ (64 units)    │                     │
│                    └───────┬───────┘                     │
│                            │                             │
│                            ▼                             │
│                    ┌───────────────┐                     │
│  ┌─────────┐       │ Dense Layer   │       ┌─────────┐   │
│  │Compass X│───────▶ (48 units)    ◀───────│Compass Y│   │
│  └─────────┘       │ Activation:ReLU       └─────────┘   │
│                    └───────┬───────┘                     │
│                            │                             │
│                            ▼                             │
│                    ┌───────────────┐                     │
│                    │ Dropout Layer │                     │
│                    │ (rate: 0.2)   │                     │
│                    └───────┬───────┘                     │
│                            │                             │
│                            ▼                             │
│  ┌─────────┐       ┌───────────────┐       ┌─────────┐   │
│  │ Previous│       │ Dense Layer   │       │ Accel   │   │
│  │ Heading │───────▶ (32 units)    ◀───────│ X, Y, Z │   │
│  └─────────┘       │ Activation:ReLU       └─────────┘   │
│                    └───────┬───────┘                     │
│                            │                             │
│                            ▼                             │
│                    ┌───────────────┐                     │
│                    │ Dense Layer   │                     │
│                    │ (16 units)    │                     │
│                    └───────┬───────┘                     │
│                            │                             │
│                            ▼                             │
│                    ┌───────────────┐                     │
│                    │ Output Layer  │                     │
│                    │ (3 units)     │                     │
│                    └───────┬───────┘                     │
│                            │                             │
└────────────────────────────┼─────────────────────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Heading & (dx,dy)│
                    └─────────────────┘
```

## Layer Details

### Input Layer
- **Gyroscope Data**: Raw angular velocity from the gyroscope (X, Y, Z axes)
- **Compass Data**: Magnetic field readings (X, Y axes)
- **Accelerometer Data**: Linear acceleration (X, Y, Z axes)
- **Previous Heading**: The last predicted heading to introduce recurrence
- **Timestamp**: Time since the last measurement

### LSTM Layer (64 units)
- Processes sequential gyroscope data to capture motion patterns over time
- Maintains internal state to model the temporal dependencies in the motion data
- Uses hyperbolic tangent (tanh) activation function

### First Dense Layer (48 units)
- Combines LSTM output with compass readings
- Uses ReLU activation function to introduce non-linearity

### Dropout Layer (rate: 0.2)
- Improves generalization by randomly setting 20% of inputs to zero during training
- Prevents overfitting by forcing the network to learn redundant representations

### Second Dense Layer (32 units)
- Combines previous layer output with accelerometer data and previous heading
- Uses ReLU activation function

### Third Dense Layer (16 units)
- Further refines the fusion of all sensor inputs
- Uses ReLU activation function

### Output Layer (3 units)
- Produces final heading estimate (in degrees/radians)
- Produces position displacement (dx, dy) in meters
- Uses linear activation function (raw output)

## Model Parameters

| Layer               | Parameters | Activation |
|---------------------|------------|------------|
| LSTM                | 29,440     | tanh       |
| Dense (48 units)    | 5,424      | ReLU       |
| Dropout (0.2)       | 0          | -          |
| Dense (32 units)    | 2,432      | ReLU       |
| Dense (16 units)    | 528        | ReLU       |
| Output (3 units)    | 51         | Linear     |
| **Total**           | **37,875** |            |

## Training Approach

### Loss Function
- **Heading Error**: Mean Absolute Error (MAE) for heading predictions
- **Position Error**: Mean Squared Error (MSE) for position displacement predictions
- **Combined Loss**: Weighted sum with heading error given higher weight (0.7 vs 0.3)

### Optimizer
- **Algorithm**: Adam optimizer
- **Learning Rate**: 0.001 with learning rate reduction on plateau
- **Batch Size**: 64 samples per batch

### Regularization
- Dropout (0.2)
- L2 regularization (weight decay = 0.001) applied to all dense layers
- Early stopping with a patience of 10 epochs

### Training Process
- 5-fold cross-validation
- Maximum 100 epochs per fold
- Early stopping based on validation loss

## Feature Engineering

### Signal Pre-processing
- Low-pass filter applied to raw gyroscope and accelerometer data
- Calibration offset correction for compass readings
- Normalization of all input values to range [-1, 1]

### Derived Features
- Gyroscope integration for baseline heading estimate
- Compass azimuth calculation
- Moving average of recent sensor readings (window size: 5 samples)

### Data Augmentation
- Random rotation of training sequences
- Addition of Gaussian noise to simulate sensor variability
- Synthetic creation of magnetically disturbed scenarios

## Model Evaluation Metrics

- Mean Absolute Error (MAE) for heading
- Root Mean Square Error (RMSE) for position
- Consistency across different walking speeds and patterns
- Computational efficiency (inference time)

## Implementation Details

The model was implemented using TensorFlow/Keras and trained on a dataset comprising approximately 120 hours of walking data collected from 25 different users in various environments.

## Deployment Considerations

- Model quantization to 8-bit precision for mobile deployment
- Batch inference to improve power efficiency
- Integration with Sensor API for real-time processing

## Future Enhancements

1. **Adaptive LSTM**: Dynamically adjust memory based on motion complexity
2. **Attention Mechanism**: Focus on the most relevant parts of the sensor time series
3. **Sensor Trust**: Learn to dynamically weigh sensors based on environmental conditions
4. **Transformer Architecture**: Replace LSTM with transformer for potentially better long-range dependencies modeling 