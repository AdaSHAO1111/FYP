# Adaptive Quasi-Static Detection for Indoor Navigation

This repository contains implementations for Phase 4 of the indoor navigation system roadmap, focusing on adaptive quasi-static detection algorithms. The goal is to improve heading estimation accuracy by optimizing the detection of quasi-static periods, where the device is relatively stationary, allowing for accurate compass calibration.

## Implemented Approaches

### 1. Genetic Algorithm Optimization

The genetic algorithm approach (`adaptive_quasi_static_detection.py`) optimizes the parameters for quasi-static detection:

- **Stability Threshold**: The variance threshold below which a period is considered quasi-static
- **Window Size**: The number of samples to consider when calculating variance

The genetic algorithm evolves a population of parameter sets through selection, crossover, and mutation to find the optimal configuration that maximizes heading accuracy.

Features:
- Population-based search of the parameter space
- Fitness function based on number of intervals, variance, and heading accuracy
- Multiple generations of evolution to refine parameters

### 2. Reinforcement Learning Optimization

The reinforcement learning approach (`adaptive_quasi_static_detection.py`) uses Q-learning to dynamically adjust the quasi-static detection parameters:

- Parameters are adjusted in real-time based on feedback
- The agent learns to make decisions that maximize heading accuracy
- Exploration-exploitation strategy for parameter space search

Features:
- State space based on current variance and recent interval count
- Action space for adjusting stability threshold and window size
- Reward function based on variance and heading accuracy

### 3. CNN-Based Classification

The CNN-based approach (`cnn_quasi_static_classifier.py`) uses a deep learning model to directly classify quasi-static states:

- Learns patterns from sensor data (compass, gyroscope, context)
- Directly predicts whether the current state is quasi-static
- Doesn't require manual parameter tuning

Features:
- 1D convolutional neural network architecture
- Multi-sensor fusion (compass and gyroscope)
- Context-aware with additional features like step and floor information

## Performance Comparison

Our evaluation shows the following results:

| Method                    | Advantages                                      | Limitations                                     |
|---------------------------|------------------------------------------------|------------------------------------------------|
| Default Parameters        | Simple, predictable behavior                    | Not adaptive to different environments          |
| Genetic Algorithm         | Good parameter optimization, high accuracy      | Requires offline training                       |
| Reinforcement Learning    | Real-time adaptation                            | Training can be unstable                        |
| CNN Classification        | Learns complex patterns, no parameter tuning    | Requires large training dataset                 |

## Usage

### Genetic Algorithm and Reinforcement Learning

```bash
python adaptive_quasi_static_detection.py
```

This will:
1. Load or generate sensor data
2. Evaluate default parameters
3. Run genetic algorithm optimization
4. Run reinforcement learning training
5. Compare all methods and determine the best one

### CNN-Based Classification

```bash
python cnn_quasi_static_classifier.py
```

This will:
1. Generate synthetic training data
2. Train a CNN model for quasi-static state classification
3. Evaluate the model on test data
4. Compare with traditional variance-based detection
5. Visualize the results

## Integration with Navigation System

To integrate the adaptive quasi-static detection into the main navigation system:

1. Import the selected detector class from the appropriate module
2. Initialize the detector with the optimized parameters
3. Call the detector's methods during navigation to identify quasi-static periods
4. Use these periods for compass calibration and heading correction

Example:

```python
from adaptive_quasi_static_detection import QuasiStaticDetector

# Use optimized parameters
detector = QuasiStaticDetector(stability_threshold=2.17, window_size=9)

# Process sensor data
for heading in compass_headings:
    detector.add_compass_heading(heading)
    is_quasi_static = detector.is_quasi_static_interval()
    
    if is_quasi_static:
        # Use this period for calibration
        mean_heading = detector.calculate_mean()
        # Update compass calibration
```

## Future Work

1. Implement federated learning for collaborative model training
2. Develop a hybrid approach that combines multiple detection methods
3. Add environmental context awareness for better adaptation
4. Implement transfer learning for new environments 