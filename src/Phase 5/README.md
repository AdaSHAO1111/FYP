# Phase 6: LSTM-Based Sensor Fusion for Improved Trajectory Estimation

This phase implements an LSTM-based sensor fusion approach to combine the corrected compass and gyroscope heading data for more accurate trajectory estimation.

## Overview

The LSTM (Long Short-Term Memory) fusion model takes advantage of the temporal nature of walking data by learning the relationship between corrected sensor data and ground truth headings. By considering sequences of sensor readings rather than individual points, the model can better capture trends and patterns in sensor behavior.

## Key Features

1. **Data Preparation**:
   - Merges corrected compass and gyroscope trajectory data
   - Calculates ground truth headings from ground truth coordinates
   - Handles the circular nature of heading data using sine/cosine encoding
   - Creates sequence-based training data for the LSTM model

2. **LSTM Fusion Model**:
   - Bidirectional LSTM architecture for capturing temporal dependencies
   - Custom angle difference loss function to handle circular data
   - Learns to predict ground truth headings from corrected sensor inputs
   - Uses early stopping to prevent overfitting

3. **Trajectory Generation**:
   - Applies the trained model to generate fused heading predictions
   - Calculates positions from fused headings
   - Compares fused trajectory with original corrected trajectories
   - Quantifies improvement through error metrics

## Implementation Details

### Files:

- `prepare_fusion_data.py`: Prepares and aligns the data for LSTM training
- `lstm_heading_fusion.py`: Implements the LSTM fusion model and trajectory generation

### Methodology:

1. **Input Features**: 
   - Corrected compass heading (sin and cos components)
   - Corrected gyro heading (sin and cos components)
   - Sequence length of 10 timesteps

2. **Output Target**:
   - Ground truth heading (sin and cos components)

3. **Training Process**:
   - Split data into training, validation, and test sets
   - Train LSTM model with early stopping
   - Evaluate model on test data
   - Generate fused trajectory and calculate error metrics

## Results

The model outputs:
- Error statistics for the original and fused trajectories
- Visualizations of the trajectories
- Performance metrics showing improvement over individual sensors

## Usage

To run the LSTM fusion model:

```
cd /Users/shaoxinyi/Downloads/FYP2/src/Phase\ 6/
python lstm_heading_fusion.py
```

This will:
1. Prepare the fusion dataset
2. Train the LSTM model
3. Generate and visualize the fused trajectory
4. Calculate error metrics and improvement percentages

## Technical Notes

- The model handles the circular nature of heading data by using sine/cosine encoding
- Ground truth headings are derived from interpolated positions
- The LSTM model captures temporal dependencies in the sensor readings
- Error reduction percentages quantify the improvement over individual corrected sensors 