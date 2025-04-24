# Finalized LSTM Fusion Model

## Performance Summary

This LSTM model fuses corrected compass and gyro headings to produce significantly improved trajectory estimation:

- Mean Error: 3.55 meters
- Improvement over Compass: 43.37%
- Improvement over Gyro: 81.10%

## Model Architecture

- Bidirectional LSTM with 64+32 units
- Sequence length: 10 timesteps
- Input features: sine and cosine components of compass and gyro headings
- Custom angle difference loss function to handle circular data

## Usage

```python
from tensorflow.keras.models import load_model
import tensorflow as tf

# Custom loss function
def angle_difference(y_true, y_pred):
    # Extract sin and cos components
    sin_true, cos_true = y_true[:, 0], y_true[:, 1]
    sin_pred, cos_pred = y_pred[:, 0], y_pred[:, 1]
    
    # Calculate angular error using dot product and cross product
    dot_product = sin_true * sin_pred + cos_true * cos_pred
    cross_product = cos_true * sin_pred - sin_true * cos_pred
    
    # Return the angle between vectors
    return tf.reduce_mean(tf.abs(tf.atan2(cross_product, dot_product)))

# Load the model
model = load_model('lstm_heading_fusion_final.h5', custom_objects={'angle_difference': angle_difference})
```

## Files Included

- `lstm_heading_fusion_final.h5`: The trained model weights and architecture
- `model_metadata.csv`: Performance metrics and model information
- `model_architecture.txt`: Detailed model architecture
- `fused_trajectory_comparison.png`: Visualization of the fused trajectory compared to other methods
- `fused_trajectory_errors.csv`: Detailed error metrics

## Date Finalized

2025-04-19
