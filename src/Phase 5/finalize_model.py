import os
import shutil
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from lstm_heading_fusion import angle_difference

def finalize_lstm_model():
    """
    Finalize and save the best LSTM model with its results for future use.
    This ensures the model with good results (43.37% improvement over compass, 
    81.10% improvement over gyro) is preserved.
    """
    print("\n=== Finalizing LSTM Fusion Model ===\n")
    
    # Paths
    source_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 6/LSTM_fusion'
    model_dir = os.path.join(source_dir, 'models')
    finalized_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 6/Finalized_Model'
    os.makedirs(finalized_dir, exist_ok=True)
    
    # Copy model file
    model_path = os.path.join(model_dir, 'lstm_heading_fusion.h5')
    if not os.path.exists(model_path):
        print("Error: Model file not found. Please run lstm_heading_fusion.py first.")
        return False
    
    # Copy model to finalized directory
    shutil.copy2(model_path, os.path.join(finalized_dir, 'lstm_heading_fusion_final.h5'))
    
    # Copy results
    result_files = [
        'fused_trajectory_errors.csv',
        'fused_trajectory_comparison.png',
        'model_evaluation.csv',
        'heading_comparison.png',
        'error_distribution.png',
        'prediction_samples.png',
        'training_history.png'
    ]
    
    for file in result_files:
        source_file = os.path.join(source_dir, file)
        if os.path.exists(source_file):
            shutil.copy2(source_file, os.path.join(finalized_dir, file))
    
    # Save metadata with performance metrics
    try:
        error_stats = pd.read_csv(os.path.join(source_dir, 'fused_trajectory_errors.csv'))
        fused_row = error_stats[error_stats['Method'] == 'LSTM Fused'].iloc[0]
        
        metadata = {
            'Mean_Error': [fused_row['Mean_Error']],
            'Median_Error': [fused_row['Median_Error']],
            'Max_Error': [fused_row['Max_Error']],
            'Improvement_over_Compass_%': [fused_row['Improvement_over_Compass_%']],
            'Improvement_over_Gyro_%': [fused_row['Improvement_over_Gyro_%']],
            'Model_Architecture': ['Bidirectional LSTM (64+32 units)'],
            'Sequence_Length': [10],
            'Input_Features': ['Compass_sin, Compass_cos, Gyro_sin, Gyro_cos'],
            'Training_Date': [pd.Timestamp.now().strftime('%Y-%m-%d')]
        }
        
        pd.DataFrame(metadata).to_csv(os.path.join(finalized_dir, 'model_metadata.csv'), index=False)
    except Exception as e:
        print(f"Warning: Could not save metadata: {str(e)}")
    
    # Test load the model to ensure it works
    try:
        model = load_model(os.path.join(finalized_dir, 'lstm_heading_fusion_final.h5'), 
                           custom_objects={'angle_difference': angle_difference})
        print("Model loaded successfully!")
        
        # Save model architecture summary
        with open(os.path.join(finalized_dir, 'model_architecture.txt'), 'w') as f:
            # Redirect summary to file
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    except Exception as e:
        print(f"Error: Failed to load model: {str(e)}")
        return False
    
    # Get values for README
    mean_error = fused_row['Mean_Error']
    compass_improvement = fused_row['Improvement_over_Compass_%'] 
    gyro_improvement = fused_row['Improvement_over_Gyro_%']
    date_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    # Create README file
    readme_content = f"""# Finalized LSTM Fusion Model

## Performance Summary

This LSTM model fuses corrected compass and gyro headings to produce significantly improved trajectory estimation:

- Mean Error: {mean_error:.2f} meters
- Improvement over Compass: {compass_improvement:.2f}%
- Improvement over Gyro: {gyro_improvement:.2f}%

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
model = load_model('lstm_heading_fusion_final.h5', custom_objects={{'angle_difference': angle_difference}})
```

## Files Included

- `lstm_heading_fusion_final.h5`: The trained model weights and architecture
- `model_metadata.csv`: Performance metrics and model information
- `model_architecture.txt`: Detailed model architecture
- `fused_trajectory_comparison.png`: Visualization of the fused trajectory compared to other methods
- `fused_trajectory_errors.csv`: Detailed error metrics

## Date Finalized

{date_str}
"""
    
    with open(os.path.join(finalized_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print(f"\nModel successfully finalized and saved to: {finalized_dir}")
    print(f"Improvements: {compass_improvement:.2f}% over compass, {gyro_improvement:.2f}% over gyro")
    
    return True

if __name__ == "__main__":
    finalize_lstm_model() 