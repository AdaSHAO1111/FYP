# Ensemble Heading Prediction and Position Tracking Analysis

## Summary

This analysis examines the performance of ensemble methods that combine traditional sensor-based heading prediction with LSTM-based machine learning approaches for indoor positioning. We explored several weighting schemes between traditional and LSTM methods, including a novel adaptive weighting system based on sensor characteristics.

## Key Findings

### Position Error Comparison

| Method | Average Error (m) | Improvement over LSTM |
|--------|-------------------|------------------------|
| Compass (Traditional) | 16.76 | 0.25% |
| Compass (Ensemble 0.7) | 16.77 | 0.20% |
| Compass (Adaptive Ensemble) | 16.76 | 0.25% |
| Compass (LSTM) | 16.80 | - |
| Gyro (Traditional) | 16.76 | 5.04% |
| Gyro (Adaptive Ensemble) | 16.77 | 4.94% |
| Gyro (Ensemble 0.7) | 16.89 | 4.26% |
| Gyro (LSTM) | 17.64 | - |

### Analysis of Results

1. **Traditional methods performed best**: The traditional compass-based method achieved the lowest average position error (16.76m), slightly outperforming all ensemble approaches.

2. **Ensemble methods bridged the gap**: For gyro-based positioning, pure LSTM performed poorly (17.64m error), but ensemble methods significantly improved performance, with the adaptive ensemble (16.77m) nearly matching traditional methods.

3. **Higher traditional weights performed better**: For fixed-weight ensembles, higher weights for traditional methods (0.6-0.7) consistently outperformed lower weights (0.3-0.5). This suggests the traditional methods provide more stable spatial consistency.

4. **Adaptive ensemble showed promise**: The adaptive ensemble, which dynamically adjusted weights based on sensor characteristics (gyro rotation speed and magnetic field stability), performed nearly identically to the traditional method for compass (16.76m vs 16.76m) and significantly better than LSTM for gyro data (16.77m vs 17.64m).

5. **Heading improvement didn't fully translate to position accuracy**: While LSTM models showed significant improvements in heading prediction accuracy in previous analysis (especially for compass data, where MAE was reduced by 46.9%), these improvements didn't translate directly to better position tracking.

## Explanation of Position Tracking Difference

The discrepancy between improved heading accuracy and unchanged/worse position accuracy can be explained by:

1. **Temporal consistency**: Traditional methods produce more temporally consistent heading changes between consecutive steps, which is critical for position tracking. LSTM may produce individually accurate headings but less consistent transitions.

2. **Error accumulation**: Position tracking involves integrating heading information over time, causing small errors to accumulate. The LSTM approach might introduce bias that compounds during integration.

3. **Sampling window effect**: The sliding window approach (20 samples) for LSTM may introduce latency in heading prediction, causing slight timing misalignment with step detection.

4. **Feature selection limitations**: The current LSTM models may not fully capture the relationship between sensor data and optimal positioning information.

## Conclusion and Future Work

The ensemble methods, particularly the adaptive approach, show significant promise for improving heading prediction reliability while maintaining position tracking accuracy. While they didn't outperform traditional methods in this dataset, they successfully mitigated the weaknesses of pure LSTM approaches.

Future work could explore:

1. **Enhanced adaptive weighting schemes** based on more complex sensor reliability metrics

2. **End-to-end position prediction** models that optimize directly for position accuracy rather than heading accuracy

3. **Integration of additional sensors** (accelerometer, barometer) for more robust fusion

4. **Online calibration mechanisms** to adapt to environmental changes during tracking

5. **Larger datasets** with more diverse movement patterns to improve generalization

The current results demonstrate that combining traditional PDR methods with deep learning approaches through intelligent ensembling offers a promising direction for indoor positioning systems. 