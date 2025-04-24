# Gyroscope-Based Indoor Positioning System - Phase 5

This phase focuses on heading and position correction techniques using gyroscope data combined with ground truth reference points. The implementation includes various algorithms for trajectory improvement and detailed analyses of their performance.

## Project Structure

```
Phase5/
├── scripts/              # Python implementation files
│   ├── gyro_heading_correction.py           # Heading correction implementation
│   ├── gyro_position_correction.py          # Position correction using corrected heading
│   ├── traditional_position_comparison.py   # Compares traditional gyro positions
│   └── enhanced_trajectory_comparison.py    # Enhanced visualization and comparison
│
├── docs/                 # Documentation and analysis
│   ├── main_readme.md                      # Original Phase 5 README
│   ├── position_readme.md                  # Position correction overview
│   ├── heading_correction_improvements.md  # Analysis of heading correction improvements
│   ├── position_correction_summary.md      # Summary of position correction results
│   ├── traditional_vs_corrected_comparison.md # Comparison between approaches
│   └── enhanced_visualization_summary.md   # Summary of enhanced visualization
│
└── output/               # Generated output directory for results
    └── ...               # Output files will be saved here when scripts are run
```

## Main Components

1. **Heading Correction**: Implementation of algorithms to correct gyroscope heading drift using ground truth calibration.

2. **Position Correction**: Methods to generate improved position estimates using the corrected heading data.

3. **Traditional vs. Corrected Comparison**: Analysis of the performance differences between traditional gyroscope-based positioning and the corrected approach.

4. **Enhanced Visualization**: Advanced visualization techniques to better understand and analyze the positioning data.

## Usage Instructions

1. Run the heading correction script:
   ```bash
   python src/Phase5/scripts/gyro_heading_correction.py
   ```

2. Run the position correction script:
   ```bash
   python src/Phase5/scripts/gyro_position_correction.py
   ```

3. For additional analysis, run:
   ```bash
   python src/Phase5/scripts/traditional_position_comparison.py
   python src/Phase5/scripts/enhanced_trajectory_comparison.py
   ```

## Documentation

For detailed information on implementation approaches, results, and analysis, refer to the documents in the `docs/` directory.

## Performance Summary

- Heading correction achieves a 65.81% reduction in mean heading error
- Position correction achieves a 24.33% reduction in average position error
- Enhanced visualization provides detailed segment analysis and improved error measurement 