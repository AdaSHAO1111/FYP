#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p /Users/shaoxinyi/Downloads/FYP2/src/Phase5/output

# Script path
SCRIPT_DIR="/Users/shaoxinyi/Downloads/FYP2/src/Phase5/scripts"

# Run the original scripts
echo "=== Running Original Analysis Scripts ==="
echo "Step 1: Running Gyro Heading Correction..."
python "${SCRIPT_DIR}/gyro_heading_correction.py"

echo "Step 2: Running Gyro Position Correction..."
python "${SCRIPT_DIR}/gyro_position_correction.py"

echo "Step 3: Running Traditional Position Comparison..."
python "${SCRIPT_DIR}/traditional_position_comparison.py"

echo "Step 4: Running Enhanced Trajectory Comparison..."
python "${SCRIPT_DIR}/enhanced_trajectory_comparison.py"

# Run the improved scripts
echo -e "\n=== Running Improved Analysis Scripts ==="
echo "Step 5: Running Enhanced QS Filtering..."
python "${SCRIPT_DIR}/enhanced_qs_filtering.py"

echo "Step 6: Running Improved Heading Correction..."
python "${SCRIPT_DIR}/improved_heading_correction.py"

echo "Step 7: Running Improved Position Correction..."
python "${SCRIPT_DIR}/improved_position_correction.py"

echo -e "\nAll scripts completed!"
echo "Output files are available in: /Users/shaoxinyi/Downloads/FYP2/src/Phase5/output" 