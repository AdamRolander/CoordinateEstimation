#!/bin/bash

# Configuration
N_SCENES=50
# ORIGINAL_SCRIPT="blender_single_scene.py"
ORIGINAL_SCRIPT="bounding_box.py"

BASE_OUTPUT_DIR="outputbb"

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

echo "Generating $N_SCENES scenes using $ORIGINAL_SCRIPT..."

for i in $(seq 0 $((N_SCENES-1))); do
    SCENE_ID=$(printf "%04d" $i)
    SCENE_OUTPUT_DIR="$BASE_OUTPUT_DIR/scene_$SCENE_ID"
    
    echo "Generating scene $SCENE_ID..."
    
    # Create scene-specific output directory
    mkdir -p "$SCENE_OUTPUT_DIR"
    
    # Run the original script with scene-specific output
    blenderproc run "$ORIGINAL_SCRIPT" "$SCENE_ID" "$SCENE_OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✓ Scene $SCENE_ID completed successfully"
    else
        echo "✗ Scene $SCENE_ID failed"
    fi
    
    # Small delay to ensure clean separation
    sleep 1
done

echo "All scenes generated! Check the $BASE_OUTPUT_DIR directory."