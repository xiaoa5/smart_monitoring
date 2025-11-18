#!/usr/bin/env python3
"""
Quick Data Generation for Phase B Testing
==========================================

Generates a small dataset (50 frames) for quick testing of Phase B/C/D.
For full dataset, use path2_phase1_2_verified.py

Usage:
    python generate_test_data.py
"""

import sys
import os

# Add path if needed
if os.path.exists('path2_phase1_2_verified.py'):
    print("✓ Found path2_phase1_2_verified.py")
else:
    print("✗ Error: path2_phase1_2_verified.py not found!")
    print("  Please ensure you're in the smart_monitoring directory")
    sys.exit(1)

# Import and run with test configuration
print("\n" + "="*60)
print("Generating Test Dataset (50 frames, 4 cameras)")
print("="*60)

# Import the main script
from path2_phase1_2_verified import (
    SimConfig,
    run_simulation,
    MOTION_PATTERNS
)

# Create test configuration
test_config = SimConfig(
    num_frames=50,      # Quick: 50 frames instead of 500
    num_objects=3,      # 3 objects
    num_cameras=4,      # 4 cameras
    width=320,          # Smaller resolution for speed
    height=240,
    fps=30,
    save_images=False,  # Don't save images (faster)
    output_dir='output'
)

print(f"\nConfiguration:")
print(f"  Frames: {test_config.num_frames}")
print(f"  Objects: {test_config.num_objects}")
print(f"  Cameras: {test_config.num_cameras}")
print(f"  Resolution: {test_config.width}x{test_config.height}")
print(f"  Output: {test_config.output_dir}/data/")

# Run simulation
print("\nGenerating data...")
run_simulation(test_config)

print("\n" + "="*60)
print("✓ Test dataset generation complete!")
print("="*60)
print(f"\nGenerated files:")
print(f"  output/data/cam_0.json")
print(f"  output/data/cam_1.json")
print(f"  output/data/cam_2.json")
print(f"  output/data/cam_3.json")
print(f"\nYou can now test Phase B/C/D!")
