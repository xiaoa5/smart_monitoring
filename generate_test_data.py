#!/usr/bin/env python3
"""
Quick Test Data Generation
===========================

Generates minimal synthetic data for testing Phase B/C/D without PyBullet.
Creates fake multi-camera bbox sequences with realistic motion patterns.

Usage:
    python generate_test_data.py
"""

import os
import json
import numpy as np
from typing import List, Dict

# Configuration
NUM_FRAMES = 50
NUM_OBJECTS = 3
NUM_CAMERAS = 4
OUTPUT_DIR = 'output/data'


def generate_circular_motion(frame: int, num_frames: int, obj_id: int) -> tuple:
    """Generate circular trajectory."""
    t = frame / num_frames
    angle = 2 * np.pi * t * 3  # 3 rotations
    radius = 1.5

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = 0.5

    return [float(x), float(y), float(z)]


def generate_sine_motion(frame: int, num_frames: int, obj_id: int) -> tuple:
    """Generate sine wave trajectory."""
    t = frame / num_frames

    x = -3.0 + 6.0 * t
    y = 1.5 * np.sin(4 * np.pi * t)
    z = 0.5

    return [float(x), float(y), float(z)]


def generate_linear_motion(frame: int, num_frames: int, obj_id: int) -> tuple:
    """Generate linear trajectory."""
    t = frame / num_frames

    x = -2.0 + 4.0 * t
    y = -1.5 + 3.0 * t
    z = 0.5

    return [float(x), float(y), float(z)]


def project_to_camera(pos_3d: List[float], camera_id: int, width: int = 640, height: int = 480) -> Dict:
    """
    Project 3D position to 2D bbox with camera-specific perspective.

    Simplified projection (no actual camera matrices needed for testing).
    """
    x, y, z = pos_3d

    # Camera positions (4 cameras around the scene)
    camera_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    cam_angle = camera_angles[camera_id % 4]
    cam_x = 5.0 * np.cos(cam_angle)
    cam_y = 5.0 * np.sin(cam_angle)

    # Vector from camera to object
    dx = x - cam_x
    dy = y - cam_y
    dz = z - 1.5  # Camera height

    # Simple perspective projection
    if abs(dz) < 0.1:
        dz = 0.1

    # Normalize to image coordinates
    scale = 2.0 / max(abs(dx), abs(dy), 0.1)

    # Project to image plane
    px = 0.5 + dx * scale * 0.2
    py = 0.5 + dy * scale * 0.2

    # Add camera-specific offset and noise
    noise_x = 0.01 * np.random.randn()
    noise_y = 0.01 * np.random.randn()

    px = np.clip(px + noise_x, 0.1, 0.9)
    py = np.clip(py + noise_y, 0.1, 0.9)

    # Bbox size (inversely proportional to distance)
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    bbox_w = np.clip(0.3 / dist, 0.05, 0.4)
    bbox_h = np.clip(0.3 / dist, 0.05, 0.4)

    # Add size variation
    bbox_w += 0.02 * np.random.randn()
    bbox_h += 0.02 * np.random.randn()
    bbox_w = np.clip(bbox_w, 0.02, 0.5)
    bbox_h = np.clip(bbox_h, 0.02, 0.5)

    # Convert to pixels for bbox_pixels
    x1 = int((px - bbox_w/2) * width)
    y1 = int((py - bbox_h/2) * height)
    x2 = int((px + bbox_w/2) * width)
    y2 = int((py + bbox_h/2) * height)

    return {
        'bbox': [float(px), float(py), float(bbox_w), float(bbox_h)],
        'bbox_pixels': [x1, y1, x2, y2]
    }


def generate_data():
    """Generate synthetic multi-camera test data."""
    print("="*60)
    print("Generating Synthetic Test Data")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Frames:  {NUM_FRAMES}")
    print(f"  Objects: {NUM_OBJECTS}")
    print(f"  Cameras: {NUM_CAMERAS}")
    print(f"  Output:  {OUTPUT_DIR}/")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Motion patterns for each object
    motion_functions = [
        generate_circular_motion,
        generate_sine_motion,
        generate_linear_motion
    ]

    object_names = [
        'red_cube',
        'green_cylinder',
        'blue_sphere'
    ]

    # Generate data for each camera
    for cam_id in range(NUM_CAMERAS):
        print(f"\nGenerating camera {cam_id}...")

        camera_data = []

        for frame in range(NUM_FRAMES):
            frame_data = {
                'frame': frame,
                'timestamp': frame / 30.0,  # 30 FPS
                'camera_id': cam_id,
                'objects': []
            }

            # Generate data for each object
            for obj_id in range(NUM_OBJECTS):
                # Get 3D position
                motion_func = motion_functions[obj_id % len(motion_functions)]
                pos_3d = motion_func(frame, NUM_FRAMES, obj_id)

                # Project to camera
                projection = project_to_camera(pos_3d, cam_id)

                # Compute velocity (finite difference)
                if frame > 0:
                    pos_prev = motion_func(frame - 1, NUM_FRAMES, obj_id)
                    velocity = [
                        30.0 * (pos_3d[i] - pos_prev[i])  # 30 FPS
                        for i in range(3)
                    ]
                else:
                    velocity = [0.0, 0.0, 0.0]

                # Create object data
                obj_data = {
                    'id': obj_id,
                    'name': object_names[obj_id],
                    'pos_3d': pos_3d,
                    'bbox': projection['bbox'],
                    'bbox_pixels': projection['bbox_pixels'],
                    'occlusion': 0.0,
                    'velocity': velocity,
                    'motion_type': ['circular', 'sine_wave', 'linear'][obj_id],
                    'orientation': [0.0, 0.0, 0.0, 1.0],  # quaternion
                    'angular_velocity': [0.0, 0.0, 0.0]
                }

                frame_data['objects'].append(obj_data)

            camera_data.append(frame_data)

        # Save camera data
        output_file = os.path.join(OUTPUT_DIR, f'cam_{cam_id}.json')
        try:
            with open(output_file, 'w') as f:
                json.dump(camera_data, f, indent=2)
            print(f"  ✓ Saved: {output_file}")
        except IOError as e:
            print(f"  ✗ Failed to save {output_file}: {e}")

    # Verify data
    print("\n" + "="*60)
    print("Verification")
    print("="*60)

    for cam_id in range(NUM_CAMERAS):
        json_file = os.path.join(OUTPUT_DIR, f'cam_{cam_id}.json')
        with open(json_file, 'r') as f:
            data = json.load(f)

        print(f"Camera {cam_id}:")
        print(f"  Frames:  {len(data)}")
        print(f"  Objects: {len(data[0]['objects'])}")
        print(f"  Sample bbox: {data[0]['objects'][0]['bbox']}")
        print(f"  Sample pos:  {data[0]['objects'][0]['pos_3d']}")

    print("\n" + "="*60)
    print("✓ Test Data Generation Complete!")
    print("="*60)
    print(f"\nGenerated files:")
    for cam_id in range(NUM_CAMERAS):
        print(f"  {OUTPUT_DIR}/cam_{cam_id}.json")

    print(f"\nNext steps:")
    print(f"  1. Test Phase B: from path2_probabilistic_lstm import ...")
    print(f"  2. Test Phase C: from path2_constraints import ...")
    print(f"  3. Test Phase D: from path2_integrated import ...")


if __name__ == '__main__':
    generate_data()
