#!/usr/bin/env python3
"""
Compare motion.npz files from MuJoCo and IsaacLab processing
"""

import numpy as np
import argparse
import os
import sys


def load_motion_data(file_path):
    """Load motion data from npz file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = np.load(file_path)
    return data


def compare_arrays(name, arr1, arr2, tolerance=1e-6):
    """Compare two arrays and return detailed statistics."""
    print(f"\n=== {name} ===")
    print(f"Shape: {arr1.shape} vs {arr2.shape}")
    print(f"Shape match: {arr1.shape == arr2.shape}")

    if arr1.shape != arr2.shape:
        print("❌ Shape mismatch!")
        return False

    # Basic statistics
    print(f"Data type: {arr1.dtype} vs {arr2.dtype}")
    print(f"Min values: {np.min(arr1):.6f} vs {np.min(arr2):.6f}")
    print(f"Max values: {np.max(arr1):.6f} vs {np.max(arr2):.6f}")
    print(f"Mean values: {np.mean(arr1):.6f} vs {np.mean(arr2):.6f}")
    print(f"Std values: {np.std(arr1):.6f} vs {np.std(arr2):.6f}")

    # Difference statistics
    diff = np.abs(arr1 - arr2)
    print(f"\nDifference statistics:")
    print(f"  Mean absolute difference: {np.mean(diff):.8f}")
    print(f"  Max absolute difference: {np.max(diff):.8f}")
    print(f"  RMS difference: {np.sqrt(np.mean(diff**2)):.8f}")

    # Relative difference (avoid division by zero)
    mask = np.abs(arr2) > 1e-10
    if np.any(mask):
        rel_diff = diff[mask] / np.abs(arr2[mask])
        print(f"  Mean relative difference: {np.mean(rel_diff):.8f}")
        print(f"  Max relative difference: {np.max(rel_diff):.8f}")

    # Check if differences are within tolerance
    within_tolerance = np.all(diff < tolerance)
    print(f"  Within tolerance ({tolerance}): {'✓' if within_tolerance else '❌'}")

    # Show first few values for inspection
    print(f"\nFirst 3 values comparison:")
    for i in range(min(3, len(arr1))):
        if arr1.ndim == 1:
            print(f"  [{i}]: {arr1[i]:.6f} vs {arr2[i]:.6f} (diff: {diff[i]:.8f})")
        elif arr1.ndim == 2:
            print(f"  [{i}]: {arr1[i][:5]} vs {arr2[i][:5]} (diff: {diff[i][:5]})")
        elif arr1.ndim == 3:
            print(f"  [{i}]: shape {arr1[i].shape}, first element: {arr1[i][0][0]:.6f} vs {arr2[i][0][0]:.6f}")

    return within_tolerance


def compare_motion_files(mjc_file, isaac_file, tolerance=1e-6):
    """Compare two motion.npz files."""
    print("=" * 80)
    print("MOTION.NPZ COMPARISON")
    print("=" * 80)
    print(f"MuJoCo file: {mjc_file}")
    print(f"IsaacLab file: {isaac_file}")
    print(f"Tolerance: {tolerance}")

    # Load data
    try:
        mjc_data = load_motion_data(mjc_file)
        isaac_data = load_motion_data(isaac_file)
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return False

    print(f"\nMuJoCo keys: {list(mjc_data.keys())}")
    print(f"IsaacLab keys: {list(isaac_data.keys())}")

    # print(mjc_data["body_pos_w"][0])
    breakpoint()

    # Check if keys match
    mjc_keys = set(mjc_data.keys())
    isaac_keys = set(isaac_data.keys())
    if mjc_keys != isaac_keys:
        print(f"❌ Key mismatch!")
        print(f"  MuJoCo only: {mjc_keys - isaac_keys}")
        print(f"  IsaacLab only: {isaac_keys - mjc_keys}")
        return False
    else:
        print("✓ All keys match")

    # Compare each array
    all_within_tolerance = True
    for key in sorted(mjc_keys):
        try:
            within_tol = compare_arrays(key, mjc_data[key], isaac_data[key], tolerance)
            all_within_tolerance = all_within_tolerance and within_tol
        except Exception as e:
            print(f"❌ Error comparing {key}: {e}")
            all_within_tolerance = False

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if all_within_tolerance:
        print("✅ All arrays are within tolerance!")
    else:
        print("❌ Some arrays exceed tolerance!")

    return all_within_tolerance


def main():
    parser = argparse.ArgumentParser(description="Compare motion.npz files from MuJoCo and IsaacLab")
    parser.add_argument("--mjc_file", type=str, required=True, help="Path to MuJoCo motion.npz file")
    parser.add_argument("--isaac_file", type=str, required=True, help="Path to IsaacLab motion.npz file")
    parser.add_argument("--tolerance", type=float, default=1e-2, help="Tolerance for comparison")

    args = parser.parse_args()

    success = compare_motion_files(args.mjc_file, args.isaac_file, args.tolerance)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
