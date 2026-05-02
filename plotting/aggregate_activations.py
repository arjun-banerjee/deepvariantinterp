#!/usr/bin/env python3
"""
Aggregate activation files from multiple samples into a single directory.

Copies activation .npz files from individual sample directories into a combined
cache directory, renaming them sequentially (activations_00000000.npz, etc.).

Each input sample directory contains files like:
  activations_00000000.npz (batch 0)
  activations_00000001.npz (batch 1)
  ...

The output directory will contain all files renamed sequentially across all samples.

Usage:
    python3 plotting/aggregate_activations.py \
        --input_dirs data/1kg_embeddings/HG00731/activations data/1kg_embeddings/NA12878/activations \
        --output_dir data/1kg_combined_cache_mixed5

    # Or with glob patterns (shell will expand):
    python3 plotting/aggregate_activations.py \
        --input_dirs data/1kg_embeddings/*/activations_mixed10 \
        --output_dir data/1kg_combined_cache_mixed10
"""

import argparse
import shutil
from pathlib import Path


def aggregate_activations(input_dirs, output_dir):
    """
    Aggregate activation files from multiple directories into one.

    Args:
        input_dirs: List of input directory paths
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert input_dirs to Path objects
    all_input_paths = []
    for dir_str in input_dirs:
        p = Path(dir_str)
        if p.exists() and p.is_dir():
            all_input_paths.append(p)
        else:
            print(f"WARNING: Skipping non-existent or invalid directory: {dir_str}")

    if not all_input_paths:
        print(f"ERROR: No valid input directories found")
        return

    print(f"Found {len(all_input_paths)} input directories")
    print(f"Output directory: {output_path.absolute()}")
    print()

    # Sort input paths to ensure consistent ordering
    all_input_paths = sorted(all_input_paths)

    file_counter = 0

    for input_dir in all_input_paths:
        # Extract sample name from path
        # e.g., "HG00731" from "data/1kg_embeddings/HG00731/activations"
        sample_name = input_dir.parent.name

        # Find all .npz files in the input directory
        npz_files = sorted(input_dir.glob("activations_*.npz"))

        if not npz_files:
            print(f"SKIP (no activation files): {input_dir} ({sample_name})")
            continue

        print(f"Processing {sample_name}: {len(npz_files)} file(s)")

        # Copy each file with sequential numbering
        for npz_file in npz_files:
            # Create output filename: activations_{counter:08d}.npz
            output_filename = f"activations_{file_counter:08d}.npz"
            output_file = output_path / output_filename

            # Copy file
            shutil.copy2(npz_file, output_file)

            file_counter += 1

    print()
    print(f"✓ Aggregation complete!")
    print(f"  Total files copied: {file_counter}")
    print(f"  Output: {output_path.absolute()}")

    # Verify output
    output_files = list(output_path.glob("activations_*.npz"))
    print(f"  Verification: {len(output_files)} .npz files in output directory")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate activation files from multiple samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From GPU script (multiple literal paths):
  python3 plotting/aggregate_activations.py \\
      --input_dirs /path/to/sample1/activations /path/to/sample2/activations \\
      --output_dir data/1kg_combined_cache

  # Manual with glob (shell expands the glob):
  python3 plotting/aggregate_activations.py \\
      --input_dirs data/1kg_embeddings/*/activations_mixed10 \\
      --output_dir data/1kg_combined_cache_mixed10
        """
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="Input directories containing activation .npz files"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for aggregated activations"
    )

    args = parser.parse_args()

    aggregate_activations(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
