#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import cv2
from PIL import Image
import pillow_heif


def extract_frames_from_mov(mov_path, output_dir, frame_interval=10):
    """
    Extract every Nth frame from a MOV file and save as HEIC.

    Args:
        mov_path: Path to the MOV file
        output_dir: Directory to save HEIC frames
        frame_interval: Extract every Nth frame (default: 10)
    """
    print(f"Processing: {mov_path}")

    # Open video file
    cap = cv2.VideoCapture(str(mov_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {mov_path}")

    # Get video properties for detailed error reporting
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info - Total frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")

    if total_frames <= 0:
        raise ValueError(f"Video has invalid frame count: {total_frames}")

    frame_number = 0
    extracted_count = 0
    mov_stem = mov_path.stem

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract every 10th frame
            if frame_number % frame_interval == 0:
                # Convert BGR to RGB (OpenCV uses BGR by default)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)

                # Generate output filename
                output_filename = f"{mov_stem}_frame_{frame_number:06d}.heic"
                output_path = output_dir / output_filename

                # Save as HEIC
                try:
                    pil_image.save(str(output_path), format="HEIF")
                    extracted_count += 1
                    print(f"Saved frame {frame_number} to {output_filename}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to save frame {frame_number} as HEIC to {output_path}: {str(e)}")

            frame_number += 1

    finally:
        cap.release()

    print(f"Extracted {extracted_count} frames from {mov_path.name}")
    return extracted_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract every 10th frame from MOV files and save as HEIC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py /path/to/videos
  python script.py /path/to/videos --output /path/to/frames
        """
    )

    parser.add_argument("input_dir",
                        help="Directory containing MOV files")
    parser.add_argument("--output", "-o",
                        help="Output directory for HEIC frames (default: input_dir/frames)")
    parser.add_argument("--interval", "-i", type=int, default=10,
                        help="Frame interval - extract every Nth frame (default: 10)")

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    # Set up output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_dir / "frames"

    # Create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Cannot create output directory {output_dir}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory {output_dir}: {str(e)}")

    # Register HEIF opener with Pillow
    pillow_heif.register_heif_opener()

    # Find all MOV files
    mov_files = list(input_dir.glob("*.mov")) + list(input_dir.glob("*.MOV"))

    if not mov_files:
        raise FileNotFoundError(f"No MOV files found in directory: {input_dir}")

    print(f"Found {len(mov_files)} MOV files in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Frame interval: every {args.interval} frames")
    print("-" * 50)

    total_frames_extracted = 0

    # Process each MOV file
    for mov_file in mov_files:
        try:
            frames_extracted = extract_frames_from_mov(mov_file, output_dir, args.interval)
            total_frames_extracted += frames_extracted
        except Exception as e:
            # Re-raise with more context about which file failed
            raise RuntimeError(f"Failed processing {mov_file.name}: {str(e)}") from e

        print("-" * 30)

    print(f"\nCompleted! Total frames extracted: {total_frames_extracted}")
    print(f"Frames saved to: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
