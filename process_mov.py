#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import cv2


def main():
    if len(sys.argv) != 2:
        raise ValueError(f"Usage: {sys.argv[0]} <input_directory>")

    input_dir = Path(sys.argv[1])

    # Fail loudly if directory doesn't exist
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {input_dir}")

    # Create output directory
    output_dir = input_dir / "extracted_frames"
    output_dir.mkdir(exist_ok=True)

    # Find all MOV files
    mov_files = list(input_dir.glob("*.mov")) + list(input_dir.glob("*.MOV"))

    if not mov_files:
        raise FileNotFoundError(f"No MOV files found in directory: {input_dir}")

    print(f"Found {len(mov_files)} MOV files")

    for mov_file in mov_files:
        print(f"Processing: {mov_file.name}")

        # Open video file
        cap = cv2.VideoCapture(str(mov_file))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {mov_file}")

        # Get total frame count for verification
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"  Total frames: {total_frames}, FPS: {fps}")

        frame_number = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Save every 10th frame starting at frame 5 (5, 15, 25, 35, ...)
            if frame_number % 10 == 5:
                # Create filename: originalname_frame_0000.png
                output_filename = f"{mov_file.stem}_frame_{frame_number:04d}.png"
                output_path = output_dir / output_filename

                # Write the frame - fail loudly if this doesn't work
                success = cv2.imwrite(str(output_path), frame)

                if not success:
                    raise RuntimeError(f"Failed to write frame {frame_number} to {output_path}")

                saved_count += 1
                print(f"  Saved frame {frame_number} -> {output_filename}")

            frame_number += 1

        cap.release()
        print(f"  Completed: saved {saved_count} frames from {mov_file.name}")

    print(f"\nAll done! Frames saved to: {output_dir}")


if __name__ == "__main__":
    main()
