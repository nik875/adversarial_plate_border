#!/usr/bin/env python3

import os
import sys
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from fast_alpr import ALPR
from typing import List, Dict, Optional
import argparse


class VideoALPRAnalyzer:
    """Analyze ALPR detection performance on video files, tracking specific plate"""

    def __init__(self, video_path: str, target_plate: str = "VRJ7774"):
        """Initialize the video ALPR analyzer

        Args:
            video_path: Path to video file (.mov or other formats)
            target_plate: License plate number to track (default: "VRJ7774")
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = video_path
        self.target_plate = target_plate
        print(f"Target plate: {self.target_plate}")

        # Initialize ALPR detector
        print("Loading ALPR detection model...")
        self.alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="cct-xs-v1-global-model"
        )
        print("ALPR model loaded successfully")

        # Get video info
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        print(f"\nVideo Info:")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Duration: {self.duration:.2f} seconds")

    def __del__(self):
        """Release video capture on cleanup"""
        if hasattr(self, 'cap'):
            self.cap.release()

    def _detect_plates_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """Run ALPR detection on a single frame

        Args:
            frame: BGR image from cv2

        Returns:
            List of detection dictionaries
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame: frame is None or empty")

        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid frame shape: {frame.shape}, expected (H, W, 3)")

        # Run ALPR detection
        predictions = self.alpr.predict(frame)

        detections = []
        if predictions:
            for pred in predictions:
                detection = pred.detection
                ocr = pred.ocr
                bbox = detection.bounding_box

                detections.append({
                    'bbox': [float(bbox.x1), float(bbox.y1), float(bbox.x2), float(bbox.y2)],
                    'confidence': float(ocr.confidence),
                    'text': ocr.text,
                    'detection_confidence': float(detection.confidence)
                })

        return detections

    def analyze_video(self, sample_interval: int = 1,
                      max_frames: Optional[int] = None) -> pd.DataFrame:
        """Analyze all frames in the video

        Args:
            sample_interval: Process every Nth frame (1 = all frames). Ignored if max_frames is set.
            max_frames: If set, process exactly this many evenly-spaced frames from the video

        Returns:
            DataFrame with per-frame detection results
        """
        results = []
        sample_frames = []  # Store frames for visualization

        # Determine which frames to process
        if max_frames is not None:
            # Calculate evenly spaced frame indices
            if max_frames >= self.frame_count:
                # Process all frames if max_frames >= total frames
                frames_to_process = set(range(self.frame_count))
                print(
                    f"\nAnalyzing video (all {self.frame_count} frames, max_frames={max_frames} >= total)...")
            else:
                # Select evenly spaced frames
                step = self.frame_count / max_frames
                frames_to_process = set(int(i * step) for i in range(max_frames))
                print(
                    f"\nAnalyzing video ({len(frames_to_process)} evenly-spaced frames out of {self.frame_count} total)...")
        else:
            # Use sample_interval
            frames_to_process = set(range(0, self.frame_count, sample_interval))
            print(
                f"\nAnalyzing video (sampling every {sample_interval} frame(s), {len(frames_to_process)} frames total)...")

        frame_idx = 0
        processed_count = 0

        # Process frames with progress bar showing frames read vs frames to process
        pbar = tqdm(total=len(frames_to_process), desc="Processing frames", unit="frame")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            # Skip frames not in our processing set
            if frame_idx not in frames_to_process:
                frame_idx += 1
                continue

            # Update progress bar for frames we're actually processing
            pbar.update(1)

            timestamp = frame_idx / self.fps if self.fps > 0 else 0

            try:
                # Run detection
                detections = self._detect_plates_in_frame(frame)

                # Check for target plate
                target_detected = False
                target_confidence = 0.0
                target_bbox = None
                best_detection = None
                best_confidence = 0.0

                for det in detections:
                    if det['text'] == self.target_plate:
                        target_detected = True
                        target_confidence = det['confidence']
                        target_bbox = det['bbox']

                    if det['confidence'] > best_confidence:
                        best_confidence = det['confidence']
                        best_detection = det

                # Store frame for visualization if it has interesting detections
                if target_detected or len(
                        detections) > 0 or processed_count < 5 or np.random.random() < 0.05:
                    sample_frames.append({
                        'frame_idx': frame_idx,
                        'frame': frame.copy(),
                        'detections': detections,
                        'target_detected': target_detected
                    })

                # Record results
                results.append({
                    'frame_idx': frame_idx,
                    'timestamp': timestamp,
                    'num_detections': len(detections),
                    'target_detected': target_detected,
                    'target_confidence': target_confidence if target_detected else 0.0,
                    'target_bbox': target_bbox,
                    'best_confidence': best_confidence,
                    'best_text': best_detection['text'] if best_detection else '',
                    'all_texts': [d['text'] for d in detections]
                })

                processed_count += 1

            except Exception as e:
                raise RuntimeError(f"Failed to process frame {frame_idx} at timestamp {timestamp:.2f}s: "
                                   f"{type(e).__name__}: {str(e)}")

            frame_idx += 1

        pbar.close()

        print(f"Processed {processed_count} frames")

        if processed_count == 0:
            raise RuntimeError("No frames were successfully processed")

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Store sample frames for visualization
        self.sample_frames = sample_frames

        return results_df

    def generate_statistics(self, results_df: pd.DataFrame) -> Dict:
        """Generate comprehensive statistics from results

        Args:
            results_df: DataFrame with detection results

        Returns:
            Dictionary of statistics
        """
        if len(results_df) == 0:
            raise ValueError("Results DataFrame is empty, cannot generate statistics")

        total_frames = len(results_df)
        frames_with_detections = (results_df['num_detections'] > 0).sum()
        frames_with_target = results_df['target_detected'].sum()

        stats = {
            'total_frames_analyzed': total_frames,
            'frames_with_any_detection': frames_with_detections,
            'frames_with_target_plate': frames_with_target,
            'detection_rate': frames_with_detections / total_frames * 100,
            'target_detection_rate': frames_with_target / total_frames * 100,
            'avg_detections_per_frame': results_df['num_detections'].mean(),
            'max_detections_in_frame': results_df['num_detections'].max(),
        }

        # Target plate specific stats
        target_frames = results_df[results_df['target_detected'] == True]
        if len(target_frames) > 0:
            stats['target_avg_confidence'] = target_frames['target_confidence'].mean()
            stats['target_min_confidence'] = target_frames['target_confidence'].min()
            stats['target_max_confidence'] = target_frames['target_confidence'].max()
            stats['target_median_confidence'] = target_frames['target_confidence'].median()
        else:
            stats['target_avg_confidence'] = 0.0
            stats['target_min_confidence'] = 0.0
            stats['target_max_confidence'] = 0.0
            stats['target_median_confidence'] = 0.0

        # Other plates detected
        all_texts = []
        for texts in results_df['all_texts']:
            all_texts.extend(texts)

        unique_plates = set(all_texts)
        stats['unique_plates_detected'] = len(unique_plates)
        stats['other_plates'] = [p for p in unique_plates if p != self.target_plate]

        return stats

    def create_visualizations(self, results_df: pd.DataFrame, stats: Dict,
                              output_dir: str = "video_alpr_results"):
        """Create comprehensive visualizations

        Args:
            results_df: DataFrame with detection results
            stats: Statistics dictionary
            output_dir: Output directory for visualizations
        """
        Path(output_dir).mkdir(exist_ok=True)

        print(f"\nCreating visualizations in {output_dir}/...")

        # Create main analysis figure
        fig = plt.figure(figsize=(20, 12))

        # 1. Detection timeline
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(results_df['timestamp'], results_df['num_detections'],
                 label='Total Detections', alpha=0.7, linewidth=1)
        ax1.scatter(results_df[results_df['target_detected']]['timestamp'],
                    results_df[results_df['target_detected']]['num_detections'],
                    color='red', s=30, label=f'Target "{self.target_plate}" Detected', zorder=5)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Number of Detections')
        ax1.set_title('Detection Timeline')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Target plate confidence over time
        ax2 = plt.subplot(3, 3, 2)
        target_frames = results_df[results_df['target_detected'] == True]
        if len(target_frames) > 0:
            ax2.scatter(target_frames['timestamp'], target_frames['target_confidence'],
                        alpha=0.6, s=20)
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Confidence')
            ax2.set_title(f'Target Plate "{self.target_plate}" Confidence Over Time')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1])
        else:
            ax2.text(0.5, 0.5, 'No target plate detections',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'Target Plate "{self.target_plate}" - Not Detected')

        # 3. Detection rate pie chart
        ax3 = plt.subplot(3, 3, 3)
        detection_categories = [
            stats['frames_with_target_plate'],
            stats['frames_with_any_detection'] - stats['frames_with_target_plate'],
            stats['total_frames_analyzed'] - stats['frames_with_any_detection']
        ]
        labels = [
            f'Target Plate\n({stats["target_detection_rate"]:.1f}%)',
            f'Other Plates\n({(detection_categories[1]/stats["total_frames_analyzed"]*100):.1f}%)',
            f'No Detection\n({(detection_categories[2]/stats["total_frames_analyzed"]*100):.1f}%)'
        ]
        colors = ['#2ca02c', '#ff7f0e', '#d62728']
        ax3.pie(
            detection_categories,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90)
        ax3.set_title('Detection Rate Distribution')

        # 4. Confidence distribution histogram
        ax4 = plt.subplot(3, 3, 4)
        if len(target_frames) > 0:
            ax4.hist(target_frames['target_confidence'], bins=20, edgecolor='black', alpha=0.7)
            ax4.axvline(stats['target_avg_confidence'], color='red', linestyle='--',
                        label=f'Mean: {stats["target_avg_confidence"]:.3f}')
            ax4.set_xlabel('Confidence')
            ax4.set_ylabel('Frequency')
            ax4.set_title(f'Target Plate Confidence Distribution')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No target plate detections',
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Confidence Distribution - N/A')

        # 5. Detections per frame histogram
        ax5 = plt.subplot(3, 3, 5)
        ax5.hist(results_df['num_detections'], bins=range(0, results_df['num_detections'].max() + 2),
                 edgecolor='black', alpha=0.7)
        ax5.set_xlabel('Number of Detections')
        ax5.set_ylabel('Number of Frames')
        ax5.set_title('Detections Per Frame Distribution')

        # 6. Detection heatmap over time (binned)
        ax6 = plt.subplot(3, 3, 6)
        bin_size = max(1, len(results_df) // 50)  # ~50 bins
        results_df['time_bin'] = results_df.index // bin_size
        heatmap_data = results_df.groupby('time_bin').agg({
            'num_detections': 'mean',
            'target_detected': 'sum'
        })
        ax6_twin = ax6.twinx()
        ax6.bar(heatmap_data.index, heatmap_data['num_detections'],
                alpha=0.5, label='Avg Detections', color='blue')
        ax6_twin.plot(heatmap_data.index, heatmap_data['target_detected'],
                      color='red', marker='o', label='Target Detections', linewidth=2)
        ax6.set_xlabel('Time Bin')
        ax6.set_ylabel('Avg Detections', color='blue')
        ax6_twin.set_ylabel('Target Detections', color='red')
        ax6.set_title('Detection Heatmap (Binned Over Time)')
        ax6.legend(loc='upper left')
        ax6_twin.legend(loc='upper right')

        # 7-9. Statistics text boxes
        ax7 = plt.subplot(3, 3, 7)
        ax7.axis('off')
        stats_text = f"""OVERALL STATISTICS

Total Frames Analyzed: {stats['total_frames_analyzed']}
Frames with Any Detection: {stats['frames_with_any_detection']} ({stats['detection_rate']:.1f}%)
Frames with Target Plate: {stats['frames_with_target_plate']} ({stats['target_detection_rate']:.1f}%)

Avg Detections/Frame: {stats['avg_detections_per_frame']:.2f}
Max Detections in Frame: {stats['max_detections_in_frame']}
Unique Plates Found: {stats['unique_plates_detected']}"""
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        if stats['frames_with_target_plate'] > 0:
            target_stats_text = f"""TARGET PLATE: "{self.target_plate}"

Detection Rate: {stats['target_detection_rate']:.1f}%
Total Detections: {stats['frames_with_target_plate']} frames

CONFIDENCE STATISTICS:
  Average: {stats['target_avg_confidence']:.4f}
  Median:  {stats['target_median_confidence']:.4f}
  Min:     {stats['target_min_confidence']:.4f}
  Max:     {stats['target_max_confidence']:.4f}"""
        else:
            target_stats_text = f"""TARGET PLATE: "{self.target_plate}"

❌ NOT DETECTED

The target plate was not detected
in any of the analyzed frames."""

        ax8.text(0.05, 0.95, target_stats_text, transform=ax8.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        other_plates_text = f"""OTHER PLATES DETECTED

Total Unique: {len(stats['other_plates'])}

"""
        if len(stats['other_plates']) > 0:
            other_plates_text += "Top plates:\n"
            # Count occurrences
            all_texts = []
            for texts in results_df['all_texts']:
                all_texts.extend([t for t in texts if t != self.target_plate])
            from collections import Counter
            plate_counts = Counter(all_texts)
            for plate, count in plate_counts.most_common(10):
                other_plates_text += f"  {plate}: {count} times\n"
        else:
            other_plates_text += "No other plates detected"

        ax9.text(0.05, 0.95, other_plates_text, transform=ax9.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        analysis_path = Path(output_dir) / "video_alpr_analysis.png"
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        print(f"Analysis visualization saved: {analysis_path}")
        plt.close()

        # Create sample frames visualization
        self._create_sample_frames_visualization(output_dir)

    def _create_sample_frames_visualization(self, output_dir: str):
        """Create visualization showing sample frames with bounding boxes"""

        if not hasattr(self, 'sample_frames') or len(self.sample_frames) == 0:
            print("No sample frames available for visualization")
            return

        print("Creating sample frames visualization...")

        # Select diverse samples
        target_samples = [s for s in self.sample_frames if s['target_detected']]
        other_samples = [
            s for s in self.sample_frames if not s['target_detected'] and len(
                s['detections']) > 0]
        no_detection_samples = [s for s in self.sample_frames if len(s['detections']) == 0]

        # Select up to 3 from each category
        selected_samples = []
        selected_samples.extend(target_samples[:3])
        selected_samples.extend(other_samples[:3])
        selected_samples.extend(no_detection_samples[:2])

        if len(selected_samples) == 0:
            print("No sample frames to visualize")
            return

        # Limit to 8 total samples for clean layout
        selected_samples = selected_samples[:8]

        # Create figure
        n_samples = len(selected_samples)
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        axes = axes.flatten()

        for idx, sample in enumerate(selected_samples):
            ax = axes[idx]

            frame = sample['frame']
            detections = sample['detections']
            frame_idx = sample['frame_idx']
            target_detected = sample['target_detected']

            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(frame_rgb)

            # Draw bounding boxes
            for det in detections:
                bbox = det['bbox']
                text = det['text']
                conf = det['confidence']

                # Color code: green for target, blue for others
                if text == self.target_plate:
                    color = 'green'
                    linewidth = 3
                else:
                    color = 'blue'
                    linewidth = 2

                rect = patches.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                    linewidth=linewidth,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)

                # Add text label
                label = f"{text}\n{conf:.3f}"
                ax.text(bbox[0], bbox[1] - 10, label,
                        color=color, fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

            # Title
            timestamp = frame_idx / self.fps if self.fps > 0 else 0
            title = f"Frame {frame_idx} ({timestamp:.2f}s)\n"
            if target_detected:
                title += f"✓ TARGET DETECTED"
                title_color = 'green'
            elif len(detections) > 0:
                title += f"{len(detections)} other plate(s)"
                title_color = 'blue'
            else:
                title += "No detections"
                title_color = 'red'

            ax.set_title(title, fontsize=10, color=title_color, weight='bold')
            ax.axis('off')

        # Hide unused subplots
        for idx in range(len(selected_samples), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Sample Frames with ALPR Detections (Target: "{self.target_plate}")',
                     fontsize=14, y=0.98)
        plt.tight_layout()

        samples_path = Path(output_dir) / "sample_frames_with_detections.png"
        plt.savefig(samples_path, dpi=200, bbox_inches='tight')
        print(f"Sample frames visualization saved: {samples_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze ALPR detection on video files')
    parser.add_argument('--video', required=True, help='Path to video file (.mov or other)')
    parser.add_argument(
        '--target',
        default='VRJ7774',
        help='Target license plate to track (default: VRJ7774)')
    parser.add_argument(
        '--output',
        default='video_alpr_results',
        help='Output directory for results')
    parser.add_argument('--sample-interval', type=int, default=1,
                        help='Process every Nth frame (default: 1, all frames). Ignored if --max-frames is set.')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Process exactly this many evenly-spaced frames from the video (overrides --sample-interval)')
    args = parser.parse_args()

    try:
        print("=== Video ALPR Analysis ===")
        print(f"Video file: {args.video}")
        print(f"Target plate: {args.target}")
        print(f"Output directory: {args.output}")
        if args.max_frames:
            print(f"Max frames: {args.max_frames} (evenly spaced)")
        else:
            print(f"Sample interval: {args.sample_interval}")

        # Initialize analyzer
        analyzer = VideoALPRAnalyzer(
            video_path=args.video,
            target_plate=args.target
        )

        # Analyze video
        results_df = analyzer.analyze_video(
            sample_interval=args.sample_interval,
            max_frames=args.max_frames
        )

        # Generate statistics
        stats = analyzer.generate_statistics(results_df)

        # Save results
        Path(args.output).mkdir(exist_ok=True)
        results_path = Path(args.output) / "detection_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")

        # Create visualizations
        analyzer.create_visualizations(results_df, stats, output_dir=args.output)

        # Print summary
        print(f"\n=== Analysis Complete ===")
        if args.max_frames:
            print(
                f"Processed {len(results_df)} evenly-spaced frames out of {analyzer.frame_count} total")
        else:
            print(f"Processed {len(results_df)} frames (every {args.sample_interval} frame(s))")
        print(f"Target plate '{args.target}' detected in {stats['frames_with_target_plate']} / {stats['total_frames_analyzed']} frames "
              f"({stats['target_detection_rate']:.1f}%)")
        if stats['frames_with_target_plate'] > 0:
            print(f"Average confidence: {stats['target_avg_confidence']:.4f}")
        print(f"\nResults saved in: {args.output}/")
        print(f"  - detection_results.csv: Per-frame detection data")
        print(f"  - video_alpr_analysis.png: Comprehensive analysis plots")
        print(f"  - sample_frames_with_detections.png: Sample frames with bounding boxes")

    except Exception as e:
        print(f"\nFATAL ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
