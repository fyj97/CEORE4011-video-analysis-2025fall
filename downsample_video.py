#!/usr/bin/env python3
"""
Video FPS Downsampling Script

This script downsamples a video to a lower frame rate (e.g., from 30 fps to 5 fps).
It preserves the original video resolution and quality while reducing the frame rate.

Usage:
    python downsample_video.py --input input_video.mp4 --output output_video.mp4 --target_fps 5
    python downsample_video.py -i test.mp4 -o test_5fps.mp4 -f 5
"""

import cv2
import argparse
import os
from pathlib import Path


def downsample_video(input_path, output_path, target_fps=5):
    """
    Downsample a video to a target frame rate.
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path to output video file
        target_fps (float): Target frame rate (fps) for output video
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open input video: {input_path}")
        return False
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Original FPS: {original_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/original_fps:.2f} seconds")
    print(f"\nTarget FPS: {target_fps}")
    
    # Validate target FPS
    if target_fps <= 0:
        print(f"Error: Target FPS must be positive, got {target_fps}")
        cap.release()
        return False
    
    if target_fps >= original_fps:
        print(f"Warning: Target FPS ({target_fps}) is >= original FPS ({original_fps:.2f})")
        print("This will not reduce the frame rate. Consider using a lower target FPS.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            cap.release()
            return False
    
    # Calculate frame skip ratio
    # If original is 30 fps and target is 5 fps, we keep every 6th frame (30/5 = 6)
    frame_skip = original_fps / target_fps
    
    # Prepare output video writer
    # Use 'mp4v' codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video: {output_path}")
        cap.release()
        return False
    
    # Process frames
    frame_count = 0
    written_count = 0
    next_frame_to_write = 0
    
    print(f"\nProcessing video...")
    print(f"Frame skip ratio: {frame_skip:.2f} (keeping every {frame_skip:.1f} frames)")
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Write frame if it's time to write based on target FPS
        if frame_count >= next_frame_to_write:
            out.write(frame)
            written_count += 1
            next_frame_to_write = int((written_count) * frame_skip)
            
            # Progress indicator
            if written_count % 10 == 0 or frame_count == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | Frames written: {written_count}/{int(total_frames/frame_skip)}", end='\r')
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    out.release()
    
    # Print summary
    print(f"\n\nDownsampling complete!")
    print(f"  Frames processed: {frame_count}")
    print(f"  Frames written: {written_count}")
    print(f"  Output video: {output_path}")
    print(f"  Output FPS: {target_fps}")
    print(f"  Output duration: {written_count/target_fps:.2f} seconds")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Downsample video frame rate (e.g., from 30 fps to 5 fps)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python downsample_video.py -i input.mp4 -o output.mp4 -f 5
  python downsample_video.py --input test.mp4 --target_fps 5
  python downsample_video.py -i video.mp4 -f 10 -o video_10fps.mp4
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input video file path'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output video file path (default: input_filename_targetfps.mp4)'
    )
    
    parser.add_argument(
        '-f', '--target_fps',
        type=float,
        default=5.0,
        help='Target frame rate (fps) for output video (default: 5.0)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {args.input}")
        return
    
    # Generate output path if not provided
    if args.output is None:
        input_stem = input_path.stem
        input_suffix = input_path.suffix
        output_path = input_path.parent / f"{input_stem}_{int(args.target_fps)}fps{input_suffix}"
    else:
        output_path = Path(args.output)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if output file already exists
    if output_path.exists():
        response = input(f"Output file already exists: {output_path}\nOverwrite? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Perform downsampling
    success = downsample_video(
        str(input_path),
        str(output_path),
        args.target_fps
    )
    
    if not success:
        print("Downsampling failed.")
        return


if __name__ == '__main__':
    main()

