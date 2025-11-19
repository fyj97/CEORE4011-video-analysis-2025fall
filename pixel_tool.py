import cv2
import argparse
import sys
from pathlib import Path

# Global variables to store clicked points
clicked_points = []
display_image = None
window_name = None

def mouse_callback(event, x, y, flags, param):
    """Mouse callback function to capture pixel coordinates on click"""
    global clicked_points, display_image, window_name
    
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        print(f"Clicked point: [{x}, {y}]")
        
        # Draw a circle at the clicked point
        cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(display_image, f'({x},{y})', (x+10, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if window_name:
            cv2.imshow(window_name, display_image)
        
        # Print all collected points
        print(f"\nAll collected points so far:")
        for i, point in enumerate(clicked_points):
            print(f"  Point {i+1}: [{point[0]}, {point[1]}]")
        print(f"Python format: {clicked_points}\n")

def main():
    global display_image, window_name
    
    parser = argparse.ArgumentParser(description='Extract first frame from video and get pixel coordinates')
    parser.add_argument('--source', type=str, 
                       default=r'PedestrianSpeedEstimator\test.mp4',
                       help='Path to video file or camera index (e.g., 0 for webcam)')
    parser.add_argument('--save', type=str, default=None,
                       help='Optional: Save the first frame to this path')
    
    args = parser.parse_args()
    
    # Open video
    source = args.source
    source_path = None
    if not source.isdigit():
        source_path = Path(source).resolve()
        source = str(source_path)
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.source}")
        if source_path:
            print(f"Resolved path: {source_path}")
            if not source_path.exists():
                print(f"File does not exist at: {source_path}")
            else:
                print("File exists but could not be opened. Possible issues:")
                print("  - File is corrupted")
                print("  - Unsupported video codec")
                print("  - Insufficient permissions")
        sys.exit(1)
    
    # Read first frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read first frame from video")
        cap.release()
        sys.exit(1)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Verify FPS by reading a few frames (if video has multiple frames)
    verified_fps = fps
    if total_frames > 1:
        frame_times = []
        for i in range(min(10, total_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, _ = cap.read()
            if ret:
                frame_times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        if len(frame_times) > 1:
            time_diff = frame_times[-1] - frame_times[0]
            if time_diff > 0:
                calculated_fps = (len(frame_times) - 1) / time_diff
                if abs(calculated_fps - fps) > 0.1:  # If significantly different
                    verified_fps = calculated_fps
    
    cap.release()
    
    # Display image info
    print("=" * 60)
    print("Video Information:")
    print(f"  Image size: {width} x {height} pixels")
    print(f"  FPS (metadata): {fps:.2f}")
    if abs(verified_fps - fps) > 0.1:
        print(f"  FPS (calculated): {verified_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    if duration > 0:
        print(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print("=" * 60)
    print("\nInstructions:")
    print("  - Click on the image to get pixel coordinates")
    print("  - Press 'q' or ESC to quit")
    print("  - Press 'c' to clear all marked points")
    print("  - Press 's' to save the current frame with markers")
    print("=" * 60)
    print()
    
    # Make a copy for display (so we can draw on it)
    display_image = frame.copy()
    
    # Add video info text on the image
    info_text = [
        f"Size: {width}x{height}",
        f"FPS: {verified_fps:.2f}",
        f"Frame: 1/{total_frames}",
        f"Duration: {duration:.1f}s"
    ]
    y_offset = 25
    for i, text in enumerate(info_text):
        cv2.putText(display_image, text, (10, y_offset + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_image, text, (10, y_offset + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Create window and set mouse callback
    window_name = f'Video Frame - FPS: {verified_fps:.2f} - Click to get pixel coordinates'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Display the image
    cv2.imshow(window_name, display_image)
    
    # Main loop
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('c'):  # Clear points
            clicked_points = []
            display_image = frame.copy()
            cv2.imshow(window_name, display_image)
            print("Cleared all points")
        elif key == ord('s'):  # Save frame
            if args.save:
                save_path = args.save
            else:
                save_path = 'first_frame_with_points.jpg'
            cv2.imwrite(save_path, display_image)
            print(f"Saved frame with markers to: {save_path}")
    
    cv2.destroyAllWindows()
    
    # Print final summary
    if clicked_points:
        print("\n" + "=" * 60)
        print("Final collected points:")
        for i, point in enumerate(clicked_points):
            print(f"  Point {i+1}: [{point[0]}, {point[1]}]")
        print("\nPython format (for roi_polygon or mapper_points):")
        print(f"  {clicked_points}")
        print("=" * 60)
    
    # Save first frame if requested
    if args.save:
        cv2.imwrite(args.save, frame)
        print(f"\nSaved first frame to: {args.save}")

if __name__ == '__main__':
    main()

