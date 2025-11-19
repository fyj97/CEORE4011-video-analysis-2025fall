import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from ultralytics import YOLO

# ==== Paths ====
input_video_path = './PedestrianSpeedEstimator/test.mp4'
# Extract filename without extension from input video path
video_filename = os.path.splitext(os.path.basename(input_video_path))[0]
output_video_path = f'./vehicle_runs/{video_filename}/output_with_overlay.mp4'
crossing_csv = f'./vehicle_runs/{video_filename}/vehicle_line_crossing.csv'
frame0_path = f'./vehicle_runs/{video_filename}/frame0_with_roi_and_line.jpg'

# Ensure output directory exists (all files are in the same directory)
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# ==== Load YOLO model ====
yolo_model = YOLO('yolov8m.pt')

# ==== Open input video ====
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {input_video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ==== Prepare output video writer ====
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# ==== Define ROI and crossing line ====
ROI_POINTS = np.array([[579, 600], [1042, 581], [867, 436], [529, 452]], dtype=np.int32)
# line_point1 is the middle of first two ROI points, line_point2 is the middle of last two ROI points
line_point1 = tuple(((ROI_POINTS[0] + ROI_POINTS[1]) / 2).astype(int))
line_point2 = tuple(((ROI_POINTS[2] + ROI_POINTS[3]) / 2).astype(int))

def in_roi(x, y, roi_polygon):
    """Check if a point (x, y) is inside the given polygon."""
    return cv2.pointPolygonTest(roi_polygon, (x, y), False) >= 0

# ==== Save the first frame with ROI and line visualization ====
ret, first_frame = cap.read()
if not ret:
    cap.release()
    out.release()
    raise RuntimeError("No readable frame found. Cannot save the first visualization frame.")

first_vis = first_frame.copy()
cv2.polylines(first_vis, [ROI_POINTS], isClosed=True, color=(0, 255, 255), thickness=2)
cv2.line(first_vis, line_point1, line_point2, color=(0, 255, 0), thickness=2)
cv2.putText(first_vis, "Crossing Line", (line_point1[0], line_point1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
cv2.putText(first_vis, "Frame 0", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

cv2.imwrite(frame0_path, first_vis)
print(f"Saved first frame visualization: {frame0_path}")

# Reset video to the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ==== State containers ====
crossing_records = {}
vehicle_last_position = {}
frame_id = 0
print("Starting video processing and overlay rendering...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[2, 5, 7],  # car, bus, truck
        verbose=False
    )

    vis_frame = frame.copy()
    # Always draw ROI and line
    cv2.polylines(vis_frame, [ROI_POINTS], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.line(vis_frame, line_point1, line_point2, color=(0, 255, 0), thickness=2)
    cv2.putText(vis_frame, "Crossing Line", (line_point1[0], line_point1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    detections = results[0].boxes
    if detections.id is not None:
        boxes = detections.xyxy.cpu().numpy()
        ids = detections.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            if not in_roi(cx, cy, ROI_POINTS):
                continue

            # Compute which side of the line the vehicle is on
            curr_side = np.sign((line_point2[0] - line_point1[0]) * (cy - line_point1[1]) -
                                (line_point2[1] - line_point1[1]) * (cx - line_point1[0]))

            prev_pos = vehicle_last_position.get(track_id)
            if prev_pos is not None:
                prev_cx, prev_cy = prev_pos
                prev_side = np.sign((line_point2[0] - line_point1[0]) * (prev_cy - line_point1[1]) -
                                    (line_point2[1] - line_point1[1]) * (prev_cx - line_point1[0]))

                # If side changes => crossing event
                if curr_side != prev_side and prev_side != 0:
                    if track_id not in crossing_records:
                        crossing_time = round(frame_id / fps, 2)
                        crossing_records[track_id] = (frame_id, crossing_time)
                        print(f"Vehicle {track_id} crossed the line at frame {frame_id}, time {crossing_time}s")

            vehicle_last_position[track_id] = (cx, cy)

            # Draw bounding box and ID
            color = (0, 0, 255) if track_id in crossing_records else (255, 255, 0)
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(vis_frame, f'ID {track_id}', (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Write the annotated frame to output video
    out.write(vis_frame)
    frame_id += 1

# ==== Finalization ====
cap.release()
out.release()

# Save crossing data to CSV
df = pd.DataFrame([
    {'vehicle_id': vid, 'cross_frame': fid, 'cross_time': time}
    for vid, (fid, time) in crossing_records.items()
])
df.to_csv(crossing_csv, index=False)

print("Video processing completed successfully.")
print(f"Crossing data saved to: {crossing_csv}")
print(f"Annotated video saved to: {output_video_path}")

