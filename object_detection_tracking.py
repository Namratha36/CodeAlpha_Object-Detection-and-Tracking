import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def main():
    print("Loading models...")
    model = YOLO('yolov8n.pt')
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Started! Press 'q' to quit.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO detection
        results = model(frame)
        result = results[0]

        # Build detections list in the correct format for DeepSORT
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                # Get bounding box coordinates
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = bbox
                
                # Get confidence and class
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                if conf > 0.5:
                    # DeepSORT expects: [[x1, y1, x2, y2], confidence, class_name]
                    detection = [[float(x1), float(y1), float(x2), float(y2)], conf, cls]
                    detections.append(detection)
                    
                    # Draw detection box (green)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Add label
                    class_name = model.names[cls]
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update tracker with detections
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw tracking results (blue boxes and IDs)
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Draw tracking box (blue)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display frame info
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Detections: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracks: {len([t for t in tracks if t.is_confirmed()])}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show frame
        cv2.imshow("Object Detection and Tracking", frame)
        
        # Quit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Program finished successfully!")

if __name__ == "__main__":
    main()
