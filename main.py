import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull


def detect_pedestrians_heatmap():
    # Open the video file
    cap = cv2.VideoCapture("Crowd Monitoring using Computer Vision!.mp4")
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Set the frame dimensions (optional)
    cap.set(3, 800)  # Width
    cap.set(4, 600)  # Height

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the YOLO model
    model = YOLO("yolov8x.pt").to(device)

    # Read the first frame to initialize variables
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    frame_height, frame_width, _ = frame.shape
    heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Run YOLO model to detect pedestrians
        results = model.predict(frame, conf=0.25, classes=[0], device=device, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # Decay the heatmap
        heatmap *= 0.5

        # Process detected pedestrians
        centers = []
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box)
            heatmap[ymin:ymax, xmin:xmax] += 10  # Increase intensity
            center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
            centers.append((center_x, center_y))

        # Apply DBSCAN clustering to group pedestrians
        if len(centers) > 1:
            dbscan = DBSCAN(eps=100, min_samples=3)  # Adjust parameters
            cluster_labels = dbscan.fit_predict(np.array(centers))

            # Draw group boundaries
            unique_clusters = set(cluster_labels)
            for cluster in unique_clusters:
                if cluster == -1:  # Ignore noise (outliers)
                    continue

                cluster_points = [
                    centers[i]
                    for i in range(len(center))
                    if cluster_labels[i] == cluster
                ]

                if len(cluster_points) > 2:
                    hull = ConvexHull(cluster_points)
                    hull_points = np.array(
                        [cluster_points[i] for i in hull.vertices], np.int32
                    )
                    hull_points = hull_points.reshape((-1, 1, 2))  # Reshape for cv2.polylines
                    cv2.polylines(
                        frame,
                        [hull_points],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2,
                    )
                else:
                    for point in cluster_points:
                        cv2.circle(
                            frame, point, 20, (0, 255, 0), 2
                        )  # Mark small groups

        # Apply Gaussian blur to the heatmap
        heatmap_blurred = cv2.GaussianBlur(heatmap, (51, 51), 0)

        # Normalize and apply color mapping
        heatmap_norm = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        # Blend the heatmap with the original frame
        final_output = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)

        # Display the output
        cv2.imshow("Crowd Heatmap with Groups", final_output)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_pedestrians_heatmap()