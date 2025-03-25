import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.cluster import DBSCAN, OPTICS
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import time
from PIL import Image, ImageTk
from collections import deque

class CrowdMonitoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Crowd Monitoring System")

        # Open the video file
        self.video_source = "Crowd Monitoring using Computer Vision!.mp4"
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print("Error: Could not open video file.")
            return

        # Set frame dimensions (optional)
        self.cap.set(3, 800)  # Width
        self.cap.set(4, 600)  # Height

        # Initialize YOLO model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = YOLO("yolov9c.pt").to(self.device)  # Updated to YOLOv9

        # Initialize variables
        self.heatmap = None
        self.frame_counter = 0
        self.start_time = time.time()
        self.num_people_history = deque(maxlen=100)  # Store historical data
        self.crowd_threshold = 20  # Threshold for high-risk crowd density
        self.risk_level = "Safe"  # Current risk level
        self.tracking_history = {}  # For tracking individuals

        # Create GUI
        self.create_gui()

        # Start the video processing loop
        self.update()

    def create_gui(self):
        # Create a canvas for video display
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack(side=tk.LEFT)

        # Create a frame for controls
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Add a label for risk level
        self.risk_label = ttk.Label(self.control_frame, text="Risk Level: Safe", font=("Arial", 16))
        self.risk_label.pack(pady=10)

        # Add a label for crowd count
        self.crowd_count_label = ttk.Label(self.control_frame, text="Crowd Count: 0", font=("Arial", 14))
        self.crowd_count_label.pack(pady=10)

        # Add a button to exit
        self.exit_button = ttk.Button(self.control_frame, text="Exit", command=self.exit)
        self.exit_button.pack(pady=10)

        # Initialize real-time graph
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], label="Number of People")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Number of People")
        self.ax.set_title("Real-Time Crowd Density")
        self.ax.legend()
        self.graph_canvas = FigureCanvasTkAgg(self.fig, master=self.control_frame)
        self.graph_canvas.get_tk_widget().pack()

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            self.exit()
            return

        # Run YOLO model to detect pedestrians
        results = self.model(frame, conf=0.25, classes=[0])  # Detect only people (class 0)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else np.array([])
        num_people = len(boxes)

        # Update historical data
        elapsed_time = time.time() - self.start_time
        self.num_people_history.append((elapsed_time, num_people))

        # Update graph
        self.line.set_data(*zip(*self.num_people_history) if self.num_people_history else ([], []))
        self.ax.relim()
        self.ax.autoscale_view()
        self.graph_canvas.draw()

        # Process detected pedestrians
        centers = []
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box)
            if self.heatmap is None:
                self.heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            self.heatmap[ymin:ymax, xmin:xmax] += 10  # Increase intensity
            center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
            centers.append((center_x, center_y))

        # Apply OPTICS clustering to group pedestrians
        if len(centers) > 1:
            optics = OPTICS(min_samples=3, xi=0.05)  # Adjust parameters
            cluster_labels = optics.fit_predict(np.array(centers))

            # Calculate cluster statistics
            unique_clusters = set(cluster_labels)
            num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)  # Exclude noise
            num_noise = list(cluster_labels).count(-1)  # Count noise points

            # Calculate the size of each cluster
            cluster_sizes = [list(cluster_labels).count(i) for i in unique_clusters if i != -1]

            # Determine risk level based on clustering
            if num_clusters == 0:
                self.risk_level = "Safe"  # Only noise points (isolated individuals)
            else:
                max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
                if max_cluster_size > 10:  # Example threshold for large clusters
                    self.risk_level = "High Risk!"
                elif max_cluster_size > 5:
                    self.risk_level = "Medium Risk"
                else:
                    self.risk_level = "Low Risk"

        else:
            self.risk_level = "Safe"  # No clusters or only one person

        # Update risk level in the GUI
        if self.risk_level == "High Risk!":
            self.risk_label.config(text=f"Risk Level: {self.risk_level}", foreground="red")
        elif self.risk_level == "Medium Risk":
            self.risk_label.config(text=f"Risk Level: {self.risk_level}", foreground="orange")
        else:
            self.risk_label.config(text=f"Risk Level: {self.risk_level}", foreground="green")

        # Update crowd count
        self.crowd_count_label.config(text=f"Crowd Count: {num_people}")

        # Apply Gaussian blur to the heatmap
        if self.heatmap is not None:
            heatmap_blurred = cv2.GaussianBlur(self.heatmap, (51, 51), 0)
            heatmap_norm = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            final_output = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
        else:
            final_output = frame

        # Convert the image to PhotoImage
        self.frame = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)
        self.img = Image.fromarray(self.frame)
        self.imgtk = ImageTk.PhotoImage(image=self.img)

        # Update the canvas with the new image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)
        self.canvas.image = self.imgtk  # Keep a reference to avoid garbage collection

        # Repeat after 10 milliseconds
        self.root.after(10, self.update)

    def exit(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CrowdMonitoringApp(root)
    root.mainloop()