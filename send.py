import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.cluster import OPTICS
from scipy.stats import poisson, expon, ttest_1samp
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from collections import deque
import networkx as nx
# from sklearn.cluster import DBSCAN  # Alternative clustering method 
import skfuzzy as fuzz
import skfuzzy.control as ctrl 

class CrowdGraph:
    def __init__(self):
        self.G = nx.Graph()

    def add_movement(self, start, end):
        self.G.add_edge(start, end, weight=1)

    def detect_choke_points(self):
        choke_points = list(nx.articulation_points(self.G))
        if choke_points:
            print(f"⚠️ Choke Points Detected: {choke_points}")
        else:
            print("✅ No major choke points detected.")

    def identify_risk_zones(self):
        centrality = nx.betweenness_centrality(self.G)
        risk_zones = {node: centrality[node] for node in centrality if centrality[node] > 0.2}
        if risk_zones:
            print(f"🔥 High-Risk Zones: {risk_zones}")
        else:
            print("✅ No high-risk zones detected.")

    def find_escape_route(self, start_zone):
        exit_nodes = ["Exit", "Emergency Exit"]
        shortest_paths = {}
        for exit in exit_nodes:
            if exit in self.G:
                try:
                    path = nx.shortest_path(self.G, source=start_zone, target=exit, weight="weight")
                    shortest_paths[exit] = path
                except nx.NetworkXNoPath:
                    pass
        if shortest_paths:
            best_exit = min(shortest_paths, key=lambda k: len(shortest_paths[k]))
            print(f"🚨 Best Escape Route: {shortest_paths[best_exit]}")
        else:
            print("❌ No available escape routes!")

class CrowdFlowPredictor:
    def __init__(self):
        self.history = []

    def update(self, current_count):
        self.history.append(current_count)
        if len(self.history) < 3:
            return current_count
        predicted_next = int(0.6 * self.history[-1] + 0.3 * self.history[-2] + 0.1 * self.history[-3])
        return predicted_next

class SymmetryAnalyzer:
    def __init__(self):
        self.movement_graph = nx.Graph()

    def add_movement(self, start, end):
        self.movement_graph.add_edge(start, end)

    def detect_symmetry(self):
        """Checks if the movement graph has symmetry by checking isomorphism."""
        gm = nx.algorithms.isomorphism.GraphMatcher(self.movement_graph, self.movement_graph)
        return gm.is_isomorphic()  # Returns True/False instead of trying to iterate

class LatticeMovement:
    def __init__(self, width, height):
        self.grid = nx.grid_2d_graph(width, height)

    def add_restriction(self, x, y):
        if (x, y) in self.grid:
            self.grid.remove_node((x, y))

    def find_best_path(self, start, end):
        if start not in self.grid or end not in self.grid:
            return "Path not possible"
        return nx.shortest_path(self.grid, source=start, target=end)

class CrowdMonitoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Crowd Monitoring System")
        self.cap = cv2.VideoCapture("Crowd Monitoring using Computer Vision!.mp4")
        if not self.cap.isOpened():
            print("Error: Could not open video file.")
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolov9c.pt").to(self.device)
        self.crowd_graph = CrowdGraph()
        self.crowd_predictor = CrowdFlowPredictor()
        self.symmetry_checker = SymmetryAnalyzer()
        self.lattice = LatticeMovement(10, 10)

        #  Add this line to initialize the crowd history
        self.crowd_history = []

        self.create_gui()
        self.update()

    def create_gui(self):
        """Create GUI layout with video + graphs."""
        self.canvas = tk.Canvas(self.root, width=600, height=400)
        self.canvas.pack(side=tk.LEFT)

        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.risk_label = ttk.Label(self.control_frame, text="Risk Level: Safe", font=("Arial", 14))
        self.risk_label.pack(pady=10)

        self.crowd_count_label = ttk.Label(self.control_frame, text="Crowd Count: 0", font=("Arial", 12))
        self.crowd_count_label.pack(pady=10)

        self.exit_button = ttk.Button(self.control_frame, text="Exit", command=self.exit)
        self.exit_button.pack(pady=10)

        # Add Matplotlib figure for graphs
        self.fig, self.axs = plt.subplots(3, 1, figsize=(5, 8))
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.control_frame)
        self.canvas_fig.get_tk_widget().pack()

    def create_graphs(self):
        """Generate real-time graphical analysis."""
        self.axs[0].clear()
        self.axs[1].clear()
        self.axs[2].clear()

        if len(self.crowd_history) > 1:
            # Graph 1: Crowd Count Over Time
            self.axs[0].plot(self.crowd_history, marker='o', linestyle='-', color='blue', label="Real-Time Count")
        
            #  Predict next crowd surge using the moving average model
            predicted_values = [self.crowd_predictor.update(c) for c in self.crowd_history]
        
            #  Plot red dashed line for predicted values
            self.axs[0].plot(predicted_values, linestyle='dashed', color='red', label="Predicted Surge")

            self.axs[0].set_title("Crowd Trend")
            self.axs[0].set_ylabel("People Count")
            self.axs[0].legend()
            self.axs[0].grid(True)

            # Graph 2: Z-Score Distribution
            if len(self.crowd_history) > 10:
                crowd_array = np.array(self.crowd_history)
                z_scores = (crowd_array - np.mean(crowd_array)) / np.std(crowd_array)
                self.axs[1].hist(z_scores, bins=10, color='orange', edgecolor='black', alpha=0.7)
                self.axs[1].axvline(2, color='red', linestyle='dashed', linewidth=1, label="High Density Alert (Z=2)")
                self.axs[1].axvline(-2, color='green', linestyle='dashed', linewidth=1, label="Low Density (Z=-2)")
                self.axs[1].legend()
                self.axs[1].set_title("Z-Score Distribution")
                self.axs[1].set_xlabel("Z-Score")
                self.axs[1].set_ylabel("Frequency")

            # Graph 3: Cluster Visualization
            x_vals = np.arange(len(self.crowd_history))
            y_vals = np.array(self.crowd_history)
            self.axs[2].scatter(x_vals, y_vals, c='purple', marker='o', alpha=0.5)
            self.axs[2].set_title("Cluster Formation")
            self.axs[2].set_xlabel("Frame Index")
            self.axs[2].set_ylabel("Crowd Size")

        self.fig.tight_layout()
        self.canvas_fig.draw()

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            self.exit()
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Fix color format
        results = self.model(frame, conf=0.25, classes=[0])
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else np.array([])

        num_people = len(boxes)

        # ✅ Store crowd count in self.crowd_history
        self.crowd_history.append(num_people)

        # ✅ Ensure history doesn't grow indefinitely
        if len(self.crowd_history) > 50:  # Keep only the last 50 values
            self.crowd_history.pop(0)

        predicted_size = self.crowd_predictor.update(num_people)
        print(f"Predicted Next Crowd Size: {predicted_size}")

        centers = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in boxes]

        if centers:
            # Apply clustering to the detected people
            optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=5)
            clusters = optics.fit_predict(centers)

            # Alternative: Using DBSCAN instead of OPTICS
            # dbscan = DBSCAN(eps=50, min_samples=5)
            # clusters = dbscan.fit_predict(centers)

            for i, center in enumerate(centers):
                cluster_id = clusters[i]
                color = (0, 255, 0) if cluster_id != -1 else (0, 0, 255)  # Green for clusters, Red for noise
                cv2.circle(frame, center, 5, color, -1)

        for i in range(len(centers) - 1):
            self.crowd_graph.add_movement(str(centers[i]), str(centers[i + 1]))
            self.symmetry_checker.add_movement(str(centers[i]), str(centers[i + 1]))

        self.crowd_graph.detect_choke_points()
        if centers:
            self.crowd_graph.find_escape_route(str(centers[0]))

        if self.symmetry_checker.detect_symmetry():
            print("⚠️ Repeating Movement Patterns Detected!")

        self.crowd_count_label.config(text=f"Crowd Count: {num_people}")

        #  Call graph update function
        self.create_graphs()

        # Fix frame rendering in Tkinter
        img = Image.fromarray(frame)
        self.imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)

        self.root.after(10, self.update)


    def exit(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CrowdMonitoringApp(root)
    root.mainloop()
