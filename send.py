import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from scipy.spatial import Voronoi, distance
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from collections import deque
import networkx as nx
import skfuzzy as fuzz
import time

class CrowdGraph:
    def __init__(self):
        self.G = nx.Graph()
        self.node_positions = {}

    def add_movement(self, start, end):
        if start not in self.node_positions:
            self.node_positions[start] = eval(start)
        if end not in self.node_positions:
            self.node_positions[end] = eval(end)
        if self.G.has_edge(start, end):
            self.G[start][end]['weight'] += 1
        else:
            self.G.add_edge(start, end, weight=1)

    def detect_choke_points(self):
        choke_points = list(nx.articulation_points(self.G))
        return choke_points

    def identify_risk_zones(self):
        centrality = nx.betweenness_centrality(self.G)
        risk_zones = {node: centrality[node] for node in centrality if centrality[node] > 0.2}
        return risk_zones

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
        self.pattern_history = deque(maxlen=10)

    def add_movement(self, start, end):
        self.movement_graph.add_edge(start, end)
        self.pattern_history.append((start, end))

    def detect_symmetry(self):
        if len(self.pattern_history) < 5:
            return False

        # Check for repeating sequences
        last_five = list(self.pattern_history)[-5:]
        if last_five.count(last_five[0]) >= 4:
            return True

        gm = nx.algorithms.isomorphism.GraphMatcher(self.movement_graph, self.movement_graph)
        return gm.is_isomorphic()

class CrowdAnalyzer:
    def __init__(self):
        self.people_positions = []
        self.cluster_centers = []
        self.low_density_zones = []
        self.voronoi_regions = []

    def update_positions(self, positions):
        self.people_positions = positions

    def detect_clusters(self, eps=100, min_samples=5):
        if len(self.people_positions) < 2:
            return []

        X = np.array(self.people_positions)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = clustering.labels_

        self.cluster_centers = []
        for k in set(labels):
            if k != -1:
                class_member_mask = (labels == k)
                self.cluster_centers.append(np.mean(X[class_member_mask], axis=0))

        return self.cluster_centers

    def calculate_voronoi_density(self, frame_shape):
        if len(self.people_positions) < 3:
            return []

        X = np.array(self.people_positions)
        h, w = frame_shape[:2]
        boundary_points = [[0,0], [0,h], [w,0], [w,h], [w/2,0], [w/2,h], [0,h/2], [w,h/2]]
        points = np.vstack([X, boundary_points])

        vor = Voronoi(points)
        self.voronoi_regions = []

        # Calculate density for each Voronoi region
        for i, region in enumerate(vor.regions):
            if not region or -1 in region:
                continue
                
            polygon = [vor.vertices[i] for i in region]
            area = cv2.contourArea(np.array(polygon, dtype=np.float32))
            
            # Count people in this region
            region_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
            cv2.fillPoly(region_mask, [np.array(polygon, dtype=np.int32)], 255)
            
            people_in_region = 0
            for person in self.people_positions:
                if region_mask[int(person[1]), int(person[0])] == 255:
                    people_in_region += 1
            
            density = people_in_region / (area + 1e-5)  # Avoid division by zero
            
            self.voronoi_regions.append({
                'polygon': polygon,
                'density': density,
                'area': area,
                'people_count': people_in_region
            })

        return self.voronoi_regions

    def visualize_voronoi_density(self, frame):
        if not self.voronoi_regions:
            return frame

        # Calculate density percentiles for coloring
        densities = [r['density'] for r in self.voronoi_regions]
        if densities:
            low_thresh = np.percentile(densities, 33)
            high_thresh = np.percentile(densities, 66)

        for region in self.voronoi_regions:
            polygon = np.array(region['polygon'], dtype=np.int32)
            
            # Determine color based on density
            if region['density'] < low_thresh:
                color = (0, 255, 0)  # Green - low density
            elif region['density'] < high_thresh:
                color = (0, 255, 255)  # Yellow - medium density
            else:
                color = (0, 0, 255)  # Red - high density
            
            # Draw the region with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Draw the region boundaries
            cv2.polylines(frame, [polygon], True, (255, 255, 255), 1)

        return frame

class ToastMessage(tk.Toplevel):
    def __init__(self, parent, message, duration=3000):
        super().__init__(parent)
        self.message = message
        self.duration = duration
        self.geometry(f"+{parent.winfo_rootx() + 50}+{parent.winfo_rooty() + 50}")
        self.overrideredirect(True)
        self.label = ttk.Label(self, text=self.message, padding=(10, 5))
        self.label.pack()
        self.after(self.duration, self.destroy)

class CrowdMonitoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crowd Monitoring System")

        # Try multiple video sources
        video_sources = [
            "Crowd Monitoring using Computer Vision!.mp4",
            0  # Webcam fallback
        ]

        self.cap = None
        for source in video_sources:
            self.cap = cv2.VideoCapture(source)
            if self.cap.isOpened():
                print(f"Using video source: {source}")
                break

        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open any video source")
            self.root.destroy()
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolov8n.pt").to(self.device)
        self.crowd_graph = CrowdGraph()
        self.crowd_predictor = CrowdFlowPredictor()
        self.symmetry_checker = SymmetryAnalyzer()
        self.crowd_analyzer = CrowdAnalyzer()
        self.crowd_history = []
        self.symmetry_persistence = 0  # Initialize symmetry persistence counter

        # Alert system
        self.alert_active = False
        self.alert_start_time = 0
        self.alert_duration = 3000  # 3 seconds

        self.create_gui()
        self.update()

    def create_gui(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.pack(pady=10)

        self.canvas = tk.Canvas(self.video_frame, width=800, height=450)
        self.canvas.pack()

        self.graphs_frame = ttk.Frame(self.main_frame)
        self.graphs_frame.pack(fill=tk.BOTH, expand=True)

        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        self.fig, self.axs = plt.subplots(1, 4, figsize=(16, 4))
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.graphs_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.risk_label = ttk.Label(self.control_frame, text="Risk Level: Safe", font=("Arial", 14))
        self.risk_label.pack(pady=10)

        self.exit_button = ttk.Button(self.control_frame, text="Exit", command=self.exit)
        self.exit_button.pack(pady=10)

    def compute_fuzzy_risk(self, count):
        x_people = np.arange(0, 201, 1)
        low_risk = fuzz.trimf(x_people, [0, 0, 5])
        medium_risk = fuzz.trimf(x_people, [3, 7, 12])
        high_risk = fuzz.trimf(x_people, [8, 15, 20])

        low_val = fuzz.interp_membership(x_people, low_risk, count)
        medium_val = fuzz.interp_membership(x_people, medium_risk, count)
        high_val = fuzz.interp_membership(x_people, high_risk, count)

        return low_val, medium_val, high_val

    def create_graphs(self):
        for ax in self.axs:
            ax.clear()

        if len(self.crowd_history) > 1:
            # Graph 1: Crowd Trend
            self.axs[0].plot(self.crowd_history, 'b-', label="Real-Time Count")
            predicted_values = [self.crowd_predictor.update(c) for c in self.crowd_history]
            self.axs[0].plot(predicted_values, 'r--', label="Predicted Surge")
            self.axs[0].set_title("Crowd Trend")
            self.axs[0].legend()
            self.axs[0].grid(True)

            # Graph 2: Histogram with KDE
            if len(self.crowd_history) > 10:
                import seaborn as sns
                sns.histplot(self.crowd_history, kde=True, ax=self.axs[1], color='blue')
                self.axs[1].set_title("Crowd Distribution")
                self.axs[1].set_xlabel("Number of People")
                self.axs[1].set_ylabel("Frequency")
                self.axs[1].grid(True)
            
                # Add vertical lines for mean and standard deviations
                mean = np.mean(self.crowd_history)
                std = np.std(self.crowd_history)
                self.axs[1].axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.1f}')
                self.axs[1].axvline(mean + std, color='g', linestyle=':', label=f'+1 Std Dev')
                self.axs[1].axvline(mean - std, color='g', linestyle=':', label=f'-1 Std Dev')
                self.axs[1].legend()

        # Graph 3: Density Visualization
        if hasattr(self.crowd_analyzer, 'voronoi_regions') and self.crowd_analyzer.voronoi_regions:
            densities = [r['density'] for r in self.crowd_analyzer.voronoi_regions]
            if densities:
                low_thresh = np.percentile(densities, 33)
                high_thresh = np.percentile(densities, 66)
                
                for region in self.crowd_analyzer.voronoi_regions:
                    polygon = np.array(region['polygon'])
                    if region['density'] < low_thresh:
                        color = 'green'
                    elif region['density'] < high_thresh:
                        color = 'yellow'
                    else:
                        color = 'red'
                    
                    self.axs[2].fill(*zip(*polygon), color=color, alpha=0.3)
                    self.axs[2].plot(*zip(*polygon, polygon[0]), color='black', linewidth=1)
            
            # Plot people positions
            if self.crowd_analyzer.people_positions:
                X = np.array(self.crowd_analyzer.people_positions)
                self.axs[2].scatter(X[:,0], X[:,1], c='blue', s=10, alpha=0.7)
            
            self.axs[2].set_title("Density Zones Visualization")
            self.axs[2].set_xlim(0, self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.axs[2].set_ylim(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 0)
            self.axs[2].set_aspect('equal')

        # Graph 4: Risk Assessment
        current_count = self.crowd_history[-1] if self.crowd_history else 0
        low, med, high = self.compute_fuzzy_risk(current_count)
        total = low + med + high
        percentages = [low/total*100, med/total*100, high/total*100] if total > 0 else [0, 0, 0]

        bars = self.axs[3].bar(['Low', 'Medium', 'High'], percentages, color=['g', 'orange', 'r'])
        self.axs[3].set_title("Risk Assessment (%)")
        self.axs[3].set_ylim(0, 100)
        for bar in bars:
            height = bar.get_height()
            self.axs[3].text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}%',
                            ha='center', va='bottom')

        # Update risk label
        max_risk = max(low, med, high)
        if max_risk == low:
            self.risk_label.config(text="Risk Level: Low", background="green")
        elif max_risk == med:
            self.risk_label.config(text="Risk Level: Medium", background="orange")
        else:
            self.risk_label.config(text="Risk Level: High", background="red")

        self.fig.tight_layout()
        self.canvas_fig.draw()

    def show_alert(self, message):
        ToastMessage(self.root, message)

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            self.exit()
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect people
        results = self.model(frame, conf=0.25, classes=[0])
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else np.array([])
        centers = [(int((x1 + x2) / 2), int((y1 + y2)/ 2)) for x1, y1, x2, y2 in boxes]

        # Update systems
        num_people = len(centers)
        self.crowd_history.append(num_people)
        if len(self.crowd_history) > 50:
            self.crowd_history.pop(0)

        self.crowd_analyzer.update_positions(centers)
        self.crowd_analyzer.detect_clusters()
        self.crowd_analyzer.calculate_voronoi_density(frame.shape)
        frame = self.crowd_analyzer.visualize_voronoi_density(frame)

        # Update movement graph and check for symmetry
        symmetry_detected = self.symmetry_checker.detect_symmetry()
        if symmetry_detected:
            self.symmetry_persistence += 1
            if self.symmetry_persistence == 5 and not self.alert_active:
                self.alert_active = True
                self.alert_start_time = time.time() * 1000
                self.show_alert("Repeating Movement Pattern Detected!")
            elif self.symmetry_persistence > 5 and (self.symmetry_persistence - 5) % 3 == 0 and not self.alert_active:
                self.alert_active = True
                self.alert_start_time = time.time() * 1000
                self.show_alert("Repeating Movement Pattern Persisting!")
        else:
            self.symmetry_persistence = 0
            self.alert_active = False

        for i in range(len(centers) - 1):
            self.crowd_graph.add_movement(str(centers[i]), str(centers[i + 1]))
            self.symmetry_checker.add_movement(str(centers[i]), str(centers[i + 1]))

        # Detect choke points (bottlenecks) and visualize as dots
        choke_points = self.crowd_graph.detect_choke_points()
        for point in choke_points:
            try:
                x, y = map(int, point.strip("()").split(","))
                dot_color = (0, 255, 0)  # Green by default
                if self.symmetry_persistence >= 5:
                    dot_color = (255, 165, 0)  # Orange if repeating 5 times
                if self.symmetry_persistence > 5 and (self.symmetry_persistence - 5) % 3 == 0:
                    dot_color = (255, 0, 0)  # Red if persisting for >3 consecutive times

                cv2.circle(frame, (x, y), 5, dot_color, -1)
            except:
                continue

        # Handle alert timeout (using the toast message duration)
        if self.alert_active and (time.time() * 1000 - self.alert_start_time) > self.alert_duration:
            self.alert_active = False

        # Update GUI
        self.create_graphs()

        img = Image.fromarray(frame)
        img = img.resize((800, 450), Image.LANCZOS)
        self.imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)

        self.root.after(30, self.update)
    
    def exit(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.state('zoomed')
    app = CrowdMonitoringApp(root)
    root.mainloop()
