import streamlit as st
import json
import time
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Real-time Crowd Monitoring Dashboard")
st.subheader("👥 Live Crowd Density Tracking with Alerts")

# Layout for Graph & Heatmap
col1, col2 = st.columns(2)

# Live count display
crowd_placeholder = st.empty()
alert_placeholder = st.empty()

# Initialize crowd data
crowd_data = []

def read_crowd_data():
    """Reads the latest crowd data from JSON file."""
    try:
        with open("crowd_data.json", "r") as file:
            data = json.load(file)
        return data.get("crowd_count", 0), data.get("timestamp", time.time())
    except:
        return 0, time.time()

while True:
    count, timestamp = read_crowd_data()
    
    # Append to list for graphing
    crowd_data.append({"timestamp": timestamp, "count": count})
    df = pd.DataFrame(crowd_data)

    # Display latest count
    crowd_placeholder.write(f"### 👥 Current Crowd Count: {count}")

    # Generate Alert if Crowd is High
    if count > 15:  # Adjust threshold as needed
        alert_placeholder.error("🚨 Alert! High crowd density detected! 🚨")

    # Plot Graph
    col1.subheader("📈 Crowd Density Over Time")
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["count"], marker='o', linestyle='-', color='b')
    ax.set_xlabel("Time")
    ax.set_ylabel("Crowd Count")
    ax.set_title("Crowd Density Over Time")
    ax.grid(True)
    col1.pyplot(fig)

    # Show Heatmap
    col2.subheader("🔥 Live Crowd Heatmap")
    col2.image("heatmap.jpg", use_column_width=True)

    time.sleep(1)  # Update every second
