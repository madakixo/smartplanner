import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans  # Note: Assuming sklearn is available or simulate if not
import statsmodels.api as sm
from threading import Thread
import queue
import logging
import json
import os

# Setup logging for robustness
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simulated integrations (for demo; in production, use actual APIs)
def sync_google_calendar():
    return [{"event": "Meeting", "start": "2025-12-24 10:00", "end": "2025-12-24 11:00"}]

def sync_outlook():
    return []

def sync_school_timetable():
    return [{"class": "Math", "start": "2025-12-25 09:00", "end": "2025-12-25 10:30"}]

# Simple NLP for motivational nudges (placeholder)
def generate_nudge(task):
    return f"Keep going on {task}! You've got this."

# ML Models

# 1. Time-series forecasting for task durations (using statsmodels ARIMA)
def forecast_duration(historical_data):
    if len(historical_data) < 3:
        return np.mean(historical_data) if historical_data else 60  # Default 1 hour in minutes
    series = pd.Series(historical_data)
    model = sm.tsa.ARIMA(series, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)
    return max(forecast[0], 30)  # Min 30 mins

# 2. Clustering for prioritization (using KMeans)
def prioritize_tasks(tasks):
    if not tasks:
        return []
    data = np.array([[t['urgency'], t['importance']] for t in tasks])
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    prioritized = sorted(tasks, key=lambda t: labels[tasks.index(t)], reverse=True)
    return prioritized

# 3. Reinforcement Learning for adaptation (simple Q-learning with torch)
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, state):
        return self.fc(state)

class RLAdapter:
    def __init__(self):
        self.state_size = 3  # e.g., completion rate, stress, productivity
        self.action_size = 4  # e.g., add break, shift task, extend time, no change
        self.q_net = QNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.001)
        self.gamma = 0.99
        self.epsilon = 0.1
    
    def adapt_schedule(self, state, reward):
        state_tensor = torch.FloatTensor(state)
        q_values = self.q_net(state_tensor)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = torch.argmax(q_values).item()
        # Simulate adaptation
        if action == 0:
            return "Add break"
        elif action == 1:
            return "Shift task"
        elif action == 2:
            return "Extend time"
        else:
            return "No change"
        
        # Update Q-network (simplified)
        next_state = state  # Placeholder
        next_q = self.q_net(torch.FloatTensor(next_state)).max()
        target = reward + self.gamma * next_q
        loss = nn.MSELoss()(q_values[action], target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Data persistence for scaling (use JSON file; in prod, use DB)
DATA_FILE = 'smartplanner_data.json'

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {"tasks": [], "historical": {}, "preferences": {}, "feedback": []}

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)

# Main Scheduler Logic
def generate_schedule(tasks, preferences, historical, calendars):
    # Forecast durations
    for task in tasks:
        hist = historical.get(task['name'], [])
        task['duration'] = forecast_duration(hist)
    
    # Prioritize
    prioritized_tasks = prioritize_tasks(tasks)
    
    # Build schedule
    start_date = datetime.now()
    schedule = []
    current_time = start_date
    for task in prioritized_tasks:
        # Check for conflicts with calendars
        while True:
            end_time = current_time + timedelta(minutes=task['duration'])
            conflict = False
            for cal in calendars:
                cal_start = datetime.strptime(cal['start'], "%Y-%m-%d %H:%M")
                cal_end = datetime.strptime(cal['end'], "%Y-%m-%d %H:%M")
                if max(current_time, cal_start) < min(end_time, cal_end):
                    conflict = True
                    break
            if not conflict:
                break
            current_time += timedelta(minutes=30)  # Shift by 30 mins
        
        schedule.append({
            "task": task['name'],
            "start": current_time.strftime("%Y-%m-%d %H:%M"),
            "end": end_time.strftime("%Y-%m-%d %H:%M"),
            "nudge": generate_nudge(task['name'])
        })
        
        # Add break if preferred
        if preferences.get('breaks', True):
            current_time = end_time + timedelta(minutes=10)
        else:
            current_time = end_time
    
    return schedule

# Streamlit UI
def main():
    st.title("SmartPlanner: AI-Driven Scheduling Tool")
    st.subheader("Optimize Your Time for Work or Study")
    
    data = load_data()
    
    # User Type
    user_type = st.selectbox("Are you a Professional or Student?", ["Professional", "Student"])
    
    # Preferences
    with st.expander("Set Preferences"):
        peak_hours = st.text_input("Peak Focus Hours (e.g., 10:00-12:00)")
        breaks = st.checkbox("Include Breaks", value=True)
        data['preferences'] = {"peak_hours": peak_hours, "breaks": breaks}
    
    # Input Tasks
    st.subheader("Add Tasks")
    task_name = st.text_input("Task Name")
    deadline = st.date_input("Deadline")
    urgency = st.slider("Urgency (1-10)", 1, 10, 5)
    importance = st.slider("Importance (1-10)", 1, 10, 5)
    if st.button("Add Task"):
        data['tasks'].append({
            "name": task_name,
            "deadline": str(deadline),
            "urgency": urgency,
            "importance": importance
        })
        save_data(data)
        st.success("Task added!")
    
    # Historical Data (simulated input)
    st.subheader("Historical Productivity (Durations in mins)")
    hist_task = st.text_input("Task for Historical Data")
    hist_dur = st.number_input("Duration", min_value=0)
    if st.button("Add Historical"):
        if hist_task not in data['historical']:
            data['historical'][hist_task] = []
        data['historical'][hist_task].append(hist_dur)
        save_data(data)
    
    # Sync Calendars
    if st.button("Sync Calendars"):
        calendars = sync_google_calendar() + sync_outlook() + (sync_school_timetable() if user_type == "Student" else [])
        st.info(f"Synced {len(calendars)} events.")
    else:
        calendars = []
    
    # Generate Schedule
    if st.button("Generate Schedule"):
        with st.spinner("Optimizing with AI..."):
            schedule = generate_schedule(data['tasks'], data['preferences'], data['historical'], calendars)
        
        st.subheader("Your Personalized Schedule")
        for item in schedule:
            st.write(f"**{item['task']}**: {item['start']} - {item['end']}")
            st.info(item['nudge'])
        
        # RL Adaptation (simulated feedback)
        st.subheader("Provide Feedback")
        completion_rate = st.slider("Completion Rate (%)", 0, 100, 80)
        stress = st.slider("Stress Level (1-10)", 1, 10, 3)
        productivity = st.slider("Productivity (1-10)", 1, 10, 8)
        if st.button("Submit Feedback"):
            rl = RLAdapter()
            state = [completion_rate / 100, stress / 10, productivity / 10]
            reward = productivity - stress  # Simple reward
            adaptation = rl.adapt_schedule(state, reward)
            data['feedback'].append({"state": state, "adaptation": adaptation})
            save_data(data)
            st.success(f"Schedule adapted: {adaptation}")
    
    # For scaling: Run heavy computations in threads
    def heavy_computation(q):
        # Simulate long-running ML training
        time.sleep(5)  # Placeholder
        q.put("Computation done")
    
    if st.button("Run Scalable Computation (Demo)"):
        q = queue.Queue()
        thread = Thread(target=heavy_computation, args=(q,))
        thread.start()
        st.write("Running in background...")
        thread.join()
        result = q.get()
        st.success(result)

if __name__ == "__main__":
    main()
