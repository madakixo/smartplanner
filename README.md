# smartplanner
A basic tool for planning activiies to boost personal productivity and rlate it to time management using machine learning

# SmartPlanner: AI-Driven Productivity Scheduler

**SmartPlanner** is a powerful, intelligent scheduling application built with Streamlit that helps **students** and **professionals** manage their time effectively using AI, productivity science, and personalized planning.

It combines machine learning for task prioritization and duration prediction, supports proven focus techniques (90-minute Sage Mode and Pomodoro), enforces deadlines, balances workload (especially credit units for students), and provides a fully editable, shareable timetable.

## Features

- **AI-Powered Prioritization**  
  Uses K-Means clustering and urgency/importance scoring to rank tasks intelligently.

- **Smart Duration Forecasting**  
  Learns from your past task completion times using ARIMA time-series modeling.

- **Two Focus Modes**  
  - **90-Minute Sage Mode** – Deep, uninterrupted focus blocks aligned to your peak hours (ideal for complex learning).  
  - **Pomodoro Technique** – 25-minute focus sessions with 5/15-minute breaks.

- **Deadline Enforcement & Workload Balancing**  
  Ensures tasks finish before due dates and respects daily credit unit limits (for students).

- **Fixed Workday Schedule**  
  Automatically schedules within 8:00 AM – 5:00 PM and rolls over excess tasks to the next day.

- **Calendar Conflict Avoidance**  
  Respects meetings, classes, and fixed events (simulated integration; ready for real APIs).

- **Intuitive Editable Timetable**  
  View and manually adjust your schedule in an interactive data table.

- **Export & Share**  
  Download your schedule as CSV or JSON for printing or sharing.

- **AI Feedback Learning**  
  Rate your day — a lightweight reinforcement learning model adapts future suggestions.

- **Persistent Local Storage**  
  All tasks, preferences, and history saved in `smartplanner_data.json`.

## Screenshots

*(Add screenshots here after running the app)*

## Installation & Setup

### Requirements
- Python 3.8+
- pip

### Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn statsmodels torch



Run the Appbash

streamlit run smartplanner.py

The app will open in your default browser (usually at http://localhost:8501).How to UseSelect User Type – Choose Student or Professional.
Set Preferences (expand the section):Define your Peak Focus Hours (e.g., 10:00-12:00)
Enable breaks
Choose Sage Mode (deep work) or Pomodoro
(Students) Set max daily credit units

Add Tasks – Include name, deadline, urgency, importance, and (for students) credit units.
(Optional) Record historical task durations for better predictions.
Sync Calendars – Load fixed events (simulated for now).
Generate Schedule – Click the button to create your optimized plan.
Edit & Refine – Adjust times directly in the interactive table.
Export – Download as CSV or JSON.
Provide Feedback – Help the AI improve for next time.

File Structure

.
├── smartplanner.py              # Main application code
├── smartplanner_data.json       # Auto-generated: stores tasks, preferences, history
├── schedule.csv                 # Exported schedule (when downloaded)
├── schedule.json                # Exported schedule (when downloaded)
└── README.md                    # This file

ContributingContributions are welcome! Feel free to:Open issues for bugs or feature requests
Submit pull requests with improvements

Please follow standard GitHub flow.LicenseThis project is licensed under the MIT License - see the LICENSE file for details.plaintext

MIT License

Copyright (c) 2025 Jamaludeen Madaki or "SmartPlanner Contributors*"

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Acknowledgements
Built with Streamlit
Machine learning powered by scikit-learn, statsmodels, and PyTorch
Inspired by productivity methods: Pomodoro Technique and ultradian rhythm deep work 

Focus better. Achieve more. Stress less.
Made with  for students and professionals everywhere.

