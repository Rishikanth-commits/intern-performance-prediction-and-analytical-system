import pandas as pd
import numpy as np

data = []
n = 10000

for i in range(n):

    # Performance distribution (Option B)
    performance = np.random.choice(
        ["High", "Medium", "Low"],
        p=[0.3, 0.4, 0.3]
    )

    # -------------------------------
    # Meetings-based Attendance
    # -------------------------------
    meetings_scheduled = np.random.randint(20, 50)

    if performance == "High":
        meetings_attended = int(meetings_scheduled * np.random.uniform(0.85, 1.0))
    elif performance == "Medium":
        meetings_attended = int(meetings_scheduled * np.random.uniform(0.6, 0.85))
    else:
        meetings_attended = int(meetings_scheduled * np.random.uniform(0.4, 0.65))

    attendance = int((meetings_attended / meetings_scheduled) * 100)

    # -------------------------------
    # Core Features (with overlap)
    # -------------------------------
    if performance == "High":
        sprint = np.random.randint(75, 100)
        quality = np.random.randint(75, 100)
        deadline = np.random.randint(70, 100)
        communication = np.random.randint(6, 10)
        tasks_assigned = np.random.randint(8, 15)
        tasks_completed = int(tasks_assigned * np.random.uniform(0.90, 1.0))

    elif performance == "Medium":
        sprint = np.random.randint(55, 90)
        quality = np.random.randint(55, 90)
        deadline = np.random.randint(50, 90)
        communication = np.random.randint(4, 9)
        tasks_assigned = np.random.randint(8, 15)
        tasks_completed = int(tasks_assigned * np.random.uniform(0.65, 0.90))

    else:
        sprint = np.random.randint(40, 75)
        quality = np.random.randint(45, 75)
        deadline = np.random.randint(40, 75)
        communication = np.random.randint(1, 7)
        tasks_assigned = np.random.randint(8, 15)
        tasks_completed = int(tasks_assigned * np.random.uniform(0.40, 0.65))

    # Clip values
    communication = max(1, min(10, communication))

    # -------------------------------
    # Punctuality
    # -------------------------------
    punctuality = np.random.choice([1, 0], p=[0.7, 0.3])

    # -------------------------------
    # Noise (avoid overfitting)
    # -------------------------------
    if np.random.rand() < 0.1:
        performance = np.random.choice(["High", "Medium", "Low"])

    # -------------------------------
    # Append row
    # -------------------------------
    data.append([
        meetings_scheduled, meetings_attended,
        attendance, punctuality,
        sprint, quality, deadline,
        communication,
        tasks_assigned, tasks_completed,
        performance
    ])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "Meetings_Scheduled", "Meetings_Attended",
    "Attendance", "Punctuality",
    "Sprint_Completion", "Task_Quality", "On_Time_Delivery",
    "Communication",
    "Tasks_Assigned", "Tasks_Completed",
    "Performance"
])

# Save CSV
df.to_csv("../dataset/intern_performance_10k.csv", index=False)

print("✅ Advanced 10K Dataset Generated Successfully!")
print(df.head())