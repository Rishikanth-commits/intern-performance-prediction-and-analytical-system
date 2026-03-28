import pandas as pd
import numpy as np

def generate_db():
    np.random.seed(42)
    n = 100
    
    # Generate 100 intern IDs (INT001, INT002, etc.)
    intern_ids = [f"INT{str(i).zfill(3)}" for i in range(1, n+1)]
    
    # Random Indian names
    first_names = ["Ramesh", "Aneesh", "Karan", "Priya", "Arjun", "Neha", "Rahul", "Anjali", "Vikram", "Sneha"]
    last_names = ["Sharma", "Patel", "Kumar", "Singh", "Gupta", "Verma", "Reddy", "Rao", "Nair", "Das"]
    names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n)]
    
    data = []
    for i in range(n):
        # Assign a hidden random performance base to make metrics realistic
        perf = np.random.choice(["High", "Medium", "Low"], p=[0.3, 0.4, 0.3])
        
        meetings_scheduled = np.random.randint(20, 50)
        if perf == "High":
            meetings_attended = int(meetings_scheduled * np.random.uniform(0.85, 1.0))
            sprint = np.random.randint(75, 100)
            quality = np.random.randint(75, 100)
            deadline = np.random.randint(70, 100)
            comm = np.random.randint(6, 10)
            assigned = np.random.randint(8, 15)
            completed = int(assigned * np.random.uniform(0.90, 1.0))
        elif perf == "Medium":
            meetings_attended = int(meetings_scheduled * np.random.uniform(0.6, 0.85))
            sprint = np.random.randint(55, 90)
            quality = np.random.randint(55, 90)
            deadline = np.random.randint(50, 90)
            comm = np.random.randint(4, 9)
            assigned = np.random.randint(8, 15)
            completed = int(assigned * np.random.uniform(0.65, 0.90))
        else:
            meetings_attended = int(meetings_scheduled * np.random.uniform(0.4, 0.65))
            sprint = np.random.randint(40, 75)
            quality = np.random.randint(45, 75)
            deadline = np.random.randint(40, 75)
            comm = np.random.randint(1, 7)
            assigned = np.random.randint(8, 15)
            completed = int(assigned * np.random.uniform(0.40, 0.65))

        attendance = int((meetings_attended / meetings_scheduled) * 100)
        punctuality = np.random.choice([1, 0], p=[0.7, 0.3])

        data.append([intern_ids[i], names[i], meetings_scheduled, meetings_attended, attendance, 
                     punctuality, sprint, quality, deadline, comm, assigned, completed])
        
    df = pd.DataFrame(data, columns=["Intern_ID", "Name", "Meetings_Scheduled", "Meetings_Attended", 
                                     "Attendance", "Punctuality", "Sprint_Completion", "Task_Quality", 
                                     "On_Time_Delivery", "Communication", "Tasks_Assigned", "Tasks_Completed"])
    df.to_csv("../dataset/intern_database.csv", index=False)
    print(f" Generated intern_database.csv with {n} interns inside the dataset folder!")

if __name__ == "__main__":
    generate_db()