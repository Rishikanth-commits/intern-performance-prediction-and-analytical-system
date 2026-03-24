import pandas as pd

def apply_feature_engineering(df):
    """
    Applies feature engineering to the dataset to create new, more powerful features.
    """
    df_engineered = df.copy()
    df_engineered['Task_Completion_Rate'] = df_engineered['Tasks_Completed'] / df_engineered['Tasks_Assigned']
    df_engineered['On_Time_Delivery_Rate'] = df_engineered['On_Time_Delivery'] / 100
    df_engineered['Overall_Quality_Score'] = (
        df_engineered['Task_Quality'] + 
        df_engineered['Sprint_Completion']
    ) / 2
    df_engineered['Communication_Level'] = df_engineered['Communication'] / 10
    df_engineered['Is_Punctual'] = df_engineered['Punctuality']
    df_engineered['Meeting_Attendance_Rate'] = (
        df_engineered['Meetings_Attended'] / df_engineered['Meetings_Scheduled']
    ) * 100
    df_engineered['Performance_Index'] = (
        (df_engineered['Sprint_Completion'] / 100 * 0.25) +
        (df_engineered['Task_Quality'] / 100 * 0.25) +
        (df_engineered['Task_Completion_Rate'] * 0.25) +
        (df_engineered['On_Time_Delivery_Rate'] * 0.25)
    ) * 100
    df_engineered['Attendance_Consistency'] = (
        (df_engineered['Attendance'] / 100) + df_engineered['Punctuality']
    ) / 2
    
    return df_engineered