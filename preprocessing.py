"""
Preprocessing and Feature Transformation Functions
Converts real-world metrics to model format
"""

import pandas as pd
import numpy as np


def transform_real_data_to_model_format(real_data):
    """
    Convert real-world intern metrics to model format
    
    Parameters:
    -----------
    real_data : dict
        Dictionary containing real-world metrics
        {
            'sprints_done': int,
            'sprints_total': int,
            'code_review_score': float (0-100),
            'meetings_attended': int,
            'meetings_total': int,
            'tasks_assigned': int,
            'tasks_completed': int,
            'deadline_met_percentage': float (0-100),
            'attendance_percentage': float (0-100),
            'punctuality': int (0 or 1),
            'communication_score': int (1-10)
        }
    
    Returns:
    --------
    dict : Transformed features in model format
    """
    
    # Sprint Completion (0-100 scale)
    sprint_completion = (real_data['sprints_done'] / real_data['sprints_total']) * 100
    
    # Task Quality (already 0-100)
    task_quality = real_data['code_review_score']
    
    # Communication (1-10 scale)
    communication = (real_data['meetings_attended'] / real_data['meetings_total']) * 10
    
    # Task Completion Rate for Tasks_Completed
    tasks_completed = real_data['tasks_completed']
    
    # Deadline Met (already 0-100)
    deadline_met = real_data['deadline_met_percentage']
    
    # Attendance (already 0-100)
    attendance = real_data['attendance_percentage']
    
    # Punctuality (0 or 1)
    punctuality = real_data['punctuality']
    
    # Meeting Attendance Rate (1-10 scale, but we return raw meetings)
    meetings_attended_scaled = (real_data['meetings_attended'] / real_data['meetings_total']) * 10
    meetings_scheduled = real_data['meetings_total']
    meetings_attended = real_data['meetings_attended']
    
    return {
        'Sprint_Completion': sprint_completion,
        'Task_Quality': task_quality,
        'Communication': meetings_attended_scaled,
        'Tasks_Assigned': real_data['tasks_assigned'],
        'Tasks_Completed': tasks_completed,
        'On_Time_Delivery': deadline_met,
        'Attendance': attendance,
        'Punctuality': punctuality,
        'Meetings_Scheduled': meetings_scheduled,
        'Meetings_Attended': meetings_attended,
        'Meeting_Attendance_Percentage': (real_data['meetings_attended'] / real_data['meetings_total']) * 100
    }


def apply_feature_engineering(df):
    """
    Apply feature engineering transformations to dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with original features
    
    Returns:
    --------
    pd.DataFrame : DataFrame with engineered features
    """
    
    df_engineered = df.copy()
    
    # 1. Task Completion Rate (0-1 scale)
    df_engineered['Task_Completion_Rate'] = df_engineered['Tasks_Completed'] / df_engineered['Tasks_Assigned']
    
    # 2. On-Time Delivery Rate (0-1 scale)
    df_engineered['On_Time_Delivery_Rate'] = df_engineered['On_Time_Delivery'] / 100
    
    # 3. Overall Quality Score (average of quality metrics)
    df_engineered['Overall_Quality_Score'] = (
        df_engineered['Task_Quality'] + 
        df_engineered['Sprint_Completion']
    ) / 2
    
    # 4. Communication Efficiency (Communication normalized 0-1)
    df_engineered['Communication_Level'] = df_engineered['Communication'] / 10
    
    # 5. Punctuality Score
    df_engineered['Is_Punctual'] = df_engineered['Punctuality']
    
    # 6. Meeting Attendance Score (derived from Meetings_Attended and Meetings_Scheduled)
    df_engineered['Meeting_Attendance_Rate'] = (
        df_engineered['Meetings_Attended'] / df_engineered['Meetings_Scheduled']
    ) * 100
    
    # 7. Combined Performance Indicator
    df_engineered['Performance_Index'] = (
        (df_engineered['Sprint_Completion'] / 100 * 0.25) +
        (df_engineered['Task_Quality'] / 100 * 0.25) +
        (df_engineered['Task_Completion_Rate'] * 0.25) +
        (df_engineered['On_Time_Delivery_Rate'] * 0.25)
    ) * 100
    
    # 8. Attendance Consistency (Attendance + Punctuality)
    df_engineered['Attendance_Consistency'] = (
        (df_engineered['Attendance'] / 100) + df_engineered['Punctuality']
    ) / 2
    
    return df_engineered


def normalize_features(df, feature_columns):
    """
    Normalize features to 0-1 scale using Min-Max normalization
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    feature_columns : list
        List of column names to normalize
    
    Returns:
    --------
    pd.DataFrame : DataFrame with normalized features
    """
    
    df_normalized = df.copy()
    
    for col in feature_columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        
        if max_val - min_val != 0:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0
    
    return df_normalized


def validate_real_data(real_data):
    """
    Validate that real-world data has all required fields
    
    Parameters:
    -----------
    real_data : dict
        Real-world data dictionary
    
    Returns:
    --------
    bool : True if valid, raises exception otherwise
    """
    
    required_fields = [
        'sprints_done', 'sprints_total',
        'code_review_score',
        'meetings_attended', 'meetings_total',
        'tasks_assigned', 'tasks_completed',
        'deadline_met_percentage',
        'attendance_percentage',
        'punctuality',
    ]
    
    missing_fields = [field for field in required_fields if field not in real_data]
    
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Validate ranges
    if not (0 <= real_data['code_review_score'] <= 100):
        raise ValueError("code_review_score must be between 0 and 100")
    
    if not (0 <= real_data['deadline_met_percentage'] <= 100):
        raise ValueError("deadline_met_percentage must be between 0 and 100")
    
    if not (0 <= real_data['attendance_percentage'] <= 100):
        raise ValueError("attendance_percentage must be between 0 and 100")
    
    if real_data['punctuality'] not in [0, 1]:
        raise ValueError("punctuality must be 0 or 1")
    
    return True


# Example usage
if __name__ == "__main__":
    # Example: Real-world data from a company
    real_intern_data = {
        'sprints_done': 3,
        'sprints_total': 4,
        'code_review_score': 92,
        'meetings_attended': 15,
        'meetings_total': 20,
        'tasks_assigned': 10,
        'tasks_completed': 9,
        'deadline_met_percentage': 85,
        'attendance_percentage': 95,
        'punctuality': 1,
    }
    
    # Validate data
    validate_real_data(real_intern_data)
    
    # Transform to model format
    transformed = transform_real_data_to_model_format(real_intern_data)
    
    print("Transformed Real-World Data:")
    for key, value in transformed.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
