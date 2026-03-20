import pandas as pd
import numpy as np

# Read the dataset
df = pd.read_csv("../dataset/intern_performance_10k.csv")

print("=" * 80)
print("DATA CLEANING SCRIPT - REMOVING INCONSISTENT RECORDS")
print("=" * 80)
print(f"\nOriginal dataset size: {len(df)} records")

# Function to validate if Performance label matches the actual metrics
def is_consistent(row):
    """
    Check if the Performance label is consistent with the actual metrics.
    High: Usually high scores in key metrics (Sprint, Quality, Communication)
    Low: Usually low scores in key metrics
    Medium: Mixed scores
    """
    performance = row['Performance']
    sprint = row['Sprint_Completion']
    quality = row['Task_Quality']
    comm = row['Communication']
    attendance = row['Attendance']
    tasks_completed = row['Tasks_Completed']
    tasks_assigned = row['Tasks_Assigned']
    on_time_delivery = row['On_Time_Delivery']
    
    if performance == 'High':
        # High performers should have high scores in at least 3 of these: Sprint, Quality, Communication
        high_score_count = sum([
            sprint >= 75,
            quality >= 75,
            comm >= 6,
            attendance >= 85,
            on_time_delivery >= 70
        ])
        return high_score_count >= 3
    
    elif performance == 'Low':
        # Low performers should have low scores in at least 3 of these: Sprint, Quality, Communication
        low_score_count = sum([
            sprint <= 60,
            quality <= 60,
            comm <= 4,
            attendance <= 60,
            on_time_delivery <= 50
        ])
        return low_score_count >= 3
    
    else:  # Medium
        # Medium performers have mixed scores (no extreme patterns)
        high_count = sum([sprint >= 75, quality >= 75, comm >= 6])
        low_count = sum([sprint <= 60, quality <= 60, comm <= 4])
        # Should not be all high or all low
        return not (high_count >= 3 or low_count >= 3)

# Apply consistency check
consistent_mask = df.apply(is_consistent, axis=1)
inconsistent_records = (~consistent_mask).sum()

print(f"\nInconsistent records found: {inconsistent_records}")
print(f"Consistent records: {(consistent_mask).sum()}")

# Show examples of removed records
if inconsistent_records > 0:
    print(f"\nExamples of REMOVED inconsistent records:")
    removed_df = df[~consistent_mask].head(10)
    for idx, row in removed_df.iterrows():
        print(f"  Label='{row['Performance']:6s}' | Sprint={row['Sprint_Completion']:3d} Quality={row['Task_Quality']:3d} Comm={row['Communication']:2d} OnTime={int(row['On_Time_Delivery'])}")

# Keep only consistent records
df_cleaned = df[consistent_mask].reset_index(drop=True)

print(f"\n✅ Cleaned dataset size: {len(df_cleaned)} records")
print(f"🗑️  Removed: {inconsistent_records} records ({(inconsistent_records/len(df)*100):.1f}%)")

# Save cleaned dataset
output_file = "../dataset/intern_performance_cleaned.csv"
df_cleaned.to_csv(output_file, index=False)

print(f"\n✅ Cleaned dataset saved to: {output_file}")

# Print summary statistics
print("\n" + "=" * 80)
print("CLEANED DATA SUMMARY")
print("=" * 80)

for perf in ['High', 'Medium', 'Low']:
    perf_df = df_cleaned[df_cleaned['Performance'] == perf]
    count = len(perf_df)
    print(f"\n{perf} Performers: {count} records ({(count/len(df_cleaned)*100):.1f}%)")
    print(f"  Avg Attendance: {perf_df['Attendance'].mean():.1f}")
    print(f"  Avg Sprint_Completion: {perf_df['Sprint_Completion'].mean():.1f}")
    print(f"  Avg Task_Quality: {perf_df['Task_Quality'].mean():.1f}")
    print(f"  Avg Communication: {perf_df['Communication'].mean():.1f}")
    print(f"  Avg On_Time_Delivery: {perf_df['On_Time_Delivery'].mean():.1f}")

print("\n" + "=" * 80)
print("DATA CLEANING COMPLETE!")
print("=" * 80)
