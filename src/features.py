"""
=============================================================
DSAI3202 - Phase 1  |  Task 5: Feature Extraction & Selection
Project: Predicting Crime Types by Location & Time-Series Data
Authors: Nur Afiqah (60306981), Elaf Marouf (60107174)
=============================================================

PURPOSE:
    Transform cleaned data into predictive features for Task 5.
    Focuses on temporal cycles, holiday impacts, and environmental density.

ENGINEERED FEATURES:
    - Holiday Status (is_holiday) using US Federal Holidays
    - Cyclical Time (hour_sin, hour_cos) to capture daily rhythm
    - Environmental Density (is_crowded: Secluded vs Busy)
    - Temporal Flags (is_weekend, season, is_night)

INPUT  : data/processed/chicago_crime_cleaned_v1.0.csv
OUTPUT : data/processed/chicago_crime_features_v1.0.csv
"""

import pandas as pd
import numpy as np
import os
import holidays
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'

# ──────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────

def run_feature_engineering():
    # Load Data
    input_path = 'data/processed/chicago_crime_cleaned_v1.0.csv'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        return

    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print("Extracting Temporal Features...")
    df['hour'] = df['Date'].dt.hour
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    
    # 1. Label Encoding 
    print("Encoding Target Variable...")
    le = LabelEncoder()
    df['crime_type_label'] = le.fit_transform(df['Primary Type'])
    df['community_area_enc'] = le.fit_transform(df['Community Area'].astype(str))
    
    # Save the mapping for later use in Task 6
    mapping = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
    os.makedirs('data/catalog', exist_ok=True)
    with open('data/catalog/label_encoding.json', 'w') as f:
        json.dump(mapping, f, indent=2)

    ca_mapping = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
    with open('data/catalog/community_area_mapping.json', 'w') as f:
        json.dump(ca_mapping, f, indent=2)

    # 2. Holidays & Weekends
    us_holidays = holidays.US()
    df['is_holiday'] = df['Date'].dt.date.apply(lambda x: 1 if x in us_holidays else 0)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int) # Sat, Sun per requirements
    
    # 3. Night Flag
    df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 22 or x <= 5 else 0)
    
    # 4. Season Mapping
    df['season'] = df['month'].apply(get_season)

    # 5. Cyclical Hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

    print("Extracting Environmental Features...")
    # 6. Busy vs Secluded
    busy_locs = ['STREET', 'SIDEWALK', 'DEPARTMENT STORE', 'RESTAURANT', 'GROCERY STORE', 'GAS STATION']
    
    # Convert text to numeric for Mutual Info calculation later
    df['is_crowded'] = df['Location Description'].apply(
        lambda x: 1 if any(k in str(x).upper() for k in busy_locs) else 0
    )

# ──────────────────────────────────────────────────────────────
# FEATURE SELECTION
# ──────────────────────────────────────────────────────────────

    print("Running Mutual Information Selection...")
    feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_night', 
                    'hour_sin', 'hour_cos', 'Latitude', 'Longitude', 'is_crowded', 'community_area_enc']
    
    # Run MI on a subset (10k rows) to keep it fast while meeting requirement
    sample_df = df.sample(n=min(50000, len(df)), random_state=42)
    X = sample_df[feature_cols].fillna(0)
    y = sample_df['crime_type_label']
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Keep features with MI > 0.005
    selected_features = [f for f, s in zip(feature_cols, mi_scores) if s > 0.005]
    selected_features.append('crime_type_label') # Must keep the target!

    # 7. Final Save
    df_final = df[selected_features]
    output_path = 'data/processed/chicago_crime_features_v1.0.csv'
    df_final.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"SUCCESS: Task 5 Complete!")
    print(f"Features Retained based on MI: {selected_features[:-1]}")
    print(f"Saved to: {output_path}")
    print("-" * 30)

if __name__ == "__main__":
    run_feature_engineering()