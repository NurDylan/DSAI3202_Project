"""
=============================================================
DSAI3202 - Phase 1  |  Task 4: Exploratory Analysis
Project: Predicting Crime Types by Location & Time-Series Data
Authors: Nur Afiqah (60306981), Elaf Marouf (60107174)
=============================================================

PURPOSE:
    Generates visual representation and statistical validations based on the 
    Chicago crime dataset to justify feature selection and model choice

All transformations are:
    - Automated (visuals directly to outputs/eda/)
    - Validated (checks class imbalance and geospatial outliers)
    - Interpretable (using colour-coded labels with legends, titles and axis titles)

INPUT  : data/processed/chicago_crime_cleaned_v1.0.csv
OUTPUT : outputs/eda/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def run_eda():
    # Setup Paths
    input_path = 'data/processed/chicago_crime_cleaned_v1.0.csv'
    output_dir = 'outputs/eda'
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Load Data
    print("Loading cleaned dataset...")
    df = pd.read_csv(input_path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])

# ──────────────────────────────────────────────────────────────
# STEP 1 - Data Analysis
# ──────────────────────────────────────────────────────────────
    print("Generating Crime Type Distribution...")
    plt.figure(figsize=(12, 8))
    top_crimes = df['Primary Type'].value_counts().head(15)
    sns.barplot(x=top_crimes.values, y=top_crimes.index, palette='viridis')
    plt.title('Top 15 Crime Types by Frequency')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/crime_type_distribution.png')
    plt.close()

# ──────────────────────────────────────────────────────────────
# STEP 2 - Temporal Pattern Analysis
# ──────────────────────────────────────────────────────────────
    print("Generating Temporal Analysis...")
    df['hour'] = df['Date'].dt.hour
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month

    # Hourly plot
    plt.figure(figsize=(10, 6))
    df.groupby('hour').size().plot(kind='line', marker='o')
    plt.title('Crime Frequency by Hour of Day')
    plt.grid(True)
    plt.savefig(f'{output_dir}/hourly_distribution.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    df.groupby('day_of_week').size().plot(kind='line', marker='o')
    plt.title('Crime Frequency by Day of Week')
    plt.grid(True)
    plt.savefig(f'{output_dir}/weekly_distribution.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    df.groupby('month').size().plot(kind='line', marker='o')
    plt.title('Crime Frequency by Month')
    plt.grid(True)
    plt.savefig(f'{output_dir}/monthly_distribution.png')
    plt.close()

# ──────────────────────────────────────────────────────────────
# STEP 3 - Geospatial Distribution
# ──────────────────────────────────────────────────────────────

    print("Generating Spatial Plot...")
    top5 = df['Primary Type'].value_counts().head(5).index
    subset = df[df['Primary Type'].isin(top5)]
    plt.figure(figsize=(10, 10))

    sns.scatterplot(
        data=subset, 
        x='Longitude', 
        y='Latitude', 
        hue='Primary Type', 
        palette='bright', 
        s=1,          
        alpha=0.4,    
        edgecolor=None
    )

    lgnd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Crime Types")
    for handle in lgnd.legend_handles:
        if hasattr(handle, 'set_sizes'):
            handle.set_sizes([50.0])
        elif hasattr(handle, 'set_markersize'):
            handle.set_markersize(10)
        
        handle.set_alpha(1) 

    plt.title('Spatial Distribution by Crime Type (Chicago)')
    plt.savefig(f'{output_dir}/spatial_distribution.png', bbox_inches='tight')
    plt.close()

# ──────────────────────────────────────────────────────────────
# STEP 4 - Correlation Analysis
# ──────────────────────────────────────────────────────────────

    print("Generating Correlation Heatmap...")
    num_cols = ['hour', 'day_of_week', 'month', 'Latitude', 'Longitude']
    plt.figure(figsize=(10, 8))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    plt.close()

# ──────────────────────────────────────────────────────────────
# STEP 5 - Outlier Detection
# ──────────────────────────────────────────────────────────────

    print("Generating Boxplots for Outliers...")
    plt.figure(figsize=(8, 6))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Latitude Boxplot
    sns.boxplot(y=df['Latitude'], ax=axes[0], color='skyblue')
    axes[0].set_title('Latitude Distribution')
    # Focus on Chicago's bounding box 
    axes[0].set_ylim(41.6, 42.1) 

    # Longitude Boxplot
    sns.boxplot(y=df['Longitude'], ax=axes[1], color='lightgreen')
    axes[1].set_title('Longitude Distribution')
    # Focus on Chicago's bounding box 
    axes[1].set_ylim(-88.0, -87.5)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/coordinate_boxplot.png')
    plt.close()

    print(f"Success! All plots saved to {output_dir}")

if __name__ == "__main__":
    run_eda()