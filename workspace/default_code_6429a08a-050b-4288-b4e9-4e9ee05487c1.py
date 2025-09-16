
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import json
import glob

# Data science imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler  
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set working directory to workspace
os.chdir("/Users/linhchi/Multi-Agent-Collaboration-for-Automated-Data-Science-Workflows/workspace")
print(f"Working directory: /Users/linhchi/Multi-Agent-Collaboration-for-Automated-Data-Science-Workflows")

# Auto-discover and load dataset
dataset_files = glob.glob('*.csv') + glob.glob('*.xlsx') + glob.glob('*.parquet')
print(f"Found datasets: {dataset_files}")

# Load the first available dataset
if dataset_files:
    dataset_path = dataset_files[0]
    print(f"Loading dataset: {dataset_path}")

    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith('.xlsx'):
        df = pd.read_excel(dataset_path)
    elif dataset_path.endswith('.parquet'):
        df = pd.read_parquet(dataset_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
else:
    print("WARNING: No dataset found in workspace")
    df = None

# ========== AGENT CODE STARTS HERE ==========

# Robust categorical encoding - handles errors gracefully
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load data
df = pd.read_csv('Housing.csv')

# Identify categorical columns

try:
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"Categorical columns: {list(categorical_cols)}")

    # Encode each categorical column safely
    df_encoded = df.copy()
    for col in categorical_cols:
        try:
            le = LabelEncoder()
            df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
            df_encoded = df_encoded.drop(col, axis=1)
            print(f"Successfully encoded {col}")
        except Exception as e:
            print(f"Error encoding {col}: {e}")

    # Save cleaned dataset
    df_encoded.to_csv('cleaned_dataset.csv', index=False)
    print("Cleaned dataset saved as cleaned_dataset.csv")
except Exception as ex:
    print(f"Error: {ex}")
