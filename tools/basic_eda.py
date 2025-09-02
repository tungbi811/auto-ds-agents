import pandas as pd
import numpy as np
import io
import sys

def analyze_csv(csv_file_path):
    """Perform basic EDA and return results as string"""
    try:
        df = pd.read_csv(csv_file_path)
        
        # Capture output in a string buffer
        output = io.StringIO()
        
        # Basic dataset info
        output.write("=== DATASET OVERVIEW ===\n")
        output.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
        output.write(f"Column names: {list(df.columns)}\n\n")
        
        # Data types and missing values
        output.write("=== DATA TYPES & MISSING VALUES ===\n")
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            output.write(f"{col}: {df[col].dtype}, {missing_pct:.1f}% missing\n")
        output.write("\n")
        
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            output.write("=== NUMERIC COLUMNS SUMMARY ===\n")
            summary = df[numeric_cols].describe()
            output.write(summary.to_string())
            output.write("\n\n")
        
        # Categorical columns info
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            output.write("=== CATEGORICAL COLUMNS ===\n")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                output.write(f"{col}: {unique_count} unique values\n")
                if unique_count <= 10:  
                    output.write(f"  Values: {df[col].value_counts().index.tolist()}\n")
            output.write("\n")
        
        # Sample data
        output.write("=== FIRST 5 ROWS ===\n")
        output.write(df.head().to_string())
        
        return output.getvalue()
        
    except Exception as e:
        return f"Error analyzing CSV: {str(e)}"

def handle_basic_eda(message, make_ai_request_func, get_prompt_config_func, client, file_path=None):
    """Handle basic EDA requests"""
    try:
        if not file_path:
            return {"response": "Error: No CSV file provided for EDA.", "search_used": False}
        
        # Analyze the CSV file
        analysis_results = analyze_csv(file_path)
        
        # Get prompt configuration and format it
        prompt_config = get_prompt_config_func("data_analysis")
        prompt = prompt_config["template"].format(data=analysis_results, message=message)
        
        response = make_ai_request_func(prompt, "data_analysis", client)
        return {"response": response, "search_used": False}
        
    except Exception as e:
        return {"response": f"EDA analysis failed: {str(e)}", "search_used": False}