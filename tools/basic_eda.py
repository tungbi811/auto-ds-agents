import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO

def generate_eda_report(csv_file_path):
    """Generate complete EDA report"""
    try:
        df = pd.read_csv(csv_file_path)
        
        # Generate structured markdown report
        report = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Header
        report.append("# Exploratory Data Analysis Report")
        report.append("")
        
        # Dataset Overview
        report.append("# I. Dataset Overview")
        report.append(f"- **Rows:** {len(df):,}")
        report.append(f"- **Columns:** {len(df.columns)}")
        report.append(f"- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        report.append(f"- **Duplicate Rows:** {df.duplicated().sum():,}")
        report.append("")
        
        # Column Types
        report.append("# II. Column Information")
        report.append("| Column | Type | Missing | Missing % | Unique Values |")
        report.append("|--------|------|---------|-----------|---------------|")
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            unique_count = df[col].nunique()
            dtype = str(df[col].dtype)
            
            report.append(f"| {col} | {dtype} | {missing_count:,} | {missing_pct:.1f}% | {unique_count:,} |")
        
        report.append("")
        
        # Numeric Columns Analysis
        if len(numeric_cols) > 0:
            report.append("## 1. Numeric Columns Summary")

            for col in numeric_cols:
                stats = df[col].describe()
                report.append(f"### {col}")
                report.append(f"- **Count:** {int(stats['count']):,} values")
                report.append(f"- **Mean:** {stats['mean']:,.2f}")
                report.append(f"- **Standard Deviation:** {stats['std']:,.2f}")
                report.append(f"- **Minimum:** {stats['min']:,.2f}")
                report.append(f"- **25th Percentile:** {stats['25%']:,.2f}")
                report.append(f"- **Median:** {stats['50%']:,.2f}")
                report.append(f"- **75th Percentile:** {stats['75%']:,.2f}")
                report.append(f"- **Maximum:** {stats['max']:,.2f}")
                report.append(f"- **Range:** {stats['max'] - stats['min']:,.2f}")
                
                # Outlier detection using IQR
                Q1, Q3 = stats['25%'], stats['75%']
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                
                if len(outliers) > 0:
                    report.append(f"- **Potential Outliers:** {len(outliers):,} values ({len(outliers)/len(df)*100:.1f}%)")
                else:
                    report.append("- **Outliers:** None detected")
                
                report.append("")
        
        # Categorical Columns Analysis
        if len(categorical_cols) > 0:
            report.append("## 2. Categorical Columns Summary")

            for col in categorical_cols:
                unique_count = df[col].nunique()
                most_common = df[col].mode().iloc[0]
                most_common_count = df[col].value_counts().iloc[0]
                most_common_pct = (most_common_count / len(df)) * 100
                
                report.append(f"### {col}")
                report.append(f"- **Unique Values:** {unique_count:,}")
                report.append(f"- **Most Common:** '{most_common}' ({most_common_count:,} occurrences, {most_common_pct:.1f}%)")
                
                if unique_count <= 10:
                    report.append("- **All Values:**")
                    value_counts = df[col].value_counts()
                    for val, count in value_counts.items():
                        pct = (count / len(df)) * 100
                        report.append(f"  - {val}: {count:,} ({pct:.1f}%)")
                elif unique_count <= 20:
                    report.append("- **Top 5 Values:**")
                    top_values = df[col].value_counts().head(5)
                    for val, count in top_values.items():
                        pct = (count / len(df)) * 100
                        report.append(f"  - {val}: {count:,} ({pct:.1f}%)")
                
                report.append("")
        
        # Correlation Analysis
        if len(numeric_cols) > 1:
            report.append("## 3. Correlation Analysis")
            corr_matrix = df[numeric_cols].corr()
            
            # Find strong correlations if there's any
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strength = "Very Strong" if abs(corr_val) > 0.9 else "Strong"
                        direction = "Positive" if corr_val > 0 else "Negative"
                        strong_correlations.append({
                            'col1': corr_matrix.columns[i],
                            'col2': corr_matrix.columns[j],
                            'correlation': corr_val,
                            'strength': strength,
                            'direction': direction
                        })
            
            if strong_correlations:
                report.append("### Strong Correlations Found:")
                for corr in strong_correlations:
                    report.append(f"- **{corr['col1']}** ↔ **{corr['col2']}**: {corr['correlation']:.3f} ({corr['strength']} {corr['direction']})")
            else:
                report.append("- No strong correlations (>0.7) found between numeric columns")
            
            report.append("")
        
        # Data Quality Issues
        report.append("## 4. Data Quality Assessment")
        
        quality_issues = []
        
        # Missing data
        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
            quality_issues.append(f"Missing values: {total_missing:,} cells ({missing_pct:.2f}% of dataset)")
        
        # Duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            quality_issues.append(f"Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.1f}%)")
        
        # Columns with high cardinality
        high_cardinality = []
        for col in categorical_cols:
            if df[col].nunique() > len(df) * 0.8:
                high_cardinality.append(col)
        
        if high_cardinality:
            quality_issues.append(f"High cardinality columns: {', '.join(high_cardinality)} (may be IDs)")
        
        # Potential constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() == 1:
                constant_cols.append(col)
        
        if constant_cols:
            quality_issues.append(f"Constant columns: {', '.join(constant_cols)} (no variation)")
        
        if quality_issues:
            for issue in quality_issues:
                report.append(f"- {issue}")
        else:
            report.append("- No major data quality issues detected")
        
        report.append("")
        
        # Recommendations
        report.append("## 5. Analysis Recommendations")
        recommendations = []
        
        if total_missing > 0:
            recommendations.append("Consider handling missing values through imputation or removal")
        
        if duplicates > 0:
            recommendations.append("Review and potentially remove duplicate rows")
        
        if len(numeric_cols) > 1:
            recommendations.append("Explore relationships between numeric variables with scatter plots")
        
        if any(df[col].nunique() > 20 for col in categorical_cols):
            recommendations.append("Consider grouping categorical variables with many categories")
        
        recommendations.append("Perform statistical tests for significant relationships")
        recommendations.append("Consider feature engineering for machine learning applications")
        
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")
        
        # Generate visualizations
        plots = create_eda_visualizations(csv_file_path)
        
        return '\n'.join(report), plots
        
    except Exception as e:
        error_msg = f"#EDA Analysis Error\n\n**Error:** {str(e)}"
        return error_msg, [("Error", f"Visualization error: {str(e)}")]

def create_eda_visualizations(csv_file_path):
    """Create visualizations for EDA"""
    try:
        df = pd.read_csv(csv_file_path)
        plots = []
        plt.switch_backend('Agg')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Dataset Overview
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Dataset Overview', fontsize=16)
            
            # Missing values
            ax1 = axes[0, 0]
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                bars = ax1.bar(range(len(missing_data)), missing_data.values, color='salmon')
                ax1.set_title('Missing Values by Column')
                ax1.set_ylabel('Count')
                if len(missing_data) <= 6:
                    ax1.set_xticks(range(len(missing_data)))
                    ax1.set_xticklabels(missing_data.index, rotation=45, ha='right', fontsize=8)
            else:
                ax1.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Missing Values by Column')
            
            # Data types
            ax2 = axes[0, 1]
            dtype_counts = df.dtypes.astype(str).value_counts()
            ax2.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
            ax2.set_title('Data Types Distribution')
            
            # Column types
            ax3 = axes[1, 0]
            col_counts = [len(numeric_cols), len(categorical_cols)]
            bars = ax3.bar(['Numeric', 'Categorical'], col_counts, color=['skyblue', 'lightcoral'])
            ax3.set_title('Column Types')
            ax3.set_ylabel('Count')
            for bar, count in zip(bars, col_counts):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        str(count), ha='center', va='bottom')
            
            # Dataset info
            ax4 = axes[1, 1]
            duplicates = df.duplicated().sum()
            info_text = f"Rows: {len(df):,}\nColumns: {len(df.columns)}\nDuplicates: {duplicates:,}"
            ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes, fontsize=12, va='center')
            ax4.set_title('Dataset Summary')
            ax4.axis('off')
            
            plt.tight_layout()
            plots.append(save_plot_as_base64('Dataset Overview', fig))
            
        except Exception as e:
            print(f"Error creating overview: {e}")
        
        return plots
        
    except Exception as e:
        return [("Error", f"Error creating visualizations: {str(e)}")]

def save_plot_as_base64(title, fig):
    """Save plot as base64 string"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return (title, plot_base64)

def handle_basic_eda(message, make_ai_request_func, get_prompt_config_func, client, file_path=None):
    """Handle EDA requests with only code analysis (no LLM)"""
    try:
        if not file_path:
            return {"response": "Error: No CSV file provided for EDA.", "search_used": False}
        
        if not file_path.lower().endswith('.csv'):
            return {"response": "Error: File must be a CSV for data analysis.", "search_used": False}
        
        # Generate pure code report
        report, plot_data = generate_eda_report(file_path)
        
        return {
            "response": report, 
            "search_used": False,
            "visualizations": plot_data
        }
        
    except Exception as e:
        return {"response": f"EDA analysis failed: {str(e)}", "search_used": False}