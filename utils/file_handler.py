import pandas as pd
import os
from pathlib import Path

def analyze_uploaded_file(uploaded_file):
    """Analyze uploaded file and return basic info"""
    if uploaded_file is None:
        return None

    # Save to workspace directly
    workspace_dir = Path("workspace")
    workspace_dir.mkdir(exist_ok=True)
    
    file_path = workspace_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Load dataset based on file type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(file_path)
        elif uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            return {"error": "Unsupported file format"}

        return {
            "filename": uploaded_file.name,
            "path": str(file_path),
            "rows": len(df),
            "columns": len(df.columns),
            "size": f"{uploaded_file.size / 1024:.1f} KB",
            "columns_list": list(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "duplicates": df.duplicated().sum(),
            "preview": df.head(3).to_dict('records')
        }
        
    except Exception as e:
        # Clean up on error
        if file_path.exists():
            os.remove(file_path)
        return {"error": f"Error analyzing file: {str(e)}"}

def dataset_upload_section():
    """Clean dataset upload section for main.py"""
    import streamlit as st
    
    st.markdown("## 📁 Dataset Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=['csv', 'xlsx', 'json', 'parquet'],
        help="Supported: CSV, Excel, JSON, Parquet (Max 200MB)"
    )
    
    if uploaded_file is not None:
        # Analyze the file
        file_info = analyze_uploaded_file(uploaded_file)
        
        if "error" in file_info:
            st.error(f"❌ {file_info['error']}")
            return None
        
        # Display file info
        st.success(f"✅ {file_info['filename']} uploaded successfully")
        
        # Quick metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", file_info['rows'])
        with col2:
            st.metric("Columns", file_info['columns'])
        with col3:
            st.metric("Missing", file_info['missing_values'])
        with col4:
            st.metric("Size", file_info['size'])
        
        # Preview
        with st.expander("📊 Data Preview", expanded=True):
            st.write("**Columns:**", ", ".join(file_info['columns_list']))
            if file_info['preview']:
                st.dataframe(pd.DataFrame(file_info['preview']))
        
        # Quality checks
        issues = []
        if file_info['missing_values'] > 0:
            issues.append(f"Missing values: {file_info['missing_values']}")
        if file_info['duplicates'] > 0:
            issues.append(f"Duplicate rows: {file_info['duplicates']}")
        
        if issues:
            st.warning("⚠️ " + " | ".join(issues))
        else:
            st.success("✅ Dataset ready for analysis")
        
        return file_info
    
    return None