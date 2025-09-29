import pandas as pd
import numpy as np
from scipy.stats import zscore
from typing import List, Annotated
from autogen.agentchat.group import ReplyResult, AgentNameTarget

def remove_columns_with_missing_data(
    columns: List[str] | str | None = None,
    thresh: float = 0.3,
) -> ReplyResult:
    """
    Remove columns containing excessive missing values from a DataFrame based on a threshold.
    
    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame object representing the dataset.
    thresh : float, optional (default=0.5)
        The minimum proportion of missing values required to drop a column.
        Should be between 0 and 1.
    columns : str, list of str, or None, optional (default=None)
        Labels of columns to consider. If None, all columns are considered.
    
    Returns
    -------
    pd.DataFrame
        A new DataFrame with columns removed if they exceed the missing value threshold.
    
    Notes
    -----
    - This function modifies the structure of your dataset by removing entire columns.
    - A low threshold may result in losing important features.
    - A high threshold may retain columns with too many missing values.
    - Consider the effect of removing columns on downstream tasks.
    """
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    
    if not (0 <= thresh <= 1):
        raise ValueError("`thresh` must be between 0 and 1.")
    
    # If columns not specified, use all columns
    if columns is None:
        columns_to_check = data.columns
    elif isinstance(columns, str):
        columns_to_check = [columns]
    elif isinstance(columns, list):
        columns_to_check = columns
    else:
        raise TypeError("`columns` must be str, list, or None.")
    
    # Proportion of missing values per column
    missing_ratios = data[columns_to_check].isna().mean()
    
    # Columns to drop (those with missing ratio >= thresh)
    cols_to_drop = missing_ratios[missing_ratios >= thresh].index

    data.drop(columns=cols_to_drop, inplace=True)
    
    return ReplyResult(
        message=f"Dropped columns due to missing data threshold ({thresh}): {list(cols_to_drop)}",
        target=AgentNameTarget("DataEngineer")
    )

def fill_missing_values(data, columns, method="auto", fill_value=None):
    """
    Fill missing values in specified columns of a DataFrame.
    
    Parameters:
        data (pd.DataFrame): The dataset.
        columns (str or list of str): Column(s) to fill.
        method (str): One of ["auto", "mean", "median", "mode", "constant"].
        fill_value (str/number): Value used when method="constant".
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    
    # Ensure columns is a list
    if isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        if method == "auto":
            if pd.api.types.is_numeric_dtype(data[col]):
                # numeric → use mean
                data[col].fillna(data[col].mean(), inplace=True)
            else:
                # categorical → use mode
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        elif method == "mean":
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col].fillna(data[col].mean(), inplace=True)
            else:
                raise TypeError(f"Column '{col}' is not numeric, cannot use mean.")
        
        elif method == "median":
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col].fillna(data[col].median(), inplace=True)
            else:
                raise TypeError(f"Column '{col}' is not numeric, cannot use median.")
        
        elif method == "mode":
            data[col].fillna(data[col].mode()[0], inplace=True)
        
        elif method == "constant":
            data[col].fillna(fill_value, inplace=True)
        
        else:
            raise ValueError("Invalid method. Choose from ['auto', 'mean', 'median', 'mode', 'constant']")
    
    return data



def detect_and_handle_outliers_zscore(data, columns, threshold=3.0, method="clip"):
    """
    Detect and handle outliers in specified columns using the Z-score method.
    
    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame object representing the dataset.
    columns : str or list of str
        The column(s) to check for outliers.
    threshold : float, optional (default=3.0)
        Z-score threshold to identify outliers.
        Values with |Z| > threshold are considered outliers.
    method : {"clip", "remove"}, optional (default="clip")
        How to handle outliers:
        - "clip"   : Replace outliers with boundary values (closest non-outlier).
        - "remove" : Drop rows containing outliers.
    
    Returns
    -------
    pd.DataFrame
        A new DataFrame with outliers handled.
    
    Notes
    -----
    - Assumes data is approximately normally distributed.
    - Sensitive to extreme outliers (affect mean & std).
    - Not suitable for highly skewed distributions.
    - Choice of threshold affects detection sensitivity.
    """
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    
    if isinstance(columns, str):
        columns = [columns]
    
    df = data.copy()
    
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric to apply Z-score method.")
        
        # Compute Z-scores
        z_scores = zscore(df[col].dropna())
        
        # Align z-scores with DataFrame index
        z_series = pd.Series(z_scores, index=df[col].dropna().index)
        
        # Identify outliers
        outliers = z_series.abs() > threshold
        
        if method == "clip":
            mean, std = df[col].mean(), df[col].std()
            upper_bound = mean + threshold * std
            lower_bound = mean - threshold * std
            df[col] = np.where(df[col] > upper_bound, upper_bound,
                               np.where(df[col] < lower_bound, lower_bound, df[col]))
        
        elif method == "remove":
            df = df.loc[~outliers]
        
        else:
            raise ValueError("Invalid method. Choose 'clip' or 'remove'.")
    
    return df