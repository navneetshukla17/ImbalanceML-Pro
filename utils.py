import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import io
import base64

def download_model(model, filename=None):
    """
    Create a download link for a trained model.
    
    Args:
        model: Trained model object
        filename: Name for the downloaded file
        
    Returns:
        Download link
    """
    if filename is None:
        filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    
    # Serialize model
    model_bytes = joblib.dumps(model)
    
    # Create download button
    st.download_button(
        label="Download Model",
        data=model_bytes,
        file_name=filename,
        mime="application/octet-stream"
    )

def load_model_from_file(uploaded_file):
    """
    Load a model from an uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Loaded model object
    """
    try:
        model = joblib.load(uploaded_file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def format_metrics(metrics):
    """
    Format metrics dictionary for display.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Formatted string
    """
    formatted = []
    for metric, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{metric.title()}: {value:.3f}")
        else:
            formatted.append(f"{metric.title()}: {value}")
    
    return " | ".join(formatted)

def create_download_link(data, filename, link_text):
    """
    Create a download link for data.
    
    Args:
        data: Data to download (string or bytes)
        filename: Name for the downloaded file
        link_text: Text for the download link
        
    Returns:
        HTML download link
    """
    if isinstance(data, str):
        data = data.encode()
    
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def validate_dataset(df, target_column):
    """
    Validate a dataset for machine learning.
    
    Args:
        df: DataFrame to validate
        target_column: Name of target column
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check if target column exists
    if target_column not in df.columns:
        errors.append(f"Target column '{target_column}' not found in dataset")
    
    # Check for minimum number of samples
    if len(df) < 10:
        errors.append("Dataset must have at least 10 samples")
    
    # Check for minimum number of features
    if len(df.columns) < 2:
        errors.append("Dataset must have at least 2 columns (features + target)")
    
    # Check for too many missing values
    missing_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if missing_percentage > 0.5:
        errors.append("Dataset has more than 50% missing values")
    
    # Check target column for valid classes
    if target_column in df.columns:
        unique_classes = df[target_column].nunique()
        if unique_classes < 2:
            errors.append("Target column must have at least 2 classes")
        elif unique_classes > 10:
            errors.append("Target column has too many classes (>10). Consider using regression instead.")
    
    return len(errors) == 0, errors

def get_memory_usage(df):
    """
    Get memory usage information for a DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with memory usage info
    """
    memory_info = {
        'total_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'per_column': df.memory_usage(deep=True).to_dict(),
        'dtypes': df.dtypes.to_dict()
    }
    
    return memory_info

def safe_divide(numerator, denominator, default=0):
    """
    Safely divide two numbers, handling division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero
        
    Returns:
        Division result or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default

def calculate_class_balance(y):
    """
    Calculate class balance metrics.
    
    Args:
        y: Target variable
        
    Returns:
        Dictionary with balance metrics
    """
    value_counts = pd.Series(y).value_counts()
    
    balance_metrics = {
        'class_counts': value_counts.to_dict(),
        'class_percentages': (value_counts / len(y) * 100).to_dict(),
        'imbalance_ratio': value_counts.max() / value_counts.min() if value_counts.min() > 0 else float('inf'),
        'majority_class': value_counts.index[0],
        'minority_class': value_counts.index[-1]
    }
    
    return balance_metrics

def suggest_model_parameters(dataset_size, n_features, imbalance_ratio):
    """
    Suggest model parameters based on dataset characteristics.
    
    Args:
        dataset_size: Number of samples
        n_features: Number of features
        imbalance_ratio: Ratio of majority to minority class
        
    Returns:
        Dictionary with parameter suggestions
    """
    suggestions = {}
    
    # Random Forest suggestions
    if dataset_size < 1000:
        suggestions['random_forest'] = {'n_estimators': 50, 'max_depth': 10}
    elif dataset_size < 10000:
        suggestions['random_forest'] = {'n_estimators': 100, 'max_depth': 15}
    else:
        suggestions['random_forest'] = {'n_estimators': 200, 'max_depth': 20}
    
    # Bagging suggestions
    suggestions['bagging'] = {
        'n_estimators': min(100, max(10, dataset_size // 50)),
        'max_samples': 0.8 if dataset_size > 1000 else 1.0
    }
    
    # Imbalance handling suggestions
    if imbalance_ratio > 10:
        suggestions['imbalance_handling'] = 'SMOTE'
    elif imbalance_ratio > 5:
        suggestions['imbalance_handling'] = 'Class Weights'
    else:
        suggestions['imbalance_handling'] = 'None'
    
    return suggestions

def export_results_summary(results, experiment_info):
    """
    Create a comprehensive results summary.
    
    Args:
        results: Training results dictionary
        experiment_info: Experiment information
        
    Returns:
        Formatted summary string
    """
    summary = f"""
Machine Learning Experiment Summary
==================================

Experiment Information:
- Model Type: {experiment_info.get('model_type', 'Unknown')}
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Dataset Size: {len(results.get('X_train', [])) + len(results.get('X_test', []))} samples

Model Performance:
"""
    
    # Add metrics
    metrics = results.get('metrics', {})
    for metric, value in metrics.items():
        if isinstance(value, float):
            summary += f"- {metric.title()}: {value:.4f}\n"
        else:
            summary += f"- {metric.title()}: {value}\n"
    
    # Add cross-validation results
    cv_scores = results.get('cv_scores', [])
    if len(cv_scores) > 0:
        summary += f"\nCross-Validation:\n"
        summary += f"- Mean Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n"
        summary += f"- Individual Scores: {[f'{score:.4f}' for score in cv_scores]}\n"
    
    return summary

def check_data_quality(df):
    """
    Perform data quality checks.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with quality assessment
    """
    quality_report = {
        'total_samples': len(df),
        'total_features': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'constant_features': (df.nunique() == 1).sum(),
        'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_features': len(df.select_dtypes(include=[object]).columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    
    # Quality score (0-100)
    score = 100
    score -= min(30, (quality_report['missing_values'] / (len(df) * len(df.columns))) * 100)
    score -= min(20, (quality_report['duplicate_rows'] / len(df)) * 100)
    score -= min(10, quality_report['constant_features'] * 5)
    
    quality_report['quality_score'] = max(0, score)
    
    return quality_report
