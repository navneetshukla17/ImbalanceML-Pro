import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io
import os
from datetime import datetime

# Import custom modules with error handling
try:
    from preprocessing import DataPreprocessor
    from training import ModelTrainer
    from visualizations import VisualizationEngine
    from datasets import DatasetManager
    from experiment_tracker import ExperimentTracker
    from utils import download_model
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Configure page
st.set_page_config(
    page_title="ML Imbalanced Classification Tool",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'experiment_tracker' not in st.session_state:
    st.session_state.experiment_tracker = ExperimentTracker()
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

def main():
    st.title("⚖️ Machine Learning Tool for Imbalanced Classification")
    st.markdown("### Build and evaluate ML models with bagging ensembles for imbalanced datasets")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Go to", [
        "📊 Data Upload & Preview",
        "🔧 Model Training",
        "📈 Results & Visualization",
        "🧠 Model Explainability",
        "📋 Experiment Tracking",
        "💾 Model Export"
    ])
    
    # Theme toggle
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    if st.sidebar.button("🌙 Toggle Dark Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    
    if tab == "📊 Data Upload & Preview":
        data_upload_tab()
    elif tab == "🔧 Model Training":
        model_training_tab()
    elif tab == "📈 Results & Visualization":
        results_visualization_tab()
    elif tab == "🧠 Model Explainability":
        explainability_tab()
    elif tab == "📋 Experiment Tracking":
        experiment_tracking_tab()
    elif tab == "💾 Model Export":
        model_export_tab()

def data_upload_tab():
    st.header("📊 Data Upload & Preview")
    
    # Data source selection
    data_source = st.radio("Choose data source:", [
        "Upload Dataset", 
        "Built-in Datasets"
    ])
    
    if data_source == "Upload Dataset":
        uploaded_file = st.file_uploader(
            "Choose a CSV or XLSX file",
            type=['csv', 'xlsx'],
            help="Upload your dataset for imbalanced classification"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.data = df
                st.success(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
    
    else:  # Built-in datasets
        dataset_manager = DatasetManager()
        dataset_options = dataset_manager.get_available_datasets()
        
        selected_dataset = st.selectbox("Select a built-in dataset:", dataset_options)
        
        if st.button("Load Dataset"):
            df, target_col = dataset_manager.load_dataset(selected_dataset)
            st.session_state.data = df
            st.session_state.target_column = target_col
            st.success(f"Successfully loaded {selected_dataset} dataset")
    
    # Display data preview if data is loaded
    if st.session_state.data is not None:
        df = st.session_state.data
        
        st.subheader("Data Preview")
        st.dataframe(df.head(10), width="stretch")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Information")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            # Data types
            st.write("**Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Type': df.dtypes.values.astype(str)
            })
            st.dataframe(dtype_df, width="stretch")
        
        with col2:
            st.subheader("Missing Values")
            missing_data = df.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': (missing_data.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, width="stretch")
        
        # Target column selection
        st.subheader("Target Column Selection")
        if st.session_state.target_column is None:
            target_column = st.selectbox(
                "Select target column for classification:",
                df.columns.tolist()
            )
            st.session_state.target_column = target_column
        else:
            target_column = st.session_state.target_column
            st.info(f"Target column: **{target_column}**")
        
        # Class distribution
        if target_column in df.columns:
            st.subheader("Class Distribution")
            class_counts = df[target_column].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Class Counts:**")
                st.dataframe(class_counts.to_frame('Count'), width="stretch")
                
                # Calculate imbalance ratio
                majority_class = class_counts.max()
                minority_class = class_counts.min()
                imbalance_ratio = majority_class / minority_class
                st.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}:1")
            
            with col2:
                fig = px.bar(
                    x=class_counts.index,
                    y=class_counts.values,
                    title="Class Distribution",
                    labels={'x': 'Class', 'y': 'Count'}
                )
                st.plotly_chart(fig, width="stretch")

def model_training_tab():
    st.header("🔧 Model Training")
    
    if st.session_state.data is None:
        st.warning("Please upload a dataset first!")
        return
    
    df = st.session_state.data
    target_column = st.session_state.target_column
    
    if target_column is None:
        st.warning("Please select a target column first!")
        return
    
    # Model configuration
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Base Models**")
        available_models = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "Bagging Ensemble", "AdaBoost", "Gradient Boosting"]
        
        # Add XGBoost if available
        try:
            import xgboost
            available_models.append("XGBoost")
        except ImportError:
            pass
            
        model_type = st.selectbox(
            "Select model type:",
            available_models
        )
        
        # Preprocessing Configuration
        st.write("**Data Preprocessing**")
        with st.expander("Preprocessing Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                scaling_method = st.selectbox(
                    "Feature Scaling:",
                    ["standard", "minmax", "robust", "normalize"],
                    help="Standard: mean=0, std=1; MinMax: [0,1] range; Robust: median-based; Normalize: unit norm"
                )
                
                missing_strategy = st.selectbox(
                    "Missing Values (Numeric):",
                    ["median", "mean", "most_frequent", "knn"],
                    help="Strategy for handling missing numeric values"
                )
            
            with col2:
                categorical_encoding = st.selectbox(
                    "Categorical Encoding:",
                    ["label", "onehot", "ordinal"],
                    help="Label: integer encoding; OneHot: binary vectors; Ordinal: ordered integers"
                )
                
                categorical_missing = st.selectbox(
                    "Missing Values (Categorical):",
                    ["most_frequent", "constant"],
                    help="Strategy for handling missing categorical values"
                )
        
        # Imbalance handling
        st.write("**Imbalance Handling**")
        imbalance_method = st.selectbox(
            "Select imbalance handling method:",
            ["None", "SMOTE", "Random Oversampling", "Random Undersampling", "Class Weights"]
        )
    
    with col2:
        st.write("**Training Parameters**")
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", value=42, min_value=0)
        
        # Cross-validation
        cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
        
        # Hyperparameter tuning
        st.write("**Hyperparameter Tuning**")
        use_tuning = st.checkbox("Enable hyperparameter tuning")
        if use_tuning:
            tuning_method = st.selectbox("Tuning method:", ["grid_search", "random_search"])
            if tuning_method == "random_search":
                n_iter = st.slider("Number of iterations", 10, 100, 20)
    
    # Advanced parameters based on model type
    st.subheader("Advanced Parameters")
    
    if model_type == "Bagging Ensemble":
        col1, col2, col3 = st.columns(3)
        with col1:
            base_estimator = st.selectbox(
                "Base estimator:",
                ["Decision Tree", "Logistic Regression", "SVM"]
            )
        with col2:
            n_estimators = st.slider("Number of estimators", 10, 200, 50, 10)
        with col3:
            max_samples = st.slider("Max samples", 0.1, 1.0, 1.0, 0.1)
    
    elif model_type == "Random Forest":
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Number of estimators", 10, 200, 100, 10)
        with col2:
            max_depth = st.slider("Max depth", 1, 20, 10)
    
    elif model_type == "SVM":
        col1, col2 = st.columns(2)
        with col1:
            svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        with col2:
            svm_c = st.slider("C parameter", 0.1, 10.0, 1.0, 0.1)
    
    elif model_type == "AdaBoost":
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Number of estimators", 10, 200, 50, 10)
        with col2:
            learning_rate = st.slider("Learning rate", 0.01, 2.0, 1.0, 0.01)
    
    elif model_type == "Gradient Boosting":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("Number of estimators", 10, 200, 100, 10)
        with col2:
            learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01)
        with col3:
            max_depth = st.slider("Max depth", 1, 10, 3)
    
    elif model_type == "XGBoost":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("Number of estimators", 10, 200, 100, 10)
        with col2:
            learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01)
        with col3:
            max_depth = st.slider("Max depth", 1, 15, 6)
    
    # Train model button
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model... This may take a few minutes."):
            try:
                # Initialize preprocessor and trainer with selected options
                preprocessor = DataPreprocessor(
                    scaling_method=scaling_method if 'scaling_method' in locals() else 'standard',
                    categorical_encoding=categorical_encoding if 'categorical_encoding' in locals() else 'label',
                    missing_value_strategy=missing_strategy if 'missing_strategy' in locals() else 'median',
                    categorical_missing_strategy=categorical_missing if 'categorical_missing' in locals() else 'most_frequent'
                )
                trainer = ModelTrainer()
                
                # Prepare data
                X = df.drop(columns=[target_column])
                y = df[target_column]
                
                # Preprocess data
                X_processed = preprocessor.preprocess_features(X)
                
                # Configure model parameters
                model_params = {
                    'test_size': test_size,
                    'random_state': random_state,
                    'cv_folds': cv_folds,
                    'imbalance_method': imbalance_method,
                    'use_tuning': use_tuning if 'use_tuning' in locals() else False,
                    'scaling_method': scaling_method if 'scaling_method' in locals() else 'standard',
                    'categorical_encoding': categorical_encoding if 'categorical_encoding' in locals() else 'label',
                    'missing_strategy': missing_strategy if 'missing_strategy' in locals() else 'median',
                    'categorical_missing': categorical_missing if 'categorical_missing' in locals() else 'most_frequent'
                }
                
                if use_tuning if 'use_tuning' in locals() else False:
                    model_params['tuning_method'] = tuning_method if 'tuning_method' in locals() else 'grid_search'
                    if 'n_iter' in locals():
                        model_params['n_iter'] = n_iter
                
                if model_type == "Bagging Ensemble":
                    model_params.update({
                        'base_estimator': base_estimator,
                        'n_estimators': n_estimators,
                        'max_samples': max_samples
                    })
                elif model_type == "Random Forest":
                    model_params.update({
                        'n_estimators': n_estimators,
                        'max_depth': max_depth
                    })
                elif model_type == "SVM":
                    model_params.update({
                        'kernel': svm_kernel,
                        'C': svm_c
                    })
                elif model_type in ["AdaBoost", "Gradient Boosting", "XGBoost"]:
                    model_params.update({
                        'n_estimators': n_estimators,
                        'learning_rate': learning_rate
                    })
                    if model_type in ["Gradient Boosting", "XGBoost"]:
                        model_params.update({'max_depth': max_depth})
                
                # Train model
                results = trainer.train_model(
                    X_processed, y, model_type, model_params
                )
                
                # Store results
                st.session_state.trained_model = results['model']
                st.session_state.training_results = results
                
                # Log experiment
                experiment_id = st.session_state.experiment_tracker.log_experiment(
                    model_type=model_type,
                    parameters=model_params,
                    metrics=results['metrics'],
                    timestamp=datetime.now()
                )
                
                st.success(f"Model trained successfully! Experiment ID: {experiment_id}")
                st.balloons()
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

def results_visualization_tab():
    st.header("📈 Results & Visualization")
    
    if st.session_state.training_results is None:
        st.warning("Please train a model first!")
        return
    
    results = st.session_state.training_results
    viz_engine = VisualizationEngine()
    
    # Metrics overview
    st.subheader("Classification Metrics")
    
    metrics = results['metrics']
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{metrics['f1']:.3f}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
    with col2:
        st.metric("AUC-PR", f"{metrics['auc_pr']:.3f}")
    
    # Visualizations
    st.subheader("Performance Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"])
    
    with tab1:
        cm_fig = viz_engine.plot_confusion_matrix(
            results['y_test'], 
            results['y_pred']
        )
        st.plotly_chart(cm_fig, width="stretch")
    
    with tab2:
        roc_fig = viz_engine.plot_roc_curve(
            results['y_test'], 
            results['y_pred_proba'][:, 1]
        )
        st.plotly_chart(roc_fig, width="stretch")
    
    with tab3:
        pr_fig = viz_engine.plot_precision_recall_curve(
            results['y_test'], 
            results['y_pred_proba'][:, 1]
        )
        st.plotly_chart(pr_fig, width="stretch")
    
    # Enhanced Cross-validation results
    if 'cv_scores' in results:
        st.subheader("Cross-Validation Results")
        cv_results = results['cv_scores']
        
        # Check if we have enhanced CV results
        if isinstance(cv_results, dict) and 'f1' in cv_results:
            # Display enhanced CV results with multiple metrics
            cv_metrics_tabs = st.tabs(["Summary", "Detailed Scores", "Visualization"])
            
            with cv_metrics_tabs[0]:
                # Summary statistics for all metrics
                st.write("**Cross-Validation Summary (Stratified K-Fold)**")
                
                summary_data = []
                for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                    if metric_name in cv_results:
                        metric_data = cv_results[metric_name]
                        summary_data.append({
                            'Metric': metric_name.replace('_', '-').title(),
                            'Mean': f"{metric_data['mean']:.4f}",
                            'Std Dev': f"{metric_data['std']:.4f}",
                            'Min': f"{metric_data['min']:.4f}",
                            'Max': f"{metric_data['max']:.4f}"
                        })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, width="stretch")
            
            with cv_metrics_tabs[1]:
                # Detailed fold-by-fold scores
                st.write("**Fold-by-Fold Scores**")
                
                detailed_data = []
                for i in range(len(cv_results['f1']['scores'])):
                    row = {'Fold': i + 1}
                    for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                        if metric_name in cv_results:
                            row[metric_name.replace('_', '-').title()] = f"{cv_results[metric_name]['scores'][i]:.4f}"
                    detailed_data.append(row)
                
                detailed_df = pd.DataFrame(detailed_data)
                st.dataframe(detailed_df, width="stretch")
            
            with cv_metrics_tabs[2]:
                # Visualization of CV results
                st.write("**Cross-Validation Score Visualization**")
                
                # Line plot for all metrics
                viz_data = []
                for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                    if metric_name in cv_results:
                        for i, score in enumerate(cv_results[metric_name]['scores']):
                            viz_data.append({
                                'Fold': i + 1,
                                'Score': score,
                                'Metric': metric_name.replace('_', '-').title()
                            })
                
                viz_df = pd.DataFrame(viz_data)
                
                fig_lines = px.line(
                    viz_df, x='Fold', y='Score', color='Metric',
                    title='Cross-Validation Scores by Fold',
                    markers=True
                )
                st.plotly_chart(fig_lines, width="stretch")
                
                # Box plot for score distribution
                fig_box = px.box(
                    viz_df, x='Metric', y='Score',
                    title='Score Distribution Across Folds'
                )
                st.plotly_chart(fig_box, width="stretch")
        
        else:
            # Fallback to simple CV results display
            cv_scores = cv_results if isinstance(cv_results, np.ndarray) else results['cv_scores']
            cv_df = pd.DataFrame({
                'Fold': range(1, len(cv_scores) + 1),
                'Score': cv_scores
            })
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(cv_df, width="stretch")
                st.metric("Mean CV Score", f"{np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
            with col2:
                cv_fig = px.line(
                    cv_df, x='Fold', y='Score',
                    title='Cross-Validation Scores',
                    markers=True
                )
                st.plotly_chart(cv_fig, width="stretch")

def explainability_tab():
    st.header("🧠 Model Explainability")
    
    if st.session_state.trained_model is None:
        st.warning("Please train a model first!")
        return
    
    if st.session_state.training_results is None:
        st.warning("No training results available!")
        return
    
    try:
        import shap
        shap_available = True
    except ImportError:
        shap_available = False
    
    model = st.session_state.trained_model
    results = st.session_state.training_results
    X_test = results['X_test']
    
    viz_engine = VisualizationEngine()
    
    # Feature importance
    st.subheader("Feature Importance")
    
    if hasattr(model, 'feature_importances_'):
        importance_fig = viz_engine.plot_feature_importance(
            model.feature_importances_,
            X_test.columns.tolist()
        )
        st.plotly_chart(importance_fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")
    
    # Model Explanations
    st.subheader("Model Explanations")
    
    explanation_type = st.selectbox("Choose explanation method:", ["Feature Importance", "SHAP", "LIME"])
    
    if explanation_type == "SHAP":
        if not shap_available:
            st.warning("SHAP library is not available. Please select another explanation method.")
        else:
            # Select samples for explanation
            n_samples = st.slider("Number of samples to explain", 1, min(100, len(X_test)), 10)
            sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
            X_sample = X_test.iloc[sample_indices]
        
            if st.button("Generate SHAP Explanations"):
                try:
                    with st.spinner("Generating SHAP explanations... This may take a moment."):
                        # Create SHAP explainer
                        explainer = shap.Explainer(model, X_test.sample(100))
                        shap_values = explainer(X_sample)
                        
                        # SHAP summary plot
                        st.write("**SHAP Summary Plot**")
                        summary_fig = viz_engine.plot_shap_summary(shap_values)
                        if summary_fig:
                            st.pyplot(summary_fig)
                        
                        # SHAP waterfall plot for first sample
                        st.write("**SHAP Waterfall Plot (First Sample)**")
                        waterfall_fig = viz_engine.plot_shap_waterfall(shap_values[0])
                        if waterfall_fig:
                            st.pyplot(waterfall_fig)
                        
                except Exception as e:
                    st.error(f"Error generating SHAP explanations: {str(e)}")
                    st.info("SHAP explanations may not be available for all model types.")
    
    elif explanation_type == "LIME":
        try:
            import lime
            import lime.lime_tabular
            
            # Select samples for explanation
            n_samples = st.slider("Number of samples to explain", 1, min(20, len(X_test)), 5)
            sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
            
            if st.button("Generate LIME Explanations"):
                with st.spinner("Generating LIME explanations... This may take a moment."):
                    try:
                        # Create LIME explainer
                        explainer = lime.lime_tabular.LimeTabularExplainer(
                            X_test.values,
                            feature_names=X_test.columns.tolist(),
                            class_names=['Class 0', 'Class 1'],
                            mode='classification'
                        )
                        
                        # Generate explanations for selected samples
                        for i, idx in enumerate(sample_indices[:3]):  # Limit to 3 for display
                            st.write(f"**LIME Explanation for Sample {i+1}**")
                            
                            exp = explainer.explain_instance(
                                X_test.iloc[idx].values,
                                model.predict_proba,
                                num_features=10
                            )
                            
                            # Display explanation as a plot
                            fig = exp.as_pyplot_figure()
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Error generating LIME explanations: {str(e)}")
                        st.info("LIME explanations may not be available for all model types.")
                        
        except ImportError:
            st.warning("LIME library is not available. Basic feature importance is shown above.")
            st.info("To enable LIME explainability features, LIME would need to be installed.")

def experiment_tracking_tab():
    st.header("📋 Experiment Tracking")
    
    tracker = st.session_state.experiment_tracker
    experiments = tracker.get_experiments()
    
    if not experiments:
        st.info("No experiments logged yet. Train some models to see them here!")
        return
    
    # Experiments table
    st.subheader("Experiment History")
    
    exp_df = pd.DataFrame([
        {
            'ID': exp['id'],
            'Model': exp['model_type'],
            'Accuracy': f"{exp['metrics']['accuracy']:.3f}",
            'F1-Score': f"{exp['metrics']['f1']:.3f}",
            'AUC-ROC': f"{exp['metrics']['auc_roc']:.3f}",
            'Timestamp': exp['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        }
        for exp in experiments
    ])
    
    st.dataframe(exp_df, width="stretch")
    
    # Enhanced Model Comparison Dashboard
    st.subheader("Model Comparison Dashboard")
    
    if len(experiments) >= 2:
        # Model selection for detailed comparison
        model_names = [f"{exp['model_type']} (ID: {exp['id']})" for exp in experiments]
        selected_models = st.multiselect(
            "Select models to compare:",
            model_names,
            default=model_names[:3] if len(model_names) >= 3 else model_names
        )
        
        if selected_models:
            # Get selected experiment IDs
            selected_ids = [name.split('ID: ')[1].rstrip(')') for name in selected_models]
            selected_experiments = [exp for exp in experiments if exp['id'] in selected_ids]
            
            # Create tabs for different comparison views
            comp_tab1, comp_tab2, comp_tab3 = st.tabs([
                "Performance Metrics", "Visual Comparison", "Statistical Analysis"
            ])
            
            with comp_tab1:
                # Performance metrics comparison
                st.write("**Performance Metrics Comparison**")
                
                # Create comparison table
                comparison_data = []
                for exp in selected_experiments:
                    row = {
                        'Model': f"{exp['model_type']} (ID: {exp['id']})",
                        'Accuracy': f"{exp['metrics']['accuracy']:.4f}",
                        'Precision': f"{exp['metrics']['precision']:.4f}",
                        'Recall': f"{exp['metrics']['recall']:.4f}",
                        'F1-Score': f"{exp['metrics']['f1']:.4f}",
                        'ROC-AUC': f"{exp['metrics']['auc_roc']:.4f}",
                        'PR-AUC': f"{exp['metrics']['auc_pr']:.4f}"
                    }
                    comparison_data.append(row)
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, width="stretch")
                
                # Highlight best performing models
                st.write("**Best Performing Models by Metric:**")
                metrics_to_check = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']
                best_models = {}
                
                for metric in metrics_to_check:
                    best_exp = max(selected_experiments, key=lambda x: x['metrics'][metric])
                    best_models[metric] = f"{best_exp['model_type']} (ID: {best_exp['id']}) - {best_exp['metrics'][metric]:.4f}"
                
                best_df = pd.DataFrame(list(best_models.items()), columns=['Metric', 'Best Model (Score)'])
                st.dataframe(best_df, width="stretch")
            
            with comp_tab2:
                # Visual comparison
                st.write("**Visual Performance Comparison**")
                
                # Radar chart for multiple metrics
                metrics_for_radar = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']
                
                radar_data = []
                for exp in selected_experiments:
                    for metric in metrics_for_radar:
                        radar_data.append({
                            'Model': f"{exp['model_type']} (ID: {exp['id']})",
                            'Metric': metric.replace('_', '-').replace('auc-', 'AUC-').title(),
                            'Score': exp['metrics'][metric]
                        })
                
                radar_df = pd.DataFrame(radar_data)
                
                # Create radar chart
                fig_radar = px.line_polar(
                    radar_df, r='Score', theta='Metric', color='Model',
                    line_close=True, title="Multi-Metric Performance Radar Chart",
                    range_r=[0, 1]
                )
                st.plotly_chart(fig_radar, width="stretch")
                
                # Bar chart comparison
                fig_bar = px.bar(
                    radar_df, x='Model', y='Score', color='Metric',
                    barmode='group', title='Side-by-Side Metric Comparison'
                )
                fig_bar.update_xaxes(tickangle=45)
                st.plotly_chart(fig_bar, width="stretch")
            
            with comp_tab3:
                # Statistical analysis and ranking
                st.write("**Performance Ranking**")
                
                ranking_data = []
                for exp in selected_experiments:
                    # Calculate composite score (weighted average of metrics)
                    composite_score = (
                        exp['metrics']['accuracy'] * 0.15 +
                        exp['metrics']['precision'] * 0.15 +
                        exp['metrics']['recall'] * 0.15 +
                        exp['metrics']['f1'] * 0.25 +
                        exp['metrics']['auc_roc'] * 0.15 +
                        exp['metrics']['auc_pr'] * 0.15
                    )
                    ranking_data.append({
                        'Model': f"{exp['model_type']} (ID: {exp['id']})",
                        'Composite Score': f"{composite_score:.4f}",
                        'F1 Score': f"{exp['metrics']['f1']:.4f}",
                        'ROC-AUC': f"{exp['metrics']['auc_roc']:.4f}",
                        'PR-AUC': f"{exp['metrics']['auc_pr']:.4f}"
                    })
                
                ranking_df = pd.DataFrame(ranking_data)
                ranking_df = ranking_df.sort_values('Composite Score', ascending=False)
                ranking_df['Rank'] = range(1, len(ranking_df) + 1)
                
                # Reorder columns
                ranking_df = ranking_df[['Rank', 'Model', 'Composite Score', 'F1 Score', 'ROC-AUC', 'PR-AUC']]
                st.dataframe(ranking_df, width="stretch")
                
                # Simple bar chart for composite scores
                composite_data = []
                for exp in selected_experiments:
                    composite_score = (
                        exp['metrics']['accuracy'] * 0.15 +
                        exp['metrics']['precision'] * 0.15 +
                        exp['metrics']['recall'] * 0.15 +
                        exp['metrics']['f1'] * 0.25 +
                        exp['metrics']['auc_roc'] * 0.15 +
                        exp['metrics']['auc_pr'] * 0.15
                    )
                    composite_data.append({
                        'Model': f"{exp['model_type']} (ID: {exp['id']})",
                        'Composite Score': composite_score
                    })
                
                composite_df = pd.DataFrame(composite_data).sort_values('Composite Score', ascending=True)
                
                fig_composite = px.bar(
                    composite_df, x='Composite Score', y='Model',
                    orientation='h', title='Overall Performance Ranking',
                    text='Composite Score'
                )
                fig_composite.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                st.plotly_chart(fig_composite, width="stretch")
        
        else:
            st.info("Please select at least one model for comparison.")
    else:
        # Simple comparison for <= 1 experiment
        if len(experiments) == 1:
            st.info("Only one experiment available. Train more models to enable comparison.")
        else:
            st.info("No experiments available for comparison.")
    
    # Download experiments
    if st.button("📥 Download Experiment Log"):
        csv_data = tracker.export_to_csv()
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def model_export_tab():
    st.header("💾 Model Export")
    
    if st.session_state.trained_model is None:
        st.warning("Please train a model first!")
        return
    
    model = st.session_state.trained_model
    results = st.session_state.training_results
    
    st.subheader("Export Trained Model")
    
    # Model information
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Information:**")
        st.write(f"Model Type: {type(model).__name__}")
        st.write(f"Training Accuracy: {results['metrics']['accuracy']:.3f}")
        st.write(f"F1-Score: {results['metrics']['f1']:.3f}")
        st.write(f"AUC-ROC: {results['metrics']['auc_roc']:.3f}")
    
    with col2:
        st.write("**Export Options:**")
        include_metadata = st.checkbox("Include metadata", value=True)
        include_preprocessing = st.checkbox("Include preprocessing steps", value=True)
    
    # Export model
    if st.button("🚀 Export Model", type="primary"):
        try:
            # Create model package
            model_package = {
                'model': model,
                'metrics': results['metrics'],
                'feature_names': results['X_test'].columns.tolist(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            if include_metadata:
                model_package['metadata'] = {
                    'model_type': type(model).__name__,
                    'training_samples': len(results['X_train']),
                    'test_samples': len(results['X_test'])
                }
            
            # Serialize model
            model_bytes = joblib.dumps(model_package)
            
            # Create download
            st.download_button(
                label="📥 Download Model (.pkl)",
                data=model_bytes,
                file_name=f"trained_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                mime="application/octet-stream"
            )
            
            st.success("Model package created successfully!")
            
        except Exception as e:
            st.error(f"Error exporting model: {str(e)}")
    
    # Usage instructions
    st.subheader("Usage Instructions")
    st.code("""
# Load the exported model
import joblib

# Load model package
model_package = joblib.load('trained_model.pkl')

# Extract components
model = model_package['model']
metrics = model_package['metrics']
feature_names = model_package['feature_names']

# Make predictions
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)
    """, language='python')

if __name__ == "__main__":
    main()
