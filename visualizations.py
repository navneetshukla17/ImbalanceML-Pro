import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

class VisualizationEngine:
    """Visualization utilities for machine learning results."""
    
    def __init__(self):
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False):
        """
        Create confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the matrix
            
        Returns:
            Plotly figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            text_format = '.2%'
        else:
            text_format = 'd'
        
        # Create labels
        labels = sorted(list(set(y_true) | set(y_pred)))
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f'Predicted {label}' for label in labels],
            y=[f'Actual {label}' for label in labels],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            showscale=True
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=500,
            height=500
        )
        
        return fig
    
    def plot_roc_curve(self, y_true, y_score):
        """
        Create ROC curve visualization.
        
        Args:
            y_true: True labels
            y_score: Prediction scores
            
        Returns:
            Plotly figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc_score = np.trapz(tpr, fpr)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            showlegend=True
        )
        
        return fig
    
    def plot_precision_recall_curve(self, y_true, y_score):
        """
        Create precision-recall curve visualization.
        
        Args:
            y_true: True labels
            y_score: Prediction scores
            
        Returns:
            Plotly figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        auc_score = np.trapz(precision, recall)
        
        fig = go.Figure()
        
        # PR curve
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AUC = {auc_score:.3f})',
            line=dict(color='green', width=2)
        ))
        
        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[baseline, baseline],
            mode='lines',
            name=f'Random Classifier (AP = {baseline:.3f})',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            showlegend=True
        )
        
        return fig
    
    def plot_feature_importance(self, importances, feature_names, top_n=20):
        """
        Create feature importance visualization.
        
        Args:
            importances: Feature importance values
            feature_names: Names of features
            top_n: Number of top features to show
            
        Returns:
            Plotly figure
        """
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(top_n)
        
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Feature Importances',
            labels={'importance': 'Importance', 'feature': 'Feature'}
        )
        
        fig.update_layout(height=max(400, top_n * 25))
        
        return fig
    
    def plot_shap_summary(self, shap_values, max_display=20):
        """
        Create SHAP summary plot.
        
        Args:
            shap_values: SHAP values
            max_display: Maximum number of features to display
            
        Returns:
            Matplotlib figure or None
        """
        try:
            import shap
            
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                shap_values, 
                max_display=max_display,
                show=False
            )
            plt.title('SHAP Summary Plot')
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            print("SHAP library not available")
            return None
        except Exception as e:
            print(f"Error creating SHAP summary plot: {e}")
            return None
    
    def plot_shap_waterfall(self, shap_values_single):
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            shap_values_single: SHAP values for single instance
            
        Returns:
            Matplotlib figure or None
        """
        try:
            import shap
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(shap_values_single, show=False)
            plt.title('SHAP Waterfall Plot')
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            print("SHAP library not available")
            return None
        except Exception as e:
            print(f"Error creating SHAP waterfall plot: {e}")
            return None
    
    def plot_class_distribution(self, y, title="Class Distribution"):
        """
        Create class distribution visualization.
        
        Args:
            y: Target variable
            title: Plot title
            
        Returns:
            Plotly figure
        """
        class_counts = pd.Series(y).value_counts().sort_index()
        
        fig = px.bar(
            x=class_counts.index,
            y=class_counts.values,
            title=title,
            labels={'x': 'Class', 'y': 'Count'}
        )
        
        # Add percentage annotations
        total = len(y)
        for i, count in enumerate(class_counts.values):
            percentage = count / total * 100
            fig.add_annotation(
                x=class_counts.index[i],
                y=count,
                text=f'{percentage:.1f}%',
                showarrow=False,
                yshift=10
            )
        
        return fig
    
    def plot_learning_curve(self, train_scores, val_scores, train_sizes):
        """
        Create learning curve visualization.
        
        Args:
            train_scores: Training scores
            val_scores: Validation scores
            train_sizes: Training set sizes
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=np.mean(train_scores, axis=1),
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue'),
            error_y=dict(
                type='data',
                array=np.std(train_scores, axis=1),
                visible=True
            )
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=np.mean(val_scores, axis=1),
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='red'),
            error_y=dict(
                type='data',
                array=np.std(val_scores, axis=1),
                visible=True
            )
        ))
        
        fig.update_layout(
            title='Learning Curve',
            xaxis_title='Training Set Size',
            yaxis_title='Score',
            showlegend=True
        )
        
        return fig
    
    def plot_model_comparison(self, results_dict):
        """
        Create model comparison visualization.
        
        Args:
            results_dict: Dictionary of model results
            
        Returns:
            Plotly figure
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=metrics,
            specs=[[{"type": "bar"} for _ in range(len(metrics))]]
        )
        
        models = list(results_dict.keys())
        colors = px.colors.qualitative.Set3[:len(models)]
        
        for i, metric in enumerate(metrics):
            values = [results_dict[model]['metrics'][metric] for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric,
                    marker_color=colors,
                    showlegend=False
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='Model Performance Comparison',
            height=400
        )
        
        return fig
