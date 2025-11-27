import pandas as pd
import numpy as np
from datetime import datetime
import json
import uuid

class ExperimentTracker:
    """Track and manage machine learning experiments."""
    
    def __init__(self):
        self.experiments = []
    
    def log_experiment(self, model_type, parameters, metrics, timestamp=None):
        """
        Log a new experiment.
        
        Args:
            model_type: Type of model used
            parameters: Dictionary of parameters
            metrics: Dictionary of metrics
            timestamp: Experiment timestamp
            
        Returns:
            Experiment ID
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        experiment_id = str(uuid.uuid4())[:8]
        
        experiment = {
            'id': experiment_id,
            'model_type': model_type,
            'parameters': parameters,
            'metrics': metrics,
            'timestamp': timestamp
        }
        
        self.experiments.append(experiment)
        
        return experiment_id
    
    def get_experiments(self):
        """Get all logged experiments."""
        return self.experiments
    
    def get_experiment_by_id(self, experiment_id):
        """
        Get a specific experiment by ID.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Experiment dictionary or None
        """
        for exp in self.experiments:
            if exp['id'] == experiment_id:
                return exp
        return None
    
    def get_best_experiment(self, metric='f1'):
        """
        Get the best experiment based on a metric.
        
        Args:
            metric: Metric to optimize for
            
        Returns:
            Best experiment dictionary or None
        """
        if not self.experiments:
            return None
        
        best_exp = max(
            self.experiments,
            key=lambda x: x['metrics'].get(metric, 0)
        )
        
        return best_exp
    
    def compare_experiments(self, experiment_ids=None, metrics=None):
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare (None for all)
            metrics: List of metrics to compare (None for all)
            
        Returns:
            DataFrame with comparison results
        """
        if experiment_ids is None:
            experiments_to_compare = self.experiments
        else:
            experiments_to_compare = [
                exp for exp in self.experiments 
                if exp['id'] in experiment_ids
            ]
        
        if not experiments_to_compare:
            return pd.DataFrame()
        
        # Get all available metrics if not specified
        if metrics is None:
            all_metrics = set()
            for exp in experiments_to_compare:
                all_metrics.update(exp['metrics'].keys())
            metrics = list(all_metrics)
        
        # Create comparison data
        comparison_data = []
        for exp in experiments_to_compare:
            row = {
                'experiment_id': exp['id'],
                'model_type': exp['model_type'],
                'timestamp': exp['timestamp']
            }
            
            # Add metrics
            for metric in metrics:
                row[metric] = exp['metrics'].get(metric, np.nan)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_experiment_summary(self):
        """
        Get summary statistics of all experiments.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.experiments:
            return {}
        
        # Count experiments by model type
        model_counts = {}
        metric_stats = {}
        
        for exp in self.experiments:
            model_type = exp['model_type']
            model_counts[model_type] = model_counts.get(model_type, 0) + 1
            
            # Collect metrics for statistics
            for metric, value in exp['metrics'].items():
                if metric not in metric_stats:
                    metric_stats[metric] = []
                metric_stats[metric].append(value)
        
        # Calculate metric statistics
        for metric in metric_stats:
            values = metric_stats[metric]
            metric_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
        
        summary = {
            'total_experiments': len(self.experiments),
            'model_type_counts': model_counts,
            'metric_statistics': metric_stats,
            'date_range': {
                'earliest': min(exp['timestamp'] for exp in self.experiments),
                'latest': max(exp['timestamp'] for exp in self.experiments)
            }
        }
        
        return summary
    
    def export_to_csv(self):
        """
        Export experiments to CSV format.
        
        Returns:
            CSV string
        """
        if not self.experiments:
            return ""
        
        # Flatten experiment data
        flattened_data = []
        for exp in self.experiments:
            row = {
                'experiment_id': exp['id'],
                'model_type': exp['model_type'],
                'timestamp': exp['timestamp'].isoformat()
            }
            
            # Add parameters (flattened)
            for param, value in exp['parameters'].items():
                row[f'param_{param}'] = value
            
            # Add metrics
            for metric, value in exp['metrics'].items():
                row[f'metric_{metric}'] = value
            
            flattened_data.append(row)
        
        # Create DataFrame and convert to CSV
        df = pd.DataFrame(flattened_data)
        return df.to_csv(index=False)
    
    def export_to_json(self):
        """
        Export experiments to JSON format.
        
        Returns:
            JSON string
        """
        # Convert datetime objects to strings for JSON serialization
        serializable_experiments = []
        for exp in self.experiments:
            serializable_exp = exp.copy()
            serializable_exp['timestamp'] = exp['timestamp'].isoformat()
            serializable_experiments.append(serializable_exp)
        
        return json.dumps(serializable_experiments, indent=2)
    
    def clear_experiments(self):
        """Clear all logged experiments."""
        self.experiments = []
    
    def delete_experiment(self, experiment_id):
        """
        Delete a specific experiment.
        
        Args:
            experiment_id: ID of the experiment to delete
            
        Returns:
            True if deleted, False if not found
        """
        for i, exp in enumerate(self.experiments):
            if exp['id'] == experiment_id:
                del self.experiments[i]
                return True
        return False
    
    def get_experiments_by_model_type(self, model_type):
        """
        Get all experiments for a specific model type.
        
        Args:
            model_type: Type of model to filter by
            
        Returns:
            List of experiments
        """
        return [
            exp for exp in self.experiments 
            if exp['model_type'] == model_type
        ]
    
    def get_top_experiments(self, metric='f1', n=5):
        """
        Get top N experiments based on a metric.
        
        Args:
            metric: Metric to rank by
            n: Number of top experiments to return
            
        Returns:
            List of top experiments
        """
        if not self.experiments:
            return []
        
        # Sort experiments by metric (descending)
        sorted_experiments = sorted(
            self.experiments,
            key=lambda x: x['metrics'].get(metric, 0),
            reverse=True
        )
        
        return sorted_experiments[:n]
