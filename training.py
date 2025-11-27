import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Machine learning model trainer for imbalanced classification."""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = {
            'Logistic Regression': LogisticRegression,
            'Decision Tree': DecisionTreeClassifier,
            'Random Forest': RandomForestClassifier,
            'SVM': SVC,
            'Bagging Ensemble': BaggingClassifier,
            'AdaBoost': AdaBoostClassifier,
            'Gradient Boosting': GradientBoostingClassifier
        }
        
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = XGBClassifier
    
    def train_model(self, X, y, model_type, params):
        """
        Train a machine learning model.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to train
            params: Training parameters
            
        Returns:
            Dictionary with training results
        """
        # Extract parameters
        test_size = params.get('test_size', 0.2)
        random_state = params.get('random_state', 42)
        cv_folds = params.get('cv_folds', 5)
        imbalance_method = params.get('imbalance_method', 'None')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Handle imbalance
        if imbalance_method.lower() != 'none':
            if imbalance_method.lower() == 'class weights':
                # Will be handled in model creation
                X_train_balanced, y_train_balanced = X_train, y_train
            else:
                X_train_balanced, y_train_balanced = self.preprocessor.handle_imbalance(
                    X_train, y_train, imbalance_method, random_state
                )
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Check if hyperparameter tuning is enabled
        use_tuning = params.get('use_tuning', False)
        
        if use_tuning and model_type in ['Random Forest', 'Logistic Regression', 'SVM', 'AdaBoost', 'Gradient Boosting', 'XGBoost']:
            # Perform hyperparameter tuning
            tuning_method = params.get('tuning_method', 'grid_search')
            n_iter = params.get('n_iter', 20)
            
            tuning_results = self.hyperparameter_tuning(
                X_train_balanced, y_train_balanced, 
                model_type, tuning_method, cv_folds, n_iter
            )
            
            model = tuning_results['best_model']
            best_params = tuning_results['best_params']
        else:
            # Create and train model normally
            model = self._create_model(model_type, params, y_train_balanced)
            model.fit(X_train_balanced, y_train_balanced)
            best_params = None
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Cross-validation
        cv_scores = self._cross_validate(model, X_train_balanced, y_train_balanced, cv_folds)
        
        # Compile results
        results = {
            'model': model,
            'X_train': X_train_balanced,
            'X_test': X_test,
            'y_train': y_train_balanced,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'metrics': metrics,
            'cv_scores': cv_scores,
            'feature_names': X.columns.tolist() if hasattr(X, 'columns') else None,
            'best_params': best_params,
            'hyperparameter_tuning': use_tuning
        }
        
        return results
    
    def _create_model(self, model_type, params, y_train):
        """
        Create a model instance based on type and parameters.
        
        Args:
            model_type: Type of model
            params: Model parameters
            y_train: Training target for class weight calculation
            
        Returns:
            Model instance
        """
        random_state = params.get('random_state', 42)
        imbalance_method = params.get('imbalance_method', 'None')
        
        # Calculate class weights if needed
        class_weight = None
        if imbalance_method.lower() == 'class weights':
            class_weight = self.preprocessor.get_class_weights(y_train)
        
        if model_type == 'Logistic Regression':
            return LogisticRegression(
                random_state=random_state,
                class_weight=class_weight,
                max_iter=1000
            )
        
        elif model_type == 'Decision Tree':
            return DecisionTreeClassifier(
                random_state=random_state,
                class_weight=class_weight,
                max_depth=params.get('max_depth', None)
            )
        
        elif model_type == 'Random Forest':
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                random_state=random_state,
                class_weight=class_weight
            )
        
        elif model_type == 'SVM':
            return SVC(
                kernel=params.get('kernel', 'rbf'),
                C=params.get('C', 1.0),
                random_state=random_state,
                class_weight=class_weight,
                probability=True
            )
        
        elif model_type == 'Bagging Ensemble':
            base_estimator_type = params.get('base_estimator', 'Decision Tree')
            
            if base_estimator_type == 'Decision Tree':
                base_estimator = DecisionTreeClassifier(random_state=random_state)
            elif base_estimator_type == 'Logistic Regression':
                base_estimator = LogisticRegression(random_state=random_state, max_iter=1000)
            elif base_estimator_type == 'SVM':
                base_estimator = SVC(random_state=random_state, probability=True)
            else:
                base_estimator = DecisionTreeClassifier(random_state=random_state)
            
            return BaggingClassifier(
                base_estimator=base_estimator,
                n_estimators=params.get('n_estimators', 50),
                max_samples=params.get('max_samples', 1.0),
                random_state=random_state
            )
        
        elif model_type == 'AdaBoost':
            return AdaBoostClassifier(
                n_estimators=params.get('n_estimators', 50),
                learning_rate=params.get('learning_rate', 1.0),
                random_state=random_state
            )
        
        elif model_type == 'Gradient Boosting':
            return GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 3),
                random_state=random_state
            )
        
        elif model_type == 'XGBoost' and XGBOOST_AVAILABLE:
            return XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 6),
                random_state=random_state,
                eval_metric='logloss'
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # AUC metrics (only for binary classification)
        if len(np.unique(y_true)) == 2:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            metrics['auc_pr'] = average_precision_score(y_true, y_pred_proba[:, 1])
        else:
            # For multiclass, use macro average
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                metrics['auc_pr'] = average_precision_score(y_true, y_pred_proba, average='macro')
            except ValueError:
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
        
        return metrics
    
    def _cross_validate(self, model, X, y, cv_folds):
        """
        Enhanced cross-validation with multiple metrics and stratified folds.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation scores for multiple metrics
        """
        try:
            # Use stratified k-fold to maintain class distribution
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Define multiple scoring metrics
            scoring_metrics = {
                'accuracy': 'accuracy',
                'precision': 'precision_weighted',
                'recall': 'recall_weighted',
                'f1': 'f1_weighted',
                'roc_auc': 'roc_auc_ovr_weighted'
            }
            
            cv_results = {}
            for metric_name, scoring in scoring_metrics.items():
                try:
                    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                    cv_results[metric_name] = {
                        'scores': scores,
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores)
                    }
                except Exception as e:
                    print(f"Failed to compute {metric_name} in CV: {e}")
                    cv_results[metric_name] = {
                        'scores': np.array([0.0] * cv_folds),
                        'mean': 0.0,
                        'std': 0.0,
                        'min': 0.0,
                        'max': 0.0
                    }
            
            # For backward compatibility, also return f1 scores as before
            cv_results['cv_scores'] = cv_results['f1']['scores']
            
            return cv_results
            
        except Exception as e:
            print(f"Cross-validation failed: {e}")
            # Return default structure for backward compatibility
            default_scores = np.array([0.0] * cv_folds)
            return {
                'f1': {
                    'scores': default_scores,
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0
                },
                'cv_scores': default_scores
            }
    
    def get_model_info(self, model):
        """
        Get information about a trained model.
        
        Args:
            model: Trained model
            
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': type(model).__name__,
            'parameters': model.get_params(),
        }
        
        # Add feature importance if available
        if hasattr(model, 'feature_importances_'):
            info['feature_importances'] = model.feature_importances_
        
        if hasattr(model, 'coef_'):
            info['coefficients'] = model.coef_
        
        return info
    
    def predict_new_data(self, model, X_new):
        """
        Make predictions on new data.
        
        Args:
            model: Trained model
            X_new: New feature matrix
            
        Returns:
            Predictions and probabilities
        """
        # Preprocess new data (without fitting)
        X_processed = self.preprocessor.preprocess_features(X_new, fit=False)
        
        # Make predictions
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)
        
        return predictions, probabilities
    
    def hyperparameter_tuning(self, X, y, model_type, tuning_method='grid_search', cv_folds=5, n_iter=10):
        """
        Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to tune
            tuning_method: 'grid_search' or 'random_search'
            cv_folds: Number of cross-validation folds
            n_iter: Number of iterations for RandomizedSearchCV
            
        Returns:
            Dictionary with best model and results
        """
        # Define parameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            },
            'AdaBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0, 2.0]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        if XGBOOST_AVAILABLE:
            param_grids['XGBoost'] = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        if model_type not in param_grids:
            raise ValueError(f"Hyperparameter tuning not implemented for {model_type}")
        
        # Create base model
        base_model = self._create_model(model_type, {}, y)
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Perform hyperparameter search
        if tuning_method == 'grid_search':
            search = GridSearchCV(
                base_model,
                param_grids[model_type],
                cv=cv,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
        else:  # random_search
            search = RandomizedSearchCV(
                base_model,
                param_grids[model_type],
                n_iter=n_iter,
                cv=cv,
                scoring='f1_weighted',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        
        # Fit the search
        search.fit(X, y)
        
        return {
            'best_model': search.best_estimator_,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
            'search_object': search
        }
