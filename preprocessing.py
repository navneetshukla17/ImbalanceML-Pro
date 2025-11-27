import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Enhanced data preprocessing utilities for machine learning pipelines."""
    
    def __init__(self, 
                 scaling_method='standard', 
                 categorical_encoding='label', 
                 missing_value_strategy='median',
                 categorical_missing_strategy='most_frequent'):
        """
        Initialize preprocessing components.
        
        Args:
            scaling_method: 'standard', 'minmax', 'robust', 'normalize'
            categorical_encoding: 'label', 'onehot', 'ordinal'
            missing_value_strategy: 'mean', 'median', 'most_frequent', 'knn'
            categorical_missing_strategy: 'most_frequent', 'constant'
        """
        self.scaling_method = scaling_method
        self.categorical_encoding = categorical_encoding
        self.missing_value_strategy = missing_value_strategy
        self.categorical_missing_strategy = categorical_missing_strategy
        
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'normalize': Normalizer()
        }
        self.scaler = self.scalers.get(scaling_method, StandardScaler())
        
        # Initialize encoders - maintain separate encoders per column
        self.label_encoders = {}  # Per-column label encoders
        self.onehot_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        # Initialize imputers
        self.imputers = {
            'mean': SimpleImputer(strategy='mean'),
            'median': SimpleImputer(strategy='median'),
            'most_frequent': SimpleImputer(strategy='most_frequent'),
            'knn': KNNImputer(n_neighbors=5)
        }
        self.imputer = self.imputers.get(missing_value_strategy, SimpleImputer(strategy='median'))
        if categorical_missing_strategy == 'constant':
            self.categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        else:
            self.categorical_imputer = SimpleImputer(strategy=categorical_missing_strategy)
        
        self.fitted = False
        self.feature_names_out = None
        self.categorical_columns = None
        self.numeric_columns = None
        
    def preprocess_features(self, X, fit=True):
        """
        Enhanced preprocessing with configurable options.
        
        Args:
            X: Feature matrix
            fit: Whether to fit the preprocessors
            
        Returns:
            Preprocessed feature matrix
        """
        X_processed = X.copy()
        
        # Identify numeric and categorical columns
        if fit:
            self.numeric_columns = X_processed.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_columns = X_processed.select_dtypes(include=[object, 'category']).columns.tolist()
        
        # Handle missing values for numeric columns
        if len(self.numeric_columns) > 0:
            if fit:
                X_processed[self.numeric_columns] = self.imputer.fit_transform(X_processed[self.numeric_columns])
            else:
                X_processed[self.numeric_columns] = self.imputer.transform(X_processed[self.numeric_columns])
        
        # Handle missing values for categorical columns
        if len(self.categorical_columns) > 0:
            if fit:
                X_processed[self.categorical_columns] = self.categorical_imputer.fit_transform(X_processed[self.categorical_columns])
            else:
                X_processed[self.categorical_columns] = self.categorical_imputer.transform(X_processed[self.categorical_columns])
        
        # Encode categorical variables based on method
        if len(self.categorical_columns) > 0:
            X_processed = self._encode_categorical(X_processed, fit)
        
        # Scale numeric features
        if len(self.numeric_columns) > 0:
            if fit:
                X_processed[self.numeric_columns] = self.scaler.fit_transform(X_processed[self.numeric_columns])
                self.fitted = True
            else:
                if self.fitted:
                    X_processed[self.numeric_columns] = self.scaler.transform(X_processed[self.numeric_columns])
        
        return X_processed
    
    def _encode_categorical(self, X, fit=True):
        """
        Encode categorical variables using the specified method.
        
        Args:
            X: Feature matrix
            fit: Whether to fit the encoders
            
        Returns:
            Encoded feature matrix
        """
        X_encoded = X.copy()
        
        if self.categorical_encoding == 'label':
            # Label encoding - use separate encoder for each column
            for col in self.categorical_columns:
                if fit:
                    # Create new encoder for this column
                    self.label_encoders[col] = LabelEncoder()
                    X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col].astype(str))
                else:
                    # Use existing encoder for this column
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_values = set(X_encoded[col].astype(str))
                        known_values = set(self.label_encoders[col].classes_)
                        
                        if unique_values - known_values:
                            most_frequent = self.label_encoders[col].classes_[0]
                            X_encoded[col] = X_encoded[col].astype(str).replace(
                                list(unique_values - known_values), most_frequent
                            )
                        
                        X_encoded[col] = self.label_encoders[col].transform(X_encoded[col].astype(str))
                    else:
                        # If encoder doesn't exist, fall back to creating one
                        self.label_encoders[col] = LabelEncoder()
                        X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col].astype(str))
        
        elif self.categorical_encoding == 'onehot':
            # One-hot encoding
            if fit:
                encoded_features = self.onehot_encoder.fit_transform(X_encoded[self.categorical_columns])
                feature_names = self.onehot_encoder.get_feature_names_out(self.categorical_columns)
                
                # Create DataFrame with one-hot encoded features
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=X_encoded.index)
                
                # Drop original categorical columns and add encoded ones
                X_encoded = X_encoded.drop(columns=self.categorical_columns)
                X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
                
                self.feature_names_out = X_encoded.columns.tolist()
            else:
                encoded_features = self.onehot_encoder.transform(X_encoded[self.categorical_columns])
                feature_names = self.onehot_encoder.get_feature_names_out(self.categorical_columns)
                
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=X_encoded.index)
                X_encoded = X_encoded.drop(columns=self.categorical_columns)
                X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
        
        elif self.categorical_encoding == 'ordinal':
            # Ordinal encoding
            if fit:
                X_encoded[self.categorical_columns] = self.ordinal_encoder.fit_transform(X_encoded[self.categorical_columns])
            else:
                X_encoded[self.categorical_columns] = self.ordinal_encoder.transform(X_encoded[self.categorical_columns])
        
        return X_encoded
    
    def handle_imbalance(self, X, y, method='smote', random_state=42):
        """
        Apply imbalance handling techniques.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Imbalance handling method
            random_state: Random state for reproducibility
            
        Returns:
            Resampled X and y
        """
        if method.lower() == 'none':
            return X, y
        
        elif method.lower() == 'smote':
            try:
                smote = SMOTE(random_state=random_state)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                return X_resampled, y_resampled
            except Exception as e:
                print(f"SMOTE failed: {e}. Using random oversampling instead.")
                return self.handle_imbalance(X, y, 'random oversampling', random_state)
        
        elif method.lower() == 'random oversampling':
            ros = RandomOverSampler(random_state=random_state)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            return X_resampled, y_resampled
        
        elif method.lower() == 'random undersampling':
            rus = RandomUnderSampler(random_state=random_state)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            return X_resampled, y_resampled
        
        else:
            return X, y
    
    def get_class_weights(self, y, method='balanced'):
        """
        Calculate class weights for imbalanced datasets.
        
        Args:
            y: Target vector
            method: Method for calculating weights
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        if method == 'balanced':
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            return dict(zip(classes, weights))
        else:
            return None
    
    def encode_target(self, y, fit=True):
        """
        Encode target variable if it's categorical.
        
        Args:
            y: Target vector
            fit: Whether to fit the encoder
            
        Returns:
            Encoded target vector
        """
        if y.dtype == 'object' or y.dtype.name == 'category':
            if fit:
                # Use a dedicated target encoder
                if not hasattr(self, 'target_encoder'):
                    self.target_encoder = LabelEncoder()
                return self.target_encoder.fit_transform(y)
            else:
                if hasattr(self, 'target_encoder'):
                    return self.target_encoder.transform(y)
                else:
                    # Fallback - create encoder
                    self.target_encoder = LabelEncoder()
                    return self.target_encoder.fit_transform(y)
        return y
    
    def get_feature_info(self, X):
        """
        Get information about features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with feature information
        """
        info = {
            'shape': X.shape,
            'numeric_columns': X.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': X.select_dtypes(include=[object]).columns.tolist(),
            'missing_values': X.isnull().sum().to_dict(),
            'dtypes': X.dtypes.to_dict()
        }
        return info
