import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DatasetManager:
    """Manager for built-in datasets and synthetic data generation."""
    
    def __init__(self):
        self.datasets = {
            'Credit Card Fraud': self._create_credit_card_fraud_dataset,
            'Customer Churn': self._create_customer_churn_dataset,
            'Email Spam Detection': self._create_email_spam_dataset,
            'Medical Diagnosis': self._create_medical_diagnosis_dataset,
            'Synthetic Imbalanced': self._create_synthetic_imbalanced_dataset
        }
    
    def get_available_datasets(self):
        """Get list of available datasets."""
        return list(self.datasets.keys())
    
    def load_dataset(self, dataset_name):
        """
        Load a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Tuple of (DataFrame, target_column_name)
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not available")
        
        return self.datasets[dataset_name]()
    
    def _create_credit_card_fraud_dataset(self):
        """Create a synthetic credit card fraud dataset."""
        np.random.seed(42)
        n_samples = 5000
        
        # Generate features
        data = {
            'transaction_amount': np.random.exponential(50, n_samples),
            'account_age_days': np.random.normal(365, 200, n_samples),
            'num_transactions_today': np.random.poisson(3, n_samples),
            'time_since_last_transaction': np.random.exponential(2, n_samples),
            'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'online', 'retail'], n_samples),
            'transaction_hour': np.random.randint(0, 24, n_samples),
            'is_weekend': np.random.choice([0, 1], n_samples, p=[5/7, 2/7]),
            'previous_failed_transactions': np.random.poisson(0.5, n_samples),
            'account_balance': np.random.lognormal(8, 1, n_samples),
            'credit_score': np.random.normal(650, 100, n_samples)
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Ensure positive values where needed
        df['account_age_days'] = np.abs(df['account_age_days'])
        df['account_balance'] = np.abs(df['account_balance'])
        df['credit_score'] = np.clip(df['credit_score'], 300, 850)
        
        # Create target variable (imbalanced - 3% fraud)
        fraud_probability = (
            0.01 +  # Base probability
            0.02 * (df['transaction_amount'] > 200) +  # High amount
            0.01 * (df['transaction_hour'] < 6) +  # Late night
            0.015 * (df['previous_failed_transactions'] > 2) +  # Previous failures
            0.01 * (df['num_transactions_today'] > 10)  # Many transactions
        )
        
        df['is_fraud'] = np.random.binomial(1, fraud_probability)
        
        return df, 'is_fraud'
    
    def _create_customer_churn_dataset(self):
        """Create a synthetic customer churn dataset."""
        np.random.seed(42)
        n_samples = 3000
        
        # Generate features
        data = {
            'customer_age': np.random.normal(40, 15, n_samples),
            'tenure_months': np.random.exponential(20, n_samples),
            'monthly_charges': np.random.normal(65, 20, n_samples),
            'total_charges': np.random.normal(1500, 1000, n_samples),
            'contract_type': np.random.choice(['month-to-month', 'one_year', 'two_year'], n_samples, p=[0.5, 0.3, 0.2]),
            'payment_method': np.random.choice(['electronic_check', 'mailed_check', 'bank_transfer', 'credit_card'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
            'online_security': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'tech_support': np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]),
            'num_support_calls': np.random.poisson(2, n_samples),
            'satisfaction_score': np.random.randint(1, 6, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples)
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Ensure positive values
        df['customer_age'] = np.clip(df['customer_age'], 18, 90)
        df['tenure_months'] = np.abs(df['tenure_months'])
        df['monthly_charges'] = np.abs(df['monthly_charges'])
        df['total_charges'] = np.abs(df['total_charges'])
        
        # Create target variable (imbalanced - 25% churn)
        churn_probability = (
            0.1 +  # Base probability
            0.2 * (df['contract_type'] == 'month-to-month') +
            0.1 * (df['satisfaction_score'] <= 2) +
            0.05 * (df['num_support_calls'] > 5) +
            0.1 * (df['tenure_months'] < 6) +
            0.05 * (df['monthly_charges'] > 80)
        )
        
        df['churn'] = np.random.binomial(1, np.clip(churn_probability, 0, 1))
        
        return df, 'churn'
    
    def _create_email_spam_dataset(self):
        """Create a synthetic email spam dataset."""
        np.random.seed(42)
        n_samples = 4000
        
        # Generate features
        data = {
            'email_length': np.random.lognormal(6, 1, n_samples),
            'num_exclamation': np.random.poisson(1, n_samples),
            'num_caps_words': np.random.poisson(3, n_samples),
            'num_links': np.random.poisson(0.5, n_samples),
            'num_images': np.random.poisson(0.3, n_samples),
            'sender_reputation': np.random.normal(0.7, 0.2, n_samples),
            'time_of_day': np.random.randint(0, 24, n_samples),
            'has_attachment': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'subject_length': np.random.normal(25, 10, n_samples),
            'num_recipients': np.random.poisson(1.5, n_samples),
            'domain_age_days': np.random.exponential(500, n_samples),
            'sender_frequency': np.random.poisson(2, n_samples)
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Ensure positive values where needed
        df['email_length'] = np.abs(df['email_length'])
        df['sender_reputation'] = np.clip(df['sender_reputation'], 0, 1)
        df['subject_length'] = np.abs(df['subject_length'])
        df['domain_age_days'] = np.abs(df['domain_age_days'])
        
        # Create target variable (imbalanced - 15% spam)
        spam_probability = (
            0.05 +  # Base probability
            0.3 * (df['num_exclamation'] > 3) +
            0.2 * (df['num_caps_words'] > 10) +
            0.15 * (df['num_links'] > 3) +
            0.1 * (df['sender_reputation'] < 0.3) +
            0.1 * (df['domain_age_days'] < 30)
        )
        
        df['is_spam'] = np.random.binomial(1, np.clip(spam_probability, 0, 1))
        
        return df, 'is_spam'
    
    def _create_medical_diagnosis_dataset(self):
        """Create a synthetic medical diagnosis dataset."""
        np.random.seed(42)
        n_samples = 2500
        
        # Generate features
        data = {
            'age': np.random.normal(50, 20, n_samples),
            'bmi': np.random.normal(25, 5, n_samples),
            'blood_pressure_systolic': np.random.normal(120, 20, n_samples),
            'blood_pressure_diastolic': np.random.normal(80, 15, n_samples),
            'cholesterol': np.random.normal(200, 40, n_samples),
            'glucose': np.random.normal(100, 30, n_samples),
            'smoking_status': np.random.choice(['never', 'former', 'current'], n_samples, p=[0.5, 0.3, 0.2]),
            'family_history': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'exercise_hours_week': np.random.exponential(3, n_samples),
            'alcohol_consumption': np.random.choice(['none', 'light', 'moderate', 'heavy'], n_samples, p=[0.3, 0.4, 0.25, 0.05]),
            'stress_level': np.random.randint(1, 11, n_samples),
            'sleep_hours': np.random.normal(7, 1.5, n_samples)
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Ensure realistic ranges
        df['age'] = np.clip(df['age'], 18, 100)
        df['bmi'] = np.clip(df['bmi'], 15, 50)
        df['blood_pressure_systolic'] = np.clip(df['blood_pressure_systolic'], 80, 200)
        df['blood_pressure_diastolic'] = np.clip(df['blood_pressure_diastolic'], 50, 130)
        df['cholesterol'] = np.clip(df['cholesterol'], 100, 400)
        df['glucose'] = np.clip(df['glucose'], 60, 300)
        df['exercise_hours_week'] = np.clip(df['exercise_hours_week'], 0, 20)
        df['sleep_hours'] = np.clip(df['sleep_hours'], 3, 12)
        
        # Create target variable (imbalanced - 20% positive diagnosis)
        disease_probability = (
            0.05 +  # Base probability
            0.2 * (df['age'] > 60) +
            0.15 * (df['bmi'] > 30) +
            0.1 * (df['blood_pressure_systolic'] > 140) +
            0.1 * (df['cholesterol'] > 240) +
            0.15 * (df['smoking_status'] == 'current') +
            0.1 * (df['family_history'] == 1) +
            0.05 * (df['stress_level'] > 7)
        )
        
        df['has_disease'] = np.random.binomial(1, np.clip(disease_probability, 0, 1))
        
        return df, 'has_disease'
    
    def _create_synthetic_imbalanced_dataset(self):
        """Create a synthetic imbalanced dataset using sklearn."""
        X, y = make_classification(
            n_samples=3000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=1,
            weights=[0.9, 0.1],  # 10% minority class
            flip_y=0.01,
            random_state=42
        )
        
        # Create feature names
        feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df, 'target'
    
    def get_dataset_info(self, dataset_name):
        """
        Get information about a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'Credit Card Fraud': {
                'description': 'Synthetic credit card fraud detection dataset',
                'samples': 5000,
                'features': 10,
                'target': 'is_fraud',
                'imbalance_ratio': '97:3 (Normal:Fraud)',
                'use_case': 'Financial fraud detection'
            },
            'Customer Churn': {
                'description': 'Synthetic customer churn prediction dataset',
                'samples': 3000,
                'features': 12,
                'target': 'churn',
                'imbalance_ratio': '75:25 (Stay:Churn)',
                'use_case': 'Customer retention'
            },
            'Email Spam Detection': {
                'description': 'Synthetic email spam classification dataset',
                'samples': 4000,
                'features': 12,
                'target': 'is_spam',
                'imbalance_ratio': '85:15 (Ham:Spam)',
                'use_case': 'Email filtering'
            },
            'Medical Diagnosis': {
                'description': 'Synthetic medical diagnosis dataset',
                'samples': 2500,
                'features': 12,
                'target': 'has_disease',
                'imbalance_ratio': '80:20 (Healthy:Disease)',
                'use_case': 'Medical screening'
            },
            'Synthetic Imbalanced': {
                'description': 'Synthetic imbalanced classification dataset',
                'samples': 3000,
                'features': 20,
                'target': 'target',
                'imbalance_ratio': '90:10 (Majority:Minority)',
                'use_case': 'General imbalanced classification'
            }
        }
        
        return info.get(dataset_name, {})
